import argparse, os, glob
import signal
import pandas as pd
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
import imageio

import time

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from torch import nn
from eval_utils import populate_queues, log_to_tensorboard
from collections import deque
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.modules.attention import CrossAttention
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from torch.profiler import profile, record_function, ProfilerActivity


@contextmanager
def skip_init():
    init_funcs = {
        'kaiming_normal_': torch.nn.init.kaiming_normal_,
        'uniform_': torch.nn.init.uniform_,
        'normal_': torch.nn.init.normal_,
        'constant_': torch.nn.init.constant_,
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'zeros_': torch.nn.init.zeros_,
        'ones_': torch.nn.init.ones_,
    }
    def no_op(*args, **kwargs): return
    for name in init_funcs:
        setattr(torch.nn.init, name, no_op)
    
    try:
        yield
    finally:
        for name, func in init_funcs.items():
            setattr(torch.nn.init, name, func)

def load_checkpoint_to_memory(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    return sd


def get_cross_attention_modules(model: nn.Module) -> list[CrossAttention]:
    cached = getattr(model, "_cross_attention_modules", None)
    if cached is None:
        cached = [
            module for module in model.model.modules()
            if isinstance(module, CrossAttention)
        ]
        model._cross_attention_modules = cached
    return cached


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): The model whose device is to be inferred.

    Returns:
        torch.device: The device of the model's parameters.
    """
    return next(iter(module.parameters())).device


def write_video(video_path: str, stacked_frames: list, fps: int) -> None:
    """Save a list of frames to a video file.

    Args:
        video_path (str): Output path for the video.
        stacked_frames (list): List of image frames.
        fps (int): Frames per second for the video.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "pkg_resources is deprecated as an API",
            category=DeprecationWarning,
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    """Return sorted list of files in a directory matching specified postfixes.

    Args:
        data_dir (str): Directory path to search in.
        postfixes (list[str]): List of file extensions to match.

    Returns:
        list[str]: Sorted list of file paths.
    """
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(
    model: nn.Module, ckpt: str, map_location: str = "cpu"
) -> nn.Module:
    if isinstance(ckpt, str):
        state_dict = torch.load(ckpt, map_location=map_location)
    else:
        state_dict = ckpt
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        new_pl_sd = OrderedDict()
        for key in state_dict["module"].keys():
            new_pl_sd[key[16:]] = state_dict["module"][key]
        model.load_state_dict(new_pl_sd)
    print(">>> model checkpoint loaded.")
    return model

def load_unet_ckpt(au, su , ckpt, map_location='cpu'):
    state_dict = torch.load(ckpt, map_location=map_location)['state_dict']
    au_dick = OrderedDict()
    su_dick = OrderedDict()
    for k, v in state_dict.items():
        if "action_unet" in k:
            k = k.replace("model.diffusion_model.action_unet.", "")
            au_dick[k] = v
        elif "state_unet" in k:
            k = k.replace("model.diffusion_model.state_unet.", "")
            su_dick[k] = v


    au_msg = au.load_state_dict(au_dick, strict=True)
    su_msg = su.load_state_dict(su_dick, strict=True)
    print(f"Action UNet load message: {au_msg}")
    print(f"State UNet load message: {su_msg}")
    return au, su



def is_inferenced(save_dir: str, filename: str) -> bool:
    video_file = os.path.join(
        save_dir, "samples_separate", f"{filename[:-4]}_sample0.mp4"
    )
    return os.path.exists(video_file)


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(
        filename, grid, fps=fps, video_codec="h264", options={"crf": "10"}
    )


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    rel_video_fp = os.path.join(sample["data_dir"], str(sample["videoid"]) + ".png")
    full_image_fp = os.path.join(data_dir, "images", rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    rel_transition_fp = os.path.join(sample["data_dir"], str(sample["videoid"]) + ".h5")
    full_transition_fp = os.path.join(data_dir, "transitions", rel_transition_fp)
    return full_transition_fp


def prepare_init_input(
    start_idx: int,
    init_frame_path: str,
    transition_dict: dict[str, torch.Tensor],
    frame_stride: int,
    wma_data,
    video_length: int = 16,
    n_obs_steps: int = 2,
) -> dict[str, Tensor]:
    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert("RGB")
    init_frame = (
        torch.tensor(np.array(init_frame)).unsqueeze(0).permute(3, 0, 1, 2).float()
    )

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict["observation.state"][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict["observation.state"][state_indices, :]

    actions = transition_dict["action"][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        "action": actions,
        "observation.state": states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict["action_type"],
        transition_dict["state_type"],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        "observation.image": init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    b, c, t, h, w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    z = model.encode_first_stage(x)
    z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
    return z


def preprocess_observation(
    model, observations: dict[str, np.ndarray]
) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # Map to expected inputs for the policy
    return_observations = {}

    if isinstance(observations["pixels"], dict):
        imgs = {
            f"observation.images.{key}": img
            for key, img in observations["pixels"].items()
        }
    else:
        imgs = {"observation.images.top": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)

        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)

        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["agent_pos"]
    ).float()
    return_observations["observation.state"] = model.normalize_inputs(
        {"observation.state": return_observations["observation.state"].to(model.device)}
    )["observation.state"]

    return return_observations


def image_guided_synthesis_sim_mode(
    model: torch.nn.Module,
    cond_ins_emb: Tensor,
    observation: dict,
    noise_shape: tuple[int, int, int, int, int],
    action_cond_step: int = 16,
    n_samples: int = 1,
    ddim_steps: int = 50,
    ddim_eta: float = 1.0,
    unconditional_guidance_scale: float = 1.0,
    fs: int | None = None,
    text_input: bool = True,
    timestep_spacing: str = "uniform",
    guidance_rescale: float = 0.0,
    sim_mode: bool = True,
    ddim_sampler: DDIMSampler = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs image-guided video generation in a simulation-style mode with optional multimodal guidance (image, state, action, text).

    Args:
        model (torch.nn.Module): The diffusion-based generative model with multimodal conditioning.
        prompts (list[str]): A list of textual prompts to guide the synthesis process.
        observation (dict): A dictionary containing observed inputs including:
            - 'observation.images.top': Tensor of shape [B, O, C, H, W] (top-down images)
            - 'observation.state': Tensor of shape [B, O, D] (state vector)
            - 'action': Tensor of shape [B, T, D] (action sequence)
        noise_shape (tuple[int, int, int, int, int]): Shape of the latent variable to generate,
            typically (B, C, T, H, W).
        action_cond_step (int): Number of time steps where action conditioning is applied. Default is 16.
        n_samples (int): Number of samples to generate (unused here, always generates 1). Default is 1.
        ddim_steps (int): Number of DDIM sampling steps. Default is 50.
        ddim_eta (float): DDIM eta parameter controlling the stochasticity. Default is 1.0.
        unconditional_guidance_scale (float): Scale for classifier-free guidance. If 1.0, guidance is off.
        fs (int | None): Frame index to condition on, broadcasted across the batch if specified. Default is None.
        text_input (bool): Whether to use text prompt as conditioning. If False, uses empty strings. Default is True.
        timestep_spacing (str): Timestep sampling method in DDIM sampler. Typically "uniform" or "linspace".
        guidance_rescale (float): Guidance rescaling factor to mitigate overexposure from classifier-free guidance.
        sim_mode (bool): Whether to perform world-model interaction or decision-making using the world-model.
        **kwargs: Additional arguments passed to the DDIM sampler.

    Returns:
        batch_variants (torch.Tensor): Predicted pixel-space video frames [B, C, T, H, W].
        actions (torch.Tensor): Predicted action sequences [B, T, D] from diffusion decoding.
        states (torch.Tensor): Predicted state sequences [B, T, D] from diffusion decoding.
    """
    batch_size = noise_shape[0]

    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation["observation.images.top"].permute(0, 2, 1, 3, 4)
    cond_img = rearrange(img, "b o c h w -> (b o) c h w")[-1:]
    cond_img_emb = model.embedder(cond_img)
    cond_img_emb = model.image_proj_model(cond_img_emb)

    if model.model.conditioning_key == "hybrid":
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
        img_cat_cond = z[:, :, -1:, :, :]
        img_cat_cond = repeat(
            img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=noise_shape[2]
        )
        cond = {"c_concat": [img_cat_cond]}



    cond_state_emb = model.state_projector(observation["observation.state"])
    cond_state_emb = cond_state_emb + model.agent_state_pos_emb


    def _get_zero_action_emb(model, batch_size, dtype):
        cache = getattr(model, "_zero_action_emb_cache", None)
        action_len = model.agent_action_pos_emb.shape[1]
        action_dim = model.agent_action_pos_emb.shape[2]
        key = (batch_size, action_len, action_dim, dtype, model.device)
        if cache is not None and cache.get("key") == key:
            return cache["value"]
        zero = torch.zeros(
            (batch_size, action_len, action_dim), dtype=dtype, device=model.device
        )
        model._zero_action_emb_cache = {"key": key, "value": zero}
        return zero
    

    if not sim_mode:
        cond_action_emb = _get_zero_action_emb(model, batch_size, cond_state_emb.dtype)
    else:
        cond_action_emb = model.action_projector(observation["action"])
        cond_action_emb = cond_action_emb + model.agent_action_pos_emb

    cond["c_crossattn"] = {
        "cond_state_emb": cond_state_emb,
        "cond_action_emb": cond_action_emb,
        "cond_ins_emb": cond_ins_emb,
        "cond_img_emb": cond_img_emb,
    }
    cond["c_crossattn_action"] = [
        observation["observation.images.top"][
            :, :, -model.n_obs_steps_acting :
        ].contiguous(),
        observation["observation.state"][:, -model.n_obs_steps_acting :].contiguous(),
        sim_mode,
        False,
    ]
    if dist.is_initialized():
        action_img = cond["c_crossattn_action"][0].contiguous()
        action_state = cond["c_crossattn_action"][1].contiguous()
        packed = torch.cat(
            [action_img.float().reshape(-1), action_state.float().reshape(-1)], dim=0
        )
        dist.isend(packed, dst=1)


    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        for module in get_cross_attention_modules(model):
            module.use_kv_cache(use_cache=True)
        samples, actions, states, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            sim_mode=sim_mode,
            **kwargs,
        )

        # Reconstruct from latent to pixel space
        if sim_mode:
            batch_variants = model.decode_first_stage(samples)
        else:
            batch_variants = None
    for module in get_cross_attention_modules(model):
        module.clear_kv_cache()
    return batch_variants, actions, states



def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int) -> None:
    wup_tensor = torch.empty(size=(16,16), device="cuda:"+str(gpu_no))
    h1 = dist.isend(wup_tensor, dst=1)
    h2 = dist.irecv(wup_tensor, src=1)

    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_future = executor.submit(load_checkpoint_to_memory, args.ckpt_path)

    config = OmegaConf.load(args.config)
    config["model"]["params"]["wma_config"]["params"]["use_checkpoint"] = False
    with skip_init():
        model = instantiate_from_config(config.model)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    loaded_state_dict = checkpoint_future.result() 
    executor.shutdown()
    model = load_model_checkpoint(model, loaded_state_dict)
    alphas_cumprod = model.alphas_cumprod
    alphas_cumprod_prev = model.alphas_cumprod_prev
    sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod
    sqrt_alphas_cumprod = model.sqrt_alphas_cumprod
    model = model.to(torch.float16)
    model.alphas_cumprod = alphas_cumprod
    model.alphas_cumprod_prev = alphas_cumprod_prev
    model.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
    model.sqrt_alphas_cumprod = sqrt_alphas_cumprod
    print(f"model.alphas_cumprod dtype: {model.alphas_cumprod.dtype}")
    model.eval()
    print(f">>> Load pre-trained model ...")

    # Build unnomalizer
    logging.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    print(">>> Dataset is successfully loaded ...")

    model = model.cuda(gpu_no)
    device = get_device_from_parameters(model)

    # Run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), (
        "Error: image size [h,w] should be multiples of 16!"
    )
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f">>> Generate {n_frames} frames under each generation ...")
    noise_shape = [args.bs, channels, n_frames, h, w]


    # h1.wait()
    # h2.wait()
    ddim_sampler = DDIMSampler(model)
    ddim_sampler.make_schedule(ddim_num_steps=args.ddim_steps,
                           ddim_discretize=args.timestep_spacing,
                           ddim_eta=args.ddim_eta,
                           verbose=False)
    # Start inference
    for idx in range(0, len(df)):
        sample = df.iloc[idx]
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            cond_ins_emb_full = model.get_learned_conditioning(sample["instruction"])
            cond_ins_emb_empty = model.get_learned_conditioning([""] * args.bs)
        # Got initial frame path
        init_frame_path = get_init_frame_path(args.prompt_dir, sample)
        ori_fps = float(sample["fps"])

        video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
        os.makedirs(video_save_dir, exist_ok=True)

        # Load transitions to get the initial state later
        transition_path = get_transition_path(args.prompt_dir, sample)
        with h5py.File(transition_path, "r") as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]
        torch.cuda.current_stream().wait_stream(s)
        # If many, test various frequence control and world-model generation

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for fs in args.frame_stride:
                wm_video = []
                # Initialize observation queues
                cond_obs_queues = {
                    "observation.images.top": deque(maxlen=model.n_obs_steps_imagen),
                    "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                    "action": deque(maxlen=args.video_length),
                }
                # Obtain initial frame and state
                start_idx = 0
                model_input_fs = ori_fps // fs
                batch, ori_state_dim, ori_action_dim = prepare_init_input(
                    start_idx,
                    init_frame_path,
                    transition_dict,
                    fs,
                    data.test_datasets[args.dataset],
                    n_obs_steps=model.n_obs_steps_imagen,
                )
                observation = {
                    "observation.images.top": batch["observation.image"]
                    .permute(1, 0, 2, 3)[-1]
                    .unsqueeze(0),
                    "observation.state": batch["observation.state"][-1].unsqueeze(0),
                    "action": torch.zeros_like(batch["action"][-1]).unsqueeze(0),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }
                # Update observation queues
                cond_obs_queues = populate_queues(cond_obs_queues, observation)

                # Multi-round interaction with the world-model
                for itr in tqdm(range(args.n_iter)):
                    # Get observation
                    observation = {
                        "observation.images.top": torch.stack(
                            list(cond_obs_queues["observation.images.top"]), dim=1
                        ).permute(0, 2, 1, 3, 4),
                        "observation.state": torch.stack(
                            list(cond_obs_queues["observation.state"]), dim=1
                        ),
                        "action": torch.stack(list(cond_obs_queues["action"]), dim=1),
                    }
                    observation = {
                        key: observation[key].to(device, non_blocking=True)
                        for key in observation
                    }

                    # Use world-model in policy to generate action
                    print(f">>> Step {itr}: generating actions ...")
                    with autocast():
                        _, pred_actions, _ = image_guided_synthesis_sim_mode(
                            model,
                            cond_ins_emb_full,
                            observation,
                            noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.unconditional_guidance_scale,
                            fs=model_input_fs,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            sim_mode=False,
                            ddim_sampler=ddim_sampler
                        )

                    # Update future actions in the observation queues
                    for idx in range(len(pred_actions[0])):
                        observation = {"action": pred_actions[0][idx : idx + 1]}
                        observation["action"][:, ori_action_dim:] = 0.0
                        cond_obs_queues = populate_queues(cond_obs_queues, observation)

                    # Collect data for interacting the world-model using the predicted actions
                    observation = {
                        "observation.images.top": torch.stack(
                            list(cond_obs_queues["observation.images.top"]), dim=1
                        ).permute(0, 2, 1, 3, 4),
                        "observation.state": torch.stack(
                            list(cond_obs_queues["observation.state"]), dim=1
                        ),
                        "action": torch.stack(list(cond_obs_queues["action"]), dim=1),
                    }
                    observation = {
                        key: observation[key].to(device, non_blocking=True)
                        for key in observation
                    }

                    # Interaction with the world-model
                    print(f">>> Step {itr}: interacting with world model ...")
                    with autocast():
                        pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                            model,
                            cond_ins_emb_empty,
                            observation,
                            noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.unconditional_guidance_scale,
                            fs=model_input_fs,
                            text_input=False,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            ddim_sampler=ddim_sampler
                        )


                    for idx in range(args.exe_steps):
                        observation = {
                            "observation.images.top": pred_videos_1[0][
                                :, idx : idx + 1
                            ].permute(1, 0, 2, 3),
                            "observation.state": torch.zeros_like(
                                pred_states[0][idx : idx + 1]
                            )
                            if args.zero_pred_state
                            else pred_states[0][idx : idx + 1],
                            "action": torch.zeros_like(pred_actions[0][-1:]),
                        }
                        observation["observation.state"][:, ori_state_dim:] = 0.0
                        cond_obs_queues = populate_queues(cond_obs_queues, observation)

                    wm_video.append(pred_videos_1[:, :, : args.exe_steps].cpu())

                full_video = torch.cat(wm_video, dim=2)
                sample_full_video_file = (
                    f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4"
                )
                save_results(full_video, sample_full_video_file, fps=args.save_fps)
        prof.export_chrome_trace(f"./profile_666.json")

@torch.no_grad()
def run_two_unet(args: argparse.Namespace, gpu_num: int, gpu_no: int) -> None:
    wup_tensor = torch.empty(size=(16,16), device="cuda:"+str(gpu_no))
    h1 = dist.irecv(wup_tensor, src=0)
    h2 = dist.isend(wup_tensor, dst=0)

    main_config = OmegaConf.load(args.config)
    wma_config = main_config.model.params.wma_config
    model_channels = wma_config.params.model_channels
    channel_mult = wma_config.params.channel_mult
    unet_head_config = main_config.model.params.wma_config.params.unet_head_config
    unet_head_config['params']['context_dims'] = [
            mult * model_channels for mult in channel_mult
        ]


    executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_future = executor.submit(load_unet_ckpt, action_unet, state_unet, args.ckpt_path, 'cpu')

    action_unet = instantiate_from_config(unet_head_config)
    state_unet = instantiate_from_config(unet_head_config)
    
    noise_scheduler_config = main_config.model.params.noise_scheduler_config
    dp_ddim_scheduler_action = instantiate_from_config(noise_scheduler_config)
    dp_ddim_scheduler_state = instantiate_from_config(noise_scheduler_config)

    device = "cuda:"+str(gpu_no)
    action_unet, state_unet = checkpoint_future.result()

    action_unet = action_unet.to(device).to(torch.float16)
    state_unet = state_unet.to(device).to(torch.float16)
    action_unet.eval()
    state_unet.eval()

    # h1.wait()
    # h2.wait()

    step = 1000 // args.ddim_steps
    timerange = list(range(999, -1, -step))

    ca_dtype = torch.float32
    context_action_1_shape = (1, 3, 2, 320, 512)
    context_action_2_shape = (1, 2, 16)
    context_action_1_numel = int(np.prod(context_action_1_shape))
    context_action_2_numel = int(np.prod(context_action_2_shape))
    with autocast(dtype=torch.float16):
        for i in range(args.n_iter):
            dp_ddim_scheduler_action.set_timesteps(len(timerange))
            if i != 0:
                ca_dtype = torch.float16
            else:
                ca_dtype = torch.float32
            packed_len = context_action_1_numel + context_action_2_numel
            packed = torch.empty((packed_len,), dtype=torch.float32, device=device)
            dist.recv(packed, src=0)
            offset = context_action_1_numel
            context_action_1 = packed[:offset].reshape(context_action_1_shape).to(ca_dtype)
            context_action_2 = packed[offset:].reshape(context_action_2_shape)

            cond = [context_action_1, context_action_2]
            cond = {'image': cond[0], 'agent_pos': cond[1]}
            cond['image'] = cond['image'].permute(0, 2, 1, 3,4)
            cond['image'] = rearrange(cond['image'], 'b t c h w -> (b t) c h w')
            cond['agent_pos'] = rearrange(cond['agent_pos'], 'b t d -> (b t) d')
            action = torch.empty((1,16,16), device=device)
            state = torch.empty((1,16,16), device=device)
            dist.recv(action, src=0)
            dist.recv(state, src=0)


            for step in timerange:
                model_output_action = action_unet(action, step, cond)
                action = dp_ddim_scheduler_action.step(
                    model_output_action,
                    step,
                    action,
                    generator=None,
                ).prev_sample

            dist.send(action, dst=0)

            dp_ddim_scheduler_state.set_timesteps(len(timerange))
            packed_len = context_action_1_numel + context_action_2_numel
            packed = torch.empty((packed_len,), dtype=torch.float32, device=device)
            dist.recv(packed, src=0)
            offset = context_action_1_numel
            context_action_1 = packed[:offset].reshape(context_action_1_shape).to(ca_dtype)
            context_action_2 = packed[offset:].reshape(context_action_2_shape)

            cond = [context_action_1, context_action_2]
            cond = {'image': cond[0], 'agent_pos': cond[1]}
            cond['image'] = cond['image'].permute(0, 2, 1, 3,4)
            cond['image'] = rearrange(cond['image'], 'b t c h w -> (b t) c h w')
            cond['agent_pos'] = rearrange(cond['agent_pos'], 'b t d -> (b t) d')
            action = torch.empty((1,16,16), device=device)
            state = torch.empty((1,16,16), device=device)
            dist.recv(action, src=0)
            dist.recv(state, src=0)

            for step in timerange:
                model_output_state = state_unet(state, step, cond)
                state = dp_ddim_scheduler_state.step(
                    model_output_state,
                    step,
                    state,
                    generator=None,
                ).prev_sample

            dist.send(state, dst=0)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", type=str, default=None, help="Path to save the results."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the model checkpoint."
    )
    parser.add_argument("--config", type=str, help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="the name of dataset to test"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.",
    )
    parser.add_argument(
        "--bs", type=int, default=1, help="Batch size for inference. Must be 1."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Height of the generated images in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the generated images in pixels.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs="+",
        required=True,
        help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=16,
        help="Number of frames in the generated video.",
    )
    parser.add_argument(
        "--num_generation", type=int, default=1, help="seed for seed_everything"
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help="Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891).",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891).",
    )
    parser.add_argument(
        "--perframe_ae",
        action="store_true",
        default=False,
        help="Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model",
    )
    parser.add_argument(
        "--zero_pred_state",
        action="store_true",
        default=False,
        help="not using the predicted states as comparison",
    )
    parser.add_argument(
        "--save_fps", type=int, default=8, help="fps for the saving video"
    )
    return parser


def worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.backends.cuda.matmul.allow_tf32 = True

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    seed_everything(seed)


    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


    if rank == 0:
        print(f"[Rank {rank}] Starting inference process...")
        run_inference(args, world_size, rank)
    elif rank == 1:
        print(f"[Rank {rank}] Starting two unet warmup process...")
        run_two_unet(args, world_size, rank)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # seed = args.seed
    # if seed < 0:
    #     seed = random.randint(0, 2**31)
    # seed_everything(seed)
    # run_inference(args, 1, 0)

    gpu_num = min(2, torch.cuda.device_count())
    if gpu_num < 2:
        print(f"gpu_num->{gpu_num}")
        raise RuntimeError("This mode requires at least 2 GPUs")

    mp.set_start_method("spawn", force=True)

    ctx = mp.spawn(worker, nprocs=gpu_num, args=(gpu_num, args), join=False)

    def _handle_sigint(_sig, _frame):
        ctx.terminate()

    signal.signal(signal.SIGINT, _handle_sigint)
    ctx.join()
