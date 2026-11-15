import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from functools import partial, lru_cache

from ops.my_gelu_op import gelu_and_mul

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from unifolm_wma.utils.common import (
    checkpoint,
    exists,
    default,
)
from unifolm_wma.utils.basics import zero_module



@lru_cache(maxsize=8)
def get_attn_mask_aa(b, l1, l2, heads, dtype, block_size=16, device='cuda'):
    num_token = l2 // block_size
    start_positions = ((torch.arange(b, device=device) % block_size) + 1) * num_token
    col_indices = torch.arange(l2, device=device)
    mask_2d = col_indices.unsqueeze(0) >= start_positions.unsqueeze(1)
    mask = mask_2d.unsqueeze(1).expand(b, l1, l2)
    attn_mask = torch.zeros_like(mask, dtype=torch.float)
    attn_mask[mask] = float('-inf')
    attn_mask = repeat(attn_mask, "b l1 l2 -> b h l1 l2", h=heads).contiguous().to(dtype)
    return attn_mask

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -self.max_relative_position,
                                           self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        relative_position=False,
        temporal_length=None,
        video_length=None,
        agent_state_context_len=2,
        agent_action_context_len=16,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        agent_state_cross_attention_scale=1.0,
        agent_action_cross_attention_scale=1.0,
        cross_attention_scale_learnable=False,
        text_context_len=77,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

        self.relative_position = relative_position
        if self.relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
            self.relative_position_v = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward
        self._cache_enabled = True
        self._kv_cache = {}
        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.agent_state_cross_attention_scale = agent_state_cross_attention_scale
        self.agent_action_cross_attention_scale = agent_action_cross_attention_scale
        self.text_context_len = text_context_len
        self.agent_state_context_len = agent_state_context_len
        self.agent_action_context_len = agent_action_context_len
        self.cross_attention_scale_learnable = cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_aa = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_aa = nn.Linear(context_dim, inner_dim, bias=False)
            if cross_attention_scale_learnable:
                self.register_parameter("alpha_ctx", nn.Parameter(torch.tensor(0.0)))
                self.register_parameter("alpha_cas", nn.Parameter(torch.tensor(0.0)))
                self.register_parameter("alpha_caa", nn.Parameter(torch.tensor(0.0)))

    def use_kv_cache(self, use_cache=False):
        self._cache_enabled = use_cache

    def clear_kv_cache(self):
        self._kv_cache.clear()

    def forward(self, x, context=None, mask=None):
        use_cache = self._cache_enabled and context
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        if use_cache and self._kv_cache:
            k = self._kv_cache["k"]
            v = self._kv_cache["v"]
        else:
            k = self.to_k(context)
            v = self.to_v(context)
            if use_cache:
                self._kv_cache["k"] = k
                self._kv_cache["v"] = v

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q, k, v))
        out = xformers.ops.memory_efficient_attention(
            q,
            k,
            v,
        )
        out = rearrange(out, "b n h d -> b n (h d)", h=h)
        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        spatial_self_attn = context is None
        k, v, out = None, None, None
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        q = self.to_q(x)
        context = default(context, x)
        use_cache = self._cache_enabled and not spatial_self_attn
        q = rearrange(q, "b n (h d) -> b n h d", h=self.heads).contiguous()

        if self.image_cross_attention and not spatial_self_attn:
            if use_cache and self._kv_cache:
                k = self._kv_cache["k"]
                v = self._kv_cache["v"]
                k_ip = self._kv_cache["k_ip"]
                v_ip = self._kv_cache["v_ip"]
                k_as = self._kv_cache["k_as"]
                v_as = self._kv_cache["v_as"]
                k_aa = self._kv_cache["k_aa"]
                v_aa = self._kv_cache["v_aa"]
            else:
                context_agent_state = context[:, : self.agent_state_context_len, :]
                context_agent_action = context[
                    :,
                    self.agent_state_context_len : self.agent_state_context_len
                    + self.agent_action_context_len,
                    :,
                ]
                context_ins = context[
                    :,
                    self.agent_state_context_len
                    + self.agent_action_context_len : self.agent_state_context_len
                    + self.agent_action_context_len
                    + self.text_context_len,
                    :,
                ]
                context_image = context[
                    :,
                    self.agent_state_context_len
                    + self.agent_action_context_len
                    + self.text_context_len :,
                    :,
                ]
                k = self.to_k(context_ins)
                v = self.to_v(context_ins)
                k, v = map(
                    lambda t: rearrange(
                        t, "b n (h d) -> b n h d", h=self.heads
                    ).contiguous(),
                    (k, v),
                )
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
                k_ip, v_ip = map(
                    lambda t: rearrange(
                        t, "b n (h d) -> b n h d", h=self.heads
                    ).contiguous(),
                    (k_ip, v_ip),
                )
                k_as = self.to_k_as(context_agent_state)
                v_as = self.to_v_as(context_agent_state)
                k_as, v_as = map(
                    lambda t: rearrange(
                        t, "b n (h d) -> b n h d", h=self.heads
                    ).contiguous(),
                    (k_as, v_as),
                )
                k_aa = self.to_k_aa(context_agent_action)
                v_aa = self.to_v_aa(context_agent_action)
                k_aa, v_aa = map(
                    lambda t: rearrange(
                        t, "b n (h d) -> b n h d", h=self.heads
                    ).contiguous(),
                    (k_aa, v_aa),
                )
                if use_cache:
                    self._kv_cache["k"] = k
                    self._kv_cache["v"] = v
                    self._kv_cache["k_ip"] = k_ip
                    self._kv_cache["v_ip"] = v_ip
                    self._kv_cache["k_as"] = k_as
                    self._kv_cache["v_as"] = v_as
                    self._kv_cache["k_aa"] = k_aa
                    self._kv_cache["v_aa"] = v_aa

            attn_mask_aa = get_attn_mask_aa(
                x.shape[0], q.shape[1], k_aa.shape[1], heads=self.heads, dtype=q.dtype, block_size=16, device="cuda"
            )
        else:
            if use_cache and self._kv_cache:
                k = self._kv_cache["k"]
                v = self._kv_cache["v"]
            else:
                k = self.to_k(context)
                v = self.to_v(context)
                k, v = map(
                    lambda t: rearrange(
                        t, "b n (h d) -> b n h d", h=self.heads
                    ).contiguous(),
                    (k, v),
                )
                if use_cache:
                    self._kv_cache["k"] = k
                    self._kv_cache["v"] = v
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=None
            )
            out = rearrange(out, "b n h d -> b n (h d)", h=self.heads).contiguous()
            return self.to_out(out)

        
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=None
        )

        out_ip = xformers.ops.memory_efficient_attention(
            q, k_ip, v_ip, attn_bias=None, op=None
        )
        out_as = xformers.ops.memory_efficient_attention(
            q, k_as, v_as, attn_bias=None, op=None
        )
            
        out_aa = xformers.ops.memory_efficient_attention(
            q, k_aa, v_aa, attn_bias=attn_mask_aa, op=None
        )
        out = out + out_ip + out_as + out_aa
        out = rearrange(out, "b n h d -> b n (h d)", h=self.heads).contiguous()
        return self.to_out(out)




class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False,
                 attention_cls=None,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 image_cross_attention_scale=1.0,
                 cross_attention_scale_learnable=False,
                 text_context_len=77):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            agent_state_context_len=agent_state_context_len,
            agent_action_context_len=agent_action_context_len,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            cross_attention_scale_learnable=cross_attention_scale_learnable,
            text_context_len=text_context_len)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x, ), self.parameters(),
                              self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x),
                       context=context if self.disable_self_attn else None,
                       mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 disable_self_attn=False,
                 use_linear=False,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 cross_attention_scale_learnable=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                agent_state_context_len=agent_state_context_len,
                agent_action_context_len=agent_action_context_len,
                image_cross_attention=image_cross_attention,
                cross_attention_scale_learnable=cross_attention_scale_learnable,
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 use_linear=False,
                 only_self_att=True,
                 causal_attention=False,
                 causal_block_size=1,
                 relative_position=False,
                 temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert (temporal_length is not None)
            attention_cls = partial(CrossAttention,
                                    relative_position=True,
                                    temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention,
                                    temporal_length=temporal_length)
        if self.causal_attention:
            assert (temporal_length is not None)
            self.mask = torch.tril(
                torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim,
                                  attention_cls=attention_cls,
                                  checkpoint=use_checkpoint)
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # Slice the from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b * h * w)
        else:
            mask = None

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, mask=mask)
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h,
                          w=w).contiguous()

        return x + x_in


# class GEGLU(nn.Module):

#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2)

#     def forward(self, x):
#         x, gate = self.proj(x).chunk(2, dim=-1)
#         return x * F.gelu(gate)

class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        input_tensor = self.proj(x).to(x.dtype)
        b,t,d = input_tensor.shape[:-1] + (input_tensor.shape[-1] // 2, )
        out = torch.empty((b*t, d), dtype=x.dtype, device=x.device)
        gelu_and_mul(out, input_tensor.reshape(-1, d*2))
        return out.reshape(b,t,d)


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv,
                            'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads=self.heads,
                            qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out,
                        'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads,
                        h=h,
                        w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_
