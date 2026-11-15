"""Validate single-case preparation without importing or starting GPU inference."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_single_case', ROOT / 'tools/run_single_case.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class SingleCaseLauncherTests(unittest.TestCase):
    def test_prepare_narrows_dataset_and_keeps_historical_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / 'weights.safetensors'
            checkpoint.touch()  # This test prepares argv; it never loads a checkpoint.
            dataset = 'unitree_z1_dual_arm_stackbox'
            (path / (dataset + '.csv')).write_text('videoid,instruction\ncase1,test\ncase2,test2\n')
            args = SimpleNamespace(checkpoint=checkpoint, prompt_dir=path, dataset=dataset,
                                   case_index=1, config=ROOT / 'project/configs/inference/world_model_interaction.yaml',
                                   output_dir=path / 'output', n_iter=None)
            original = Path(args.config).read_bytes()
            command, env = launcher.prepare(args)
            def option(key):
                return command[command.index(key) + 1]
            generated = yaml.safe_load(Path(option('--config')).read_text())
            self.assertEqual(generated['data']['params']['dataset_and_weights'], {dataset: 1.0})
            self.assertEqual(generated['data']['params']['test']['params']['data_dir'], str(path.resolve()))
            self.assertEqual(option('--case_index'), '1')
            self.assertEqual(option('--n_iter'), '7')
            self.assertEqual(option('--ddim_steps'), '50')
            self.assertEqual(option('--height'), '320')
            self.assertEqual(option('--width'), '512')
            self.assertIn(str(ROOT / 'project/src'), env['PYTHONPATH'])
            self.assertEqual(Path(args.config).read_bytes(), original)
            args.case_index = 2
            with self.assertRaises(ValueError):
                launcher.prepare(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
