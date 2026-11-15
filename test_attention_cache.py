"""CPU regression of the real attention methods extracted from their AST.

No CUDA extension or model is imported. PyTorch supplies real small tensors;
only the xFormers attention call and the two einops layouts are substituted by
their CPU mathematical equivalents. This does not validate the GPU kernels.
"""
import ast
import copy
import difflib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parent
RELATIVE = 'project/src/unifolm_wma/modules/attention.py'
SOURCE = ROOT / RELATIVE
ORIGINAL = ROOT / 'archived/before-maintenance' / RELATIVE


def rearrange(value, pattern, h):
    if pattern == 'b n (h d) -> b n h d':
        b, n, width = value.shape
        return value.reshape(b, n, h, width // h)
    if pattern == 'b n h d -> b n (h d)':
        b, n, heads, width = value.shape
        return value.reshape(b, n, heads * width)
    raise AssertionError('Unexpected attention layout: ' + pattern)


def cpu_attention(q, k, v, attn_bias=None, op=None):
    scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(q.shape[-1])
    if attn_bias is not None:
        scores = scores + attn_bias
    return torch.einsum('bhnm,bmhd->bnhd', scores.softmax(-1), v)


def class_node(path):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == 'CrossAttention')


def load_methods(path):
    methods = {'forward', 'efficient_forward', 'use_kv_cache', 'clear_kv_cache'}
    cls = class_node(path)
    harness = ast.ClassDef(name='AttentionHarness', bases=[], keywords=[],
                          body=[copy.deepcopy(n) for n in cls.body
                                if isinstance(n, ast.FunctionDef) and n.name in methods],
                          decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    env = {'rearrange': rearrange,
           'default': lambda value, fallback: value if value is not None else fallback,
           'xformers': SimpleNamespace(ops=SimpleNamespace(memory_efficient_attention=cpu_attention))}
    exec(compile(module, str(path), 'exec'), env)
    return env['AttentionHarness']


class Projection:
    def __init__(self, matrix):
        self.matrix = matrix
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return value @ self.matrix


def attention(path=SOURCE):
    model = load_methods(path)()
    model.heads = 2
    model.image_cross_attention = False
    model._cache_enabled = True
    model._kv_cache = {}
    model.to_q = Projection(torch.eye(4))
    model.to_k = Projection(torch.diag(torch.tensor([0.5, 1.0, 1.5, 2.0])))
    model.to_v = Projection(torch.tensor([[1., 0., .5, 0.], [0., 1., 0., .5],
                                         [.2, 0., 1., 0.], [0., .2, 0., 1.]]))
    model.to_out = lambda value: value
    return model


class AttentionCacheTests(unittest.TestCase):
    def setUp(self):
        self.query = torch.tensor([[[1., 2., -1., 0.], [.5, -2., 1., 3.]]])
        self.context = torch.tensor([[[1., 0., 2., 1.], [0., 2., 1., -1.], [2., -1., 0., 3.]]])

    def test_original_tensor_condition_reproduces_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            attention(ORIGINAL).forward(self.query, self.context)

    def test_cached_output_matches_uncached_for_both_methods(self):
        for name in ['forward', 'efficient_forward']:
            with self.subTest(method=name):
                cached, uncached = attention(), attention()
                uncached.use_kv_cache(False)
                for query in [self.query, self.query + .25]:
                    torch.testing.assert_close(getattr(cached, name)(query, self.context),
                                               getattr(uncached, name)(query, self.context))
                self.assertEqual(cached.to_k.calls, 1)
                self.assertEqual(cached.to_v.calls, 1)
                self.assertEqual(uncached.to_k.calls, 2)

    def test_clearing_between_sessions_uses_new_conditions(self):
        for name in ['forward', 'efficient_forward']:
            with self.subTest(method=name):
                model = attention()
                old = getattr(model, name)(self.query, self.context)
                model.clear_kv_cache()
                new_context = self.context + 4
                actual = getattr(model, name)(self.query, new_context)
                reference = attention()
                reference.use_kv_cache(False)
                torch.testing.assert_close(actual, getattr(reference, name)(self.query, new_context))
                self.assertFalse(torch.allclose(old, actual))
                self.assertEqual(model.to_k.calls, 2)

    def test_self_attention_never_reuses_cross_condition_cache(self):
        for name in ['forward', 'efficient_forward']:
            with self.subTest(method=name):
                model = attention()
                getattr(model, name)(self.query, self.context)
                for query in [self.query, self.query + 2]:
                    reference = attention()
                    reference.use_kv_cache(False)
                    torch.testing.assert_close(getattr(model, name)(query), getattr(reference, name)(query))
                self.assertEqual(model.to_k.calls, 3)

    def test_efficient_method_is_unchanged_from_snapshot(self):
        def method(path):
            return next(n for n in class_node(path).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'efficient_forward')
        self.assertEqual(ast.dump(method(SOURCE)), ast.dump(method(ORIGINAL)))


class MaintenanceTests(unittest.TestCase):
    def test_recorded_patches_match_actual_changes(self):
        manifest = json.loads((ROOT / 'maintenance.json').read_text())
        for item in manifest['files']:
            original = (ROOT / item['original_copy']).read_text(encoding='utf-8')
            maintained = (ROOT / item['path']).read_text(encoding='utf-8')
            expected = ''.join(difflib.unified_diff(original.splitlines(True), maintained.splitlines(True),
                                                  fromfile='a/' + item['path'], tofile='b/' + item['path']))
            self.assertEqual((ROOT / item['patch']).read_text(encoding='utf-8'), expected)

    def test_real_case_selector_preserves_original_metadata(self):
        path = ROOT / 'project/scripts/evaluation/world_model_interaction.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'select_case_frame')
        env = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), env)
        class Frame:
            def __init__(self, rows):
                self.rows = rows
                self.iloc = self
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, indexes):
                return Frame([self.rows[i] for i in indexes])
        frame = Frame(['case1', 'case2', 'case3'])
        select = env['select_case_frame']
        self.assertIs(select(frame, None), frame)
        self.assertEqual(select(frame, 1).rows, ['case2'])
        self.assertEqual(len(frame), 3)
        for invalid in [-1, 3]:
            with self.assertRaises(ValueError):
                select(frame, invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
