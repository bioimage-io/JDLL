"""Device resolution must warn before CPU fallback, without requiring a real GPU."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('jdll_devices',
    Path(__file__).resolve().parents[2] / 'main/resources/python/device_selection.py')
devices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(devices)


class DeviceSelectionTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.update = lambda **event: self.events.append(event)

    def torch(self, cuda=False, mps=False):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: mps, is_available=lambda: mps)),
            ones=MagicMock(return_value=MagicMock()))

    def tensorflow(self, gpu=True, placed='GPU:0'):
        value = MagicMock()
        value.__add__.return_value = SimpleNamespace(device=placed, numpy=lambda: None)
        return SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda kind: ['gpu'] if gpu else [],
            experimental=SimpleNamespace(set_memory_growth=MagicMock()), set_visible_devices=MagicMock()),
            constant=MagicMock(return_value=value), device=lambda dev: nullcontext())

    def assertFallback(self, requested):
        event = self.events[-1]
        self.assertIn('WARNING', event['message'])
        self.assertIn('Continuing on CPU', event['message'])
        self.assertEqual('device', event['info']['type'])
        self.assertEqual('fallback', event['info']['status'])
        self.assertEqual(requested, event['info']['requested_device'])
        self.assertEqual('cpu', event['info']['resolved_device'])

    def test_cpu_does_not_probe_gpu(self):
        torch = self.torch()
        with patch.dict(sys.modules, torch=torch):
            self.assertEqual('cpu', devices.torch_device('cpu', self.update))
        torch.ones.assert_not_called()
        self.assertEqual('selected', self.events[-1]['info']['status'])

    def test_unavailable_cuda_and_mps_warn(self):
        with patch.dict(sys.modules, torch=self.torch()):
            for requested in ('cuda', 'mps'):
                self.assertEqual('cpu', devices.torch_device(requested, self.update))
                self.assertFallback(requested)

    def test_explicit_no_fallback_policy_warns_before_failing(self):
        with patch.dict(sys.modules, torch=self.torch()):
            with self.assertRaisesRegex(RuntimeError, 'CUDA acceleration is unavailable'):
                devices.torch_device('cuda', self.update, allow_cpu_fallback=False)
        self.assertEqual('unavailable', self.events[-1]['info']['status'])
        self.assertIn('WARNING', self.events[-1]['message'])
        self.assertNotIn('Continuing on CPU', self.events[-1]['message'])

    def test_gpu_boolean_is_not_enough_if_driver_or_kernel_fails(self):
        torch = self.torch(cuda=True)
        torch.ones.side_effect = RuntimeError('driver/library version mismatch')
        with patch.dict(sys.modules, torch=torch):
            self.assertEqual('cpu', devices.torch_device('cuda', self.update))
        self.assertFallback('cuda')
        self.assertIn('driver/library version mismatch', self.events[-1]['message'])

    def test_successful_cuda_and_mps_are_preserved(self):
        for requested in ('cuda', 'mps'):
            torch = self.torch(cuda=True, mps=True)
            with patch.dict(sys.modules, torch=torch):
                self.assertEqual(requested, devices.torch_device(requested, self.update))
            torch.ones.assert_called_once_with(1, device=requested)
            self.assertEqual('selected', self.events[-1]['info']['status'])

    def test_automatic_pytorch_requests_remain_supported(self):
        for cuda, mps, expected in [(True, False, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
            with patch.dict(sys.modules, torch=self.torch(cuda=cuda, mps=mps)):
                self.assertEqual(expected, devices.torch_device('auto', self.update))

    def test_tensorflow_no_gpu_warns(self):
        tf = self.tensorflow(gpu=False)
        with patch.dict(sys.modules, tensorflow=tf):
            self.assertEqual('/CPU:0', devices.tensorflow_device('cuda', self.update))
        tf.constant.assert_not_called()
        self.assertFallback('cuda')

    def test_tensorflow_soft_cpu_placement_warns(self):
        with patch.dict(sys.modules, tensorflow=self.tensorflow(placed='CPU:0')):
            self.assertEqual('/CPU:0', devices.tensorflow_device('mps', self.update))
        self.assertFallback('mps')

    def test_tensorflow_cuda_and_metal_report_resolved_device(self):
        for requested, resolved in [('cuda', 'cuda:0'), ('mps', 'mps')]:
            tf = self.tensorflow()
            tf.config.experimental.set_memory_growth.side_effect = RuntimeError('Already initialized')
            with patch.dict(sys.modules, tensorflow=tf):
                self.assertEqual('/GPU:0', devices.tensorflow_device(requested, self.update))
            self.assertEqual(resolved, self.events[-1]['info']['resolved_device'])

    def test_tensorflow_cpu_does_not_create_gpu_tensor(self):
        tf = self.tensorflow()
        with patch.dict(sys.modules, tensorflow=tf):
            self.assertEqual('/CPU:0', devices.tensorflow_device('cpu', self.update))
        tf.constant.assert_not_called()
        tf.config.set_visible_devices.assert_called_once_with([], 'GPU')

    def test_tensorflow_plugin_without_memory_growth_still_uses_gpu(self):
        tf = self.tensorflow()
        tf.config.experimental.set_memory_growth.side_effect = ValueError('Unsupported by this plug-in')
        with patch.dict(sys.modules, tensorflow=tf):
            self.assertEqual('/GPU:0', devices.tensorflow_device('mps', self.update))
        self.assertEqual('selected', self.events[-1]['info']['status'])


if __name__ == '__main__':
    unittest.main()
