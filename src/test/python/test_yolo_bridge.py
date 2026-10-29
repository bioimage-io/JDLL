"""Run the exact Java-generated training task and check its exported contract."""

import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

script, output = map(Path, sys.argv[1:3])
inference = len(sys.argv) > 3
events = []


class Task:
    outputs = {}

    def export(self, **values):
        scope.update(values)

    def update(self, **event):
        events.append(event)
        normalization = (event.get('info') or {}).get('preprocessing')
        if normalization:
            path = output.parent / 'config.json'
            config = json.loads(path.read_text())
            config['preprocessing'] = normalization
            path.write_text(json.dumps(config))


task = Task()
scope = {'task': task}
try:
    exec(compile(script.read_text(), str(script), 'exec'), scope)
finally:
    if inference:
        from multiprocessing import shared_memory, resource_tracker
        for value in list(scope.values()):
            if isinstance(value, shared_memory.SharedMemory):
                value.close()
                if value is scope.get('image_shm'):
                    resource_tracker.unregister(value._name, 'shared_memory')
                else:
                    value.unlink()
if inference:
    assert task.outputs['preprocessing'] == json.loads((output.parent / 'config.json').read_text())['preprocessing']
    assert scope['image_torch'].shape == (1, 3, 640, 640)
    assert 0 <= float(scope['image_torch'].min()) < float(scope['image_torch'].max()) <= 1
    print('Generated YOLO inference task reused training intensity statistics and returned detections')
    sys.exit(0)
assert output.is_file()
normalization = task.outputs['preprocessing']
assert normalization['float_min'] == 0
assert normalization['float_max'] == 255
assert json.loads((output.parent / 'config.json').read_text())['preprocessing'] == normalization
previews = [event['info']['preview_path'] for event in events if event.get('info', {}).get('type') == 'preview']
assert previews, 'Java-generated bridge did not publish validation previews'
for path in previews:
    preview = json.loads(Path(path).read_text())
    assert preview['images']
    for image in preview['images']:
        assert image['source_path'].endswith('.npy')
        assert image['path'].endswith('.png') and Path(image['path']).is_file()
        assert Path(image['path']).parent == output.parent / 'validation_images'
assert any(e.get('info', {}).get('type') == 'checkpoint' for e in events)
assert task.outputs['completed_epochs'] == 1
print('Generated YOLO training task, normalization metadata and NumPy validation previews passed')
