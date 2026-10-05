"""Execute Java-generated training tasks against the real jdll-unet API on CPU."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import tifffile
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

for fixture in json.loads(os.environ['JDLL_UNET_TASKS']):
    root = Path(fixture['dataset_path'])
    output = Path(fixture['output_dir'])
    requests = Path(fixture['_jdll_validation_requests'])
    shape = (8, 32, 32)
    image = np.random.RandomState(42).randint(0, 1000, shape).astype(np.uint16)
    mask = np.zeros(shape, np.uint16)
    mask[:, 5:24, 6:25] = 1
    for split in ('train', 'val'):
        for kind, array in (('images', image), ('masks', mask)):
            folder = root / split / kind
            folder.mkdir(parents=True)
            tifffile.imwrite(folder / 'sample.tif', array, photometric='minisblack', metadata={'axes': 'ZYX'})
    events = []

    def update(**event):
        info = event.get('info') or {}
        events.append(info)
        if info.get('type') == 'full_validation':
            if info.get('status') == 'ready':
                (requests / 'first.request').touch()
            elif info.get('status') == 'started':
                assert (output / 'weights_last.pt').is_file()
                if info['epoch'] == 1:
                    (requests / 'second.request').touch()

    task = SimpleNamespace(outputs={}, update=update)
    scope = {'task': task}
    script = Path(fixture['script'])
    exec(compile(script.read_text(), str(script), 'exec'), scope)
    full = [e for e in events if e.get('type') == 'full_validation']
    assert [e['epoch'] for e in full if e.get('status') == 'completed'] == [1, 2], full
    assert full[-1]['status'] == 'closed'
    assert (output / 'weights_best.pt').is_file()
    assert any(e.get('type') == 'validation_plan' for e in events)
    assert any(e.get('type') == 'preview' for e in events)
    assert any(e.get('type') == 'progress' and e.get('losses', {}).get('val/total_loss') is not None for e in events)
    preview = next(e for e in reversed(events) if e.get('type') == 'preview')
    manifest = json.loads(Path(preview['preview_path']).read_text())
    item = manifest['items'][0]
    for asset in item['assets'].values():
        array = np.load(asset['path'])
        assert list(array.shape) == asset['shape']
    assert not list(requests.iterdir())
    assert task.outputs['model_dir'] == str(output)
    print('UNet Java/Python validation bridge passed for', fixture['architecture'])
