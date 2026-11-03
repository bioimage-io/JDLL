"""Runtime device checks shared by JDLL's bridges; no Appose dependency."""


def report(requested, resolved, backend, update, reason=None, allow_cpu_fallback=True):
    fallback = requested != 'cpu' and resolved == 'cpu'
    action = 'Continuing on CPU.' if allow_cpu_fallback else 'The operation cannot start on the requested device.'
    message = ('WARNING: %s acceleration was requested but cannot be used by %s. '
               '%s %s' % (requested.upper(), backend, action, reason or 'No usable GPU was found.')) if fallback else (
               '%s execution device: %s (requested: %s).' % (backend, resolved, requested))
    update(message=message, info=dict(type='device', phase='device', requested_device=requested,
           resolved_device=resolved if allow_cpu_fallback or not fallback else None, backend=backend,
           status=('fallback' if allow_cpu_fallback else 'unavailable') if fallback else 'selected',
           reason=reason, severity='warning' if fallback else 'info'))


def torch_device(requested, update, allow_cpu_fallback=True):
    import torch
    requested = str(requested or 'cpu').strip().lower()
    resolved, reason = 'cpu', None
    if requested != 'cpu':
        try:
            target = requested
            if requested in ('auto', 'acceleration', 'accelerated'):
                mps = getattr(torch.backends, 'mps', None)
                if torch.cuda.is_available():
                    target = 'cuda'
                elif mps is not None and mps.is_built() and mps.is_available():
                    target = 'mps'
                else:
                    raise RuntimeError('Neither CUDA nor MPS is available in this environment.')
            if target.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise RuntimeError('PyTorch cannot initialize CUDA; check the NVIDIA driver and CUDA-enabled environment.')
            elif target == 'mps':
                mps = getattr(torch.backends, 'mps', None)
                if mps is None or not mps.is_built() or not mps.is_available():
                    raise RuntimeError('This PyTorch environment or system does not provide usable MPS acceleration.')
            else:
                raise ValueError('Unsupported acceleration device: ' + requested)
            # Availability alone does not catch all driver/kernel initialization failures.
            test = torch.ones(1, device=target)
            (test + test).cpu()
            del test
            resolved = target
        except Exception as error:
            reason = str(error)
    report(requested, resolved, 'PyTorch', update, reason, allow_cpu_fallback)
    if requested != 'cpu' and resolved == 'cpu' and not allow_cpu_fallback:
        raise RuntimeError('Requested %s acceleration is unavailable: %s' % (requested.upper(), reason))
    return resolved


def tensorflow_device(requested, update, allow_cpu_fallback=True):
    import tensorflow as tf
    requested = str(requested or 'cpu').strip().lower()
    resolved, reason = 'cpu', None
    if requested != 'cpu':
        try:
            devices = tf.config.list_physical_devices('GPU')
            if not devices:
                raise RuntimeError('TensorFlow cannot initialize a GPU; check the driver and GPU-enabled environment.')
            for device in devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except (RuntimeError, ValueError):
                    pass  # Existing or plug-in devices may not allow changing this setting.
            with tf.device('/GPU:0'):
                test = tf.constant([1.0])
                result = test + test
                result.numpy()
            if 'GPU:' not in result.device.upper():
                raise RuntimeError('TensorFlow placed the GPU test on CPU instead.')
            resolved = 'mps' if requested == 'mps' else 'cuda:0'
        except Exception as error:
            reason = str(error)
    else:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            pass
    report(requested, resolved, 'TensorFlow', update, reason, allow_cpu_fallback)
    if requested != 'cpu' and resolved == 'cpu' and not allow_cpu_fallback:
        raise RuntimeError('Requested %s acceleration is unavailable: %s' % (requested.upper(), reason))
    return '/CPU:0' if resolved == 'cpu' else '/GPU:0'
