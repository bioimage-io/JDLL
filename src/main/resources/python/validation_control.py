"""JDLL transport adapter. No Appose dependency and no commands via callback returns."""

from pathlib import Path
import uuid


def file_control(directory):
    directory = Path(directory)

    def poll():
        tokens = []
        # Snapshot once: arrivals after this poll belong to the next boundary.
        for path in sorted(directory.glob('*.request')):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            tokens.append(path.stem)
        return tokens

    return poll


class FullValidationSchedule:
    """Epoch-boundary snapshot; completing one pass never clears newer requests."""

    def __init__(self, interval=0, request_path=None, update=lambda **event: None, next_epoch=None):
        self.interval = int(interval)
        if self.interval < 0:
            raise ValueError('validation.full_every must be >= 0.')
        self.next_epoch = next_epoch if next_epoch is not None else (self.interval or None)
        self.request_path = Path(request_path) if request_path else None
        self.update = update
        self.run_id = uuid.uuid4().hex
        self.seen, self.pending, self.active = set(), [], []
        self.reason = None

    def emit(self, status, epoch=None, message=None, **fields):
        info = dict(type='full_validation', status=status, epoch=epoch, run_id=self.run_id, **fields)
        self.update(info=info, **({'message': message} if message else {}))

    def poll(self):
        if self.request_path is None:
            return
        for token in file_control(self.request_path)():
            if token not in self.seen:
                self.seen.add(token)
                self.pending.append(token)
                self.emit('pending', request_id=token)

    def consume(self, epoch):
        self.poll()
        periodic = self.next_epoch is not None and epoch >= self.next_epoch
        if not self.pending and not periodic:
            return None
        self.active, self.pending = self.pending, []
        self.reason = 'requested' if self.active else 'periodic'
        self.emit('accepted', epoch, 'Full validation scheduled for epoch %d.' % epoch,
                  request_ids=list(self.active), reason=self.reason)
        return self.reason

    def started(self, epoch):
        self.next_epoch = epoch + self.interval if self.interval else None
        self.emit('started', epoch, 'Starting full-volume validation for epoch %d.' % epoch,
                  request_ids=list(self.active), reason=self.reason, next_epoch=self.next_epoch)

    def finish(self, epoch, status, **fields):
        self.emit(status, epoch, request_ids=list(self.active), next_epoch=self.next_epoch, **fields)
        self.active = []
        self.poll()

    def close(self):
        self.poll()
        self.emit('closed', unserved_request_ids=self.pending + self.active,
                  message=('A full-validation request arrived after the final epoch boundary or training stopped; '
                           'no extra epoch will be created.') if self.pending or self.active else None)

    def state(self):
        return dict(interval=self.interval, next_epoch=self.next_epoch)
