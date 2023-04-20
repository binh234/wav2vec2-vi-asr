from speechbrain.utils.checkpoints import register_checkpoint_hooks, mark_as_saver, mark_as_loader

@register_checkpoint_hooks
class StepCounter:
    def __init__(self):
        self.current = 0

    def update(self):
        self.current += 1

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @mark_as_loader
    def _recover(self, path, end_of_epoch=False, device=None):
        del end_of_epoch
        del device  # Not used.
        with open(path) as fi:
            self.current = int(fi.read())