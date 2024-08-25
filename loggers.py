from typing_extensions import override
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger

from lightning.fabric.utilities.types import _PATH


class TextLogger(DummyLogger):

    def __init__(self, save_dir: _PATH):
        # A quick hack to the version number
        self._version = FabricCSVLogger(root_dir=save_dir).version
        self.samples = []

    @property
    @override
    def version(self):
        return self._version

    def log_samples(self, *samples):
        self.samples = samples
