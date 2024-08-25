from typing import Optional, Union, Any, Dict
from argparse import Namespace
import tempfile

from typing_extensions import override
from lightning.pytorch.loggers.logger import Logger, DummyLogger
from lightning.pytorch.utilities import rank_zero_only
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