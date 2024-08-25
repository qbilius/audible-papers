from threading import RLock
from typing import Any, Callable, Dict, Optional, Union

from lightning_utilities.core.imports import RequirementCache

import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme, CustomProgress, MetricsTextColumn


GetTimeCallable = Callable[[], float]
_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")


if _RICH_AVAILABLE:
    from rich import get_console, reconfigure
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import ProgressColumn, Task, TaskID

    class ProgressWithLayout(CustomProgress):
        """
        Overrides ``CustomProgress`` to support passing layout as a renderable.

        Presumably `get_renderable` would provide the same functionality without having to copy-paste the entire `__init__` method, but I cannot get it to work.
        """

        def __init__(
            self,
            *columns: Union[str, ProgressColumn],
            renderable=None,  # new parameter
            console: Optional[Console] = None,
            auto_refresh: bool = True,
            refresh_per_second: float = 10,
            speed_estimate_period: float = 30.0,
            transient: bool = False,
            redirect_stdout: bool = True,
            redirect_stderr: bool = True,
            get_time: Optional[GetTimeCallable] = None,
            disable: bool = False,
            expand: bool = False,
        ) -> None:
            assert refresh_per_second > 0, "refresh_per_second must be > 0"
            self._lock = RLock()
            self.columns = columns or self.get_default_columns()
            self.speed_estimate_period = speed_estimate_period

            self.disable = disable
            self.expand = expand
            self._tasks: Dict[TaskID, Task] = {}
            self._task_index: TaskID = TaskID(0)
            self.live = Live(
                renderable=renderable,  # pass it here
                console=console or get_console(),
                auto_refresh=auto_refresh,
                refresh_per_second=refresh_per_second,
                transient=transient,
                redirect_stdout=redirect_stdout,
                redirect_stderr=redirect_stderr,
            )
            self.get_time = get_time or self.console.get_time
            self.print = self.console.print
            self.log = self.console.log

    class BaseDashboard(RichProgressBar):

        """Create a dashboard with a progress bar with `rich text formatting <https://github.com/Textualize/rich>`_ and any other rich components.

        This is a base class that should be inherited when composing your own dashboard. Your own subclass must define `layout` (`rich.layout.Layout`) that is split into subpanels (with unique names) and a `components` dictionary that for each of these names defines a subclass of `rich.jupyter.JupyterMixin` where the component's behavior is defined.

        Args:
            refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
                Set it to ``0`` to disable the display.
            leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
            theme: Contains styles used to stylize the progress bar.
            console_kwargs: Args for constructing a `Console`

        Raises:
            ModuleNotFoundError:
                If required `rich` package is not installed on the device.

        Note:
            PyCharm users will need to enable “emulate terminal” in output console option in
            run/debug configuration to see styled output.
            Reference: https://rich.readthedocs.io/en/latest/introduction.html#requirements

        """

        def __init__(
            self,
            refresh_rate: int = 1,
            leave: bool = False,
            theme: RichProgressBarTheme = RichProgressBarTheme(),
            console_kwargs: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__(refresh_rate=refresh_rate,
                             leave=leave,
                             theme=theme,
                             console_kwargs=console_kwargs
                             )
            self.layout: Optional[Layout] = None
            self.components: Dict[str, Any] = {}

        def _init_progress(self, trainer: "pl.Trainer") -> None:
            if self.is_enabled and (self.progress is None or self._progress_stopped):
                self._reset_progress_bar_ids()
                reconfigure(**self._console_kwargs)
                self._console = get_console()
                self._console.clear_live()
                self._metric_component = MetricsTextColumn(
                    trainer,
                    self.theme.metrics,
                    self.theme.metrics_text_delimiter,
                    self.theme.metrics_format,
                )
                self.progress = ProgressWithLayout(
                    *self.configure_columns(trainer),
                    self._metric_component,
                    auto_refresh=False,
                    disable=self.is_disabled,
                    console=self._console,
                    renderable=self.layout
                )
                self.progress.start()
                # progress has started
                self._progress_stopped = False

        def _update_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            metrics = self.get_metrics(trainer, pl_module)
            if self._metric_component:
                self._metric_component.update(metrics)

            for component in self.components.values():
                component.update(trainer)
