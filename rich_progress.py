from re import L
from typing import Any, Callable, Dict, Generator, Optional, Union, cast
from collections import deque, defaultdict
from threading import RLock

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme, CustomProgress, MetricsTextColumn, _detect_light_colab_theme
from lightning.pytorch.utilities.types import STEP_OUTPUT


GetTimeCallable = Callable[[], float]
_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")


if _RICH_AVAILABLE:
    from rich import get_console, reconfigure
    from rich.ansi import AnsiDecoder
    from rich.console import Console, Group, RenderableType
    from rich.jupyter import JupyterMixin
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn
    from rich.progress_bar import ProgressBar as _RichProgressBar
    from rich.style import Style
    from rich.text import Text

    import plotext as plt

    class RichProgressBarWithPlot(RichProgressBar):

        """Create a progress bar with `rich text formatting <https://github.com/Textualize/rich>`_.

        Install it with pip:

        .. code-block:: bash

            pip install rich

        .. code-block:: python

            from lightning.pytorch import Trainer
            from lightning.pytorch.callbacks import RichProgressBar

            trainer = Trainer(callbacks=RichProgressBar())

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
            window: Optional[int] = None,
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
            self._plot_component: Optional["Plot"] = None
            self.window = window
            self.layout = Layout(name='root')
            self.layout.split(
                Layout(name='plot', ratio=1),
                Layout(name='progress_bar', size=1),
            )
            self.live: Optional[Live] = None

        def _init_progress(self, trainer: "pl.Trainer") -> None:
            if self.is_enabled and (self.progress is None or self._progress_stopped):
                self._reset_progress_bar_ids()
                reconfigure(**self._console_kwargs)
                self._console = get_console()
                self._console.clear_live()
                self._metric_component = MetricsTextColumnFix(
                    trainer,
                    self.theme.metrics,
                    self.theme.metrics_text_delimiter,
                    self.theme.metrics_format,
                )
                self._plot_component = Plot(window=self.window)
                self.progress = ProgressWithPlot(
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

        def _update(self, progress_bar_id: Optional["TaskID"], current: int, visible: bool = True) -> None:
            if self.progress is not None and self.is_enabled:
                assert progress_bar_id is not None

                # make room for multiple progress bars
                self.layout['progress_bar'].size = sum([t.visible for t in self.progress.tasks])

                total = self.progress.tasks[progress_bar_id].total
                assert total is not None
                if not self._should_update(current, total):
                    return

                leftover = current % self.refresh_rate
                advance = leftover if (current == total and leftover != 0) else self.refresh_rate
                self.progress.update(progress_bar_id, advance=advance, visible=visible)

                self.layout['progress_bar'].update(self.progress)
                self.layout['plot'].update(self._plot_component)

                self.refresh()

        def _update_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            metrics = self.get_metrics(trainer, pl_module)
            if self._metric_component:
                self._metric_component.update(metrics)
            if self._plot_component:
                self._plot_component.update(trainer.global_step, trainer.progress_bar_metrics)

    class Plot(JupyterMixin):

        def __init__(self, window=None):
            self.window = window

            self._metrics = defaultdict(lambda: [
                deque(maxlen=window), deque(maxlen=window)])
            self.decoder = AnsiDecoder()

        def __rich_console__(self, console, options):
            self.width = options.max_width or console.width
            self.height = options.height or console.height
            canvas = self.make_plot()
            self.rich_canvas = Group(*self.decoder.decode(canvas))
            yield self.rich_canvas

        def update(self, global_step: int, metrics: Dict[Any, Any]) -> None:
            # Called when metrics are ready to be rendered.
            # This is to prevent render from causing deadlock issues by requesting metrics
            # in separate threads.
            for name, value in metrics.items():
                self._metrics[name][0].append(global_step)
                self._metrics[name][1].append(value)

        def make_plot(self):
            plt.clear_data()
            plt.clear_figure()

            xs = []
            ys = []
            for name, (x, y) in self._metrics.items():
                plt.plot(x, y, label=name)
                xs.extend(list(x))
                if name.startswith('val'):
                    ys.extend(list(y))

            if not _detect_light_colab_theme():
                plt.theme('dark')
            plt.plotsize(self.width, self.height)

            if self.window is not None:
                start = max(0, max(xs, default=0) - self.window)
                plt.xlim(start, start + self.window)

            plt.title(f'{len(ys)}')

            return plt.build()

    class MetricsTextColumnFix(MetricsTextColumn):
        """A column containing text."""

        def _generate_metrics_texts(self) -> Generator[str, None, None]:
            for name, value in self._metrics.items():
                if name != 'v_num' and not isinstance(value, str):
                    value = f"{value:{self._metrics_format}}"
                yield f"{name}: {value}"

    class ProgressWithPlot(CustomProgress):
        """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

        def __init__(
            self,
            *columns: Union[str, ProgressColumn],
            renderable=None,
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
                renderable=renderable,
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
