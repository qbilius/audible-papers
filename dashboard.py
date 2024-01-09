from typing import Any, Dict, Optional
from collections import deque, defaultdict

from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme, _detect_light_colab_theme

from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.progress import TaskID
from rich.table import Table

import plotext as plt

import rich_progress


class Dashboard(rich_progress.Dashboard):

    def __init__(self,
                 plot_window: Optional[int] = None,
                 refresh_rate: int = 1,
                 leave: bool = False,
                 theme: RichProgressBarTheme = RichProgressBarTheme(),
                 console_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__(refresh_rate=refresh_rate,
                         leave=leave,
                         theme=theme,
                         console_kwargs=console_kwargs
                         )

        self.layout: Layout = Layout(name='root')
        self.layout.split(
            Layout(name='plot', ratio=1),
            Layout(name='progress_bar', size=1),
            Layout(name='samples', size=2),
        )
        self.components = {
            'plot': Plot(window=plot_window),
            'samples': TextSamples()
        }

    def _update(self,
                progress_bar_id: Optional['TaskID'],
                current: int,
                visible: bool = True
                ) -> None:

        super()._update(
            progress_bar_id=progress_bar_id,
            current=current,
            visible=visible
        )

        # make room for multiple progress bars
        self.layout['progress_bar'].size = sum([t.visible for t in self.progress.tasks])
        self.layout['progress_bar'].update(self.progress)

        for name, component in self.components.items():
            self.layout[name].update(component)

        self.refresh()


class Plot(JupyterMixin):

    def __init__(self, window: Optional[int] = None):
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

    def update(self, trainer) -> None:
        for name, value in trainer.progress_bar_metrics.items():
            self._metrics[name][0].append(trainer.global_step)
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


class TextSamples(JupyterMixin):

    def __init__(self):
        self.samples = []

    def __rich_console__(self, console, options):
        grid = Table.grid()
        for sample in self.samples:
            grid.add_row(sample)
        yield grid

    def update(self, trainer) -> None:
        self.samples = trainer.loggers[1].samples
