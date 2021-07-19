from __future__ import annotations
import os
import shutil
import matplotlib.pyplot as plt


class Plotter:
    def __init__(
        self: Plotter,
        filename: str,
        plt: plt,
        backend: str = None,
        should_save: bool = True,
        should_plot: bool = False,
    ) -> None:
        if backend is not None:
            allowed_backends = ["WebAgg"]
            assert (
                backend in allowed_backends
            ), f'Backend "{backend}" not in allowed backends: {allowed_backends}'
        self._filename = filename
        self._backend = backend
        self._should_save = should_save
        self._should_plot = should_plot
        if self._should_plot and self._backend is None:
            raise "If should plot default is set to True, backend must not be None"
        if self._backend is not None:
            plt.switch_backend(backend)
        self._clear_old_plots(f"./plots/{filename}")

    def __call__(
        self: Plotter,
        plotname: str,
        sub_path: str = "",
        should_save: bool = None,
        should_plot: bool = None,
        verbose: bool = False,
        dpi: int = 100,
    ) -> None:
        should_save = self._should_save if should_save is None else should_save
        should_plot = self._should_plot if should_plot is None else should_plot
        if should_save is True:
            dir_path = f"./plots/{self._filename}{sub_path}"
            file_path = dir_path + f"/{plotname}"
            self._ensure_path_exists(dir_path)
            plt.savefig(file_path, dpi=dpi)
            if verbose:
                print(f"Saved plot to {file_path}")
        if self._backend is not None and should_plot is True:
            plt.show()
        plt.clf()

    def _ensure_path_exists(self: Plotter, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

    def _clear_old_plots(self: Plotter, dir_path: str) -> None:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
