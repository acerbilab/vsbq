import logging
import os
import time
from pathlib import Path

import dill
import jaxgp
import matplotlib.pyplot as plt
import numpy as np
import psutil
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf.dictconfig import DictConfig
from scipy.special import erfc
from scipy.stats import chi2

from benchflow.algorithms import Algorithm
from benchflow.posteriors import PyVBMCPosterior
from benchflow.utilities.cfg import cfg_to_args
from benchflow.utilities.dataset import (
    combine_data_from_files,
    nstd_threshold,
    read_from_file,
    read_generated_initial_set,
)
from pyvbmc.vbmc import VBMC


class VSBQ(Algorithm):
    """``benchmark`` algorithm for PyVBMC."""

    def run(self, checkpoint_path=None) -> PyVBMCPosterior:
        """Run the inference and return a ``PyVBMCPosterior``.

        Returns
        -------
        posterior : benchflow.posteriors.PyVBMCPosterior
            The ``Posterior`` object containing relevant information about the
            algorithm's execution and inference.
        """
        args, kwargs = cfg_to_args(self.cfg, self.task)
        logging.info(f"PID: {os.getpid()}")
        print(os.system("nvidia-smi"))

        if kwargs is None:
            kwargs = {}
        if "user_options" not in kwargs:
            kwargs["user_options"] = {}
        options = kwargs.get("user_options", {})

        # Switch to noisy algorithm for noisy tasks, if not already specified:
        if self.cfg.task.get("noisy") and not options.get(
            "specifytargetnoise"
        ):
            logging.warn("Switching to noisy pyvbmc algorithm for noisy task.")
            kwargs["user_options"]["specifytargetnoise"] = True

        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN":
            kwargs["user_options"]["experimentfolder"] = os.path.join(
                hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir
            )
            logging.info(kwargs["user_options"]["experimentfolder"])
        else:
            kwargs["user_options"]["experimentfolder"] = hydra_cfg.run.dir

        if self.cfg.algorithm.get("use_grid"):
            n1 = 40
            n2 = 50
            print(n1, n2)
            plb = np.array([[-4.0, -3.0]])  # Plausible LB
            pub = np.array([[4.0, 8]])  # Plausible UB
            x1 = np.linspace(plb[0, 0], pub[0, 0], n1)
            x2 = np.linspace(plb[0, 1], pub[0, 1], n2)
            x1v, x2v = np.meshgrid(x1, x2)
            Xs = np.stack([x1v.flatten(), x2v.flatten()], 1)
            kwargs["x0"] = Xs
        elif self.cfg.algorithm.get("initial_set"):
            x, y, S_orig = read_generated_initial_set(self.task, self.cfg)
            # Ideally (noisy) duplicates should be combined in a better way and we get noise reduction, we simply remove duplicates here.
            N = x.shape[0]
            x, inds = np.unique(x, axis=0, return_index=True)
            logging.info(f"{np.unique(x, axis=0).shape[0]}/{N} unique points.")
            y = y[inds]
            if S_orig is not None:
                S_orig = S_orig[inds]

            kwargs["x0"] = x
            kwargs["user_options"]["fvals"] = y
            assert np.isfinite(x).all()
            assert np.isfinite(y).all()

            if self.cfg.task.get("noisy", False):
                assert (
                    S_orig is not None
                ), "S_orig need to be parsed. Typically it's self.cfg.task.options.noise_sd."
            if S_orig is not None:
                kwargs["user_options"]["S_orig"] = S_orig
                assert np.all(S_orig >= 0)
        else:
            pass

        if self.cfg.task.name == "bimodal":
            # In case of multi-modality, the variational components need to be initialized more exploratively. Other than multi-modality, it's better to be more conservative and initialize variational components around the MAP/top log-density points. In general one doesn't know whether multi-modality exists in advance. The suggestion would be, if one wants to deal with multi-modality, then try to initialize like below. If the algorithm fails then revert to more conservative initialization and intend to find an useful unimodality posterior.
            kwargs["user_options"]["clustering_init_hpdfrac"] = 0.8
            kwargs["user_options"]["n_clusters"] = 50

        if self.cfg.algorithm.get("plot_gt", True):
            # Plot ground truth posterior samples and train points for debugging
            from benchflow.function_logger import ParameterTransformer

            parameter_transformer = ParameterTransformer(
                self.task.D,
                self.task.lb,
                self.task.ub,
                self.task.plb,
                self.task.pub,
            )
            samples = self.task.get_posterior_samples()
            fig = corner_plot_samples(samples, train_X=x)
            experiment_folder = Path(
                kwargs["user_options"]["experimentfolder"]
            )
            fig.savefig(experiment_folder / "gt_posterior_with_X.png")
            plt.close(fig)
            samples_trans = parameter_transformer(samples)
            x_trans = parameter_transformer(x)
            fig = corner_plot_samples(samples_trans, train_X=x_trans)
            fig.savefig(
                experiment_folder
                / "gt_posterior_with_X(transformed space).png"
            )
            plt.close(fig)
        start_time = time.time()
        # Initialize vbmc object:
        vbmc = VBMC(*args, **kwargs)
        # Run inference:
        posterior = vbmc.optimize()
        posterior[-1]["runtime"] = time.time() - start_time
        # Construct PyVBMCPosterior from results:
        return self.PosteriorClass(self.cfg, self.task, *posterior, vbmc=vbmc)


def corner_plot_samples(
    samples,
    title=None,
    train_X=None,
    highlight_data=None,
    plot_style=None,
    figure_size=None,
    extra_data=None,
):
    from corner import corner

    D = samples.shape[1]
    # cornerplot with samples of vp
    if figure_size is None:
        figure_size = (3 * D, 3 * D)
    fig = plt.figure(figsize=figure_size, dpi=100)
    labels = ["$x_{}$".format(i) for i in range(D)]
    corner_style = dict({"fig": fig, "labels": labels})

    if plot_style is None:
        plot_style = dict()

    if "corner" in plot_style:
        corner_style.update(plot_style.get("corner"))

    # suppress warnings for small datasets with quiet=True
    fig = corner(samples, quiet=True, **corner_style)

    # style of the gp data
    data_style = dict({"s": 15, "color": "blue", "facecolors": "none"})

    if "data" in plot_style:
        data_style.update(plot_style.get("data"))

    highlighted_data_style = dict(
        {
            "s": 15,
            "color": "orange",
        }
    )
    axes = np.array(fig.axes).reshape((D, D))

    # plot train data
    if train_X is not None:
        # highlight nothing when argument is None
        if highlight_data is None or highlight_data.size == 0:
            highlight_data = np.array([False] * len(train_X))
            normal_data = ~highlight_data
        else:
            normal_data = [
                i for i in range(len(train_X)) if i not in highlight_data
            ]

        orig_X_norm = train_X[normal_data]
        orig_X_highlight = train_X[highlight_data]

        for r in range(1, D):
            for c in range(D - 1):
                if r > c:
                    axes[r, c].scatter(
                        orig_X_norm[:, c], orig_X_norm[:, r], **data_style
                    )
                    axes[r, c].scatter(
                        orig_X_highlight[:, c],
                        orig_X_highlight[:, r],
                        **highlighted_data_style,
                    )

    if title is not None:
        fig.suptitle(title)

    # adjust spacing between subplots
    fig.tight_layout(pad=0.5)
    return fig
