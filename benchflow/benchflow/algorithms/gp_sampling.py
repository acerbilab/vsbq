# from benchflow.utilities.cfg import cfg_to_task, cfg_to_metrics
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
from gpyreg.slice_sample import SliceSampler
from hydra.core.hydra_config import HydraConfig
from scipy.stats import bootstrap

from benchflow.algorithms import Algorithm
from benchflow.algorithms.vsbq_bench import read_from_file
from benchflow.metrics import gskl, mmtv, mtv
from benchflow.plotting import load_result
from benchflow.plotting.utils import corner_plot
from pyvbmc.stats import get_hpd
from pyvbmc.vbmc.variational_optimization import get_cluster_centers_inds


class GPSampling(Algorithm):
    def run(self):
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN":
            experiment_folder = os.path.join(
                hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir
            )
            logging.info(experiment_folder)
        else:
            experiment_folder = hydra_cfg.run.dir
        experiment_folder = Path(experiment_folder)

        with open(experiment_folder / "results.pkl", "rb") as f:
            algo_result = dill.load(f)

        gp = algo_result.vbmc.gp
        gp.post = gp.posterior(gp.params_cache)

        def _predict_f(x):
            if x.ndim == 1:
                x = x[None, :]
            fval, fs2 = gp.post.predict_f_with_precomputed(x)
            return fval.squeeze()

        if self.cfg.algorithm.get("debugging"):
            Ns_slice_sampling = 100
        else:
            Ns_slice_sampling = 20000

        start_time = time.time()
        results = {}
        Xs_trans = gp.X
        ys_trans = gp.y
        X_star, y_star, _, _ = get_hpd(Xs_trans, ys_trans, 0.01)
        # Get cluster centers
        inds = get_cluster_centers_inds(X_star, 1)
        X_centroids = X_star[inds]
        y_centroids = y_star[inds]

        slicer = SliceSampler(
            lambda x: _predict_f(x).item(),
            X_centroids.squeeze(),
        )
        sampling_result = slicer.sample(Ns_slice_sampling, burn=0)
        results["sampling_time"] = time.time() - start_time
        with open(
            experiment_folder / "gp_slice_sampling_result.pkl", "wb"
        ) as f:
            pickle.dump(sampling_result, f)
        reference_samples = self.task.get_reference_samples(10000)
        parameter_transformer = algo_result.vbmc.parameter_transformer
        samples_gp = sampling_result["samples"][-10000:]
        samples_gp = parameter_transformer.inverse(samples_gp)

        try:
            fig = corner_plot(reference_samples, samples_gp)
            fig.savefig(experiment_folder / "corner_plot_gp_sampling.png")
        except Exception as e:
            logging.error(f"corner plot failed: {e}")

        mtv_value = mtv(samples_gp, reference_samples)
        mmtv_value = np.mean(mtv_value)
        gskl_value = gskl(samples_gp, reference_samples)
        results.update(
            {
                "mtv": list(mtv_value),
                "mmtv": mmtv_value,
                "gskl": gskl_value,
            }
        )

        with open(experiment_folder / "results_gp_sampling.json", "w") as f:
            json.dump(results, f)

        sys.exit()
