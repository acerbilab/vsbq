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
import pandas as pd
import psutil
import pyvbmc
import torch
import torch.nn.functional as F
import torchmetrics
from gpyreg.slice_sample import SliceSampler
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import get_hpd
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.options import Options
from pyvbmc.vbmc.variational_optimization import (
    get_cluster_centers_inds,
    optimize_vp,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from benchflow.algorithms import Algorithm
from benchflow.algorithms.nn_regression import *
from benchflow.algorithms.nn_regression.mlp_utils import (
    LightningModel,
    PyTorchMLP,
)
from benchflow.metrics import gskl, mmtv, mtv
from benchflow.plotting.utils import corner_plot
from benchflow.posteriors import PyVBMCPosterior
from benchflow.utilities.cfg import cfg_to_args, cfg_to_seed, cfg_to_task
from benchflow.utilities.dataset import (
    combine_data_from_files,
    nstd_threshold,
    read_from_file,
    read_generated_initial_set,
)


class NNSampling(Algorithm):
    def run(self):
        args, kwargs = cfg_to_args(self.cfg, self.task)

        if kwargs is None:
            kwargs = {}
        if "user_options" not in kwargs:
            kwargs["user_options"] = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN" or self.cfg.algorithm.get(
            "debugging"
        ):
            experiment_folder = os.path.join(
                hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir
            )
            logging.info(experiment_folder)
        else:
            experiment_folder = hydra_cfg.run.dir
        experiment_folder = Path(experiment_folder)
        kwargs["user_options"]["experimentfolder"] = experiment_folder

        if self.cfg.algorithm.get("debugging"):
            max_epochs = 1
            Ns_slice_sampling = 100
            vp_opt_iters = 1
        else:
            max_epochs = 200
            Ns_slice_sampling = 20000
            vp_opt_iters = 4

        X_orig, y_orig, S_orig = read_generated_initial_set(
            self.task, self.cfg
        )

        results = {}
        (
            Xs_trans,
            ys_trans,
            S_trans,
            parameter_transformer,
            options,
            noise_shaping_options,
        ) = prepare_for_mlp_training(
            self.task,
            X_orig,
            y_orig,
            S_orig,
            user_options=kwargs["user_options"],
        )

        ## Load trained model
        with open(experiment_folder / "results_nn_fitting.json", "rb") as f:
            result_nn_fitting = json.load(f)

        best_ind = result_nn_fitting["best_index"]
        checkpoint = (
            experiment_folder / f"checkpoints/best_model_{best_ind}.ckpt"
        )

        D = self.task.D
        cfg = {"learning_rate": 0.0, "weight_decay": 0.0}
        mlp_options = {
            "use_quadratic_mean": True,
            "train_quadratic_mean": True,
            "num_hidden_layers": 4,
            "num_hidden_units": 1024,
        }
        model = LightningModel.load_from_checkpoint(
            checkpoint,
            cfg=cfg,
            model=PyTorchMLP(D, **mlp_options),
        )

        start_time = time.time()
        X_star, y_star, _, _ = get_hpd(
            Xs_trans, ys_trans, options["clustering_init_hpdfrac"]
        )
        # Get cluster centers
        inds = get_cluster_centers_inds(X_star, 1)
        X_centroids = X_star[inds]
        y_centroids = y_star[inds]

        slicer = SliceSampler(
            lambda x: model.predict(
                torch.from_numpy(x).float().to(model.device)
            ).item(),
            X_centroids.squeeze(),
        )
        sampling_result = slicer.sample(Ns_slice_sampling, burn=0)
        results["sampling_time"] = time.time() - start_time

        with open(
            experiment_folder / "nn_slice_sampling_result.pkl", "wb"
        ) as f:
            pickle.dump(sampling_result, f)

        reference_samples = self.task.get_reference_samples(10000)
        samples_nn = sampling_result["samples"][-10000:]
        samples_nn = parameter_transformer.inverse(samples_nn)

        try:
            fig = corner_plot(reference_samples, samples_nn)
            fig.savefig(experiment_folder / "corner_plot_nn_sampling.png")
        except Exception as e:
            logging.error(f"corner plot failed: {e}")

        mtv_value = mtv(samples_nn, reference_samples)
        mmtv_value = np.mean(mtv_value)
        gskl_value = gskl(samples_nn, reference_samples)
        results.update(
            {
                "mtv": list(mtv_value),
                "mmtv": mmtv_value,
                "gskl": gskl_value,
            }
        )

        with open(experiment_folder / "results_nn_sampling.json", "w") as f:
            json.dump(results, f)

        sys.exit()
