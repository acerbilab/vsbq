import json
import logging
import os
import time
from pathlib import Path

import dill
import jaxgp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from gpyreg.slice_sample import SliceSampler
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

import pyvbmc
from benchflow.algorithms import Algorithm
from benchflow.algorithms.nn_regression import *
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
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import get_hpd
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.options import Options
from pyvbmc.vbmc.variational_optimization import (
    get_cluster_centers_inds,
    optimize_vp,
)


class NNRegression(Algorithm):
    def run(self):
        # torch.manual_seed(cfg_to_seed(self.cfg))
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.benchmark = False
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        logging.info(f"SLURM_JOB_ID: {slurm_job_id}")
        logging.info(
            f"SLURM_ARRAY_JOB_ID: {os.environ.get('SLURM_ARRAY_JOB_ID')}"
        )
        logging.info(
            f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID')}"
        )
        D = self.task.D
        args, kwargs = cfg_to_args(self.cfg, self.task)
        logging.info(f"PID: {os.getpid()}")
        print(os.system("nvidia-smi"))
        if kwargs is None:
            kwargs = {}
        if "user_options" not in kwargs:
            kwargs["user_options"] = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN":
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
        start_time = time.time()
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

        logger = TensorBoardLogger(
            experiment_folder / "lightning_logs", name="mlp", version=0
        )
        model_paths = []
        val_losses = []
        for i, weight_decay in enumerate([1e-1, 1e-2, 0]):
            tic = time.time()

            training_options = {
                "noise_shaping_options": noise_shaping_options,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
                "logger": logger,
                "experiment_folder": experiment_folder,
                "mlp_options": {
                    "use_quadratic_mean": True,
                    "train_quadratic_mean": True,
                    "num_hidden_layers": 4,
                    "num_hidden_units": 1024,
                },
                "early_stopping": True,
                "checkpoint_name": f"best_model_{i}",
                "learning_rate": 1e-3,
            }

            trainer, model = train_mlp(
                Xs_trans, ys_trans, S_trans, training_options
            )[:2]
            if np.isnan(trainer.logged_metrics["val_loss"].item()):
                logging.info(
                    "Get NaN val_loss. Retrain and try to avoid exploding gradients"
                )
                training_options["learning_rate"] = (
                    training_options["learning_rate"] / 10
                )
                training_options["gradient_clip_val"] = 1.0
                trainer, model = train_mlp(
                    Xs_trans, ys_trans, S_trans, training_options
                )[:2]

            model_paths.append(model)

            val_losses.append(trainer.logged_metrics["val_loss"])
            results[f"fit_{i}"] = {
                "train_loss": trainer.logged_metrics["train_loss"].item(),
                "val_loss": trainer.logged_metrics["val_loss"].item(),
                "weight_decay": weight_decay,
                "epoch": trainer.current_epoch,
                "fitting_time": time.time() - tic,
            }

        best_index = int(np.nanargmin(val_losses))
        results["best_index"] = best_index
        logging.info(f"val_losses: {val_losses}")
        logging.info(f"argmin val_loss: {best_index}")

        # Load the best model
        model_path = model_paths[best_index]
        results["best_model_path"] = str(model_path)
        model = LightningModel.load_from_checkpoint(
            model_path,
            cfg={"learning_rate": 0, "weight_decay": 0.0},  # dummy cfg
            model=PyTorchMLP(D, **training_options["mlp_options"]),
        )
        for param in model.parameters():
            param.requires_grad = False
        results["fitting_time"] = time.time() - start_time

        with open(experiment_folder / "results_nn_fitting.json", "w") as f:
            json.dump(results, f)

        ## SVI or slice sampling
        tic = time.time()

        if self.cfg.algorithm.get("posterior_sampling") == "SVI":
            logging.info("Sampling from SVI")
            target_model = model
            target_model.X = Xs_trans
            target_model.y = ys_trans
            target_model.posteriors = [None]
            if self.cfg.task.name == "bimodal":
                # More explorative. See the comments in `pyvbmc_bench.py`
                options.__setitem__("clustering_init_hpdfrac", 0.8, force=True)
                options.__setitem__("n_clusters", 50, force=True)

            if self.cfg.algorithm.get("debugging"):
                # For faster debugging
                options.__setitem__("maxiterstochastic", 10, force=True)

            K = 50
            vp = VariationalPosterior(
                D=D,
                K=K,
                x0=Xs_trans,
                parameter_transformer=parameter_transformer,
            )
            N_fastopts = 0
            N_slowopts = 1
            optim_state = {
                "delta": None,
                "entropy_switch": False,
                "building_vp": True,
            }
            f_vals = []
            results["adam_iters"] = []
            for i in tqdm(range(0, vp_opt_iters)):
                logging.info(f"Optimizing VP iteration {i}")
                optim_state["iter"] = i
                vp, varss, pruned, f_val_list = optimize_vp(
                    options,
                    optim_state,
                    vp,
                    target_model,
                    N_fastopts,
                    N_slowopts,
                    K,
                    debug=True,
                )
                f_vals.extend(f_val_list)
                results["adam_iters"].append(len(f_vals))
            results["sampling_time(svbmc iters)"] = time.time() - tic

            fig, ax = plt.subplots()
            ax.plot(f_vals)
            fig.savefig(experiment_folder / "adam_svbmc_iters.png")
            plt.close(fig)

            # SVI might not have converged yet, so keep running it for more iterations with early stopping
            additional_svi_iters = self.cfg.algorithm.get(
                "additional_svi_iters", 0
            )
            if additional_svi_iters > 0:
                optim_state["iter"] = i + 1
                options.__setitem__(
                    "maxiterstochastic", additional_svi_iters, force=True
                )
                vp, varss, pruned, f_val_list = optimize_vp(
                    options,
                    optim_state,
                    vp,
                    target_model,
                    N_fastopts,
                    N_slowopts,
                    K,
                    debug=True,
                )
                f_vals.extend(f_val_list)
                results["adam_iters"].append(len(f_vals))
            results["sampling_time"] = time.time() - tic

            fig, ax = plt.subplots()
            ax.plot(f_vals)
            fig.savefig(experiment_folder / "adam.png")
            plt.close(fig)

            # Save results
            with open(experiment_folder / "svi_result.pkl", "wb") as f:
                dill.dump({"vp": vp, "objective_vals": f_vals}, f)

            samples_nn = vp.sample(10000)[0]
            ELBO, _, _ = compute_elbo(
                target_model, vp, ns_reparamization=1000, ns_entropy=1000
            )
            results["lml"] = ELBO
            results["lml_error"] = abs(self.task.posterior_log_Z - ELBO)
        elif self.cfg.algorithm.get("posterior_sampling") == "slice sampling":
            logging.info("Use slice sampling")
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
            results["sampling_time"] = time.time() - tic

            # Save results
            with open(
                experiment_folder / "slice_sampling_result.pkl", "wb"
            ) as f:
                dill.dump(sampling_result, f)

            samples_nn = sampling_result["samples"][-10000:]
            samples_nn = parameter_transformer.inverse(samples_nn)
        else:
            raise ValueError(
                f"Invalid posterior sampling method {self.cfg.algorithm.get('posterior_sampling')}"
            )

        total_runtime = time.time() - start_time
        results["runtime"] = total_runtime

        reference_samples = self.task.get_reference_samples(10000)

        try:
            fig = corner_plot(reference_samples, samples_nn)
            fig.savefig(experiment_folder / "corner_plot.png")
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
        with open(experiment_folder / "results.json", "w") as f:
            json.dump(results, f)
