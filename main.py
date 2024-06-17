# %%
# %load_ext autoreload
# %autoreload 2

# %%
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from omegaconf import OmegaConf

from benchflow.plotting.utils import corner_plot
from benchflow.utilities.cfg import cfg_to_task

# %%
task_name = "bimodal"
task_cfg = OmegaConf.load(
    f"./benchflow/benchflow/config/task/{task_name}.yaml"
)

task = cfg_to_task(task_cfg)

# %%
from benchflow.algorithms.generate_initial_set import (
    GenerateInitialSet,
    read_from_file,
)

algo_cfg = OmegaConf.load(
    "./benchflow/benchflow/config/algorithm/generate_initial_set.yaml"
)
N_samples = 2000
algo_cfg["method"] = "CMA-ES"
algo_cfg.map_optimization.stop_after_first = False
algo_cfg.map_optimization.N_fun_evals = N_samples
initial_points_path = (
    f"./data/initial_points/{task_name}/initial_train_set_{N_samples}.pkl"
)
algo_cfg["data_save_path"] = initial_points_path

try:
    data = read_from_file(initial_points_path)
    print("Loaded initial points from file")
except FileNotFoundError:
    print("Generating initial points")
    cfg = OmegaConf.create({"task": task_cfg, "algorithm": algo_cfg})
    data = GenerateInitialSet(cfg).run()

print(data["X"].shape)

# %%
data.keys()

# %%
from pyvbmc.vbmc.vbmc import VBMC

kwargs = {
    "x0": data["X"],
    "lower_bounds": task.lb,
    "upper_bounds": task.ub,
    "plausible_lower_bounds": task.plb,
    "plausible_upper_bounds": task.pub,
    # See vsbq_bench.py for more details on the options
    "user_options": {
        "num_ips": 30,
        "fvals": data["y"],
        "clustering_init_hpdfrac": 0.8,
        "n_clusters": 50,
    },
}
vbmc = VBMC(task.log_joint, **kwargs)

# %%
result = vbmc.optimize()
