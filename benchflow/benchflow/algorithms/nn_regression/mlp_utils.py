import logging
import numbers
from copy import deepcopy
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from benchflow.utilities.dataset import nstd_threshold


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.std + self.epsilon) + self.mean

    def to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


class PyTorchMLP(torch.nn.Module):
    def __init__(
        self,
        D,
        num_hidden_layers=4,
        num_hidden_units=1024,
        use_quadratic_mean=True,
        train_quadratic_mean=True,
    ):
        super().__init__()

        layers = [torch.nn.Linear(D, num_hidden_units), torch.nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(num_hidden_units, num_hidden_units))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(num_hidden_units, 1))
        self.all_layers = torch.nn.Sequential(*layers)
        self.use_quadratic_mean = use_quadratic_mean
        if use_quadratic_mean:
            if train_quadratic_mean:
                require_grad = True
            else:
                require_grad = False
            self.m0 = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=require_grad
            )
            self.xm = torch.nn.Parameter(
                torch.zeros(1, D), requires_grad=require_grad
            )
            self.raw_lengthscale = torch.nn.Parameter(
                torch.zeros(1, D), requires_grad=require_grad
            )

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        # with torch.no_grad():
        #     self.raw_lengthscale.clamp_(min=-1.0, max=1000.0)
        if self.use_quadratic_mean:
            lengthscale = torch.nn.functional.softplus(self.raw_lengthscale)
            mean = self.m0 - 0.5 * (x - self.xm).div(lengthscale).pow(2).sum(
                -1
            )
        else:
            mean = 0
        preds = torch.flatten(self.all_layers(x))
        preds += mean
        return preds

    def freeze_quadrative_mean(self):
        self.m0.requires_grad = False
        self.xm.requires_grad = False
        self.raw_lengthscale.requires_grad = False


class LightningModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()

        self.learning_rate = cfg["learning_rate"]
        self.weight_decay = cfg["weight_decay"]
        self.optimizer = cfg.get("optimizer", "AdamW")
        self.model = model

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        self.sc_x = None
        self.sc_y = None

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, ys_true, noise = batch
        preds = self(features)
        if not torch.isfinite(noise).all():
            loss = F.mse_loss(preds, ys_true)
        else:
            loss = F.mse_loss(preds / noise, ys_true / noise)
        return loss, ys_true, preds, noise

    def training_step(self, batch, batch_idx):
        loss, ys_true, preds, noise = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)

        # if self.current_epoch == 20:
        #     self.model.freeze_quadrative_mean()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ys_true, preds, noise = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, ys_true, preds, noise = self._shared_step(batch)
        self.test_mse(preds / noise, ys_true / noise)
        self.log("test_mse", self.test_mse, prog_bar=True)
        return preds

    def predict(self, X):
        features = X
        self.sc_x.to_device(self.device)
        self.sc_y.to_device(self.device)
        features = self.sc_x.transform(features)
        preds = self(features).reshape(-1, 1)
        preds = self.sc_y.inverse_transform(preds)
        preds = preds.ravel()
        return preds

    def configure_optimizers(self):
        # Modified from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # print(fpn)
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, whitelist_weight_modules
                ):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, blacklist_weight_modules
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the quadratic mean parameters are not decayed
        no_decay.update({"model.m0", "model.xm", "model.raw_lengthscale"})

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )
        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "AdamW":
            logging.info(
                f"Using AdamW optimizer lr={self.learning_rate}, wd={self.weight_decay}"
            )
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not recognized")
            # optimizer = torch.optim.SGD(
            #     self.parameters(),
            #     lr=self.learning_rate,
            #     weight_decay=self.weight_decay,
            # )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sc_x"] = self.trainer.datamodule.sc_x

        checkpoint["sc_y"] = self.trainer.datamodule.sc_y

    def on_load_checkpoint(self, checkpoint):
        self.sc_x = checkpoint["sc_x"]
        self.sc_y = checkpoint["sc_y"]


class CustomDataset(Dataset):
    def __init__(self, X, y, noise):
        self.x = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float).flatten()
        if noise is None:
            self.noise = None
        else:
            self.noise = torch.tensor(noise, dtype=torch.float).flatten()

    def __getitem__(self, index):
        if self.noise is None:
            return self.x[index], self.y[index], None
        return self.x[index], self.y[index], self.noise[index]

    def __len__(self):
        return self.y.shape[0]


class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        X,
        y,
        noise=None,
        noise_shaping_options={},
        batch_size=32,
    ):
        super().__init__()
        self.X_all = X
        self.y_all = y.ravel()
        if noise is None:
            noise = np.ones_like(self.y_all) * np.sqrt(1e-5)
            # noise = np.full_like(self.y_all, None)
        self.noise_obs_all = noise
        assert (
            self.X_all.shape[0] == self.y_all.shape[0]
            and self.X_all.shape[0] == noise.shape[0]
        )
        assert self.y_all.ndim == 1
        self.noise_shaping_options = noise_shaping_options

        self.batch_size = batch_size
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()

    def prepare_data(self) -> None:
        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            self.noise_train,
            self.noise_val,
        ) = train_test_split(
            self.X_all,
            self.y_all,
            self.noise_obs_all,
            test_size=0.2,
            random_state=0,
        )

        self.y_train_max = self.y_all.max()
        self.noise_train = noise_shaping(
            self.noise_train,
            self.y_train,
            self.noise_shaping_options,
            self.y_train_max,
        )
        self.noise_val = noise_shaping(
            self.noise_val,
            self.y_val,
            self.noise_shaping_options,
            self.y_train_max,
        )
        assert self.noise_train.size == self.y_train.size

        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_val = torch.from_numpy(self.X_val).float()
        self.y_train = torch.from_numpy(self.y_train).float()
        self.y_val = torch.from_numpy(self.y_val).float()

    def setup(self, stage: str):
        if stage == "fit":
            X_train = self.sc_x.fit_transform(self.X_train)
            X_val = self.sc_x.transform(self.X_val)
            y_train = self.sc_y.fit_transform(
                self.y_train.reshape(-1, 1)
            ).flatten()
            y_val = self.sc_y.transform(self.y_val.reshape(-1, 1)).flatten()
            self.train = CustomDataset(X_train, y_train, self.noise_train)
            self.val = CustomDataset(X_val, y_val, self.noise_val)
        elif stage == "test":
            noise_pred_set = noise_shaping(
                self.noise_obs_all,
                self.y_all,
                self.noise_shaping_options,
            )
            X_pred_set = self.sc_x.transform(self.X_all)
            y_pred_set = self.sc_y.transform(
                self.y_all.reshape(-1, 1)
            ).flatten()
            self.pred = CustomDataset(X_pred_set, y_pred_set, noise_pred_set)
        else:
            raise ValueError(f"stage {stage} not recognized")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.pred, batch_size=self.batch_size, shuffle=False)


def noise_shaping(s, y, options: dict, ymax: Optional[float] = None):
    DIVIDE_BY_MIN = True  # Test If needed for MLP fitting
    # Increase noise for low density points
    if s is None:
        s2 = options["tolgpnoise"] ** 2 * np.ones_like(y)
    else:
        s2 = np.array(s**2).squeeze()
    if not bool(options):
        s = np.sqrt(s2)
        if DIVIDE_BY_MIN:
            s /= s.min()
        return s
    min_lnsigma = np.log(options["noiseshapingmin"])
    med_lnsigma = np.log(options["noiseshapingmed"])

    if ymax is None:
        ymax = np.max(y)
    frac = np.minimum(1, (ymax - y) / options["noiseshapingthreshold"])
    sigma_shape = np.exp((min_lnsigma * (1 - frac) + frac * med_lnsigma))

    delta_y = np.maximum(0, ymax - y - options["noiseshapingthreshold"])
    sigma_shape += options["noiseshapingfactor"] * delta_y

    sn2extra = sigma_shape**2

    s2s = s2 + sn2extra
    # Excessive difference between low and high noise might cause numerical
    # instabilities, so we give the option of capping the ratio
    maxs2 = np.min(s2s) * options["noiseshapingmaxratio"] ** 2
    s2s = np.minimum(s2s, maxs2)
    s = np.sqrt(s2s)
    if DIVIDE_BY_MIN:
        s /= s.min()
    return s


def prepare_for_mlp_training(task, X_orig, y_orig, S_orig=None, **kwargs):
    import pyvbmc
    from pyvbmc.parameter_transformer import ParameterTransformer
    from pyvbmc.vbmc.options import Options

    # Use options from SVBMC for noise shaping
    pyvbmc_path = Path(pyvbmc.__file__).parent / "vbmc"

    D = task.D
    lower_bounds = task.lb
    upper_bounds = task.ub
    plausible_lower_bounds = task.plb
    plausible_upper_bounds = task.pub
    options = Options(
        pyvbmc_path / "./option_configs/basic_vbmc_options.ini",
        evaluation_parameters={"D": D},
        user_options=kwargs.get("user_options", {}),
    )
    options.load_options_file(
        pyvbmc_path / "./option_configs/advanced_vbmc_options.ini",
        evaluation_parameters={"D": D},
    )
    options.update_defaults()
    parameter_transformer = ParameterTransformer(
        D,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
        transform_type=options["boundedtransform"],
    )
    Xs_trans = parameter_transformer(X_orig)
    log_abs_dets = parameter_transformer.log_abs_det_jacobian(Xs_trans)
    ys_trans = y_orig + log_abs_dets

    noise_shaping_factors = options["noiseshapingfactors"]
    if options.get("noiseshapingthresholds_instd"):
        ns_thresholds = nstd_threshold(options["noiseshapingthresholds"], D)
    else:
        ns_thresholds = options["noiseshapingthresholds"]
    if isinstance(noise_shaping_factors, numbers.Number):
        noise_shaping_factors = [noise_shaping_factors]
    if isinstance(ns_thresholds, numbers.Number):
        ns_thresholds = [ns_thresholds]
    assert len(ns_thresholds) == 1 and len(noise_shaping_factors) == 1
    options.__setitem__(
        "noiseshapingfactor", noise_shaping_factors[0], force=True
    )
    options.__setitem__(
        "noiseshapingthreshold",
        ns_thresholds[0],
        force=True,
    )

    if options.get("noiseshaping"):
        logging.info("using noise shaping")
        noise_shaping_options = options
    else:
        logging.info("not using noise shaping")
        noise_shaping_options = {}

    return (
        Xs_trans,
        ys_trans,
        S_orig,  # The noise doesn't change with transformation
        parameter_transformer,
        options,
        noise_shaping_options,
    )
    # return {
    #     "Xs_trans": Xs_trans,
    #     "ys_trans": ys_trans,
    #     "S_trans": S_orig,  # The noise doesn't change with transformation
    #     "parameter_transformer": parameter_transformer,
    #     "noise_shaping_options": noise_shaping_options,
    # }


def train_mlp(Xs_trans, ys_trans, S_trans, training_options):
    D = Xs_trans.shape[1]
    weight_decay = training_options["weight_decay"]
    max_epochs = training_options["max_epochs"]
    noise_shaping_options = training_options["noise_shaping_options"]
    logger = training_options.get("logger", False)
    experiment_folder = training_options.get("experiment_folder", Path("."))
    verbose = training_options.get("verbose", True)
    checkpoint_name = training_options.get("checkpoint_name", "best_model")
    mlp_options = training_options.get("mlp_options", {})
    early_stopping = training_options.get("early_stopping", True)
    experiment_folder = Path(experiment_folder)
    learning_rate = training_options.get("learning_rate", 1e-3)
    gradient_clip_val = training_options.get("gradient_clip_val", 0.0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints",
        filename=checkpoint_name,
        monitor="val_loss",
        save_top_k=1,
        every_n_epochs=1,
    )
    callbacks = [checkpoint_callback, LitProgressBar(refresh_rate=300)]

    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=20,
            verbose=verbose,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    dm = CustomDataModule(
        Xs_trans,
        ys_trans,
        S_trans,
        batch_size=32,
        noise_shaping_options=noise_shaping_options,
    )
    pytorch_model = PyTorchMLP(D, **mlp_options)
    cfg = {"learning_rate": learning_rate, "weight_decay": weight_decay}
    logging.info(f"weight_decay: {weight_decay}")
    model = LightningModel(pytorch_model, cfg)
    trainer = L.Trainer(
        # overfit_batches=0.01,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        logger=logger,
        # log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)
    model.sc_x = deepcopy(trainer.datamodule.sc_x)
    model.sc_y = deepcopy(trainer.datamodule.sc_y)

    train_loss = trainer.logged_metrics["train_loss"]
    val_loss = trainer.logged_metrics["val_loss"]

    logging.info(
        f"current epoch: {trainer.current_epoch}, train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}"
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    return trainer, best_model_path, dm
