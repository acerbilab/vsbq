import numpy as np
import scipy.io
import scipy.special
import scipy.stats

from benchflow.distributions import SplineTrapezoidal
from benchflow.function_logger import ParameterTransformer

from .task import Task


class Multisensory_6D(Task):
    def __init__(
        self,
        cfg=None,
        prior=None,
        transform_to_unconstrained_coordinates=False,
    ):
        self.D = 6
        self.transform_to_unconstrained_coordinates = (
            transform_to_unconstrained_coordinates
        )
        task_cfg = cfg.get("task", cfg)
        # Initialize general attributes:
        super().__init__(cfg=cfg, D=self.D)

        self.subject_id = task_cfg.get("subject_id", 0)
        basepath = self.task_info_dir / "multisensory_6D"
        self.observed_data = scipy.io.loadmat(
            basepath / "acerbidokka2018_data.mat", squeeze_me=True
        )["unity_data"][self.subject_id]

        # x: (\sigma_{vis,1}, \sigma_{vis,2}, \sigma_{vis,3}, \sigma_{vest}, \lambda, \kappa)
        plb = np.array([[1.0, 1.0, 1.0, 1.0, 0.01, 1.0]])
        pub = np.array([[40.0, 40, 40, 40, 0.2, 45]])
        lb = np.array([[0.5, 0.5, 0.5, 0.5, 0.005, 0.25]])
        ub = np.array([[80.0, 80, 80, 80, 0.5, 180]])

        # transform to unconstrained coordinates
        if transform_to_unconstrained_coordinates:
            self.transform = ParameterTransformer(self.D, lb, ub, plb, pub)
            self.lb = self.transform(lb)
            self.ub = self.transform(ub)
            self.plb = self.transform(plb)
            self.pub = self.transform(pub)
        else:
            self.transform = ParameterTransformer(self.D)
            self.lb = lb
            self.ub = ub
            self.plb = plb
            self.pub = pub

        if prior == "SplineTrapezoidal":
            self.prior = SplineTrapezoidal(lb, plb, pub, ub)
            self.posterior_log_Z = -502.4790984812846
            self.posterior_mode = np.array(
                [
                    2.3969857,
                    1.37271846,
                    8.43661978,
                    7.00615081,
                    0.02525963,
                    10.13859551,
                ]
            )
            self.posterior_mode_val = -503.4863062430452
        else:
            raise ValueError(f"Unknown prior: {prior}")

        self.mcmc_info["bounded"] = True

    def log_prior(self, theta):
        """Compute the log density of the prior.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point, of dimension ``(D,)`` or ``(1,D)``,
            where ``D`` is the problem dimension.

        Returns
        -------
            The log density of the prior at the input point, of dimension
            ``(1,1)``.
        """
        dy = np.atleast_2d(self.transform.log_abs_det_jacobian(theta)).T

        if self.transform_to_unconstrained_coordinates:
            u = self.transform.inverse(theta)
        else:
            u = theta
        return super().log_prior(u) + dy

    def log_likelihood(self, theta):
        theta = np.atleast_2d(theta)
        x_orig = self.transform.inverse(theta)
        assert (
            len(x_orig.shape) == 2 and x_orig.shape[1] == 6
        ), f"x_orig must be 6d: x_orig.shape={x_orig.shape}, d={x_orig.shape[-1]}"

        sigma_vis = x_orig[:, None, :3]
        sigma_vest = x_orig[:, 3, None]
        lambd = x_orig[:, 4, None]
        kappa = x_orig[:, 5, None]
        # (m,)
        log_likelihood = np.zeros(x_orig.shape[0])
        for noise_level in range(3):
            # (n,)
            s_vest = self.observed_data[noise_level][:, 2]
            s_vis = self.observed_data[noise_level][:, 3]
            success = self.observed_data[noise_level][:, 4] == 2

            # (m, n)
            a_plus = (s_vest - s_vis + kappa) / sigma_vis[:, :, noise_level]
            a_minus = (s_vest - s_vis - kappa) / sigma_vis[:, :, noise_level]

            # (m, 1)
            b = sigma_vest / sigma_vis[:, :, noise_level]

            # (m, n)
            p_resp = lambd * 1 / 2 + (1 - lambd) * (
                scipy.stats.norm.cdf(a_plus / np.sqrt(1 + b**2))
                - scipy.stats.norm.cdf(a_minus / np.sqrt(1 + b**2))
            )

            # (m,)
            log_likelihood += scipy.stats.bernoulli.logpmf(
                success, p=1 - p_resp
            ).sum(1)

        return np.atleast_2d(log_likelihood)

    def _data(self):
        data = dict(C=self.parallel_factor)
        for i in range(3):
            data[f"n{i+1}"] = self.observed_data[i].shape[0]
            data[f"s_vest{i+1}"] = self.observed_data[i][:, 2]
            data[f"s_vis{i+1}"] = self.observed_data[i][:, 3]
            data[f"obs{i+1}"] = (self.observed_data[i][:, 4] == 2).astype(int)
        return data
