from os.path import join

import numpy as np
import scipy.stats as sps
from scipy.io import loadmat

from benchflow.distributions import Normal, SplineTrapezoidal, Uniform
from benchflow.function_logger import ParameterTransformer, log_function

from .task import Task


class BayesianTiming(Task):
    """Bayesian time perception model from Acerbi et al., 2012.

    Attributes
    ----------
    self.dataset : int
        The integer label of a particular dataset. 1 through 6 corresponds to
        fake datasets (used in parameter recovery analysis in Goris et al.
        2015), while 7 through 12 correspond to real datasets.
    lb : np.ndarray
        The absolute lower bounds of the task, of shape ``(1,D)``.
    ub : np.ndarray
        The absolute upper bounds of the task, of shape ``(1,D)``.
    plb : np.ndarray
        The plausibe lower bounds of the task, of shape ``(1,D)``.
    pub : np.ndarray
        The plausible upper bounds of the task, of shape ``(1,D)``.
    posterior_log_Z : float
        The log marginal likelihood / normalization constant, as estimated from
        MCMC samples (if available).
    """

    def __init__(
        self,
        cfg=None,
        prior=None,
        transform_to_unconstrained_coordinates=False,
    ):
        """Initialize Bayesian timing Task.

        Parameters
        ----------
        cfg : omegaconf.DictConfig, optional
            The section of the Hydra config specifying the task information.
        prior : benchflow.distributions.Distribution, string, or callable; optional
            The prior Distribution, or a callable which returns the log density
            of the prior. If ``None`` or ``"uniform"``, defaults to a box-
            uniform prior (which corresponds to the pre-computed log marginal
            likelihood). If ``"normal"``, defaults to a independent normal
            prior based on plausible bounds.
        """
        self.transform_to_unconstrained_coordinates = (
            transform_to_unconstrained_coordinates
        )

        self.D = 5
        # Initialize general attributes:
        super().__init__(cfg=cfg, D=self.D)

        basepath = join(self.task_info_dir, "timing")
        timing_setup = loadmat(join(basepath, "timing.mat"))

        # Load parameter upper/lower bounds
        lb = timing_setup["y"]["LB"][0, 0]
        ub = timing_setup["y"]["UB"][0, 0]
        plb = timing_setup["y"]["PLB"][0, 0]
        plb[0, 1] = 0.02
        pub = timing_setup["y"]["PUB"][0, 0]
        # transform to unconstrained coordinates
        if transform_to_unconstrained_coordinates:
            self.transform = ParameterTransformer(self.D, lb, ub, plb, pub)
        else:
            self.transform = ParameterTransformer(self.D)  # Identity
        self.lb = self.transform(lb)
        self.ub = self.transform(ub)
        self.plb = self.transform(plb)
        self.pub = self.transform(pub)

        data = timing_setup["y"]["Data"][0, 0]
        self.data = {}
        self.data["X"] = data["X"][0, 0]
        self.data["S"] = data["S"][0, 0].squeeze()
        self.data["R"] = data["R"][0, 0]
        self.data["binsize"] = data["binsize"][0, 0].item()

        self.likelihood_mode = timing_setup["y"]["Mode"][0, 0]
        self.likelihood_mode_val = timing_setup["y"]["ModeFval"][0, 0].item()

        self.prior_mean = timing_setup["y"]["Prior"][0, 0]["Mean"][0, 0]
        self.prior_cov = timing_setup["y"]["Prior"][0, 0]["Cov"][0, 0]

        # NB: Prior is over untransformed space.
        # The parameters are not transformed for Bayesian timing model.
        if callable(prior):  # User-defined prior function
            self.prior = prior
        elif prior is None or prior.lower() == "uniform":  # Default prior
            self.prior = Uniform(self.D, lb, ub)

            self.posterior_mean = timing_setup["y"]["Post"][0, 0]["Mean"][0, 0]
            self.posterior_mode = timing_setup["y"]["Post"][0, 0]["Mode"][0, 0]
            self.posterior_mode_val = timing_setup["y"]["Post"][0, 0][
                "ModeFval"
            ][0, 0].item()

            self.log_Z_mcmc = timing_setup["y"]["Post"][0, 0]["lnZ"][
                0, 0
            ].item()
            self.posterior_log_Z = self.log_Z_mcmc
            self.posterior_cov = timing_setup["y"]["Post"][0, 0]["Cov"][0, 0]

            self.posterior_marginal_bounds = timing_setup["y"]["Post"][0, 0][
                "MarginalBounds"
            ][
                0, 0
            ]  # shape (2, D)
            self.posterior_marginal_pdf = timing_setup["y"]["Post"][0, 0][
                "MarginalPdf"
            ][
                0, 0
            ].T  # transpose to shape (n, D)
        elif prior.lower() == "normal":  # Default normal prior
            prior_mean = 0.5 * (pub + plb)
            prior_sigma = 0.5 * (pub - plb)
            self.prior = Normal(self.D, prior_mean, prior_sigma)
        elif prior == "SplineTrapezoidal":
            self.prior = SplineTrapezoidal(lb, plb, pub, ub, self.D)
            self.posterior_log_Z = -3860.956679584825
        else:
            raise ValueError(
                "Provided prior is not a callable, nor 'uniform' / 'normal'."
            )

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
        """Compute the log density of the likelihood.

        Parameters
        ----------
        theta : np.ndarray
            The array of input point, of dimension ``(D,)`` or ``(1,D)``, where
            ``D`` is the problem dimension.

        Returns
        -------
            The log density of the likelihood at the input point, of
            dimension ``(1,1)``.
        """
        assert theta.shape[-1] == np.size(theta) and theta.ndim <= 2
        # Transform unconstrained variables to original space
        x_orig = self.transform.inverse(theta)

        MAXSD = 5
        Ns = 101
        Nx = 401

        ws = x_orig[0]
        wm = x_orig[1]
        mu_prior = x_orig[2]
        sigma_prior = x_orig[3]
        if len(x_orig) < 5:
            lambd = 0.01
        else:
            lambd = x_orig[4]

        dr = self.data["binsize"]

        srange = np.linspace(0, 2, Ns)[:, None]
        ds = srange[1, 0] - srange[0, 0]

        ll = np.zeros((self.data["X"].shape[0], 1))
        Nstim = np.size(self.data["S"])

        for iStim in range(0, Nstim):
            mu_s = self.data["S"][iStim]
            sigma_s = ws * mu_s
            xrange = np.linspace(
                max(0, mu_s - MAXSD * sigma_s), mu_s + MAXSD * sigma_s, Nx
            )[None, :]
            dx = xrange[0, 1] - xrange[0, 0]
            xpdf = sps.norm.pdf(xrange, mu_s, sigma_s)
            xpdf = xpdf / np.trapz(xpdf, dx=dx)

            like = sps.norm.pdf(
                xrange, srange, ws * srange + np.finfo(float).eps
            )
            prior = sps.norm.pdf(srange, mu_prior, sigma_prior)

            post = like * prior
            post = post / np.trapz(post, axis=0, dx=ds)

            post_mean = np.trapz(post * srange, axis=0, dx=ds)
            s_hat = post_mean / (1 + wm**2)
            s_hat = s_hat[None, :]

            idx = self.data["X"][:, 2] == iStim + 1

            sigma_m = wm * s_hat
            if dr > 0:
                pr = sps.norm.cdf(
                    self.data["R"][idx] + 0.5 * dr, s_hat, sigma_m
                ) - sps.norm.cdf(
                    self.data["R"][idx] - 0.5 * dr, s_hat, sigma_m
                )
            else:
                pr = sps.norm.pdf(self.data["R"][idx], s_hat, sigma_m)

            ll[idx] = np.trapz(xpdf * pr, axis=1, dx=dx)[:, None]

        if dr > 0:
            ll = np.log(
                ll * (1 - lambd) + lambd / ((srange[-1] - srange[0]) / dr)
            )
        else:
            ll = np.log(ll * (1 - lambd) + lambd / (srange[-1] - srange[0]))

        ll = np.sum(ll)

        return np.atleast_2d(ll)


class NoisyBayesianTiming(BayesianTiming):
    def __init__(self, *args, **kwargs):
        self.noise_function = kwargs.pop("noise_function", None)
        # Default to constant normal noise, if not given noise function:
        if self.noise_function is None:
            self.noise_sd = kwargs.pop("noise_sd", 1.0)
            self.noise_function = lambda t: (
                np.random.normal(
                    scale=self.noise_sd,
                    size=(np.atleast_2d(t).shape[0], 1),
                ),
                self.noise_sd,
            )
        # Finish initializing:
        self.is_noisy = True
        super().__init__(*args, **kwargs)
        self.id = "noisy_" + self.id

    # Save handles to the deterministic log-likelihood and log-joint,
    # e.g. for testing:
    def _deterministic_log_likelihood(self, theta):
        return super().log_likelihood(theta)

    def _deterministic_log_joint(self, theta):
        log_likelihoods = self._deterministic_log_likelihood(theta)
        log_priors = self.log_prior(theta)
        if not np.isscalar(log_priors) and not np.isscalar(log_likelihoods):
            assert log_likelihoods.shape == log_priors.shape, (
                "likelihood(s) and prior(s) have incompatible shapes:"
                f"{log_likelihoods.shape} and {log_priors.shape}."
            )
        return log_likelihoods + log_priors

    # Override the log-likelihood and log-joint methods to add noise:
    def log_likelihood(self, theta):
        noise, noise_est = self.noise_function(theta)
        return super().log_likelihood(theta) + noise, noise_est

    @log_function
    def log_joint(self, theta):
        lls, noise_est = self.log_likelihood(theta)
        return lls + super().log_prior(theta), noise_est
