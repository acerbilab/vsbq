import gc
import json
import logging
import math
import os
import resource
import shutil
import sys
import time
from copy import deepcopy

import dill
import gpyreg as gpr
import jax
import jax.numpy as jnp
import jaxgp as jgp
import matplotlib.pyplot as plt
import numpy as np
import psutil
from jax._src.api import clear_backends

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kldiv_mvn
from pyvbmc.timer import Timer
from pyvbmc.variational_posterior import VariationalPosterior

from .gaussian_process_train import (
    check_train_set_predictions,
    gpyreg_params_to_jaxgp,
    jaxgp_params_to_gpyreg,
    reupdate_gp,
    search_noise_shaping_hyperparams,
    train_gp,
    train_sgpr,
)
from .iteration_history import IterationHistory
from .options import Options
from .variational_optimization import optimize_vp

logger_root = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger_root.addFilter(CheckTypesFilter())


def print_live_buffers(logger=None) -> None:
    lbs = jax.lib.xla_bridge.get_backend().live_arrays()
    lbs.sort(key=jnp.size)
    if logger is None:
        print_fun = print
    else:
        print_fun = logger.debug
    print_fun(f"Number live buffers: {len(lbs)}")


def copy_sgpr(gp):
    memo = {id(gp.post_cache): None}
    output = deepcopy(gp, memo)  # Deep copy of everything else
    return output


class VBMC:
    """
    Posterior and model inference via Variational Sparse Bayesian Quadrature (VSBQ). VSBQ is closely related to VBMC but uses a sparse GP instead of an exact GP for scalability. (Modified from the PyVBMC algorithm: https://github.com/acerbilab/pyvbmc)

    Parameters
    ----------
    fun : callable
        A given target log posterior `fun`. `fun` accepts input `x` and returns
        the value of the target log-joint, that is the unnormalized
        log-posterior density, at `x`.
    x0 : np.ndarray, optional
        Starting point for the inference. Ideally `x0` is a point in the
        proximity of the mode of the posterior. Default is ``None``.
    lower_bounds, upper_bounds : np.ndarray, optional
        `lower_bounds` (`LB`) and `upper_bounds` (`UB`) define a set
        of strict lower and upper bounds for the coordinate vector, `x`, so
        that the posterior has support on `LB` < `x` < `UB`.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for `LB` and `UB` if no bounds exist. Set `LB` [`i`] = -``inf``
        and `UB` [`i`] = ``inf`` if the `i`-th coordinate is unbounded (while
        other coordinates may be bounded). Note that if `LB` and `UB` contain
        unbounded variables, the respective values of `PLB` and `PUB` need to
        be specified (see below), by default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of `plausible_lower_bounds` (`PLB`) and
        `plausible_upper_bounds` (`PUB`) such that `LB` < `PLB` < `PUB` < `UB`.
        Both `PLB` and `PUB` need to be finite. `PLB` and `PUB` represent a
        "plausible" range, which should denote a region of high posterior
        probability mass. Among other things, the plausible box is used to
        draw initial samples and to set priors over hyperparameters of the
        algorithm. When in doubt, we found that setting `PLB` and `PUB` using
        the topmost ~68% percentile range of the prior (e.g, mean +/- 1 SD
        for a Gaussian prior) works well in many cases (but note that
        additional information might afford a better guess). Both are
        by default ``None``.
    user_options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        VBMC options page for the default options. If no `user_options` are
        passed, the default options are used.
    """

    def __init__(
        self,
        fun: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        user_options: dict = None,
        aux: dict = None,
    ):
        # Initialize variables and algorithm structures
        if x0 is None:
            if (
                plausible_lower_bounds is None
                or plausible_upper_bounds is None
            ):
                raise ValueError(
                    """vbmc:UnknownDims If no starting point is
                 provided, PLB and PUB need to be specified."""
                )
            else:
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        self.D = x0.shape[1]
        # load basic and advanced options and validate the names
        pyvbmc_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=user_options,
        )

        advanced_path = (
            pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.update_defaults()
        self.options.validate_option_names([basic_path, advanced_path])
        if not os.path.exists(self.options["experimentfolder"]):
            os.makedirs(self.options["experimentfolder"])

        self.logger = self._init_logger()
        _ = self._init_logger("_debug")

        # variable to keep track of logging actions
        self.logging_action = []

        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, self.D)) * -np.inf

        if upper_bounds is None:
            upper_bounds = np.ones((1, self.D)) * np.inf

        # Check/fix boundaries and starting points
        (
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
        ) = self._boundscheck(
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

        self.K = self.options.get("kstart")

        # starting point
        if not np.all(np.isfinite(self.x0)):
            # print('Initial starting point is invalid or not provided.
            # Starting from center of plausible region.\n');
            self.x0 = 0.5 * (
                self.plausible_lower_bounds + self.plausible_upper_bounds
            )

        # Initialize transformation to unbounded parameters
        self.parameter_transformer = ParameterTransformer(
            self.D,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
            transform_type=self.options["boundedtransform"],
        )

        self.x0_trans = self.parameter_transformer(self.x0)

        # Initialize variational posterior
        self.vp = VariationalPosterior(
            D=self.D,
            K=self.K,
            x0=self.x0_trans,
            parameter_transformer=self.parameter_transformer,
        )

        plt.close("all")
        self.vp.optimize_mu = self.options.get("variablemeans")
        self.vp.optimize_weights = self.options.get("variableweights")

        self.aux = aux
        self.optim_state = self._init_optim_state()

        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cachesize"),
            parameter_transformer=self.parameter_transformer,
        )

        self.random_state = np.random.get_state()
        self.iteration_history = IterationHistory(
            [
                "elcbo_impro",
                "stable",
                "elbo",
                "vp",
                "iter",
                "elbo_sd",
                "lcbmax",
                "data_trim_list",
                "gp",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "sKL",
                "sKL_true",
                "pruned",
                "varss",
                "func_count",
                "n_eff",
                "logging_action",
                "gp_elbo_withouts",
                "gp_elbo_withs",
                "vp_elbo_withouts",
                "vp_elbo_withs",
                "xnews",
                "ynews",
                "ymaxs",
                "delta_LMLs",
                "delta_ELBOs",
                "delta_ELBO_us",
                "KL_trues",
                "pred_means",
                "pred_sigma2",
                "sigma2_obss",
                "add_ip",
                "N_ips",
                "function_logger",
                "random_state",
                "N_train",
            ]
        )

        self.is_finished = False
        # the iterations of pyvbmc start at 0
        self.iteration = -1
        self.timer = Timer()

        self.gp = None
        self.hyp_dict = {}

        # Record provided points to function_logger
        Xs = np.copy(self.optim_state["cache"]["x_orig"])
        ys = np.copy(self.optim_state["cache"]["y_orig"])
        if self.optim_state["uncertainty_handling_level"] == 2:
            S_orig = np.copy(self.optim_state["cache"]["S_orig"])

        Xs_trans = self.parameter_transformer(Xs)
        log_abs_dets = self.parameter_transformer.log_abs_det_jacobian(
            Xs_trans
        )
        ys_trans = ys + log_abs_dets
        assert np.all(np.isfinite(ys)) & np.all(np.isfinite(Xs_trans))
        if self.optim_state["uncertainty_handling_level"] == 0:
            self.function_logger.initialize(Xs, Xs_trans, ys, ys_trans)
        elif self.optim_state["uncertainty_handling_level"] == 2:
            self.function_logger.initialize(Xs, Xs_trans, ys, ys_trans, S_orig)
        else:
            raise NotImplementedError()

    def _boundscheck(
        self,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
    ):
        """
        Private function to do the initial check of the VBMC bounds.
        """

        N0, D = x0.shape

        if plausible_lower_bounds is None or plausible_upper_bounds is None:
            if N0 > 1:
                self.logger.warning(
                    "PLB and/or PUB not specified. Estimating"
                    + "plausible bounds from starting set X0..."
                )
                width = x0.max(0) - x0.min(0)
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = x0.min(0) - width / N0
                    plausible_lower_bounds = np.maximum(
                        plausible_lower_bounds, lower_bounds
                    )
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = x0.max(0) + width / N0
                    plausible_upper_bounds = np.minimum(
                        plausible_upper_bounds, upper_bounds
                    )

                idx = plausible_lower_bounds == plausible_upper_bounds
                if np.any(idx):
                    plausible_lower_bounds[idx] = lower_bounds[idx]
                    plausible_upper_bounds[idx] = upper_bounds[idx]
                    self.logger.warning(
                        "vbmc:pbInitFailed: Some plausible bounds could not be"
                        " determined from starting set. Using hard upperlower"
                        " bounds for those instead."
                    )
            else:
                self.logger.warning(
                    "vbmc:pbUnspecified: Plausible lower/upper bounds PLB and"
                    "/or PUB not specified and X0 is not a valid starting set."
                    " Using hard upper/lower bounds instead."
                )
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # check that all bounds are row vectors with D elements
        if (
            np.ndim(lower_bounds) != 2
            or np.ndim(upper_bounds) != 2
            or np.ndim(plausible_lower_bounds) != 2
            or np.ndim(plausible_upper_bounds) != 2
            or lower_bounds.shape != (1, D)
            or upper_bounds.shape != (1, D)
            or plausible_lower_bounds.shape != (1, D)
            or plausible_upper_bounds.shape != (1, D)
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be row vectors with D elements."""
            )

        # check that plausible bounds are finite
        if np.any(np.invert(np.isfinite(plausible_lower_bounds))) or np.any(
            np.invert(np.isfinite(plausible_upper_bounds))
        ):
            raise ValueError(
                "Plausible interval bounds PLB and PUB need to be finite."
            )

        # Test that all vectors are real-valued
        if (
            np.any(np.invert(np.isreal(x0)))
            or np.any(np.invert(np.isreal(lower_bounds)))
            or np.any(np.invert(np.isreal(upper_bounds)))
            or np.any(np.invert(np.isreal(plausible_lower_bounds)))
            or np.any(np.invert(np.isreal(plausible_upper_bounds)))
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be real valued."""
            )

        # Fixed variables (all bounds equal) are not supported
        fixidx = (
            (lower_bounds == upper_bounds)
            & (upper_bounds == plausible_lower_bounds)
            & (plausible_lower_bounds == plausible_upper_bounds)
        )
        if np.any(fixidx):
            raise ValueError(
                """vbmc:FixedVariables VBMC does not support fixed
            variables. Lower and upper bounds should be different."""
            )

        # Test that plausible bounds are different
        if np.any(plausible_lower_bounds == plausible_upper_bounds):
            raise ValueError(
                """vbmc:MatchingPB:For all variables,
            plausible lower and upper bounds need to be distinct."""
            )

        # Check that all X0 are inside the bounds
        if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
            raise ValueError(
                """vbmc:InitialPointsNotInsideBounds: The starting
            points X0 are not inside the provided hard bounds LB and UB."""
            )

        # % Compute "effective" bounds (slightly inside provided hard bounds)
        bounds_range = upper_bounds - lower_bounds
        bounds_range[np.isinf(bounds_range)] = 1e3
        scale_factor = 1e-3
        realmin = sys.float_info.min
        LB_eff = lower_bounds + scale_factor * bounds_range
        LB_eff[np.abs(lower_bounds) <= realmin] = (
            scale_factor * bounds_range[np.abs(lower_bounds) <= realmin]
        )
        UB_eff = upper_bounds - scale_factor * bounds_range
        UB_eff[np.abs(upper_bounds) <= realmin] = (
            -scale_factor * bounds_range[np.abs(upper_bounds) <= realmin]
        )
        # Infinities stay the same
        LB_eff[np.isinf(lower_bounds)] = lower_bounds[np.isinf(lower_bounds)]
        UB_eff[np.isinf(upper_bounds)] = upper_bounds[np.isinf(upper_bounds)]

        if np.any(LB_eff >= UB_eff):
            raise ValueError(
                """vbmc:StrictBoundsTooClose: Hard bounds LB and UB
                are numerically too close. Make them more separate."""
            )

        # Fix when provided X0 are almost on the bounds -- move them inside
        # if np.any(x0 < LB_eff) or np.any(x0 > UB_eff):
        #     self.logger.warning(
        #         "vbmc:InitialPointsTooClosePB: The starting points X0 are on "
        #         + "or numerically too close to the hard bounds LB and UB. "
        #         + "Moving the initial points more inside..."
        #     )
        #     x0 = np.maximum((np.minimum(x0, UB_eff)), LB_eff)

        # Test order of bounds (permissive)
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """vbmc:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Test that plausible bounds are reasonably separated from hard bounds
        if np.any(LB_eff > plausible_lower_bounds) or np.any(
            plausible_upper_bounds > UB_eff
        ):
            self.logger.warning(
                "vbmc:TooCloseBounds: For each variable, hard "
                + "and plausible bounds should not be too close. "
                + "Moving plausible bounds."
            )
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Check that all X0 are inside the plausible bounds,
        # move bounds otherwise
        # if np.any(x0 <= plausible_lower_bounds) or np.any(
        #     x0 >= plausible_upper_bounds
        # ):
        #     self.logger.warning(
        #         "vbmc:InitialPointsOutsidePB. The starting points X0"
        #         + " are not inside the provided plausible bounds PLB and "
        #         + "PUB. Expanding the plausible bounds..."
        #     )
        #     plausible_lower_bounds = np.minimum(
        #         plausible_lower_bounds, x0.min(0)
        #     )
        #     plausible_upper_bounds = np.maximum(
        #         plausible_upper_bounds, x0.max(0)
        #     )

        # Test order of bounds
        ordidx = (
            (lower_bounds < plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds < upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """vbmc:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Check that variables are either bounded or unbounded
        # (not half-bounded)
        if np.any(
            (np.isfinite(lower_bounds) & np.isinf(upper_bounds))
            | (np.isinf(lower_bounds) & np.isfinite(upper_bounds))
        ):
            raise ValueError(
                """vbmc:HalfBounds: Each variable needs to be unbounded or
            bounded. Variables bounded only below/above are not supported."""
            )

        return (
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _init_optim_state(self):
        """
        A private function to init the optim_state dict that contains
        information about VBMC variables.
        """
        # Record starting points (original coordinates)
        y_orig = np.array(self.options.get("fvals")).flatten()
        if self.options.get("specifytargetnoise"):
            sigma = np.array(self.options.get("S_orig")).ravel()
            if np.size(sigma) == 1:
                sigma = np.repeat(sigma, len(y_orig))
        if len(y_orig) == 0:
            y_orig = np.full([self.x0.shape[0]], np.nan)
            sigma = np.full([self.x0.shape[0]], np.nan)
        if len(self.x0) != len(y_orig):
            raise ValueError(
                """vbmc:MismatchedStartingInputs The number of
            points in X0 and of their function values as specified in
            self.options.fvals are not the same."""
            )

        optim_state = dict()
        optim_state["cache"] = dict()
        optim_state["cache"]["x_orig"] = self.x0
        optim_state["cache"]["y_orig"] = y_orig
        if self.options.get("specifytargetnoise"):
            optim_state["cache"]["S_orig"] = sigma
            assert np.all(
                ~np.isnan(
                    optim_state["cache"]["S_orig"][
                        ~np.isnan(optim_state["cache"]["y_orig"])
                    ]
                )
            ), "S_orig need to be provided for all points in X0 with non-nan log likelihoods."

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("y_orig"))
        )
        assert optim_state[
            "cache_active"
        ], "Log density values at x0 need to be provided via kwargs['user_options']['fvals'] = y_orig."

        # fprintf('Index of variable restricted to integer values: %s.\n'
        optim_state["lb_orig"] = self.lower_bounds
        optim_state["ub_orig"] = self.upper_bounds
        optim_state["plb_orig"] = self.plausible_lower_bounds
        optim_state["pub_orig"] = self.plausible_upper_bounds
        eps_orig = (self.upper_bounds - self.lower_bounds) * self.options.get(
            "tolboundx"
        )
        # inf - inf raises warning in numpy, but output is correct
        with np.errstate(invalid="ignore"):
            optim_state["lb_eps_orig"] = self.lower_bounds + eps_orig
            optim_state["ub_eps_orig"] = self.upper_bounds - eps_orig

        # Transform variables (Transform of lower_bounds and upper bounds can
        # create warning but we are aware of this and output is correct)
        with np.errstate(divide="ignore"):
            optim_state["lb"] = self.parameter_transformer(self.lower_bounds)
            optim_state["ub"] = self.parameter_transformer(self.upper_bounds)
        optim_state["plb"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optim_state["pub"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        # here. In MATLAB this was 0.
        optim_state["iter"] = -1

        # Estimate of GP observation noise around the high posterior
        # density region
        optim_state["sn2hpd"] = np.inf

        # Number of warpings performed
        optim_state["warping_count"] = 0

        optim_state["stop_sampling"] = np.Inf

        # Fully recompute variational posterior
        optim_state["recompute_var_post"] = True

        # Quality of the variational posterior
        optim_state["R"] = np.inf

        # Start with adaptive sampling
        optim_state["skip_active_sampling"] = False

        # Running mean and covariance of variational posterior
        # in transformed space
        optim_state["run_mean"] = []
        optim_state["run_cov"] = []
        # Last time running average was updated
        optim_state["last_run_avg"] = np.NaN

        # Current number of components for variational posterior
        optim_state["vpK"] = self.K

        # Number of variational components pruned in last iteration
        optim_state["pruned"] = 0

        # Need to switch from stochastic entropy to deterministic entropy
        optim_state["entropy_switch"] = False

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specifytargetnoise"):
            optim_state["uncertainty_handling_level"] = 2
        elif len(self.options.get("uncertaintyhandling")) > 0:
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        # Repository of variational solutions (not used in Python)
        optim_state["vp_repo"] = []

        # Repeated measurement streak
        optim_state["repeated_observations_streak"] = 0

        # List of data trimming events
        optim_state["data_trim_list"] = []

        # Initialize Gaussian process settings
        # Squared exponential kernel with separate length scales
        optim_state["gp_covfun"] = 1

        if optim_state.get("uncertainty_handling_level") == 0:
            # Observation noise for stability
            optim_state["gp_noisefun"] = [1, 0, 0]
        elif optim_state.get("uncertainty_handling_level") == 1:
            # Infer noise
            optim_state["gp_noisefun"] = [1, 2, 0]
        elif optim_state.get("uncertainty_handling_level") == 2:
            # Provided heteroskedastic noise
            optim_state["gp_noisefun"] = [1, 1, 0]

        if (
            self.options.get("noiseshaping")
            and optim_state["gp_noisefun"][1] == 0
        ):
            optim_state["gp_noisefun"][1] = 1

        optim_state["gp_meanfun"] = self.options.get("gpmeanfun")
        valid_gpmeanfuns = [
            "negquad",
        ]
        if optim_state["gp_meanfun"] not in valid_gpmeanfuns:
            raise ValueError(
                """vbmc:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optim_state["int_meanfun"] = self.options.get("gpintmeanfun")
        optim_state["building_vp"] = False
        return optim_state

    def optimize(self):
        """
        Run inference.

        VSBQ computes a variational approximation of the full posterior and the
        ELBO (evidence lower bound), a lower bound on the log normalization
        constant (log marginal likelhood or log model evidence) for the provided
        unnormalized log posterior.

        Returns
        -------
        vp : VariationalPosterior
            The ``VariationalPosterior`` computed by VBMC.
        elbo : float
            An estimate of the ELBO for the returned `vp`.
        elbo_sd : float
            The standard deviation of the estimate of the ELBO. Note that this
            standard deviation is *not* representative of the error between the
            `elbo` and the true log marginal likelihood.
        results_dict : dict
            A dictionary with additional information about the VBMC run.
        """

        # set up strings for logging of the iteration
        display_format = self._setup_logging_display_format()

        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                "Beginning variational optimization assuming NOISY observations of the log-joint"
            )
        else:
            self.logger.info(
                "Beginning variational optimization assuming EXACT observations of the log-joint."
            )

        self._log_column_headers()

        while not self.is_finished:
            gc.collect()
            self.logger.debug(
                f"Maxrss: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)} MB"
            )
            self.logger.debug(
                f"Iteration start, {psutil.Process().memory_info().rss / (1024 * 1024) :.2f} MB"
            )
            self.iteration += 1
            self.optim_state["iter"] = self.iteration
            self.optim_state["redo_roto_scaling"] = False

            if self.iteration == 0:
                # Fix gp and build vp
                self.optim_state["building_vp"] = True

            vp_old = deepcopy(self.vp)

            self.logging_action = []

            # Switch to stochastic entropy towards the end if still
            # deterministic.
            if self.optim_state.get("entropy_switch") and (
                self.function_logger.func_count
                >= self.optim_state.get("entropy_force_switch")
                * self.optim_state.get("max_fun_evals")
            ):
                self.optim_state["entropy_switch"] = False
                self.logging_action.append("entropy switch")

            ## Actively sample new points into the training set
            gc.collect()
            self.logger.debug(
                f"activeSampling start, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
            )
            self.timer.start_timer("activeSampling")
            print_live_buffers(self.logger)
            self.parameter_transformer = self.vp.parameter_transformer
            self.function_logger.parameter_transformer = (
                self.parameter_transformer
            )

            # Careful with Xn, in MATLAB this condition is > 0
            # due to 1-based indexing.
            if self.function_logger.Xn >= 0:
                self.function_logger.ymax = np.max(
                    self.function_logger.y[self.function_logger.X_flag]
                )

            self.optim_state["hyp_dict"] = self.hyp_dict

            # Number of training inputs
            self.optim_state["N"] = self.function_logger.Xn + 1
            self.optim_state["n_eff"] = np.sum(
                self.function_logger.nevals[self.function_logger.X_flag]
            )

            ## Train gp
            self.timer.start_timer("gpTrain")
            ts = time.time()
            gc.collect()
            self.logger.debug(
                f"gpTrain start, {ts:.2f}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
            )
            print_live_buffers(self.logger)

            Ns_gp = 0
            self.optim_state["stop_sampling"] = (
                1  # No MCMC sampling for SGPR hyperparameters
            )

            # Utility for searching good noise shaping hyperparameters (if multiple values are provided), and increase number of inducing points if the sparse GP is not fitting well
            if self.iteration == 0:
                bad_fitting_alarm = True
                count = 0

                self.logger.debug("First iteration, train SGPR")
                while bad_fitting_alarm:
                    optim_state = deepcopy(self.optim_state)
                    options = deepcopy(self.options)
                    num_ips = (
                        self.options["num_ips"]
                        + count * self.options["num_ips_increment"]
                    )  # Increase number of inducing points
                    options.__setitem__("num_ips", num_ips, force=True)
                    self.logger.debug(f"Number of inducing points: {num_ips}")
                    (
                        noise_shaping_factor,
                        noise_shaping_threshold,
                        gp,
                        sn2hpd,
                        hyp_dict,
                        optim_state,
                    ) = search_noise_shaping_hyperparams(
                        self.function_logger,
                        optim_state,
                        options,
                        optim_state["plb"],
                        optim_state["pub"],
                    )
                    bad_fitting_alarm = check_train_set_predictions(
                        gp,
                        self.function_logger.noise_flag,
                        {
                            "noiseshapingthreshold": noise_shaping_threshold,
                            "fast_debugging": options.get("fast_debugging"),
                        },
                    )
                    count += 1

                    if (
                        count > self.options["max_retries_gp_retrain"]
                        and bad_fitting_alarm
                    ):
                        raise RuntimeError(
                            f'Could not find good noise shaping hyperparams with {gp.params_cache["inducing_points"].shape[0]} inducing points.'
                        )

                # Set the best noise shaping hyperparameters
                self.options.__setitem__(
                    "noiseshapingfactor", noise_shaping_factor, force=True
                )
                self.options.__setitem__(
                    "noiseshapingthreshold",
                    noise_shaping_threshold,
                    force=True,
                )
                self.options.__setitem__(
                    "num_ips",
                    gp.params_cache["inducing_points"].shape[0],
                    force=True,
                )
                self.gp = gp
                self.logger.debug(
                    f"Found good noise shaping hyperparams {(noise_shaping_factor, noise_shaping_threshold)} with {gp.params_cache['inducing_points'].shape[0]} inducing points."
                )
                self.optim_state["sn2hpd"] = sn2hpd
                self.hyp_dict = hyp_dict
                self.optim_state = optim_state

            if self.iteration == 0:
                gp_is_retrained = False
                count = 0
                bad_fitting_alarm = True
                while bad_fitting_alarm:
                    # Need retraining
                    self.logger.debug(
                        f"Retraining SGPR. Current number of inducing points: {self.gp.params_cache['inducing_points'].shape[0]}"
                    )
                    optim_state = deepcopy(self.optim_state)
                    options = deepcopy(self.options)
                    options.__setitem__(
                        "num_ips",
                        self.gp.params_cache["inducing_points"].shape[0]
                        + count * self.options["num_ips_increment"],
                        force=True,
                    )
                    # print("num_ips: ", options["num_ips"])
                    hyp_dict = deepcopy(self.hyp_dict)
                    iteration_history = deepcopy(self.iteration_history)

                    optim_state["reselect_inducing_points"] = True
                    gp, sn2hpd, hyp_dict = train_sgpr(
                        hyp_dict,
                        optim_state,
                        self.function_logger,
                        iteration_history,
                        options,
                        optim_state["plb"],
                        optim_state["pub"],
                    )
                    gp_is_retrained = True
                    bad_fitting_alarm = check_train_set_predictions(
                        gp, self.function_logger.noise_flag, options
                    )
                    count += 1

                    if count > self.options["max_retries_gp_retrain"]:
                        # TODO: dump to file for debugging?
                        raise RuntimeError(
                            f'Could not fit a good SGPR with {gp.params_cache["inducing_points"].shape} inducing points on current train set.'
                        )
                if gp_is_retrained:
                    self.gp = gp
                    self.optim_state["sn2hpd"] = sn2hpd
                    self.hyp_dict = hyp_dict
                    self.optim_state = optim_state
                    self.options.__setitem__(
                        "num_ips",
                        self.gp.params_cache["inducing_points"].shape[0],
                        force=True,
                    )

            self.timer.stop_timer("gpTrain")
            gc.collect()
            self.logger.debug(
                f"gpTrain finish, {time.time() - ts :.2f}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
            )
            print_live_buffers(self.logger)

            ## Optimize variational parameters
            self.timer.start_timer("variationalFit")

            # Update number of variational mixture components
            Knew = self.vp.K

            # Turn off fast sieve optimization. fast_opts_N = 0, slow_opts_N=1
            N_fastopts = 0
            N_slowopts = 1

            ts = time.time()
            gc.collect()
            self.logger.debug(
                f"optimize_vp start, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            print_live_buffers(self.logger)
            # Run optimization of variational parameters
            self.vp, varss, pruned = optimize_vp(
                self.options,
                self.optim_state,
                self.vp,
                self.gp,
                N_fastopts,
                N_slowopts,
                Knew,
            )
            gc.collect()
            self.logger.debug(
                f"optimize_vp finish, {time.time() - ts:.2f}s, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            print_live_buffers(self.logger)
            self.optim_state["vpK"] = self.vp.K
            # Save current entropy
            self.optim_state["H"] = self.vp.stats["entropy"]

            elbo = self.vp.stats["elbo"]
            elbo_sd = self.vp.stats["elbo_sd"]

            self.timer.stop_timer("variationalFit")
            gc.collect()
            self.logger.debug(
                f"Variational fit time: {self.timer.get_duration('variationalFit')}"
            )

            # Finalize iteration
            self.timer.start_timer("finalize")
            gc.collect()
            self.logger.debug(
                f"kldiv start, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            print_live_buffers(self.logger)
            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5

            sKL = max(
                0,
                0.5
                * np.sum(
                    self.vp.kldiv(
                        vp2=vp_old,
                        N=Nkl,
                        gaussflag=self.options.get("klgauss"),
                    )
                ),
            )
            gc.collect()
            self.logger.debug(
                f"kldiv finish, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            print_live_buffers(self.logger)
            ts = time.time()
            gc.collect()

            self.optim_state["lcbmax"] = None
            print_live_buffers(self.logger)
            self.timer.stop_timer("finalize")

            # store current gp in vp
            vp_save = self.vp
            ts = time.time()
            gc.collect()
            self.logger.debug(
                f"record iteration value, {ts}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            gp_save = copy_sgpr(self.gp)
            assert gp_save.post_cache is None

            iteration_values = {
                "iter": self.iteration,
                "optim_state": self.optim_state,
                "vp": vp_save,
                "elbo": elbo,
                "elbo_sd": elbo_sd,
                "varss": varss,
                "sKL": sKL,
                "gp": gp_save,
                "Ns_gp": Ns_gp,
                "pruned": pruned,
                "timer": self.timer,
                "func_count": self.function_logger.func_count,
                "lcbmax": self.optim_state["lcbmax"],
                "n_eff": self.optim_state["n_eff"],  # number of evaluations
                "N_train": self.gp.X.shape[0],  # number of train points
                # "function_logger": self.function_logger,  # For debug
            }

            # Record all useful stats
            self.iteration_history.record_iteration(
                iteration_values,
                self.iteration,
            )
            gc.collect()
            self.logger.debug(
                f"record iteration value, {time.time() -ts:.2f}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB memory"
            )
            print_live_buffers(self.logger)

            # Check termination conditions
            (
                self.is_finished,
                termination_message,
            ) = self._check_termination_conditions()

            # Write iteration output
            if self.options.get("printiterationheader") is None:
                # Default behavior, try to guess based on plotting options:
                reprint_headers = (
                    self.options.get("plot")
                    and self.iteration > 0
                    and "inline" in plt.get_backend()
                )
            elif self.options["printiterationheader"]:
                # Re-print every iteration after 0th
                reprint_headers = self.iteration > 0
            else:
                # Never re-print headers
                reprint_headers = False
            # Reprint the headers if desired:
            if reprint_headers:
                self._log_column_headers()

            if self.optim_state["cache_active"]:
                self.logger.info(
                    display_format.format(
                        self.iteration,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                    )
                )

            else:
                if (
                    self.optim_state["uncertainty_handling_level"] > 0
                    and self.options.get("maxrepeatedobservations") > 0
                ):
                    self.logger.info(
                        display_format.format(
                            self.iteration,
                            self.optim_state["N"],
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                        )
                    )
                else:
                    self.logger.info(
                        display_format.format(
                            self.iteration,
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                        )
                    )
            self.iteration_history.record(
                "logging_action", self.logging_action, self.iteration
            )

            # Plot iteration
            if self.options.get("plot"):
                if (
                    self.iteration > 0
                    and self.options.get("funevalsperiter") != 0
                ):
                    previous_gp = self.iteration_history["gp"][
                        self.iteration - 1
                    ]
                    # find points that are new in this iteration
                    # (hacky cause numpy only has 1D set diff)
                    # future fix: active sampling should return the set of
                    # indices of the added points
                    highlight_data = np.array(
                        [
                            i
                            for i, x in enumerate(self.gp.X)
                            if tuple(x) not in set(map(tuple, previous_gp.X))
                        ]
                    )
                else:
                    highlight_data = None

                if len(self.logging_action) > 0:
                    title = "VBMC iteration {} ({})".format(
                        self.iteration, "".join(self.logging_action)
                    )
                else:
                    title = "VBMC iteration {}".format(self.iteration)

                fig = self.vp.plot_with_extra_data(
                    train_X=self.gp.X,
                    highlight_data=highlight_data,
                    plot_vp_centres=True,
                    title=title,
                    figure_size=(3 * self.vp.D, 3 * self.vp.D),
                    original_space=None,
                    save_path=os.path.join(
                        self.options["experimentfolder"],
                        f"vp_{self.iteration}.png",
                    ),
                )

            self.random_state = np.random.get_state()
            try:
                self.iteration_history.record_iteration(
                    {"random_state": self.random_state},
                    self.iteration,
                )
            except:
                pass

            if np.isfinite(self.options["checkpointiters"]) and (
                self.iteration % self.options["checkpointiters"] == 0
                or self.is_finished
            ):
                self.save(
                    os.path.join(
                        self.options["experimentfolder"],
                        "vbmc-0.pkl",
                    )
                )

            self.logger.debug("End of iteration.")
            print_live_buffers(self.logger)

            # clear_backends()  # Something seems wrong with this function, https://github.com/google/jax/issues/10828#issuecomment-1835805181
            gc.collect()

        # Pick "best" variational solution to return
        self.vp, elbo, elbo_sd, idx_best = self.determine_best_vp()
        gp = self.iteration_history["gp"][idx_best]

        # Print final message
        self.logger.warning(termination_message)
        self.logger.warning(
            "Estimated ELBO: {:.3f} +/-{:.3f}.".format(elbo, elbo_sd)
        )

        result_dict = self._create_result_dict(idx_best, termination_message)

        self.logger.info("Plotting final variational posterior...")
        fig = self.vp.plot_with_extra_data(
            n_samples=int(1e5),
            train_X=self.gp.X,
            figure_size=(3 * self.D, 3 * self.D),
            extra_data=self.gp.params_cache["inducing_points"],
            title="final vp",
            plot_vp_centres=True,
            original_space=None,
            save_path=os.path.join(
                self.options["experimentfolder"], "vp_final.png"
            ),
        )
        self.logger.info(
            f"Plotting done. Saved to folder {self.options['experimentfolder']}"
        )

        # jax.profiler.stop_trace()
        return (
            deepcopy(self.vp),
            self.vp.stats["elbo"],
            self.vp.stats["elbo_sd"],
            "Check values of ELBO and ELBO_SD at each iterations to see if the optimization converged.",
            result_dict,
        )

    def _check_termination_conditions(self):
        """
        Private method to determine the status of termination conditions.

        It also saves the reliability index, ELCBO improvement and stableflag
        to the iteration_history object.
        """
        is_finished_flag = False
        termination_message = ""
        # Maximum number of iterations
        iteration = self.optim_state.get("iter")

        if iteration + 1 >= self.options.get("maxiter"):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of iterations options.maxiter."
            )
        return (
            is_finished_flag,
            termination_message,
        )

    def _compute_reliability_index(self, tol_stable_iters):
        """
        Private function to compute the reliability index.
        """
        iteration_idx = self.optim_state.get("iter")
        # Was < 3 in MATLAB due to different indexing.
        if self.optim_state.get("iter") < 2:
            rindex = np.Inf
            ELCBO_improvement = np.NaN
            return rindex, ELCBO_improvement

        sn = np.sqrt(self.optim_state.get("sn2hpd"))
        tol_sn = np.sqrt(sn / self.options.get("tolsd")) * self.options.get(
            "tolsd"
        )
        tol_sd = min(
            max(self.options.get("tolsd"), tol_sn),
            self.options.get("tolsd") * 10,
        )

        rindex_vec = np.full((3), np.NaN)
        rindex_vec[0] = (
            np.abs(
                self.iteration_history.get("elbo")[iteration_idx]
                - self.iteration_history.get("elbo")[iteration_idx - 1]
            )
            / tol_sd
        )
        rindex_vec[1] = (
            self.iteration_history.get("elbo_sd")[iteration_idx] / tol_sd
        )
        rindex_vec[2] = self.iteration_history.get("sKL")[
            iteration_idx
        ] / self.options.get("tolskl")

        # Compute average ELCBO improvement per fcn eval in the past few iters
        # TODO: off by one error
        idx0 = int(
            max(
                0,
                self.optim_state.get("iter")
                - math.ceil(0.5 * tol_stable_iters),
            )
        )
        # Remember than upper end of range is exclusive in Python, so +1 is
        # needed.
        xx = self.iteration_history.get("func_count")[idx0 : iteration_idx + 1]
        yy = (
            self.iteration_history.get("elbo")[idx0 : iteration_idx + 1]
            - self.options.get("elcboimproweight")
            * self.iteration_history.get("elbo_sd")[idx0 : iteration_idx + 1]
        )
        # need to casts here to get things to run
        try:
            ELCBO_improvement = np.polyfit(
                list(map(float, xx)), list(map(float, yy)), 1
            )[0]
        except:
            ELCBO_improvement = np.NaN
        self.logger.debug(f"rindex: {rindex_vec}")
        return np.mean(rindex_vec), ELCBO_improvement

    def determine_best_vp(
        self,
        max_idx: int = None,
        safe_sd: float = 5,
        frac_back: float = 0.25,
        rank_criterion_flag: bool = False,
    ):
        """
        Return the best VariationalPosterior found during the optimization of
        VBMC as well as its ELBO, ELBO_SD and the index of the iteration.

        Parameters
        ----------
        max_idx : int, optional
            Check up to this iteration, by default None which means last iter.
        safe_sd : float, optional
            Penalization for uncertainty, by default 5.
        frac_back : float, optional
            If no past stable iteration, go back up to this fraction of
            iterations, by default 0.25.
        rank_criterion_flag : bool, optional
            If True use new ranking criterion method to pick best solution.
            It finds a solution that combines ELCBO, stability, and recency,
            by default False.

        Returns
        -------
        vp : VariationalPosterior
            The VariationalPosterior found during the optimization of VBMC.
        elbo : float
            The ELBO of the iteration with the best VariationalPosterior.
        elbo_sd : float
            The ELBO_SD of the iteration with the best VariationalPosterior.
        idx_best : int
            The index of the iteration with the best VariationalPosterior.
        """
        # Check up to this iteration (default, last)
        if max_idx is None:
            max_idx = self.iteration_history.get("iter")[-1]

        if self.options["funevalsperiter"] == 0:
            idx_best = max_idx
        else:
            raise NotImplementedError("")

        # Return best variational posterior, its ELBO and SD
        vp = self.iteration_history.get("vp")[idx_best]
        elbo = self.iteration_history.get("elbo")[idx_best]
        elbo_sd = self.iteration_history.get("elbo_sd")[idx_best]
        return vp, elbo, elbo_sd, int(idx_best)

    def _create_result_dict(self, idx_best: int, termination_message: str):
        """
        Private method to create the result dict.
        """
        output = dict()
        # output["function"] = str(self.function_logger.fun)
        if np.all(np.isinf(self.optim_state["lb"])) and np.all(
            np.isinf(self.optim_state["ub"])
        ):
            output["problemtype"] = "unconstrained"
        else:
            output["problemtype"] = "boundconstraints"

        output["iterations"] = self.optim_state["iter"]
        output["gp_elbo"] = self.optim_state["gp_elbo"]
        output["funccount"] = self.function_logger.func_count
        output["bestiter"] = idx_best
        output["n_eff"] = self.iteration_history["n_eff"][idx_best].item()
        output["N_train"] = self.iteration_history["N_train"][idx_best]
        output["components"] = self.vp.K
        output["message"] = termination_message

        output["elbo"] = self.vp.stats["elbo"]
        output["elbo_sd"] = self.vp.stats["elbo_sd"]

        return output

    def _log_column_headers(self):
        """
        Private method to log column headers for the iteration log.
        """
        # We only want to log the column headers once when writing to a file,
        # but we re-write them to the stream (stdout) when plotting.
        if self.optim_state.get("iter") > 0:
            logger = self.logger.stream_only
        else:
            logger = self.logger

        if self.optim_state["cache_active"]:
            logger.info(
                " Iteration     Mean[ELBO]     Std[ELBO]     "
                + "sKL-iter[q]   K[q]"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                logger.info(
                    " Iteration   f-count (x-count)   Mean[ELBO]     Std[ELBO]"
                    + "     sKL-iter[q]   K[q]"
                )
            else:
                logger.info(
                    " Iteration  f-count    Mean[ELBO]    Std[ELBO]    "
                    + "sKL-iter[q]   K[q]"
                )

    def _setup_logging_display_format(self):
        """
        Private method to set up the display format for logging the iterations.
        """
        if self.optim_state["cache_active"]:
            display_format = " {:5.0f}     {:12.2f}  "
            display_format += "{:12.2f}  {:12.2f}     {:4.0f}  "
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                display_format = " {:5.0f}       {:5.0f} {:5.0f} {:12.2f}  "
                display_format += (
                    "{:12.2f}  {:12.2f}     {:4.0f} {:10.3g}     "
                )
                display_format += "{}"
            else:
                display_format = " {:5.0f}      {:5.0f}   {:12.2f} {:12.2f} "
                display_format += "{:12.2f}     {:4.0f} {:10.3g}     {}"

        return display_format

    def _init_logger(self, substring=""):
        """
        Private method to initialize the logging object.

        Parameters
        ----------
        substring : str
            A substring to append to the logger name (used to create separate
            logging objects for initialization and optimization, in case
            options change in between). Default "" (empty string).

        Returns
        -------
        logger : logging.Logger
            The main logging interface.
        """
        # set up VBMC logger
        logger = logging.getLogger("VBMC" + substring)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        level = logging.INFO
        if self.options.get("display") == "off":
            level = logging.WARN
        elif self.options.get("display") == "iter":
            level = logging.INFO
        elif self.options.get("display") == "full":
            level = logging.DEBUG
        logger.setLevel(level)

        # Add a special logger for sending messages only to the default stream:
        logger.stream_only = logging.getLogger("VBMC.stream_only")

        # Options and special handling for writing to a file:

        # If logging for the first time, get write mode from user options
        # (default "a" for append)
        if substring == "_init":
            log_file_mode = self.options.get("logfilemode", "a")
        # On subsequent writes, switch to append mode:
        else:
            log_file_mode = "a"

        # Avoid duplicating a handler for the same log file
        # (remove duplicates, re-add below)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for filter in logger.filters[:]:
            logger.removeFilter(filter)

        if self.options.get("logfilename") and self.options.get(
            "logfilelevel"
        ):
            file_handler = logging.FileHandler(
                filename=os.path.join(
                    self.options["experimentfolder"],
                    self.options["logfilename"],
                ),
                mode=log_file_mode,
            )

            # Set file logger level according to string or logging level:
            log_file_level = self.options.get("logfilelevel", logging.INFO)
            if log_file_level == "off":
                file_handler.setLevel(logging.WARN)
            elif log_file_level == "iter":
                file_handler.setLevel(logging.INFO)
            elif log_file_level == "full":
                file_handler.setLevel(logging.DEBUG)
            elif log_file_level in [0, 10, 20, 30, 40, 50]:
                file_handler.setLevel(log_file_level)
            else:
                raise ValueError(
                    "Log file logging level is not a recognized"
                    + "string or logging level."
                )

            # Add a filter to ignore messages sent to logger.stream_only:
            def log_file_filter(record):
                return record.name != "VBMC.stream_only"

            file_handler.addFilter(log_file_filter)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        consoleHandler.setLevel(level)
        logger.addHandler(consoleHandler)
        logger.propagate = False
        return logger

    def save(self, filepath):
        with open(filepath, "wb") as f:
            dill.dump(self, f)
