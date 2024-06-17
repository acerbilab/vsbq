import logging
import math
import numbers
import time
from copy import deepcopy
from typing import Dict, Optional

import gpyreg as gpr
import jax
import jax.numpy as jnp
import jaxgp as jgp
import jaxopt
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax._src.api import clear_backends
from jax.tree_util import tree_leaves
from jaxgp.contrib.train_utils import train_model
from jaxgp.kernels import cross_covariance
from jaxgp.priors import TruncatedPrior
from jaxgp.utils import concat_dictionaries
from plum import dispatch

# %%
from scipy.special import erfc
from scipy.stats import chi2
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.contrib.itertools import product

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.stats import get_hpd

from .. import settings
from .gaussian_process_train_utils import *
from .gaussian_process_train_utils import (
    _cov_identifier_to_covariance_function,
    _meanfun_name_to_mean_function,
)
from .iteration_history import IterationHistory
from .options import Options

logger = logging.getLogger("VBMC_debug")


def nstd_threshold(n1, d):
    """Ref: https://arxiv.org/abs/2211.02045"""
    return chi2.isf(erfc(n1 / np.sqrt(2)), d) / 2


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def check_train_set_predictions(
    model: jgp.HeteroskedasticSGPR, noisy, options=None
):
    """Return True if the model is bad."""
    if options.get("fast_debugging", False):
        return False
    y_pred, y_pred_s2 = model.post_cache.predict_f_with_precomputed(
        model.train_data.X
    )
    y_pred = y_pred.squeeze()
    y_train = model.train_data.Y.squeeze()
    if noisy:
        assert options is not None
        # Check outside 2-sigma interval fraction
        n_std = 2
        s_train = np.sqrt(model.sigma_sq_user.squeeze())
        y_train_max = np.max(y_train)
        mask = y_train >= y_train_max - options["noiseshapingthreshold"]

        f1 = np.sum(
            (
                (y_pred > y_train + n_std * s_train)
                | (y_train - n_std * s_train > y_pred)
            )[mask]
        ) / np.sum(mask)
        if np.sum(~mask) > 0:
            f2 = np.sum((y_pred > y_train + n_std * s_train)[~mask]) / np.sum(
                ~mask
            )
        else:
            f2 = 0
        f = np.maximum(f1, f2)
        logger.debug(f"Fraction of points outside: {f1:.3f}, {f2:.3f}")
        return (
            f > 0.5
        )  # 50% of points outside 2-sigma interval, a very empirical number
    else:
        # Check max absolute error
        ymax = y_train.max()
        y_train_pdf = np.exp(y_train - ymax)
        y_pred_pdf = np.exp(y_pred - ymax)
        # abs_err = mean_absolute_error(y_train_pdf, y_pred_pdf)
        # squared_err = mean_squared_error(y_train_pdf, y_pred_pdf)
        err = np.max(np.abs(y_train_pdf - y_pred_pdf))
        logger.debug(f"Max absolute error: {err}")
        return err > 0.5


def search_noise_shaping_hyperparams(
    function_logger, optim_state, options, plb, pub
):
    D = function_logger.D
    # TODO: read from ini file
    noise_shaping_factors = options["noiseshapingfactors"]
    if options.get("noiseshapingthresholds_instd"):
        ns_thresholds = nstd_threshold(options["noiseshapingthresholds"], D)
    else:
        ns_thresholds = options["noiseshapingthresholds"]

    if isinstance(noise_shaping_factors, numbers.Number):
        noise_shaping_factors = [noise_shaping_factors]
    if isinstance(ns_thresholds, numbers.Number):
        ns_thresholds = [ns_thresholds]

    logger = settings.get_default_debug_logger()
    logger.debug(f"noise_shaping_factors = {noise_shaping_factors}")
    logger.debug(f"ns_thresholds = {ns_thresholds}")

    best_nll = np.inf
    best_noise_shape_factor = None
    best_ns_threshold = None
    best_model = None
    best_sn2hpd = None
    best_hyp_dict = None
    for noise_shape_factor, ns_threshold in product(
        noise_shaping_factors,
        ns_thresholds,
        disable=settings.progress_bar.off(),
    ):
        options_local = deepcopy(options)
        options_local.is_initialized = False
        options_local["noiseshapingthreshold"] = ns_threshold
        options_local["noiseshapingfactor"] = noise_shape_factor
        options_local["maxgpoptsN"] = (
            1  # only one optimization for faster search
        )

        optim_state_local = deepcopy(optim_state)
        function_logger_local = deepcopy(
            function_logger
        )  # TODO: no need for deepcopy?

        with settings.return_aux(True):
            model, sn2hpd, hyp_dict, nll = train_sgpr(
                None,
                optim_state_local,
                function_logger_local,
                {},
                options_local,
                plb,
                pub,
            )
        neg_elbo = np.min(nll)
        if neg_elbo < best_nll:
            best_nll = neg_elbo
            best_noise_shape_factor = noise_shape_factor
            best_ns_threshold = ns_threshold
            best_model = model
            best_sn2hpd = sn2hpd
            best_hyp_dict = hyp_dict
            best_optim_state = optim_state_local

    return (
        best_noise_shape_factor,
        best_ns_threshold,
        best_model,
        best_sn2hpd,
        best_hyp_dict,
        best_optim_state,
    )


def kl_univariate_gaussian(m1, sigma1, m2, sigma2):
    """KL(p,q)
    p ~ N(m1,sigma1^2), q~N(m2,sigma2^2)
    """
    m1 = np.array(m1).squeeze()
    m2 = np.array(m2).squeeze()
    sigma1 = np.array(sigma1).squeeze()
    sigma2 = np.array(sigma2).squeeze()
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (m1 - m2) ** 2) / (2 * sigma2**2)
        - 0.5
    )


def count_params(params: Optional[Dict] = None) -> int:
    if params is None:
        return 0
    l = tree_leaves(params)
    cnt = 0
    for p in l:
        cnt += p.size
    return cnt


def noise_shaping(s2, y, options: Options, ymax: Optional[float] = None):
    # Increase noise for low density points
    if s2 is None:
        s2 = options["tolgpnoise"] ** 2 * np.ones_like(y)

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
    return s2s


def train_sgpr(
    hyp_dict: dict,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
):
    if hyp_dict is None:
        hyp_dict = {}

    current_iter = optim_state["iter"]

    warped_flag = False  # Warp the inference space, not supported
    reselect_ip_flag = False
    if (
        optim_state.get("reselect_inducing_points", False)
        or warped_flag
        or optim_state.get("inducing_points") is None
    ):
        # Need to reselect new inducing points
        reselect_ip_flag = True
        # Reset to False for the future iterations
        optim_state["reselect_inducing_points"] = False

    if optim_state.get("use_fixed_inducing_points"):
        reselect_ip_flag = False

    hyp_dict_exact = None
    if "last_exact_gp" not in optim_state or warped_flag or reselect_ip_flag:
        select_subset = False
        if optim_state["N"] >= 500:
            select_subset = True

        hyp_dict_exact = deepcopy(hyp_dict)
        if "last_exact_gp" in optim_state:
            hyp = optim_state["last_exact_gp"].get_hyperparameters(
                as_array=True
            )
            if hyp_dict_exact.get("hyp") is None:
                hyp_dict_exact["hyp"] = np.empty((0, hyp.shape[1]))
            hyp_dict_exact["hyp"] = np.concatenate(
                [hyp_dict_exact["hyp"], hyp]
            )
        gp, _, _, hyp_dict_exact = train_gp(
            hyp_dict_exact,
            optim_state,
            function_logger,
            iteration_history,
            options,
            optim_state["plb"],
            optim_state["pub"],
            select_subset,
        )
        optim_state["last_exact_gp_iter"] = current_iter - 1
        optim_state["last_exact_gp"] = gp
        if "first_exact_gp" not in optim_state or warped_flag:
            # keep the first exact gp and the first exact gp after rotoscaling
            optim_state["first_exact_gp"] = gp

    # Get training dataset.
    x_train, y_train, s2_train, t_train = _get_training_data(
        function_logger, options
    )

    ################
    x_sub, y_sub, s2_sub = select_training_subset(
        x_train, y_train, s2_train, options["subsetsize"]
    )

    D = x_train.shape[1]

    # Pick the mean function
    mean_f = _meanfun_name_to_mean_function(optim_state["gp_meanfun"])

    # Pick the covariance function.
    covariance_f = _cov_identifier_to_covariance_function(
        optim_state["gp_covfun"]
    )

    # Pick the noise function.
    const_add = optim_state["gp_noisefun"][0] == 1
    user_add = optim_state["gp_noisefun"][1] == 1
    user_scale = optim_state["gp_noisefun"][1] == 2
    rlod_add = optim_state["gp_noisefun"][2] == 1
    noise_f = gpr.noise_functions.GaussianNoise(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    # Setup a GP.
    exact_gp = gpr.GP(D=D, covariance=covariance_f, mean=mean_f, noise=noise_f)

    exact_gp, hyp0, gp_s_N = _gp_hyp(
        optim_state, options, plb, pub, exact_gp, x_train, y_train
    )
    # Set data inputs for exact gp
    exact_gp.X = x_train
    exact_gp.y = y_train
    exact_gp.s2 = s2_train

    logger.debug(f"{x_train.shape[0]} training points for gp")
    logger.debug(
        f"top log joint values: {np.sort(function_logger.y_orig[function_logger.X_flag], axis=None)[::-1][:10]}"
    )
    logger.debug(
        f"bottom log joint values: {np.sort(function_logger.y_orig[function_logger.X_flag], axis=None)[:10]}"
    )
    logger.debug(f"x_train max: {np.max(x_train, 0)}")
    logger.debug(f"x_train min: {np.min(x_train, 0)}")

    ## Set up SGPR model
    if (
        optim_state["gp_meanfun"] == "negquad"
        and optim_state["gp_covfun"] == 1
    ):
        mean = jgp.means.Quadratic(input_dim=D)
        kernel = jgp.kernels.RBF(active_dims=tuple(range(x_train.shape[-1])))
    else:
        raise NotImplementedError

    likelihood = jgp.likelihoods.HeteroskedasticGaussianVBMC(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    gpyreg_bounds, LB, UB, PLB, PUB = get_bounds_gpyreg(
        exact_gp, x_train, y_train, options
    )
    jaxgp_bounds = gpyreg_bounds_to_jaxgp(gpyreg_bounds)
    hyp_prior = gpyreg_priors_to_jaxgp(exact_gp.get_priors(), gpyreg_bounds)
    train_data = jgp.Dataset(X=x_train, Y=y_train)

    model = jgp.HeteroskedasticSGPR(
        train_data=train_data,
        gprior=jgp.GPrior(kernel=kernel, mean_function=mean),
        likelihood=likelihood,
        inducing_points=x_train,  # just a default value for inducing points
        sigma_sq_user=s2_train,
        hyp_prior=hyp_prior,
    )

    assert gp_s_N == 0

    # # Get GP training options.
    init_N = 9
    gp_train = {}
    # Set other hyperparameter fitting parameters
    gp_train["init_N"] = init_N
    gp_train["opts_N"] = options["maxgpoptsN"]

    logger.debug(f"gp_train option is {gp_train}")

    # Build starting points
    hyp0 = np.reshape(hyp0, (-1, hyp0.T.shape[0]))

    # Clunky workaround for different shaped arrays:
    if hyp_dict_exact is not None:
        if len(hyp_dict_exact["hyp"].shape) == 2:
            hyp0 = np.concatenate((hyp0, np.array(hyp_dict_exact["hyp"])))
        else:
            hyp0 = np.concatenate((hyp0, np.array([hyp_dict_exact["hyp"]])))

    hyp0 = np.unique(hyp0, axis=0)

    # In some cases the model can change so be careful. (Not supported)
    if hyp0.shape[1] != np.size(exact_gp.hyper_priors["mu"]):
        raise ValueError("hyp0 has wrong shape.")
        # hyp0 = None

    if reselect_ip_flag:
        Z, select_info = select_inducing_points(
            model, optim_state, iteration_history, options
        )
    else:
        Z = optim_state["inducing_points"]

    # Define model
    model_gpr = jgp.HeteroskedasticSGPR(
        train_data=jgp.Dataset(x_sub, y_sub),
        gprior=jgp.GPrior(kernel=kernel, mean_function=mean),
        likelihood=likelihood,
        inducing_points=x_sub,
        sigma_sq_user=s2_sub,
        hyp_prior=hyp_prior,
    )
    neg_elbo_gpr = jax.jit(model_gpr.build_elbo(sign=-1.0))

    neg_elbo_raw = jax.jit(model.build_elbo(sign=-1.0))
    neg_elbo = jax.jit(model.build_elbo(sign=-1.0, raw_flag=False))

    # Create aliases for vbmc
    model.X = train_data.X
    model.y = train_data.Y
    model.D = D
    model.temporary_data = {}
    model.posteriors = [None]  # for np.size(gp.posteriors)

    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    constrain_trans = jax.jit(constrain_trans)
    unconstrain_trans = jax.jit(unconstrain_trans)

    Z, inducing_bounds = keep_inducing_points_with_bounds(Z, x_train, options)
    bounds = get_jaxgp_unconstrained_bounds(
        jaxgp_bounds, inducing_bounds, unconstrain_trans
    )

    if optim_state.get("oracle_gp_params"):
        print("load gp params")
        # test with an "oracle" gp
        final_params = optim_state.get("oracle_gp_params")
        sn2hpd = _estimate_noise_sgpr(model, final_params)

        model.params_cache = final_params
        model.bounds = bounds
        optim_state["inducing_points"] = final_params["inducing_points"]
        for k, v in final_params.items():
            if k != "inducing_points":
                logger.debug(f"{k}: {v}")
            else:
                logger.debug(f"Number of inducing points: {v.shape[0]}")

        model.post_cache = model.posterior(final_params)
        logger.debug("Posterior cached.")
        final_params_gpyreg = jaxgp_params_to_gpyreg(final_params)
        final_params_gpyreg = exact_gp.hyperparameters_from_dict(
            final_params_gpyreg
        )
        hyp_dict["hyp"] = final_params_gpyreg
        hyp_dict["inducing_points"] = final_params["inducing_points"]

        nll = [neg_elbo(final_params).item()]
        optim_state["gp_elbo"] = np.min(nll).item()
        if settings.return_aux.on():
            return model, sn2hpd, hyp_dict, nll
        return model, sn2hpd, hyp_dict

    # The initial hyperparameters for optimizations come from three sources: (Z, hyp_select), (Z, hyp_random), and (Z_pre, hyp_pre).
    tol = gp_train.get("tol_opt", 1e-5)
    init_N = gp_train.get("init_N", 2**10)
    init_method = gp_train.get("init_method", "sobol")
    opts_N = gp_train.get("opts_N", 3)
    ts = time.time()
    # 1. (Z, hyp_random)
    if init_N >= 0:
        X0, y0 = f_min_fill(
            lambda x: np.inf,  # hack
            None,  # Purely random initialization without including hyp0 to avoid duplicate initial hyperparameters
            LB,
            UB,
            PLB,
            PUB,
            exact_gp.hyper_priors,
            init_N,
            init_method,
        )
        for i in range(X0.shape[0]):
            ts_1 = time.time()
            hyp_cur = X0[i]
            hyp_cur = exact_gp.hyperparameters_to_dict(hyp_cur)
            hyp_cur = gpyreg_params_to_jaxgp(hyp_cur[0])
            hyp_cur = concat_dictionaries({"inducing_points": Z}, hyp_cur)
            hyp_cur["likelihood"]["noise_add"] = jnp.array([1e-5])
            y0[i] = np.array(neg_elbo(hyp_cur))
            if i % 50 == 0:
                if i == 0:
                    logger.debug(f"first eval: {time.time() - ts_1:.2f}")
                else:
                    print(f"first eval: {time.time() - ts_1}")
        y0[np.isnan(y0)] = np.inf
        order = np.argsort(y0)
        X0, y0 = X0[order, :], y0[order]

        # Make sure we have at least one hyperparameter to use later.
        hyp = X0[0 : np.maximum(opts_N, 1), :]
        hyp_neg_elbos = y0[0 : np.maximum(opts_N, 1)]
    else:
        raise ValueError
    logger.debug(f"Time for init_N: {time.time() - ts}")
    # Check that hyperparameters are within bounds.
    # Note that with infinite upper and lower bounds we have to be careful
    # with spacing since it returns NaN. Furthermore, if LB == UB then
    # we have to be careful about the lower bound not being larger than
    # the upper bounds. Also, copy is necessary to avoid LB or UB
    # getting modified.
    eps_LB = np.reshape(LB.copy(), (1, -1))
    eps_UB = np.reshape(UB.copy(), (1, -1))
    LB_idx = (eps_LB != eps_UB) & np.isfinite(eps_LB)
    UB_idx = (eps_LB != eps_UB) & np.isfinite(eps_UB)
    # np.spacing could return negative numbers so use nextafter
    eps_LB[LB_idx] = np.nextafter(eps_LB[LB_idx], np.inf)
    eps_UB[UB_idx] = np.nextafter(eps_UB[UB_idx], -np.inf)
    hyp = np.minimum(eps_UB, np.maximum(eps_LB, hyp))

    hyp_starts = []
    hyp_starts_neg_elbos = []
    hyp_starts_sources = []
    bounds_opts = []
    for i in range(hyp.shape[0]):
        hyp_cur = hyp[i]
        # to jaxgp
        hyp_cur = exact_gp.hyperparameters_to_dict(hyp_cur)
        hyp_cur = gpyreg_params_to_jaxgp(hyp_cur[0])
        hyp_cur = concat_dictionaries({"inducing_points": Z}, hyp_cur)
        hyp_starts.append(hyp_cur)
        bounds_opts.append(bounds)
        hyp_starts_neg_elbos.append(hyp_neg_elbos[i])
        hyp_starts_sources.append("random")

    if reselect_ip_flag:
        # 2. (Z, hyp_select)
        # Add the hyperparameters associated with inducing points selection as a potential candidate for optimization initialization. Only applicable when ipselectmethod=dpp.
        hyp_select = select_info.get("hyp_select")
        if hyp_select is not None:
            hyp_select["inducing_points"] = Z  # new inducing points
            hyp_starts.append(hyp_select)
            bounds_opts.append(bounds)
            neg_elbo_select = neg_elbo(hyp_select).item()
            hyp_starts_neg_elbos.append(neg_elbo_select)
            hyp_starts_sources.append("inducing points selection")

    # 3. (Z_pre, hyp_pre)
    # Add inducing points and hyperparameters from the previous iteration as a potential candidate for optimization initialization.
    if optim_state["iter"] > 0:
        if optim_state.get("last_warping") != current_iter:
            hyp_pre = iteration_history["gp"][-1].params_cache
            Z_pre = hyp_pre["inducing_points"]
            hyp_starts_sources.append("previous iteration")
        else:
            # this iteration is a warping iteration
            assert (
                hyp_dict.get("hyp") is not None
            ), "If warping, then hyp_dict is supposed to contain warped hyperparameters."
            # to jaxgp
            hyp_pre = exact_gp.hyperparameters_to_dict(
                hyp_dict["hyp"]
            )  # hyp_dict["hyp"] is the last iteration's hyperparameters after warping
            hyp_pre = gpyreg_params_to_jaxgp(hyp_pre[0])
            Z_pre = hyp_pre["inducing_points"]
            hyp_pre = concat_dictionaries({"inducing_points": Z_pre}, hyp_pre)
            hyp_starts_sources.append(
                "previous iteration (warped hyp and inducing points)"
            )
        Z_pre, inducing_bounds_pre = keep_inducing_points_with_bounds(
            Z_pre, x_train, options
        )
        bounds_pre = get_jaxgp_unconstrained_bounds(
            jaxgp_bounds, inducing_bounds_pre, unconstrain_trans
        )
        hyp_starts.append(hyp_pre)
        bounds_opts.append(bounds_pre)
        hyp_starts_neg_elbos.append(neg_elbo(hyp_pre).item())

    # 4. from hyp_dict
    if bool(hyp_dict):  # if not empty
        # to jaxgp
        hyp = exact_gp.hyperparameters_to_dict(hyp_dict["hyp"])
        assert (
            len(hyp) == 1
        ), "Not implemented for multiple hyperparameter vectors."
        hyp = gpyreg_params_to_jaxgp(hyp[0])
        if hyp_dict.get("inducing_points") is not None:
            Z = hyp_dict["inducing_points"]
        else:
            raise NotImplementedError(
                "inducing_points must be provided in hyp_dict."
            )

        hyp = concat_dictionaries({"inducing_points": Z}, hyp)
        hyp_starts_sources.append("hyp_dict")
        Z, inducing_bounds = keep_inducing_points_with_bounds(
            Z, x_train, options
        )
        bounds = get_jaxgp_unconstrained_bounds(
            jaxgp_bounds, inducing_bounds, unconstrain_trans
        )
        hyp_starts.append(hyp)
        bounds_opts.append(bounds)
        hyp_starts_neg_elbos.append(neg_elbo(hyp).item())

    # Remove duplicates
    # Simply use elbo values to remove duplicates such that no need to check duplicates in hyp_starts
    # print("hyp_starts_neg_elbos: ", hyp_starts_neg_elbos)
    hyp_starts_neg_elbos, inds = np.unique(
        hyp_starts_neg_elbos, return_index=True
    )
    if len(hyp_starts) != len(hyp_starts_neg_elbos):
        print(
            f"Duplicates found in hyp_starts_neg_elbos. Removing duplicates which is caused by duplicate initial hyperparameters. Unique indices are {inds} for sources {hyp_starts_sources}."
        )
    hyp_starts = [hyp_starts[i] for i in inds]
    bounds_opts = [bounds_opts[i] for i in inds]
    hyp_starts_sources = [hyp_starts_sources[i] for i in inds]

    # Sort
    inds = np.argsort(hyp_starts_neg_elbos)
    hyp_starts = [hyp_starts[i] for i in inds]
    bounds_opts = [bounds_opts[i] for i in inds]
    hyp_starts_sources = [hyp_starts_sources[i] for i in inds]
    hyp_starts_neg_elbos = [hyp_starts_neg_elbos[i] for i in inds]

    logger.debug(
        f"np.argsort(hyp_starts_neg_elbos) = {hyp_starts_sources}, hyp_starts_neg_elbos = {hyp_starts_neg_elbos}"
    )

    # Make sure we don't overshoot.
    opts_N = np.minimum(opts_N, len(hyp_starts))
    hyp_optimized = [None] * opts_N  # Store the optimized hyperparameters
    nll = np.full((np.maximum(opts_N, 1),), np.inf)

    reinit_steps = options.get("reinit_steps", 0)
    if reinit_steps > 1:
        assert options.get(
            "no_bounds_for_inducing_points"
        ), "reinit_steps > 1 is only supported when no_bounds_for_inducing_points is True"

    for i in range(opts_N):
        hyp_cur = hyp_starts[i]
        for j in range(reinit_steps + 1):
            old_hyp = deepcopy(hyp_cur)
            old_neg_ELBO = neg_elbo(old_hyp).item()
            old_Z = hyp_cur["inducing_points"].copy()

            fixed_params = {
                "inducing_points": hyp_cur["inducing_points"],
                "likelihood": {"noise_add": jnp.array([1e-5])},
            }

            soln = train_model(
                model,
                fixed_params=fixed_params,
                init_params=hyp_cur,
                tol=tol,
                bounds=bounds_opts[i],
                transforms_jitted=(constrain_trans, unconstrain_trans),
                return_soln=True,
                neg_elbo=neg_elbo_raw,
                options={"disp": settings.debug.on()},
                logger=logger,
                jit=True,
            )

            hyp_cur = constrain_trans(soln.params)
            if np.isnan(soln.state.fun_val):
                # Restore initial hyperparameters and break the optimization
                nll[i] = old_neg_ELBO
                hyp_optimized[i] = old_hyp
                logger.debug(
                    "lbfgs fails to optimize elbo. Init params for lbfgs are returned and used."
                )
                break
            else:
                # Store the optimized hyperparameters
                if soln.state.fun_val.item() < nll[i]:
                    nll[i] = soln.state.fun_val.item()
                    hyp_optimized[i] = hyp_cur
            logger.debug(f"Nit: {soln[-1].iter_num}")

            if reinit_steps >= 1:
                # Reinitialize inducing points
                Z, _ = select_inducing_points(
                    model,
                    optim_state,
                    iteration_history,
                    options,
                    hyp_select=hyp_cur,
                )
                hyp_cur["inducing_points"] = Z
                if options["break_reinit_if_not_improve"]:
                    neg_ELBO = neg_elbo(hyp_cur).item()
                    if np.isnan(neg_ELBO):
                        neg_ELBO = np.inf
                    if neg_ELBO >= soln.state.fun_val.item():
                        # Restore old inducing points and break
                        hyp_cur["inducing_points"] = old_Z
                        break
        logger.debug(f"Reinit steps: {j}")
    # Take the best hyperparameter vector.
    if opts_N > 0:
        final_params = hyp_optimized[np.argmin(nll)]
    else:
        raise NotImplementedError
    logger.debug(f"NLL: {nll}")
    optim_state["gp_elbo"] = np.min(nll).item()

    sn2hpd = _estimate_noise_sgpr(model, final_params)

    model.params_cache = final_params
    model.bounds = bounds
    optim_state["inducing_points"] = final_params["inducing_points"]
    for k, v in final_params.items():
        if k != "inducing_points":
            logger.debug(f"{k}: {v}")
        else:
            logger.debug(f"Number of inducing points: {v.shape[0]}")

    logger.debug(f"LB for gp hyp: {jaxgp_bounds[0]}")
    logger.debug(f"UB for gp hyp: {jaxgp_bounds[1]}")

    model.post_cache = model.posterior(final_params)
    logger.debug("Posterior cached.")
    # fmu, fs2 = model.post_cache.predict_f_with_precomputed(
    #     model.X, full_cov=False
    # )
    final_params_gpyreg = jaxgp_params_to_gpyreg(final_params)
    final_params_gpyreg = exact_gp.hyperparameters_from_dict(
        final_params_gpyreg
    )
    hyp_dict["hyp"] = final_params_gpyreg
    hyp_dict["inducing_points"] = final_params["inducing_points"]
    if settings.return_aux.on():
        return model, sn2hpd, hyp_dict, nll
    return model, sn2hpd, hyp_dict


def train_gp(
    hyp_dict: dict,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
    select_subset: Optional[bool] = False,
):
    """
    Train Gaussian process model.

    Parameters
    ==========
    hyp_dict : dict
        Hyperparameter summary statistics dictionary.
        If it does not contain the appropriate keys they will be added
        automatically.
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    plb : ndarray, shape (hyp_N,)
        Plausible lower bounds for hyperparameters.
    pub : ndarray, shape (hyp_N,)
        Plausible upper bounds for hyperparameters.

    Returns
    =======
    gp : GP
        The trained GP.
    gp_s_N : int
        The number of samples for fitting.
    sn2hpd : float
        An estimate of the GP noise variance at high posterior density.
    hyp_dict : dict
        The updated summary statistics.
    """

    # Initialize hyp_dict if empty.
    if hyp_dict is None:
        hyp_dict = {}
    if "hyp" not in hyp_dict:
        hyp_dict["hyp"] = None
    if "warp" not in hyp_dict:
        hyp_dict["warp"] = None
    if "logp" not in hyp_dict:
        hyp_dict["logp"] = None
    if "full" not in hyp_dict:
        hyp_dict["full"] = None
    if "run_cov" not in hyp_dict:
        hyp_dict["run_cov"] = None

    # Get training dataset.
    x_train, y_train, s2_train, _ = _get_training_data(
        function_logger, options
    )

    if select_subset:
        x_train, y_train, s2_train = select_training_subset(
            x_train, y_train, s2_train, options["subsetsize"]
        )

    D = x_train.shape[1]

    # Heuristic fitness shaping (unused even in MATLAB)
    # if options.FitnessShaping
    #     [y_train,s2_train] = outputwarp_vbmc(X_train,y_train,s2_train,
    #                                           optimState,options);
    #  end

    # Pick the mean function
    mean_f = _meanfun_name_to_mean_function(optim_state["gp_meanfun"])

    # Pick the covariance function.
    covariance_f = _cov_identifier_to_covariance_function(
        optim_state["gp_covfun"]
    )

    # Pick the noise function.
    const_add = optim_state["gp_noisefun"][0] == 1
    user_add = optim_state["gp_noisefun"][1] == 1
    user_scale = optim_state["gp_noisefun"][1] == 2
    rlod_add = optim_state["gp_noisefun"][2] == 1
    noise_f = gpr.noise_functions.GaussianNoise(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    # Setup a GP.
    gp = gpr.GP(D=D, covariance=covariance_f, mean=mean_f, noise=noise_f)
    # Get number of samples and set priors and bounds.
    gp, hyp0, gp_s_N = _gp_hyp(
        optim_state, options, plb, pub, gp, x_train, y_train
    )
    # Initial GP hyperparameters.
    if hyp_dict["hyp"] is None:
        hyp_dict["hyp"] = hyp0.copy()

    # Get GP training options.
    gp_train = _get_gp_training_options(
        optim_state, iteration_history, options, hyp_dict, gp_s_N
    )

    # In some cases the model can change so be careful.
    if gp_train["widths"] is not None and np.size(
        gp_train["widths"]
    ) != np.size(hyp0):
        gp_train["widths"] = None

    # Build starting points
    # hyp0 = np.empty((0, np.size(hyp_dict["hyp"])))
    hyp0 = np.empty((0, hyp_dict["hyp"].T.shape[0]))
    if gp_train["init_N"] > 0 and optim_state["iter"] > 0:
        # Be very careful with off-by-one errors compared to MATLAB in the
        # range here.
        for i in range(
            math.ceil((np.size(iteration_history["gp"]) + 1) / 2) - 1,
            np.size(iteration_history["gp"]),
        ):
            gp_past = iteration_history["gp"][i]
            if isinstance(gp_past, jgp.HeteroskedasticSGPR):
                hyp_past = jaxgp_params_to_gpyreg(gp_past.params_cache)
                hyp_past = gp.hyperparameters_from_dict(hyp_past)
                hyp0 = np.concatenate([hyp0, hyp_past])
            else:
                hyp0 = np.concatenate(
                    (
                        hyp0,
                        iteration_history["gp"][i].get_hyperparameters(
                            as_array=True
                        ),
                    )
                )
        N0 = hyp0.shape[0]
        if N0 > gp_train["init_N"] / 2:
            hyp0 = hyp0[
                np.random.choice(
                    N0, math.ceil(gp_train["init_N"] / 2), replace=False
                ),
                :,
            ]
    hyp0 = np.concatenate((hyp0, np.atleast_2d(hyp_dict["hyp"])))
    hyp0 = np.unique(hyp0, axis=0)

    # In some cases the model can change so be careful.
    if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]):
        hyp0 = None

    if (
        "hyp_vp" in hyp_dict
        and hyp_dict["hyp_vp"] is not None
        and gp_train["sampler"] == "npv"
    ):
        hyp0 = hyp_dict["hyp_vp"]

    # print(hyp0.shape)
    hyp_dict["hyp"], _, res = gp.fit(
        x_train, y_train, s2_train, hyp0=hyp0, options=gp_train
    )

    if res is not None:
        # Pre-thinning GP hyperparameters
        hyp_dict["full"] = res["samples"]
        hyp_dict["logp"] = res["log_priors"]

        # Missing port: currently not used since we do
        # not support samplers other than slice sampling.
        # if isfield(gpoutput,'hyp_vp')
        #     hypstruct.hyp_vp = gpoutput.hyp_vp;
        # end

        # if isfield(gpoutput,'stepsize')
        #     optimState.gpmala_stepsize = gpoutput.stepsize;
        #     gpoutput.stepsize
        # end

    # TODO: think about the purpose of this line elsewhere in the program.
    # gp.t = t_train

    # Update running average of GP hyperparameter covariance (coarse)
    if hyp_dict["full"] is not None and hyp_dict["full"].shape[1] > 1:
        hyp_cov = np.cov(hyp_dict["full"].T)
        if hyp_dict["run_cov"] is None or options["hyprunweight"] == 0:
            hyp_dict["run_cov"] = hyp_cov
        else:
            w = options["hyprunweight"] ** options["funevalsperiter"]
            hyp_dict["run_cov"] = (1 - w) * hyp_cov + w * hyp_dict["run_cov"]
    else:
        hyp_dict["run_cov"] = None

    # Missing port: sample for GP for debug (not used)

    # Estimate of GP noise around the top high posterior density region
    # We don't modify optim_state to contain sn2hpd here.
    sn2hpd = _estimate_noise(gp)

    return gp, gp_s_N, sn2hpd, hyp_dict


def _gp_hyp(
    optim_state: dict,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
    gp: gpr.GP,
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Define bounds, priors and samples for GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    plb : ndarray, shape (hyp_N,)
        Plausible lower bounds for the hyperparameters.
    pub : ndarray, shape (hyp_N,)
        Plausible upper bounds for the hyperparameters.
    gp : GP
        Gaussian process for which we are making the bounds,
        priors and so on.
    X : ndarray, shape (N, D)
        Training inputs.
    y : ndarray, shape (N, 1)
        Training targets.

    Returns
    =======
    gp : GP
        The GP with updates priors, bounds and so on.
    hyp0 : ndarray, shape (hyp_N,)
        Initial guess for the hyperparameters.
    gp_s_N : int
        The number of samples for GP fitting.

    Raises
    ------
    TypeError
        Raised if the mean function is not supported by gpyreg.
    """

    # Get high posterior density dataset.
    hpd_X, hpd_y, _, _ = get_hpd(X, y, options["hpdfrac"])
    D = hpd_X.shape[1]
    # s2 = None

    ## Set GP hyperparameter defaults for VBMC.

    cov_bounds_info = gp.covariance.get_bounds_info(hpd_X, hpd_y)
    mean_bounds_info = gp.mean.get_bounds_info(hpd_X, hpd_y)
    noise_bounds_info = gp.noise.get_bounds_info(hpd_X, hpd_y)
    # Missing port: output warping hyperparameters not implemented
    cov_x0 = cov_bounds_info["x0"]
    mean_x0 = mean_bounds_info["x0"]

    noise_x0 = noise_bounds_info["x0"]
    min_noise = options["tolgpnoise"]
    noise_mult = None
    if optim_state["uncertainty_handling_level"] == 0:
        if options["noisesize"] != []:
            noise_size = max(options["noisesize"], min_noise)
        else:
            noise_size = min_noise
        noise_std = 0.5
    elif optim_state["uncertainty_handling_level"] == 1:
        # This branch is not used and tested at the moment.
        if options["noisesize"] != []:
            noise_mult = max(options["noisesize"], min_noise)
            noise_mult_std = np.log(10) / 2
        else:
            noise_mult = 1
            noise_mult_std = np.log(10)
        noise_size = min_noise
        noise_std = np.log(10)
    elif optim_state["uncertainty_handling_level"] == 2:
        noise_size = min_noise
        noise_std = 0.5
    noise_x0[0] = np.log(noise_size)
    hyp0 = np.concatenate([cov_x0, noise_x0, mean_x0])

    # Missing port: output warping hyperparameters not implemented

    ## Change default bounds and set priors over hyperparameters.

    bounds = gp.get_bounds()
    if options["uppergplengthfactor"] > 0:
        # Max GP input length scale
        bounds["covariance_log_lengthscale"] = (
            -np.inf,
            np.log(options["uppergplengthfactor"] * (pub - plb)),
        )
    # Increase minimum noise.
    bounds["noise_log_scale"] = (np.log(min_noise), np.inf)

    # Missing port: we only implement the mean functions that gpyreg supports.
    if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
        pass
    elif isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        # Lower maximum constant mean
        bounds["mean_const"] = (-np.inf, np.min(hpd_y))
    elif isinstance(gp.mean, gpr.mean_functions.NegativeQuadratic):
        if options["gpquadraticmeanbound"]:
            delta_y = max(
                options["tolsd"],
                min(D, np.max(hpd_y) - np.min(hpd_y)),
            )
            bounds["mean_const"] = (-np.inf, np.max(hpd_y) + delta_y)
    else:
        raise TypeError("The mean function is not supported by gpyreg.")

    # Set lower bound for GP's outputscale
    assert isinstance(
        gp.covariance, gpr.covariance_functions.SquaredExponential
    )
    bounds["covariance_log_outputscale"] = (cov_bounds_info["LB"][D], np.inf)
    # Low lower bounds for lenthscales could cause numerical issues especially for training points from sliced sampling, therefore +3 here in log space.
    bounds["covariance_log_lengthscale"] = (
        cov_bounds_info["LB"][:D] + 3,
        np.inf,
    )
    # Set priors over hyperparameters (might want to double-check this)
    priors = gp.get_priors()

    # Hyperprior over observation noise
    priors["noise_log_scale"] = (
        "student_t",
        (np.log(noise_size), noise_std, 3),
    )
    if noise_mult is not None:
        priors["noise_provided_log_multiplier"] = (
            "student_t",
            (np.log(noise_mult), noise_mult_std, 3),
        )

    # Missing port: hyperprior over mixture of quadratics mean function

    # Change bounds and hyperprior over output-dependent noise modulation
    # Note: currently this branch is not used.
    if optim_state["gp_noisefun"][2] == 1:
        bounds["noise_rectified_log_multiplier"] = (
            [np.min(np.min(y), np.max(y) - 20 * D), -np.inf],
            [np.max(y) - 10 * D, np.inf],
        )

        # These two lines were commented out in MATLAB as well.
        # If uncommented add them to the stuff below these two lines
        # where we have np.nan
        # hypprior.mu(Ncov+2) = max(y_hpd) - 10*D;
        # hypprior.sigma(Ncov+2) = 1;

        # Only set the first of the two parameters here.
        priors["noise_rectified_log_multiplier"] = (
            "student_t",
            ([np.nan, np.log(0.01)], [np.nan, np.log(10)], [np.nan, 3]),
        )

    # Missing port: priors and bounds for output warping hyperparameters
    # (not used)

    # VBMC used to have an empirical Bayes prior on some GP hyperparameters,
    # such as input length scales, based on statistics of the GP training
    # inputs. However, this approach could lead to instabilities. From the
    # 2020 paper, we switched to a fixed prior based on the plausible bounds.
    priors["covariance_log_lengthscale"] = (
        "student_t",
        (
            np.log(options["gplengthpriormean"] * (pub - plb)),
            options["gplengthpriorstd"],
            3,
        ),
    )

    stop_sampling = optim_state["stop_sampling"]
    gp_s_N = 0
    assert (
        stop_sampling != 0
    ), "We don't support stop_sampling == 0 which means MCMC sampling for GP hyperparameters. GP is always fitted with optimization."

    gp.set_bounds(bounds)
    gp.set_priors(priors)

    return gp, hyp0, round(gp_s_N)


def _get_gp_training_options(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
    gp_s_N: int,
):
    """
    Get options for training GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    hyp_dict : dict
        Hyperparameter summary statistic dictionary.
    gp_s_N : int
        Number of samples for the GP fitting.

    Returns
    =======
    gp_train : dic
        A dictionary of GP training options.

    Raises
    ------
    ValueError
        Raised if the MCMC sampler for GP hyperparameters is unknown.

    """

    iteration = optim_state["iter"]
    if iteration > 0:
        r_index = iteration_history["rindex"][iteration - 1]
    else:
        r_index = np.inf

    gp_train = {}
    gp_train["init_method"] = options["gptraininitmethod"]
    gp_train["tol_opt"] = options["gptolopt"]
    gp_train["widths"] = None

    # Get hyperparameter posterior covariance from previous iterations
    hyp_cov = _get_hyp_cov(optim_state, iteration_history, options, hyp_dict)

    gp_train["sampler"] = None

    init_N = options["gptrainninit"]

    # Set other hyperparameter fitting parameters
    if optim_state["recompute_var_post"]:
        gp_train["init_N"] = init_N
        if gp_s_N > 0:
            gp_train["opts_N"] = 1
        else:
            gp_train["opts_N"] = 2
    else:
        raise ValueError()
        # gp_train["burn"] = gp_train["thin"] * 3
        # if (
        #     iteration > 1
        #     and iteration_history["rindex"][iteration - 1]
        #     < options["gpretrainthreshold"]
        # ):
        #     gp_train["init_N"] = 0
        #     if options["gphypsampler"] == "slicelite":
        #         # TODO: gpretrainthreshold is by default 1, so we get
        #         #       division by zero. what should the default be?
        #         gp_train["burn"] = (
        #             max(
        #                 1,
        #                 math.ceil(
        #                     gp_train["thin"]
        #                     * np.log(
        #                         iteration_history["rindex"][iteration - 1]
        #                         / np.log(options["gpretrainthreshold"])
        #                     )
        #                 ),
        #             )
        #             * gp_s_N
        #         )
        #         gp_train["thin"] = 1
        #     if gp_s_N > 0:
        #         gp_train["opts_N"] = 0
        #     else:
        #         gp_train["opts_N"] = 1
        # else:
        #     gp_train["init_N"] = init_N
        #     if gp_s_N > 0:
        #         gp_train["opts_N"] = 1
        #     else:
        #         gp_train["opts_N"] = 2

    gp_train["n_samples"] = round(gp_s_N)
    return gp_train


def _get_hyp_cov(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
):
    """
    Get hyperparameter posterior covariance.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    hyp_dict : dict
        Hyperparameter summary statistic dictionary.

    Returns
    =======
    hyp_cov : ndarray, optional
        The hyperparameter posterior covariance if it can be computed.
    """

    if optim_state["iter"] > 0:
        if options["weightedhypcov"]:
            w_list = []
            hyp_list = []
            w = 1
            for i in range(0, optim_state["iter"]):
                if i > 0:
                    # Be careful with off-by-ones compared to MATLAB here
                    diff_mult = max(
                        1,
                        np.log(
                            iteration_history["sKL"][optim_state["iter"] - i]
                            / options["tolskl"]
                            * options["funevalsperiter"]
                        ),
                    )
                    w *= options["hyprunweight"] ** (
                        options["funevalsperiter"] * diff_mult
                    )
                # Check if weight is getting too small.
                if w < options["tolcovweight"]:
                    break

                hyp = iteration_history["gp_hyp_full"][
                    optim_state["iter"] - 1 - i
                ]
                hyp_n = hyp.shape[1]
                if len(hyp_list) == 0 or np.shape(hyp_list)[2] == hyp.shape[0]:
                    hyp_list.append(hyp.T)
                    w_list.append(w * np.ones((hyp_n, 1)) / hyp_n)

            w_list = np.concatenate(w_list)
            hyp_list = np.concatenate(hyp_list)

            # Normalize weights
            w_list /= np.sum(w_list, axis=0)
            # Weighted mean
            mu_star = np.sum(hyp_list * w_list, axis=0)

            # Weighted covariance matrix
            hyp_n = np.shape(hyp_list)[1]
            hyp_cov = np.zeros((hyp_n, hyp_n))
            for j in range(0, np.shape(hyp_list)[0]):
                hyp_cov += np.dot(
                    w_list[j],
                    np.dot((hyp_list[j] - mu_star).T, hyp_list[j] - mu_star),
                )

            hyp_cov /= 1 - np.sum(w_list**2)

            return hyp_cov

        return hyp_dict["run_cov"]

    return None


def _get_training_data(function_logger: FunctionLogger, options: Options):
    """
    Get training data for building GP surrogate.

    Parameters
    ==========
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.

    Returns
    =======
    x_train, ndarray
        Training inputs.
    y_train, ndarray
        Training targets.
    s2_train, ndarray, optional
        Training data noise variance, if noise is used.
    t_train, ndarray
        Array of the times it took to evaluate the function on the training
        data.
    """

    x_train = function_logger.X[function_logger.X_flag, :]
    y_train = function_logger.y[function_logger.X_flag]
    if function_logger.noise_flag:
        s2_train = function_logger.S[function_logger.X_flag] ** 2
    else:
        s2_train = options["tolgpnoise"] ** 2 * np.ones_like(y_train)

    if options["noiseshaping"]:
        s2_train = noise_shaping(s2_train, y_train, options)

    t_train = function_logger.fun_evaltime[function_logger.X_flag]

    return x_train, y_train, s2_train, t_train


def _estimate_noise_sgpr(gp: jgp.HeteroskedasticSGPR, params: dict):
    """Estimate GP observation noise at high posterior density for SGPR model.

    Parameters
    ==========
    gp : GP
        The GP for which to perform the estimate.

    Returns
    =======
    est : float
        The estimate of observation noise.
    """

    hpd_top = 0.2
    X, Y = gp.train_data.X, gp.train_data.Y
    y = Y.squeeze()

    N, _ = X.shape
    # Subsample high posterior density dataset
    # Sort by descending order, not ascending.
    order = np.argsort(y, axis=None)[::-1]
    hpd_N = math.ceil(hpd_top * N)
    hpd_X = X[order[0:hpd_N]]
    hpd_y = y[order[0:hpd_N]]

    if gp.sigma_sq_user is not None:
        hpd_s2 = gp.sigma_sq_user[order[0:hpd_N]]
    else:
        hpd_s2 = None

    sn2 = gp.likelihood.compute(params["likelihood"], hpd_s2)
    return np.median(sn2)


def _estimate_noise(gp: gpr.GP):
    """Estimate GP observation noise at high posterior density.

    Parameters
    ==========
    gp : GP
        The GP for which to perform the estimate.

    Returns
    =======
    est : float
        The estimate of observation noise.
    """

    hpd_top = 0.2
    N, _ = gp.X.shape

    # Subsample high posterior density dataset
    # Sort by descending order, not ascending.
    order = np.argsort(gp.y, axis=None)[::-1]
    hpd_N = math.ceil(hpd_top * N)
    hpd_X = gp.X[order[0:hpd_N]]
    hpd_y = gp.y[order[0:hpd_N]]

    if gp.s2 is not None:
        hpd_s2 = gp.s2[order[0:hpd_N]]
    else:
        hpd_s2 = None

    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    s_N = np.size(gp.posteriors)

    sn2 = np.zeros((hpd_X.shape[0], s_N))

    for s in range(0, s_N):
        hyp = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        sn2[:, s : s + 1] = gp.noise.compute(hyp, hpd_X, hpd_y, hpd_s2)

    return np.median(np.mean(sn2, axis=1))


@dispatch
def reupdate_gp(function_logger: FunctionLogger, gp: gpr.GP, options: Options):
    """
    Quick posterior reupdate of Gaussian process.

    Parameters
    ==========
    gp : GP
        The GP to update.
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.
    Returns
    =======
    gp : GP
        The updated Gaussian process.
    """

    x_train, y_train, s2_train, t_train = _get_training_data(
        function_logger, options
    )
    gp.X = x_train
    gp.y = y_train
    gp.s2 = s2_train
    # Missing port: gp.t = t_train
    gp.update(compute_posterior=True)

    # Missing port: intmean part

    return gp


@dispatch
def reupdate_gp(
    function_logger: FunctionLogger,
    gp: jgp.HeteroskedasticSGPR,
    options: Options,
):
    """
    Posterior reupdate of SGPR. TODO: this can be more efficient if only one new point is added. Implement more efficient one if needed.

    Parameters
    ==========
    gp : GP
        The GP to update.
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.
    Returns
    =======
    gp : GP
        The updated Gaussian process.
    """

    x_train, y_train, s2_train, t_train = _get_training_data(
        function_logger, options
    )
    # TODO: test after fix
    gp.train_data = jgp.Dataset(X=x_train, Y=y_train)

    gp.sigma_sq_user = s2_train

    gp.X = x_train
    gp.y = y_train
    gp.s2 = s2_train
    # Missing port: gp.t = t_train
    # gp.update(compute_posterior=True)

    gp.post_cache = gp.posterior(gp.params_cache)
    # Missing port: intmean part

    return gp


def select_training_subset(x_train, y_train, s2_train, N_subset):
    N, D = x_train.shape
    if N_subset >= N:
        return x_train, y_train, s2_train

    inds = np.argsort(y_train.squeeze())[::-1]
    partition = [0, 0.1, 0.25, 0.45, 0.7, 1]
    num = len(partition) - 1
    selected_inds = []
    for i in range(num):
        start = int(N * partition[i])
        end = int(N * partition[i + 1])
        Xt = x_train[start:end]
        n_clusters = min(int(N_subset / num), Xt.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(Xt)
        # distance to cluster center
        dist = kmeans.transform(Xt)
        # for each cluster, find the closest point to cluster center
        ind = np.argmin(dist, 0)
        selected_inds.extend(ind.tolist())

    selected_inds = set(selected_inds)
    N_remained = N_subset - len(selected_inds)
    if N_remained > 0:
        for i in range(N):
            if i in selected_inds:
                continue
            selected_inds.add(i)
            N_remained -= 1
            if N_remained == 0:
                break
    selected_inds = list(selected_inds)
    if s2_train is not None:
        return (
            x_train[selected_inds],
            y_train[selected_inds],
            s2_train[selected_inds],
        )
    else:
        return (
            x_train[selected_inds],
            y_train[selected_inds],
            None,
        )
