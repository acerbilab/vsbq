"""Module for helper functions for GP training."""

import copy
import logging
import operator
import re
import time
import warnings
from typing import Dict, List, Union

import gpyreg as gpr
import jax
import jax.numpy as jnp
import jaxgp as jgp
import jaxopt
import numpy as np
import scipy as sp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax._src.api import clear_backends
from jax.tree_util import tree_leaves
from jaxgp.contrib.train_utils import train_model
from jaxgp.kernels import cross_covariance, gram
from jaxgp.priors import TruncatedPrior
from jaxgp.utils import concat_dictionaries
from plum import dispatch
from sklearn.cluster import KMeans
from tqdm.autonotebook import tqdm

import pyvbmc.settings as settings
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.stats import get_hpd

logger = logging.getLogger("VBMC_debug")


class TransformedDistribution(tfd.TransformedDistribution):
    def __reduce__(self):
        return (self.__class__, (self.distribution, self.bijector))


def get_spgr_bounds_for_lbfgs(sgpr_model, inducing_points, options):
    """This function reuses gpyreg's bounds for SGPR."""
    x_train = sgpr_model.train_data.X
    y_train = sgpr_model.train_data.Y.flatten()
    s2_train = sgpr_model.sigma_sq_user
    D = x_train.shape[1]
    # Pick the mean function
    assert isinstance(sgpr_model.gprior.mean_function, jgp.means.Quadratic)
    mean_f = _meanfun_name_to_mean_function("negquad")

    # Pick the covariance function.
    assert isinstance(sgpr_model.gprior.kernel, jgp.kernels.RBF)
    covariance_f = _cov_identifier_to_covariance_function(1)

    # Pick the noise function.
    const_add = sgpr_model.likelihood.constant_add
    user_add = sgpr_model.likelihood.user_provided_add
    user_scale = sgpr_model.likelihood.scale_user_provided
    rlod_add = sgpr_model.likelihood.rectified_linear_output_dependent_add

    noise_f = gpr.noise_functions.GaussianNoise(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    # Setup a GP.
    exact_gp = gpr.GP(D=D, covariance=covariance_f, mean=mean_f, noise=noise_f)
    # Set data inputs for exact gp
    exact_gp.X = x_train
    exact_gp.y = y_train
    exact_gp.s2 = s2_train

    # Set bounds for GP
    gp = exact_gp
    bounds = gp.get_bounds()
    if options["uppergplengthfactor"] > 0:
        # Max GP input length scale
        bounds["covariance_log_lengthscale"] = (
            -np.inf,
            np.log(options["uppergplengthfactor"] * (pub - plb)),
        )
    # Increase minimum noise.
    min_noise = options["tolgpnoise"]
    bounds["noise_log_scale"] = (np.log(min_noise), np.inf)

    hpd_X, hpd_y, _, _ = get_hpd(x_train, y_train, options["hpdfrac"])
    cov_bounds_info = gp.covariance.get_bounds_info(hpd_X, hpd_y)
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
    gp.set_bounds(bounds)

    # For unspecified bounds (np.inf), we replace them with finite recommened bounds in gpyreg
    gpyreg_bounds, LB, UB, PLB, PUB = get_bounds_gpyreg(
        exact_gp, x_train, y_train, options
    )
    # Convert to jaxgp's format
    jaxgp_bounds = gpyreg_bounds_to_jaxgp(gpyreg_bounds)

    Z = inducing_points
    Z, inducing_bounds = keep_inducing_points_with_bounds(Z, x_train)

    _, constrain_trans, unconstrain_trans = jgp.initialise(sgpr_model)

    bounds = get_jaxgp_unconstrained_bounds(
        jaxgp_bounds, inducing_bounds, unconstrain_trans
    )
    return bounds


def pivoted_cholesky_init(
    kernel_matrix: np.ndarray,
    max_length: int,
    min_length: int = 0,
    epsilon: float = 1e-10,
    x_train: np.ndarray = None,
) -> np.ndarray:
    r"""
    A pivoted cholesky initialization method for the inducing points, originally proposed in
    [burt2020svgp] with the algorithm itself coming from [chen2018dpp].
    Args:
        train_inputs: training inputs
        kernel_matrix: kernel matrix on the training inputs
        max_length: number of inducing points to initialize
        epsilon: numerical jitter for stability.
    """
    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration

    if callable(kernel_matrix):
        assert x_train is not None
        item_size = x_train.shape[-2]
    else:
        item_size = kernel_matrix.shape[-2]

    if np.isinf(max_length) or max_length > item_size:
        max_length = item_size
    assert min_length <= max_length
    cis = np.zeros((max_length, item_size))

    if callable(kernel_matrix):
        di2s = kernel_matrix("diag")
    else:
        di2s = np.diag(kernel_matrix)
    tr_init = np.sum(di2s)

    selected_items = []
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    i = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        if callable(kernel_matrix):
            elements = kernel_matrix(selected_item)
        else:
            elements = kernel_matrix[selected_item, :]
        eis = (elements - ci_optimal @ cis[:k, :]) / di_optimal
        cis[k, :] = eis
        try:
            di2s = di2s - eis**2
        except FloatingPointError:
            pass
        di2s = np.clip(di2s, 0, None)
        # tr(K_ff - K_fu K_uu^-1 K_uf)
        tr_cur = np.sum(di2s)
        fraction = tr_cur / tr_init
        # if fraction < epsilon:
        if fraction < epsilon or np.isnan(tr_cur):
            if len(selected_items) >= min_length:
                break
            else:
                if np.isnan(tr_cur):
                    N_rest = min_length - len(selected_items)
                    logger.debug(
                        f"Sample randomly for the rest {N_rest} points"
                    )
                    mask = np.full(len(di2s), True)
                    mask[selected_items] = False
                    mask = np.flatnonzero(mask)
                    inds = np.random.choice(mask, N_rest, replace=False)
                    selected_items.extend(inds)
                    break
                # else:
                #     print(
                #         "Continue selection since not reach minimum number of inducing points yet"
                #     )
        di2s_copy = di2s.copy()
        di2s_copy[selected_items] = -np.inf
        selected_item = np.argmax(di2s_copy)
        selected_items.append(selected_item)
    logger.debug(f"fraction: {fraction}, tr_cur: {tr_cur}")
    logger.debug(f"last di^2 = {di2s[selected_item]}")
    # logger.debug(f"selected inds: {np.sort(selected_items)}")
    # ind_points = x_train[np.stack(selected_items)]
    return selected_items


def pivoted_cholesky_init_torch(
    kernel_matrix,
    max_length: int,
    min_length: int = 0,
    epsilon: float = 1e-10,
    x_train=None,
):
    import torch

    if callable(kernel_matrix):
        assert x_train is not None
        device = x_train.device()
        item_size = x_train.shape[-2]
        if x_train.dtype == np.float64:
            dtype = torch.float64
        else:
            assert x_train.dtype == np.float32
            dtype = torch.float32
        NEG_INF = torch.tensor(float("-inf"), dtype=dtype)

    else:
        assert x_train is None
        device = kernel_matrix.device
        item_size = kernel_matrix.shape[-2]
        NEG_INF = torch.tensor(float("-inf"), dtype=kernel_matrix.dtype).to(
            device
        )

    if np.isinf(max_length) or max_length > item_size:
        max_length = item_size
    assert min_length <= max_length
    cis = torch.zeros((max_length, item_size))
    if "cuda" in str(device):
        NEG_INF = NEG_INF.cuda()
        cis = cis.cuda()

    if callable(kernel_matrix):
        di2s = kernel_matrix("diag")
    else:
        di2s = kernel_matrix.diag()
    tr_init = torch.sum(di2s)

    selected_items = []
    selected_item = torch.argmax(di2s).item()
    selected_items.append(selected_item)
    i = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        if callable(kernel_matrix):
            elements = kernel_matrix(selected_item)
        else:
            elements = kernel_matrix[selected_item, :]
        eis = (elements - ci_optimal @ cis[:k, :]) / di_optimal
        cis[k, :] = eis
        try:
            di2s = di2s - eis**2
        except FloatingPointError:
            pass
        di2s = torch.clip(di2s, 0, None)
        # tr(K_ff - K_fu K_uu^-1 K_uf)
        tr_cur = torch.sum(di2s)
        fraction = tr_cur / tr_init
        if fraction < epsilon or torch.isnan(tr_cur):
            if len(selected_items) >= min_length:
                break
            else:
                if torch.isnan(tr_cur):
                    # Sample randomly for the rest
                    print(
                        "Due to numerical precision, sample randomly for the rest"
                    )
                    N_rest = min_length - len(selected_items)
                    mask = np.full(len(di2s), True)
                    mask[selected_items] = False
                    mask = np.flatnonzero(mask)
                    inds = np.random.choice(mask, N_rest, replace=False)
                    selected_items.extend(list(inds))
                    break
        di2s_copy = di2s.clone()
        di2s_copy[selected_items,] = NEG_INF
        selected_item = torch.argmax(di2s_copy).item()
        selected_items.append(selected_item)
    logger.debug(f"fraction: {fraction}, tr_cur: {tr_cur}")
    logger.debug(f"last di^2 = {di2s[selected_item]}")
    return selected_items


def select_inducing_points(
    sgpr_model, optim_state, iteration_history, options, hyp_select=None
):
    """Select inducing points from the train points with the specified method in options["ipselectmethod"]."""
    method = options["ipselectmethod"]
    if hyp_select is not None:
        assert method == "dpp"
    x_train = jnp.array(sgpr_model.train_data.X)
    y_train = jnp.array(sgpr_model.train_data.Y.flatten())
    s2_train = jnp.array(sgpr_model.sigma_sq_user)

    model_tmp = copy.deepcopy(sgpr_model)

    ts = time.time()
    elbo_fun_tmp = model_tmp.build_elbo(raw_flag=False)

    first_exact_gp_hyp = gpyreg_params_to_jaxgp(
        optim_state["first_exact_gp"].get_hyperparameters()[0]
    )
    if hyp_select is None:
        if optim_state["iter"] > 0:
            # Select the hyperparameter with maximum ELBO for selecting inducing points
            last_exact_gp: gpr.GP = optim_state["last_exact_gp"]
            last_exact_gp_hyp = gpyreg_params_to_jaxgp(
                last_exact_gp.get_hyperparameters()[0]
            )
            hyps_select = [
                last_exact_gp_hyp,
                first_exact_gp_hyp,
            ]
            past_sgpr_hyps = []
            for i in range(
                len(iteration_history["gp"]) - 1, len(iteration_history["gp"])
            ):
                past_sgpr_hyps.append(iteration_history["gp"][i].params_cache)
            hyps_select.extend(past_sgpr_hyps)

            hyps_select = [
                concat_dictionaries(
                    {"inducing_points": optim_state["inducing_points"]},
                    hyp_select,
                )
                for hyp_select in hyps_select
            ]
            elbos_select = np.array(
                [elbo_fun_tmp(hyp_select) for hyp_select in hyps_select]
            )
            max_elbo_select_ind = np.argmax(elbos_select)
            max_elbo_select = elbos_select[max_elbo_select_ind]
            hyp_select = hyps_select[max_elbo_select_ind]

            select_info = {
                "hyp_select": hyp_select,
                "elbo_select(with_old_Z)": max_elbo_select,
            }

            logger.debug(
                f"ELBOs of hyp candidates for inducing points selection: {elbos_select}"
            )
            logger.debug(f"{np.argmax(elbos_select)} is chosen")
        else:
            hyp_select = first_exact_gp_hyp
            select_info = {"hyp_select": hyp_select}
    else:
        # Use the specified hyperparameters for selecting inducing points
        select_info = {"hyp_select": hyp_select}

    logger.debug(
        f"hyperparameters for selecting inducing points: {hyp_select['kernel']}"
    )

    if method == "all":
        Z = x_train.copy()
    elif method == "dpp":
        if x_train.shape[0] >= 20000:
            logger.debug(
                "use kernel_matrix function instead of the entire kernel matrix for saving memory"
            )

            def _kernel_matrix(indice):
                if indice == "diag":
                    v = gram(
                        model_tmp.gprior.kernel,
                        x_train,
                        hyp_select["kernel"],
                        False,
                    )
                    v = v.squeeze()
                    if s2_train is not None:
                        v = v / s2_train.squeeze()
                else:
                    v = cross_covariance(
                        model_tmp.gprior.kernel,
                        x_train,
                        x_train[indice].reshape(1, -1),
                        hyp_select["kernel"],
                    )
                    v = v.squeeze()
                    if s2_train is not None:
                        s_inv = 1 / np.sqrt(s2_train.squeeze())
                        v = s_inv[indice] * v * s_inv
                return v

            if "cuda" in str(x_train.device()):
                # Will use torch and GPU
                def kernel_matrix(indice):
                    v = _kernel_matrix(indice)
                    return jax_array_to_torch(v)

            else:

                def kernel_matrix(indice):
                    return np.array(_kernel_matrix(indice))

        else:
            logger.debug("Compute kernel matrix...")
            kernel_matrix = cross_covariance(
                model_tmp.gprior.kernel, x_train, x_train, hyp_select["kernel"]
            )
            if s2_train is not None:
                s_inv = 1 / np.sqrt(s2_train.squeeze())
                kernel_matrix = s_inv[:, None] * kernel_matrix * s_inv[None, :]

        if options.get("num_ips"):
            # Use the same number of inducing points, it's chosen to be the largest number one can accept according to computation time and may increase by if not able to model the train set.
            num_min_ip = options["num_ips"]
            num_max_ip = options["num_ips"]
        else:
            raise ValueError(
                "Need to specify number of inducing points `num_ips`."
            )
        if num_min_ip > x_train.shape[0]:
            num_min_ip = x_train.shape[0]

        if "cuda" in str(x_train.device()):
            # Implement this function efficiently in Jax is tricky, therefore simply use torch when gpu is available.
            if callable(kernel_matrix):
                inds = pivoted_cholesky_init_torch(
                    kernel_matrix, num_max_ip, num_min_ip, x_train=x_train
                )
            else:
                kernel_matrix = jax_array_to_torch(kernel_matrix)
                inds = pivoted_cholesky_init_torch(
                    kernel_matrix, num_max_ip, num_min_ip
                )

        else:
            if callable(kernel_matrix):
                inds = pivoted_cholesky_init(
                    kernel_matrix, num_max_ip, num_min_ip, x_train=x_train
                )
            else:
                kernel_matrix = np.array(kernel_matrix)
                inds = pivoted_cholesky_init(
                    kernel_matrix,
                    num_max_ip,
                    num_min_ip,
                )
        inds = jnp.array(inds)
        Z = x_train[inds]
    else:
        raise NotImplementedError
    logger.debug(
        f"Time for initalizing {Z.shape[0]} inducing points: {time.time() - ts:.2f}"
    )

    return Z, select_info


def jax_array_to_torch(x):
    import torch

    x_dlpack = jax.dlpack.to_dlpack(x)
    x_torch = torch.utils.dlpack.from_dlpack(x_dlpack)
    return x_torch


def get_jaxgp_unconstrained_bounds(
    params_bound, inducing_bounds, unconstrain_trans
):
    """Get bounds for SGPR's params in unconstrained space.
    params_bounds: bounds for hyperparameters
    inducing_bounds: bounds for inducing points
    unconstrain_trans: transformation function from constrained to uncontrained space
    """
    lower_bounds = concat_dictionaries(params_bound[0], inducing_bounds[0])
    upper_bounds = concat_dictionaries(params_bound[1], inducing_bounds[1])

    def replace_nan_with_inf(x, sign=1):
        x = x.at[jnp.isnan(x)].set(sign * jnp.inf)
        return x

    lower_bounds = unconstrain_trans(lower_bounds)
    upper_bounds = unconstrain_trans(upper_bounds)
    lower_bounds = jax.tree_map(
        lambda x: replace_nan_with_inf(x, -1), lower_bounds
    )
    upper_bounds = jax.tree_map(
        lambda x: replace_nan_with_inf(x, 1), upper_bounds
    )
    bounds = (lower_bounds, upper_bounds)
    return bounds


def keep_inducing_points_with_bounds(Z, x_train, options=None):
    """Compute inducing points bounds based on x_train and keep inducing points Z that are inside the bounds."""
    N_iv = Z.shape[0]
    if options is not None and options.get("no_bounds_for_inducing_points"):
        logger.debug("No bounds for inducing points")
        inducing_bounds = [{}, {}]
        inducing_bounds[0]["inducing_points"] = -np.inf * np.ones_like(Z)
        inducing_bounds[1]["inducing_points"] = np.inf * np.ones_like(Z)
        return Z, inducing_bounds

    inducing_bounds = compute_inducing_points_bounds(x_train, N_iv)

    inds_keep = np.all(Z >= inducing_bounds[0]["inducing_points"], 1) & np.all(
        Z <= inducing_bounds[1]["inducing_points"], 1
    )
    logger.debug(
        f"{np.sum(inds_keep)}/{Z.shape[0]} inducing points are kept because of inducing bounds constraints."
    )
    Z = Z[inds_keep]
    inducing_bounds[0]["inducing_points"] = inducing_bounds[0][
        "inducing_points"
    ][inds_keep]
    inducing_bounds[1]["inducing_points"] = inducing_bounds[1][
        "inducing_points"
    ][inds_keep]
    return Z, inducing_bounds


def get_bounds_gpyreg(exact_gp, x_train, y_train, options):
    """A helper function to get gpyreg's recommended bounds for the GP's hyperparameters."""
    hpd_X, hpd_y, _, _ = get_hpd(x_train, y_train, options["hpdfrac"])
    cov_bounds_info = exact_gp.covariance.get_bounds_info(hpd_X, hpd_y)
    mean_bounds_info = exact_gp.mean.get_bounds_info(hpd_X, hpd_y)
    noise_bounds_info = exact_gp.noise.get_bounds_info(hpd_X, hpd_y)
    use_recommended_bounds = True
    if use_recommended_bounds:
        recommended_bounds = exact_gp.get_recommended_bounds()
        exact_gp.set_bounds(recommended_bounds)
    LB = exact_gp.lower_bounds
    UB = exact_gp.upper_bounds
    gpyreg_bounds = exact_gp.get_bounds()

    # Plausible bounds for generation of starting points
    PLB = np.concatenate(
        [
            cov_bounds_info["PLB"],
            noise_bounds_info["PLB"],
            mean_bounds_info["PLB"],
        ]
    )
    PUB = np.concatenate(
        [
            cov_bounds_info["PUB"],
            noise_bounds_info["PUB"],
            mean_bounds_info["PUB"],
        ]
    )
    PLB = np.minimum(np.maximum(PLB, LB), UB)
    PUB = np.maximum(np.minimum(PUB, UB), LB)

    return gpyreg_bounds, LB, UB, PLB, PUB


def f_min_fill(
    f,
    x0,
    LB: np.ndarray,
    UB: np.ndarray,
    PLB: np.ndarray,
    PUB: np.ndarray,
    hprior: dict,
    N: int,
    design: str = None,
):
    """
    Create a space-filling design, evaluates the function ``f``
    on the points of the design and sorts the points from smallest
    value of ``f`` to largest.

    Parameters
    ==========
    f : callable
        The function to evaluate on the design points.
    x0 : ndarray, shape (M, hyp_N)
        A 2D array of points to include in the design, with each row
        containing a design point.
    LB : ndarray, shape (hyp_N,)
        The lower bounds.
    UB : ndarray, shape (hyp_N,)
        The upper bounds.
    PLB : ndarray, shape (hyp_N,)
        The plausible lower bounds.
    PUB : ndarray, shape (hyp_N,)
        The plausible upper bounds.
    hprior : dict
        Hyperparameter prior dictionary.
    N : int
        Design size to use.
    init_method : {'sobol', 'rand'}, defaults to 'sobol'
        Specify what kind of method to use to construct the space-filling
        design.

    Returns
    =======
    X : ndarray, shape (N, hyp_N)
        An array of the design points sorted according to the value
        ``f`` has at those points. N = M if N <= M, i.e. only x0
        will be returned.
    y : ndarray, shape (N,)
        An array of the sorted values of ``f`` at the design points.
    """
    if design is None:
        design = "sobol"

    # Helper for comparing version numbers.
    def ge_versions(version1, version2):
        def normalize(v):
            return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

        return operator.ge(normalize(version1), normalize(version2))

    # Check version number to make sure qmc exists.
    # Remove in the future when Anaconda has SciPy 1.7.0
    if design == "sobol" and not ge_versions(sp.__version__, "1.7.0"):
        design = "rand"

    if x0 is None:
        x0 = np.reshape(
            np.minimum(np.maximum((PLB + PUB) / 2, LB), UB), (1, -1)
        )

    N0 = x0.shape[0]
    n_vars = np.max(
        [x0.shape[1], np.size(LB), np.size(UB), np.size(PLB), np.size(PUB)]
    )

    # Force provided points to be inside bounds
    x0 = np.minimum(np.maximum(x0, LB), UB)

    sX = None

    if N > N0:
        # First test hyperparameters on a space-filling initial design
        if design == "sobol":
            sampler = sp.stats.qmc.Sobol(d=n_vars, scramble=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Get rid of first zero.
                S = sampler.random(n=N - N0 + 1)[1:, :]
            # Randomly permute columns
            np.random.shuffle(S.T)
        elif design == "rand":
            S = np.random.uniform(size=(N - N0, n_vars))
        else:
            raise ValueError(
                "Unknown design: got "
                + design
                + ' and expected either "sobol" or "rand"'
            )
        sX = np.zeros((N - N0, n_vars))

        # If a prior is specified use that.
        for i in range(0, n_vars):
            mu = hprior["mu"][i]
            sigma = hprior["sigma"][i]
            a = hprior["a"][i]
            b = hprior["b"][i]

            if not np.isfinite(mu) and not np.isfinite(
                sigma
            ):  # Uniform distribution?
                if np.isfinite(LB[i]) and np.isfinite(UB[i]):
                    # Fixed dimension
                    if LB[i] == UB[i]:
                        sX[:, i] = LB[i]
                    else:
                        # Mixture of uniforms
                        # (full bounds and plausible bounds)

                        # Half of all starting points from inside the
                        # plausible box
                        w = 0.5 ** (1 / n_vars)

                        sX[:, i] = uuinv(
                            S[:, i], [LB[i], PLB[i], PUB[i], UB[i]], w
                        )
                else:
                    # All starting points from inside the plausible box
                    sX[:, i] = S[:, i] * (PUB[i] - PLB[i]) + PLB[i]
            elif np.isfinite(a) and np.isfinite(
                b
            ):  # Smooth box student's t prior
                df = hprior["df"][i]
                # Force fat tails
                if not np.isfinite(df):
                    df = 3
                df = np.minimum(df, 3)
                if df == 0:
                    cdf_lb = smoothbox_cdf(LB[i], sigma, a, b)
                    cdf_ub = smoothbox_cdf(UB[i], sigma, a, b)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_ppf(S_scaled[j], sigma, a, b)
                else:
                    tcdf_lb = smoothbox_student_t_cdf(LB[i], df, sigma, a, b)
                    tcdf_ub = smoothbox_student_t_cdf(UB[i], df, sigma, a, b)
                    S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_student_t_ppf(
                            S_scaled[j], df, sigma, a, b
                        )
            else:  # Student's t prior
                df = hprior["df"][i]
                # Force fat tails
                if not np.isfinite(df):
                    df = 3
                df = np.minimum(df, 3)
                if df == 0:
                    cdf_lb = sp.stats.norm.cdf((LB[i] - mu) / sigma)
                    cdf_ub = sp.stats.norm.cdf((UB[i] - mu) / sigma)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    sX[:, i] = sp.stats.norm.ppf(S_scaled) * sigma + mu
                else:
                    tcdf_lb = sp.stats.t.cdf((LB[i] - mu) / sigma, df)
                    tcdf_ub = sp.stats.t.cdf((UB[i] - mu) / sigma, df)
                    S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
                    sX[:, i] = sp.stats.t.ppf(S_scaled, df) * sigma + mu
    else:
        N = N0

    if sX is None:
        X = x0
    else:
        X = np.concatenate([x0, sX])
    y = np.full((N,), np.inf)
    for i in tqdm(range(0, N), disable=settings.progress_bar.off()):
        y[i] = f(X[i, :])

    order = np.argsort(y)

    return X[order, :], y[order]


def uuinv(p, B, w):
    """
    Inverse of cumulative distribution function of mixture of uniform
    distributions. The mixture is:
    .. math::
    w \text{Uniform}(B[1], B[2]) +
    \frac{1 - w}{2} (\text{Uniform}(B[0], B[1]) + \text{Uniform}(B[2], B[3]))

    Parameters
    ----------
    p : ndarray
        1D array of cumulative function values.
    B : ndarray, list
        1D array or list containing [LB, PLB, PUB, UB].
    w : float
        The coefficient for mixture of uniform distributions.
        :math: `0 \leq w \leq 1`.

    Returns
    -------
    x : ndarray
        1D array of samples corresponding to `p`.
    """
    assert B[0] <= B[1] <= B[2] <= B[3]
    assert 0 <= w <= 1
    x = np.zeros(p.shape)
    L = B[3] - B[0] + B[1] - B[2]

    if w == 1:
        x = p * (B[2] - B[1]) + B[1]
        return x

    if L == 0:
        # Degenerate to mixture of delta and uniform distributions
        i1 = p <= (1 - w) / 2
        x[i1] = B[0]

        if w != 0:
            i2 = (p <= (1 - w) / 2 + w) & ~i1
            x[i2] = (p[i2] - (1 - w) / 2) * (B[2] - B[1]) / w + B[1]

        i3 = p > (1 - w) / 2 + w
        x[i3] = B[3]
        return x

    # First step
    i1 = p <= (1 - w) * (B[1] - B[0]) / L
    x[i1] = B[0] + p[i1] * L / (1 - w)

    # Second step
    i2 = (p <= (1 - w) * (B[1] - B[0]) / L + w) & ~i1
    if w != 0:
        x[i2] = (p[i2] - (1 - w) * (B[1] - B[0]) / L) * (B[2] - B[1]) / w + B[
            1
        ]

    # Third step
    i3 = p > (1 - w) * (B[1] - B[0]) / L + w
    x[i3] = (p[i3] - w - (1 - w) * (B[1] - B[0]) / L) * L / (1 - w) + B[2]

    x[p < 0] = np.nan
    x[p > 1] = np.nan

    return x


def smoothbox_cdf(x: float, sigma: float, a: float, b: float):
    """
    Compute the value of the cumulative distribution function
    for the smooth box distribution.

    Parameters
    ==========
    x : float
        The point where we want the value of the cdf.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if x < a:
        return sp.stats.norm.cdf(x, loc=a, scale=sigma) / C

    if x <= b:
        return (0.5 + (x - a) / (sigma * np.sqrt(2 * np.pi))) / C

    return (C - 1.0 + sp.stats.norm.cdf(x, loc=b, scale=sigma)) / C


def smoothbox_student_t_cdf(
    x: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the cumulative distribution function
    for the smooth box student t distribution.

    Parameters
    ==========
    x : float
        The point where we want the value of the cdf.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c

    if x < a:
        return sp.stats.t.cdf(x, df, loc=a, scale=sigma) / C

    if x <= b:
        return (0.5 + (x - a) * c) / C

    return (C - 1.0 + sp.stats.t.cdf(x, df, loc=b, scale=sigma)) / C


def smoothbox_ppf(q: float, sigma: float, a: float, b: float):
    """
    Compute the value of the percent point function for
    the smooth box distribution.

    Parameters
    ==========
    q : float
        The quantile where we want the value of the ppf.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if q < 0.5 / C:
        return sp.stats.norm.ppf(C * q, loc=a, scale=sigma)

    if q <= (C - 0.5) / C:
        return (q * C - 0.5) * sigma * np.sqrt(2 * np.pi) + a

    return sp.stats.norm.ppf(C * q - (C - 1), loc=b, scale=sigma)


def smoothbox_student_t_ppf(
    q: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the percent point function for
    the smooth box student t distribution.

    Parameters
    ==========
    q : float
        The quantile where we want the value of the ppf.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c

    if q < 0.5 / C:
        return sp.stats.t.ppf(C * q, df, loc=a, scale=sigma)

    if q <= (C - 0.5) / C:
        return (q * C - 0.5) / c + a

    return sp.stats.t.ppf(C * q - (C - 1), df, loc=b, scale=sigma)


def _meanfun_name_to_mean_function(name: str):
    """
    Transforms a mean function name to an instance of that mean function.

    Parameters
    ==========
    name : str
        Name of the mean function.

    Returns
    =======
    mean_f : object
        An instance of the specified mean function.

    Raises
    ------
    ValueError
        Raised when the mean function name is unknown.
    """
    if name == "zero":
        mean_f = gpr.mean_functions.ZeroMean()
    elif name == "const":
        mean_f = gpr.mean_functions.ConstantMean()
    elif name == "negquad":
        mean_f = gpr.mean_functions.NegativeQuadratic()
    else:
        raise ValueError("Unknown mean function!")

    return mean_f


def _cov_identifier_to_covariance_function(identifier):
    """
    Transforms a covariance function identifer to an instance of the
    corresponding covariance function.

    Parameters
    ==========
    identifier : object
        Either an integer, or a list such as [3, 3] where the first
        number is the identifier and the further numbers are parameters
        of the covariance function.

    Returns
    =======
    cov_f : object
        An instance of the specified covariance function.

    Raises
    ------
    ValueError
        Raised when the covariance function identifier is unknown.
    """
    if identifier == 1:
        cov_f = gpr.covariance_functions.SquaredExponential()
    elif identifier == 3:
        cov_f = gpr.covariance_functions.Matern(5)
    elif isinstance(identifier, list) and identifier[0] == 3:
        cov_f = gpr.covariance_functions.Matern(identifier[1])
    else:
        raise ValueError("Unknown covariance function")

    return cov_f


def gpyreg_params_to_jaxgp(gpyreg_params: Union[Dict, List[Dict]]):
    # if isinstance(gpyreg_params, list):
    #     gpyreg_params = gpyreg_params[0]  # take first posterior sample only
    lengthscale = jnp.exp(gpyreg_params["covariance_log_lengthscale"])
    outputscale = jnp.exp(2 * gpyreg_params["covariance_log_outputscale"])
    noise_scale = jnp.exp(gpyreg_params["noise_log_scale"])
    mean_const = jnp.array(gpyreg_params["mean_const"])
    mean_location = jnp.array(gpyreg_params["mean_location"])
    mean_scale = jnp.exp(gpyreg_params["mean_log_scale"])
    jaxgp_params = {
        "kernel": {"lengthscale": lengthscale, "outputscale": outputscale},
        "likelihood": {"noise_add": noise_scale**2},
        "mean_function": {
            "mean_const": mean_const,
            "xm": mean_location,
            "scale": mean_scale,
        },
    }
    return jaxgp_params


def jaxgp_params_to_gpyreg(jaxgp_params: Dict):
    gpyreg_params = {}
    gpyreg_params["covariance_log_lengthscale"] = np.log(
        jaxgp_params["kernel"]["lengthscale"]
    )
    gpyreg_params["covariance_log_outputscale"] = (
        np.log(jaxgp_params["kernel"]["outputscale"]) / 2
    )
    gpyreg_params["noise_log_scale"] = (
        np.log(jaxgp_params["likelihood"]["noise_add"]) / 2
    )
    gpyreg_params["mean_const"] = np.array(
        jaxgp_params["mean_function"]["mean_const"]
    )
    gpyreg_params["mean_location"] = np.array(
        jaxgp_params["mean_function"]["xm"]
    )
    gpyreg_params["mean_log_scale"] = np.log(
        jaxgp_params["mean_function"]["scale"]
    )
    return gpyreg_params


def gpyreg_bounds_to_jaxgp(bounds: dict):
    lengthscale = list(map(jnp.exp, bounds["covariance_log_lengthscale"]))
    outputscale = list(
        map(lambda x: jnp.exp(2 * x), bounds["covariance_log_outputscale"])
    )
    noise_scale = list(map(jnp.exp, bounds["noise_log_scale"]))
    mean_const = list(map(jnp.array, bounds["mean_const"]))
    mean_location = list(map(jnp.array, bounds["mean_location"]))
    mean_scale = list(map(jnp.exp, bounds["mean_log_scale"]))
    res = [None, None]
    for i in range(2):
        res[i] = {
            "kernel": {
                "lengthscale": lengthscale[i],
                "outputscale": outputscale[i],
            },
            "likelihood": {"noise_add": noise_scale[i] ** 2},
            "mean_function": {
                "mean_const": mean_const[i],
                "xm": mean_location[i],
                "scale": mean_scale[i],
            },
        }
    return res[0], res[1]


def gpyreg_priors_to_jaxgp(priors: Dict, bounds: Dict):
    assert priors["covariance_log_lengthscale"][0] == "student_t"
    assert priors["noise_log_scale"][0] == "student_t"

    res = {
        "kernel": {
            "lengthscale": TransformedDistribution(
                tfd.StudentT(
                    df=priors["covariance_log_lengthscale"][1][2],
                    loc=priors["covariance_log_lengthscale"][1][0],
                    scale=priors["covariance_log_lengthscale"][1][1],
                ),
                bijector=tfb.Exp(),
            ),
            "outputscale": None,
        },
        "likelihood": {
            "noise_add": TransformedDistribution(
                tfd.StudentT(
                    df=priors["noise_log_scale"][1][2],
                    loc=priors["noise_log_scale"][1][0],
                    scale=priors["noise_log_scale"][1][1],
                ),
                bijector=tfb.Chain([tfb.Power(2), tfb.Exp()]),
            )
        },
        "mean_function": {
            "mean_const": None,
            "xm": None,
            "scale": None,
        },
    }
    lower_bounds, upper_bounds = gpyreg_bounds_to_jaxgp(bounds)
    res["kernel"]["lengthscale"] = TruncatedPrior(
        res["kernel"]["lengthscale"],
        lower_bounds["kernel"]["lengthscale"],
        upper_bounds["kernel"]["lengthscale"],
    )
    res["likelihood"]["noise_add"] = TruncatedPrior(
        res["likelihood"]["noise_add"],
        lower_bounds["likelihood"]["noise_add"],
        upper_bounds["likelihood"]["noise_add"],
    )
    return res


def compute_inducing_points_bounds(X: np.ndarray, N_iv: int):
    LB = X.min(0)
    UB = X.max(0)
    delta = (UB - LB) * 0.1
    LB -= delta
    UB += delta
    LB = np.repeat(LB[None, :], N_iv, 0)
    UB = np.repeat(UB[None, :], N_iv, 0)
    return {"inducing_points": LB}, {"inducing_points": UB}
