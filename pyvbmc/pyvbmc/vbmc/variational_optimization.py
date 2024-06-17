"""Variational optimization / training of variational posterior"""

import copy
import logging
import math
import os
import time
from functools import partial
from typing import Union

import gpyreg as gpr
import jax
import jax.numpy as jnp
import jaxgp as jgp
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy as sp
from plum import dispatch

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.stats import get_hpd
from pyvbmc.variational_posterior import VariationalPosterior

from .iteration_history import IterationHistory
from .minimize_adam import minimize_adam
from .options import Options

logger = logging.getLogger("VBMC_debug")


def optimize_vp(
    options: Options,
    optim_state: dict,
    vp: VariationalPosterior,
    gp: gpr.GP,
    fast_opts_N: int,
    slow_opts_N: int,
    K: int = None,
    debug: bool = False,
):
    """
    Optimize variational posterior.

    Parameters
    ==========
    options : Options
        Options from the VBMC instance we are calling this from.
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    vp : VariationalPosterior
        The variational posterior we want to optimize.
    fast_opts_N : int
        Number of fast optimizations.
    slow_opts_N : int
        Number of slow optimizations.
    K : int, optional
        Number of mixture components. If not given defaults to the number
        of mixture components the given VP has.

    Returns
    =======
    vp : VariationalPosterior
        The optimized variational posterior.
    var_ss : int
        To be written by Luigi.
    pruned : int
        Number of pruned components.
    """

    if K is None:
        K = vp.K

    # Quick sieve optimization to determine starting point(s)
    ts = time.time()
    logger.debug(
        f"_sieve start, {ts}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
    )

    assert fast_opts_N == 0
    assert slow_opts_N == 1
    vp0_vec, vp0_type, elcbo_beta, compute_var, nsent_K, _ = _sieve(
        options,
        optim_state,
        vp,
        gp,
        K=K,
        init_N=fast_opts_N,
        best_N=slow_opts_N,
    )
    logger.debug(
        f"_sieve finish, {time.time() - ts:.2f}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
    )
    # Compute soft bounds for variational parameter optimization.
    # hpd_X, hpd_y, _, _ = get_hpd(gp.X, gp.y, 0.8)
    # theta_bnd = vp.get_bounds(hpd_X, options, K)
    if not options.get("turn_off_vp_bound_loss"):
        theta_bnd = vp.get_bounds(gp.X, options, K)
    else:
        theta_bnd = None

    ## Perform optimization starting from one or few selected points.

    # Set up an empty stats struct for optimization
    theta_N = np.size(vp0_vec[0].get_parameters())
    Ns = np.size(gp.posteriors)
    elbo_stats = _initialize_full_elcbo(slow_opts_N * 2, theta_N, K, Ns)

    # For the moment no gradient available for variance
    gradient_available = compute_var == 0

    # if gradient_available:
    #     # Set basic options for deterministic (?) optimizer
    #     compute_grad = True
    # else:
    #     if nsent_K > 0:
    #         raise ValueError(
    #             """Gradients must be available when nsent_K is > 0."""
    #         )
    #     else:
    #         compute_grad = False

    ts = time.time()
    logger.debug(
        f"slow_opts_N start, {ts}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
    )
    vp0_fine = {}
    for i in range(0, slow_opts_N):
        # Be careful with off-by-one errors here, Python is zero-based.
        i_mid = (i + 1) * 2 - 2
        i_end = (i + 1) * 2 - 1

        # Select points from best ones depending on subset
        if slow_opts_N == 1:
            idx = 0
        elif slow_opts_N == 2:
            if i == 0:
                idx = np.where(vp0_type == 1)[0][0]
            else:
                idx = np.where((vp0_type == 2) | (vp0_type == 3))[0][0]
        else:
            idx = np.where(vp0_type == (i % 3) + 1)[0][0]

        vp0 = vp0_vec[idx]

        # if optim_state.get("finalboost"):
        #     file_path = os.path.join(
        #         options["experimentfolder"], f"vp0_{i}_finalboost.png"
        #     )
        # else:
        #     file_path = os.path.join(
        #         options["experimentfolder"],
        #         f'vp0_{i}_iter_{optim_state["iter"]}.png',
        #     )
        # fig = vp0.plot_with_extra_data(
        #     n_samples=int(1e5),
        #     train_X=gp.X,
        #     plot_vp_centres=True,
        #     title=f"type: {vp0_type[idx]}",
        #     original_space=None,
        #     save_path=file_path,
        # )

        vp0_vec = np.delete(vp0_vec, idx)
        vp0_type = np.delete(vp0_type, idx)
        theta0 = vp0.get_parameters()

        if nsent_K == 0:
            # Fast optimization via deterministic entropy approximation
            raise NotImplementedError()
        else:
            # Objective function, should only return value and gradient.
            def vbtrain_mc_fun(theta_):
                res = _negelcbo(
                    theta_,
                    gp,
                    vp0,
                    elcbo_beta,
                    nsent_K,
                    compute_grad=True,
                    compute_var=compute_var,
                    theta_bnd=theta_bnd,
                )
                return res[0], res[1]

            # Optimization via unbiased stochastic entroply approximation
            theta_opt = theta0

            if options["stochasticoptimizer"] == "adam":
                master_min = min(options["sgdstepsize"], 0.001)
                if not vp.optimize_weights:
                    scaling_factor = min(0.1, options["sgdstepsize"] * 10)
                else:
                    scaling_factor = min(0.1, options["sgdstepsize"])

                # Fixed master stepsize
                master_max = scaling_factor

                # Note: we tried to adapt the stepsizes guided by the GP
                # hyperparameters, but this did not seem to help (the former
                # experimental option was "GPStochasticStepsize").
                master_max = max(master_min, master_max)
                master_decay = 200
                max_iter = min(10000, options["maxiterstochastic"])
                # print(master_max, master_min)
                theta_opt, _, theta_lst, f_val_lst, _ = minimize_adam(
                    vbtrain_mc_fun,
                    theta_opt,
                    tol_fun=options["tolfunstochastic"],
                    max_iter=max_iter,
                    master_min=master_min,
                    master_max=master_max,
                    master_decay=master_decay,
                    use_early_stopping=options.get("use_early_stopping", True),
                )

                if options["elcbomidpoint"]:
                    # Recompute ELCBO at best midpoint with full variance
                    # and more precision.
                    idx_mid = np.argmin(f_val_lst)
                    elbo_stats = _eval_full_elcbo(
                        i_mid,
                        theta_lst[:, idx_mid],
                        vp0,
                        gp,
                        elbo_stats,
                        elcbo_beta,
                        options,
                    )
            else:
                raise ValueError("Unknown stochastic optimizer!")

        # Recompute ELCBO at endpoint with full variance and more precision
        elbo_stats = _eval_full_elcbo(
            i_end, theta_opt, vp0, gp, elbo_stats, elcbo_beta, options
        )

        vp0_fine[i_mid] = copy.deepcopy(vp0)
        vp0_fine[i_end] = copy.deepcopy(vp0)  # Parameters get assigned later

    logger.debug(
        f"slow_opts_N finish, {time.time() - ts: .2f}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
    )

    ## Finalize optimization by taking variational parameters with best ELCBO

    idx = np.argmin(elbo_stats["nelcbo"])
    elbo = -elbo_stats["nelbo"][idx]
    elbo_sd = np.sqrt(elbo_stats["varF"][idx])
    G = elbo_stats["G"][idx]
    H = elbo_stats["H"][idx]
    varss = elbo_stats["varss"][idx]
    varG = elbo_stats["varG"][idx]
    varH = elbo_stats["varH"][idx]
    I_sk = np.zeros((Ns, K))
    J_sjk = np.zeros((Ns, K, K))
    I_sk[:, :] = elbo_stats["I_sk"][idx, :, :].copy()
    J_sjk[:, :, :] = elbo_stats["J_sjk"][idx, :, :, :].copy()
    vp = vp0_fine[idx]
    vp.set_parameters(elbo_stats["theta"][idx, :])

    ## Potentionally prune mixture components
    pruned = 0
    ts = time.time()
    logger.debug(
        f"Prune start, {ts}, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
    )
    if vp.optimize_weights and not optim_state["building_vp"]:
        already_checked = np.full((vp.K,), False)
        prune_cnt = 0
        while np.any((vp.w < options["tolweight"]) & ~already_checked):
            prune_cnt += 1
            vp_pruned = copy.deepcopy(vp)

            # Choose a random component below threshold
            idx = np.argwhere(
                (vp.w < options["tolweight"]).flatten() & ~already_checked
            ).flatten()
            idx = idx[np.random.randint(0, np.size(idx))]
            vp_pruned.w = np.delete(vp_pruned.w, idx)
            vp_pruned.eta = np.delete(vp_pruned.eta, idx)
            vp_pruned.sigma = np.delete(vp_pruned.sigma, idx)
            vp_pruned.mu = np.delete(vp_pruned.mu, idx, axis=1)
            vp_pruned.K -= 1
            theta_pruned = vp_pruned.get_parameters()
            # Recompute ELCBO
            elbo_stats = _eval_full_elcbo(
                0,
                theta_pruned,
                vp_pruned,
                gp,
                elbo_stats,
                elcbo_beta,
                options,
            )
            elbo_pruned = -elbo_stats["nelbo"][0]
            elbo_pruned_sd = np.sqrt(elbo_stats["varF"][0])

            # Difference in ELCBO (before and after pruning)
            delta_elcbo = np.abs(
                (elbo_pruned - options["elcboimproweight"] * elbo_pruned_sd)
                - (elbo - options["elcboimproweight"] * elbo_sd)
            )
            # Prune component if it has neglible influence on ELCBO
            pruning_threshold = options["tolimprovement"] * options.eval(
                "pruningthresholdmultiplier", {"K": K}
            )

            if delta_elcbo < pruning_threshold:
                vp = vp_pruned
                elbo = elbo_pruned
                elbo_sd = elbo_pruned_sd
                G = elbo_stats["G"][0]
                H = elbo_stats["H"][0]
                varss = elbo_stats["varss"][0]
                varG = elbo_stats["varG"][0]
                varH = elbo_stats["varH"][0]
                pruned += 1
                already_checked = np.delete(already_checked, idx)
                I_sk = np.delete(I_sk, idx, axis=1)
                J_sjk = np.delete(J_sjk, idx, axis=2)
            else:
                already_checked[idx] = True
        logger.debug(
            f"Check pruning {prune_cnt} times, {pruned} components pruned, {time.time() - ts:.2f}s, {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
        )

    vp.stats = {}
    vp.stats["elbo"] = elbo  # ELBO
    vp.stats["elbo_sd"] = elbo_sd  # Error on the ELBO
    vp.stats["elogjoint"] = G  # Expected log joint
    vp.stats["elogjoint_sd"] = np.sqrt(varG)  # Error on expected log joint
    vp.stats["entropy"] = H  # Entropy
    vp.stats["entropy_sd"] = np.sqrt(varH)  # Error on the entropy
    vp.stats["stable"] = False  # Unstable until proven otherwise
    vp.stats["I_sk"] = I_sk  # Expected log joint per component
    vp.stats["J_sjk"] = J_sjk  # Covariance of expected log joint

    if debug:
        return vp, varss, pruned, f_val_lst
    return vp, varss, pruned


def _initialize_full_elcbo(max_idx: int, D: int, K: int, Ns: int):
    """Initialize a dictionary for keeping track of full ELCBO output.

    Parameters
    ==========
    max_idx : int
        Maximum number of full ELCBO evaluations.
    D : int
        The dimension.
    K : int
        Number of mixture components.
    Ns : int
        Number of samples for entropy approximation.

    Returns
    =======
    elbo_stats : dict
        A dictionary with entries for all output variables of full ELCBO.
    """
    elbo_stats = {}
    elbo_stats["nelbo"] = np.full((max_idx,), np.inf)
    elbo_stats["G"] = np.full((max_idx,), np.nan)
    elbo_stats["H"] = np.full((max_idx,), np.nan)
    elbo_stats["varF"] = np.full((max_idx,), np.nan)
    elbo_stats["varG"] = np.full((max_idx,), np.nan)
    elbo_stats["varH"] = np.full((max_idx,), np.nan)
    elbo_stats["varss"] = np.full((max_idx,), np.nan)
    elbo_stats["nelcbo"] = np.full((max_idx,), np.inf)
    elbo_stats["theta"] = np.full((max_idx, D), np.nan)
    elbo_stats["I_sk"] = np.full((max_idx, Ns, K), np.nan)
    elbo_stats["J_sjk"] = np.full((max_idx, Ns, K, K), np.nan)
    return elbo_stats


def _eval_full_elcbo(
    idx: int,
    theta: np.ndarray,
    vp: VariationalPosterior,
    gp: gpr.GP,
    elbo_stats: dict,
    beta: float,
    options: Options,
    entropy_alpha: float = 0.0,
):
    """Evaluate full ELCBO and store the results in a dictionary.

    Parameters
    ==========
    idx : int
        Index in the dictionary to which store the evaluated values.
    theta : np.ndarray
        VP parameters for which to evaluate full ELCBO.
    vp : VariationalPosterior
        The variational posterior in question.
    gp : GP
        Gaussian process from VBMC main loop.
    elbo_stats : dict
        The dictionary for storing full ELCBO stats.
    beta : float
        Confidence weight.
    options : Options
        Options from the VBMC instance we are calling from.
    entropy_alpha : float, defaults to 0.0
        To be written by Luigi

    Returns
    =======
    elbo_stats : dict
        The updated dictionary.
    """
    # Number of samples per component for MC approximation of the entropy.
    K = vp.K
    nsent_fine_K = math.ceil(options.eval("nsentfine", {"K": K}) / K)

    if "skipelbovariance" in options and options["skipelbovariance"]:
        compute_var = False
    else:
        compute_var = True

    nelbo, _, G, H, varF, _, varss, varG, varH, I_sk, J_sjk = _negelcbo(
        theta,
        gp,
        vp,
        0,
        nsent_fine_K,
        False,
        compute_var,
        None,
        entropy_alpha,
        True,
    )
    nelcbo = nelbo + beta * np.sqrt(varF)

    elbo_stats["nelbo"][idx] = nelbo
    elbo_stats["G"][idx] = G
    elbo_stats["H"][idx] = H
    elbo_stats["varF"][idx] = varF
    elbo_stats["varG"][idx] = varG
    elbo_stats["varH"][idx] = varH
    elbo_stats["varss"][idx] = varss
    elbo_stats["nelcbo"][idx] = nelcbo
    elbo_stats["theta"][idx, 0 : np.size(theta)] = theta
    elbo_stats["I_sk"][idx, :, 0:K] = I_sk
    elbo_stats["J_sjk"][idx, :, 0:K, 0:K] = J_sjk

    return elbo_stats


def _vp_bound_loss(
    vp: VariationalPosterior,
    theta: np.ndarray,
    theta_bnd: dict,
    tol_con: float = 1e-3,
    compute_grad: bool = True,
):
    """
    Variational parameter loss function for soft optimization bounds.

    Parameters
    ==========
    vp : VariationalPosterior
        The variational posterior for which we are interested in computing
        the loss function on.
    theta : np.ndarray, shape (N,)
        The parameters at which we want to compute the loss function.
    theta_bnd : dict
        Variational posterior soft bounds.
    tol_con : float, defaults to 1e-3
        Penalization relative scale.
    compute_grad : bool, defaults to True
        Whether to compute gradients.

    Returns
    =======
    L : float
        The value of the loss function.
    dL : np.ndarray, shape (N,), optional
        The gradient of the loss function.

    """

    if vp.optimize_mu:
        mu = theta[: vp.D * vp.K]
        start_idx = vp.D * vp.K
    else:
        mu = vp.mu.flatten(order="F")
        start_idx = 0

    if vp.optimize_sigma:
        ln_sigma = theta[start_idx : start_idx + vp.K]
        start_idx += vp.K
    else:
        ln_sigma = np.log(vp.sigma.flatten())

    if vp.optimize_lambd:
        ln_lambd = theta[start_idx : start_idx + vp.D].T
    else:
        ln_lambd = np.log(vp.lambd.flatten())

    if vp.optimize_weights:
        eta = theta[-vp.K :]

    ln_scale = np.reshape(ln_lambd, (-1, 1)) + np.reshape(ln_sigma, (1, -1))
    theta_ext = []
    if vp.optimize_mu:
        theta_ext.append(mu.flatten())
    if vp.optimize_sigma or vp.optimize_lambda:
        theta_ext.append(ln_scale.flatten(order="F"))
    if vp.optimize_weights:
        theta_ext.append(eta.flatten())
    theta_ext = np.concatenate(theta_ext)

    if compute_grad:
        L, dL = _soft_bound_loss(
            theta_ext,
            theta_bnd["lb"].flatten(),
            theta_bnd["ub"].flatten(),
            tol_con,
            compute_grad=True,
        )

        dL_new = np.array([])
        if vp.optimize_mu:
            dL_new = np.concatenate((dL_new, dL[0 : vp.D * vp.K].flatten()))
            start_idx = vp.D * vp.K
        else:
            start_idx = 0

        if vp.optimize_sigma or vp.optimize_lambda:
            dlnscale = np.reshape(
                dL[start_idx : start_idx + vp.D * vp.K], (vp.D, vp.K)
            )

            if vp.optimize_sigma:
                dL_new = np.concatenate((dL_new, np.sum(dlnscale, axis=0)))

            if vp.optimize_lambd:
                dL_new = np.concatenate((dL_new, np.sum(dlnscale, axis=1)))

        if vp.optimize_weights:
            dL_new = np.concatenate((dL_new, dL[-vp.K :].flatten()))

        return L, dL_new

    L = _soft_bound_loss(
        theta_ext,
        theta_bnd["lb"].flatten(),
        theta_bnd["ub"].flatten(),
        tol_con,
    )

    return L


def _soft_bound_loss(
    x: np.ndarray,
    slb: np.ndarray,
    sub: np.ndarray,
    tol_con: float = 1e-3,
    compute_grad: bool = False,
):
    """
    Loss function for soft bounds for function minimization.

    Parameters
    ==========
    x : np.ndarray, shape (D,)
        Point for which we want to know the loss function value.
    slb : np.ndarray, shape (D,)
        Soft lower bounds.
    sub : np.ndarray, shape (D,)
        Soft upper bounds.
    tol_con : float, defaults to 1e-3
        Penalization relative scale.
    compute_grad : bool, defaults to False
        Whether to compute gradients.

    Returns
    =======
    y : float
        The value of the loss function.
    dy : np.ndarray, shape (D,), optional
        The gradient of the loss function.
    """
    ell = (sub - slb) * tol_con
    y = 0.0
    dy = np.zeros(x.shape)

    idx = x < slb
    if np.any(idx):
        y += 0.5 * np.sum(((slb[idx] - x[idx]) / ell[idx]) ** 2)
        if compute_grad:
            dy[idx] = (x[idx] - slb[idx]) / ell[idx] ** 2

    idx = x > sub
    if np.any(idx):
        y += 0.5 * np.sum(((x[idx] - sub[idx]) / ell[idx]) ** 2)
        if compute_grad:
            dy[idx] = (x[idx] - sub[idx]) / ell[idx] ** 2

    if compute_grad:
        return y, dy
    return y


def get_cluster_centers_inds(X: np.ndarray, n_clusters: int, method="K-means"):
    """
    Get indices of cluster centers.

    Parameters
    ==========
    X : np.ndarray, shape (N, D)
        The training points.
    n_clusters : int
        The number of clusters.

    Returns
    =======
    inds : np.ndarray, shape (n_clusters,)
        The indices of the cluster centers.
    """
    if X.shape[0] <= n_clusters:
        inds = np.arange(X.shape[0])
        inds = np.tile(inds, int(n_clusters // X.shape[0]) + 1)
        inds = inds[:n_clusters]
        return inds
    if method == "K-medoids":
        from sklearn_extra.cluster import KMedoids

        kmedoids = KMedoids(n_clusters=n_clusters).fit(X)
        inds = kmedoids.medoid_indices_
        return inds
    elif method == "K-means":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        dist = kmeans.transform(X)
        # for each cluster, find the closest point to cluster center
        inds = np.argmin(dist, 0)
    else:
        raise ValueError(f"{method} is not supported.")
    return inds


# def count_nearby_points(centers, X, distance=0.05):
#     N = centers.shape[0]
#     for i in range(N):
#         center = centers[i]
#         X <= center + distance


def _sieve(
    options: Options,
    optim_state: dict,
    vp: VariationalPosterior,
    gp: gpr.GP,
    init_N: int = None,
    best_N: int = 1,
    K: int = None,
):
    """
    Preliminary 'sieve' method for fitting variational posterior.

    Parameters
    ==========
    options : Options
        Options from the VBMC instance we are calling this from.
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    vp : VariationalPosterior
        The variational posterior to use as a basis for new candidates.
    gp : GP
        Current GP from optimization.
    init_N : int, optional
        Number of initial starting points.
    best_N : int, defaults to 1
        To be written by Luigi.  # TODO: ?
    K : int, optional
        Number of mixture components. If not given defaults to the number
        of mixture components the given VP has.

    Returns
    =======
    vp0_vec : np.ndarray, shape (init_N,)
        Vector of candidate variational posteriors.
    vp0_type : np.ndarray, shape (init_N,)
        Vector of types of candidate variational posteriors.
    elcbo_beta : float
        Confidence weight.
    compute_var : bool
        Whether to compute variance in later optimization.
    nsent_K : int
        Number of samples per component for MC approximation of the entropy.
    nsent_K_fast : int
        Number of samples per component for preliminary MC approximation of
        the entropy.
    """
    if K is None:
        K = vp.K

    if optim_state["iter"] == 0:
        if optim_state.get("X_centroids") is None:
            # Initialize with K-means on top x% points
            X_star, y_star, _, _ = get_hpd(
                gp.X, gp.y, options["clustering_init_hpdfrac"]
            )
            # Get cluster centers
            inds = get_cluster_centers_inds(
                X_star, min(options["n_clusters"], vp.K)
            )
            optim_state["X_centroids"] = X_star[inds]
            optim_state["y_centroids"] = y_star[inds]
            X_centroids = optim_state["X_centroids"]
        else:
            # TODO: delete a centroid that has few points surrounding it, unless its log density value is the maximum among all centroids
            X_centroids = optim_state["X_centroids"]
            # count_nearby_points(X_centroids, gp.X)
            # y_centroids = optim_state["y_centroids"]
            num_repeats = int(np.ceil(vp.K / X_centroids.shape[0]))
            X_centroids = np.tile(X_centroids, (num_repeats, 1))

        x0 = X_centroids[: vp.K].T
        vp.w = np.ones((1, vp.K)) / vp.K
        vp.eta = np.ones((1, vp.K)) / vp.K
        vp.mu = x0 + 1e-6 * np.random.randn(vp.D, vp.K)
        vp.sigma = 1e-3 * np.ones((1, vp.K))
        vp.lambd = np.ones((vp.D, 1))

    # Missing port: assign default values to optim_state (since
    #               this doesn't seem to be necessary)

    ## Set up optimization variables and options.

    vp.delta = None

    # Number of initial starting points
    if init_N is None:
        init_N = math.ceil(options.eval("nselbo", {"K": K}))
    nelcbo_fill = np.zeros((init_N,))

    # Number of samples per component for MC approximation of the entropy.
    nsent_K = math.ceil(options.eval("nsent", {"K": K}) / K)

    # Number of samples per component for preliminary MC approximation
    # of the entropy.
    nsent_K_fast = math.ceil(options.eval("nsentfast", {"K": K}) / K)

    # Deterministic entropy if entropy switch is on or only one component
    if optim_state["entropy_switch"] or K == 1:
        nsent_K = 0
        nsent_K_fast = 0

    # Confidence weight
    # Missing port: elcboweight does not exist
    # elcbo_beta = self._eval_option(self.options["elcboweight"],
    #                                self.optim_state["n_eff"])
    elcbo_beta = options.get("elcbo_beta", 0)
    compute_var = elcbo_beta != 0

    # Compute soft bounds for variational parameter optimization
    # hpd_X, hpd_y, _, _ = get_hpd(gp.X, gp.y, 0.8)
    # theta_bnd = vp.get_bounds(hpd_X, options, K)
    theta_bnd = vp.get_bounds(gp.X, options, K)

    ## Perform quick shotgun evaluation of many candidate parameters

    if init_N > 0:
        # Get high-posterior density points
        X_star, y_star, _, _ = get_hpd(gp.X, gp.y, options["hpdfrac"])
        # Get cluster centers
        # if optim_state.get("X_centroids") is None:
        #     inds = get_cluster_centers_inds(gp.X, 5)
        #     optim_state["X_centroids"] = gp.X[inds]
        #     optim_state["y_centroids"] = gp.y[inds]

        # Generate a bunch of random candidate variational parameters.

        if init_N < 3:
            vp0_vec, vp0_type = _vbinit(vp, 1, init_N, K, X_star, y_star)
        else:
            # Fix random seed here if trying to reproduce MATLAB numbers
            vp0_vec1, vp0_type1 = _vbinit(
                vp, 1, math.ceil(init_N / 3), K, X_star, y_star
            )
            vp0_vec2, vp0_type2 = _vbinit(
                vp, 2, math.ceil(init_N / 3), K, X_star, y_star
            )
            vp0_vec3, vp0_type3 = _vbinit(
                vp,
                3,
                init_N - 2 * math.ceil(init_N / 3),
                K,
                optim_state["X_centroids"],
                optim_state["y_centroids"],
            )
            vp0_vec = np.concatenate([vp0_vec1, vp0_vec2, vp0_vec3])
            vp0_type = np.concatenate([vp0_type1, vp0_type2, vp0_type3])

        # in MATLAB the vp_repo is used here

        # Quickly estimate ELCBO at each candidate variational posterior.
        for i, vp0 in enumerate(vp0_vec):
            theta = vp0.get_parameters()
            nelbo_tmp, _, _, _, varF_tmp = _negelcbo(
                theta,
                gp,
                vp0,
                0,
                nsent_K_fast,
                0,
                compute_var,
                theta_bnd,
            )
            nelcbo_fill[i] = nelbo_tmp + elcbo_beta * np.sqrt(varF_tmp)

        # Sort by negative ELCBO
        order = np.argsort(nelcbo_fill)
        vp0_vec = vp0_vec[order]
        vp0_type = vp0_type[order]

        return (
            vp0_vec,
            vp0_type,
            elcbo_beta,
            compute_var,
            nsent_K,
            nsent_K_fast,
        )

    return (
        [copy.deepcopy(vp)],
        np.ones(1),
        elcbo_beta,
        compute_var,
        nsent_K,
        nsent_K_fast,
    )


def _vbinit(
    vp: VariationalPosterior,
    vbtype: int,
    opts_N: int,
    K_new: int,
    X_star: np.ndarray,
    y_star: np.ndarray,
):
    """
    Generate array of random starting parameters for variational posterior.

    Parameters
    ==========
    vp : VariationalPosterior
        Variational posterior to use as base.
    vbtype : {1, 2, 3}
        Type of method to create new starting parameters. Here 1 means
        starting from old variational parameters, 2 means starting from
        highest-posterior density training points, and 3 means starting
        from random provided training points.
    opts_N : int
        Number of random starting parameters.
    K_new : int
        New number of mixture components.
    X_star : np.ndarray, shape (N, D)
        Training inputs, usually HPD regions.
    y_star : np.ndarray, shape (N, 1)
        Training targets, usually HPD regions.

    Returns
    =======
    vp0_vec : np.ndarray, shape (opts_N, )
        The array of random starting parameters.
    type_vec : np.ndarray, shape (opts_N, )
        The array of type of each random starting parameter.
    """

    D = vp.D
    K = vp.K
    N_star = X_star.shape[0]
    add_jitter = True
    type_vec = vbtype * np.ones((opts_N))
    lambd0 = vp.lambd.copy()
    mu0 = vp.mu.copy()
    w0 = vp.w.copy()

    if vbtype == 1:
        # Start from old variational parameters
        sigma0 = vp.sigma.copy()
    elif vbtype == 2:
        # Start from highest-posterior density training points
        if vp.optimize_mu:
            order = np.argsort(y_star, axis=None)[::-1]
            idx_order = np.tile(
                range(0, min(K_new, N_star)), (math.ceil(K_new / N_star),)
            )
            mu0 = X_star[order[idx_order[0:K_new]], :].T
        if K > 1:
            V = np.var(mu0, axis=1, ddof=1)
        else:
            V = np.var(X_star, axis=0, ddof=1)
        sigma0 = np.sqrt(np.mean(V / lambd0**2) / K_new) * np.exp(
            0.2 * np.random.randn(1, K_new)
        )
    else:
        # Start from random provided training points.
        if vp.optimize_mu:
            mu0 = np.zeros((D, K))
        sigma0 = np.zeros((1, K))

    vp0_list = []
    for i in range(0, opts_N):
        mu = mu0.copy()
        sigma = sigma0.copy()
        lambd = lambd0.copy()
        if vp.optimize_weights:
            w = w0.copy()

        if vbtype == 1:
            # Start from old variational parameters

            # Copy previous parameters verbatim.
            if i == 0:
                add_jitter = False
            else:
                add_jitter = True
            if K_new > vp.K:
                # Spawn a new component near an existing one
                for i_new in range(K, K_new):
                    idx = np.random.randint(0, K)
                    mu = np.hstack((mu, mu[:, idx : idx + 1]))
                    sigma = np.hstack((sigma, sigma[0:1, idx : idx + 1]))
                    mu[:, i_new : i_new + 1] += (
                        0.5 * sigma[0, i_new] * lambd * np.random.randn(D, 1)
                    )

                    if vp.optimize_sigma:
                        sigma[0, i_new] *= np.exp(0.2 * np.random.randn())

                    if vp.optimize_weights:
                        xi = 0.25 + 0.25 * np.random.rand()
                        w = np.hstack((w, xi * w[0:1, idx : idx + 1]))
                        w[0, idx] *= 1 - xi
        elif vbtype == 2:
            # Start from highest-posterior density training points
            if i == 0:
                add_jitter = False
            else:
                add_jitter = True
            if vp.optimize_lambd:
                lambd = np.reshape(np.std(X_star, axis=0, ddof=1), (-1, 1))
                lambd *= np.sqrt(D / np.sum(lambd**2))
            if vp.optimize_weights:
                w = np.ones((1, K_new)) / K_new
        elif vbtype == 3:
            # Start from random provided training points
            if vp.optimize_mu:
                order = np.random.permutation(N_star)
                idx_order = np.tile(
                    range(0, min(K_new, N_star)),
                    (math.ceil(K_new / N_star),),
                )
                mu = X_star[order[idx_order[0:K_new]], :].T
            else:
                mu = mu0.copy()

            if vp.optimize_sigma:
                if K > 1:
                    V = np.var(mu, axis=1, ddof=1)
                else:
                    V = np.var(X_star, axis=0, ddof=1)
                sigma = np.sqrt(np.mean(V) / K_new) * np.exp(
                    0.2 * np.random.randn(1, K_new)
                )

            if vp.optimize_lambd:
                lambd = np.reshape(np.std(X_star, axis=0, ddof=1), (-1, 1))
                lambd *= np.sqrt(D / np.sum(lambd**2))

            if vp.optimize_weights:
                w = np.ones((1, K_new)) / K_new
        else:
            raise ValueError(
                "Unknown type for initialization of variational posteriors."
            )

        if add_jitter:
            if vp.optimize_mu:
                # When reproducing MATLAB numbers we need to do Fortran order
                # here, adding .T works with square shape.
                mu += sigma * lambd * np.random.randn(mu.shape[0], mu.shape[1])
            if vp.optimize_sigma:
                sigma *= np.exp(0.2 * np.random.randn(1, K_new))
            if vp.optimize_lambd:
                lambd *= np.exp(0.2 * np.random.randn(D, 1))
            if vp.optimize_weights:
                w *= np.exp(0.2 * np.random.randn(1, K_new))
                w /= np.sum(w)

        new_vp = copy.deepcopy(vp)
        new_vp.K = K_new

        if vp.optimize_weights:
            new_vp.w = w
        else:
            new_vp.w = np.ones((1, K_new)) / K_new
        if vp.optimize_mu:
            new_vp.mu = mu
        else:
            new_vp.mu = mu0.copy()
        new_vp.sigma = sigma
        new_vp.lambd = lambd
        # TODO: just set to None?
        new_vp.eta = np.ones((1, K_new)) / K_new
        new_vp.bounds = None
        new_vp.stats = None
        vp0_list.append(new_vp)

    return np.array(vp0_list), type_vec


def _negelcbo(
    theta: np.ndarray,
    gp: gpr.GP,
    vp: VariationalPosterior,
    beta: float = 0.0,
    Ns: int = 0,
    compute_grad: bool = True,
    compute_var: int = None,
    theta_bnd: dict = None,
    _entropy_alpha: float = 0.0,
    separate_K: bool = False,
):
    """
    Negative evidence lower confidence bound objective.

    Parameters
    ==========
    theta : np.ndarray
        Vector of variational parameters at which to evaluate NELCBO.
        Note that these should be transformed parameters.
    gp : GP
        Gaussian process from optimization
    vp : VariationalPosterior
        Variational posterior for which to evaluate NELCBO.
    beta : float, defaults to 0.0
        Confidence weight.
    Ns : int, defaults to 0
        Number of samples for entropy.
    compute_grad : bool, defaults to True
        Whether to compute gradient.
    compute_var : bool, optional
        Whether to compute variance. If not given this is
        determined automatically.
    theta_bnd : dict, optional
        Soft bounds for theta.
    entropy_alpha : float, defaults to 0.0
        To be written by Luigi.
    separate_K : bool, defaults to False
        Whether to return expected log joint per component.

    Returns
    =======
    F : float
        Negative evidence lower confidence bound objective.
    dF : np.ndarray
        Gradient of NELCBO.
    G : object
        The expected variational log joint probability.
    H : float
        Entropy term.
    varF : float
        Variance of NELCBO.
    dH : np.ndarray
        Gradient of entropy term.
    varGss :
        To be written by Luigi.
    varG :
        Variance of the expected variational log joint
        probability.
    varH : float
        Variance of entropy term.
    I_sk : np.ndarray
        To be written by Luigi.
    J_sjk : np.ndarray
        To be written by Luigi.
    """
    if not np.isfinite(beta):
        beta = 0
    if compute_var is None:
        compute_var = beta != 0

    # if compute_grad and beta != 0 and compute_var != 2:
    #     raise NotImplementedError(
    #         "Computation of the gradient of ELBO with full variance not "
    #         "supported"
    #     )

    K = vp.K

    # Average over multiple GP hyperparameters if provided
    avg_flag = True
    # Variational parameters are transformed
    jacobian_flag = True

    # Reformat variational parameters from theta.
    vp.set_parameters(theta)

    if vp.optimize_weights:
        vp.eta = theta[-K:]
        vp.eta -= np.amax(vp.eta)
        vp.eta = np.reshape(vp.eta, (1, -1))
        # Doing the above is more numerically robust than
        # below, but it might cause slightly different results
        # to MATLAB in some cases.
        # vp.eta = np.reshape(theta[-K:], (1, -1))

    # Which gradients should be computed, if any?
    if compute_grad:
        grad_flags = (
            vp.optimize_mu,
            vp.optimize_sigma,
            vp.optimize_lambd,
            vp.optimize_weights,
        )
    else:
        grad_flags = (False, False, False, False)

    # Only weight optimization?
    # Not currently used, since it is only a speed optimization.
    # onlyweights_flag = (
    #     vp.optimize_weights
    #     and not vp.optimize_mu
    #     and not vp.optimize_sigma
    #     and not vp.optimize_lambd
    # )

    # Missing port: block below does not have branches for only weight
    #               optimization
    ts = time.time()
    if separate_K:
        if compute_grad:
            raise ValueError(
                "Computing the gradient of variational parameters and "
                "requesting per-component results at the same time."
            )

        if compute_var:
            G, _, varG, _, varGss, I_sk, J_sjk = _gplogjoint(
                vp,
                gp,
                grad_flags,
                avg_flag,
                jacobian_flag,
                compute_var,
                True,
            )
        else:
            G, dG, _, _, _, I_sk, _ = _gplogjoint(
                vp, gp, grad_flags, avg_flag, jacobian_flag, False, True
            )
            varG = varGss = 0
            J_sjk = None
    else:
        if compute_var:
            if compute_grad:
                G, dG, varG, dvarG, varGss = _gplogjoint(
                    vp,
                    gp,
                    grad_flags,
                    avg_flag,
                    jacobian_flag,
                    compute_var,
                )
            else:
                G, _, varG, _, varGss = _gplogjoint(
                    vp,
                    gp,
                    grad_flags,
                    avg_flag,
                    jacobian_flag,
                    compute_var,
                )
        else:
            G, dG, _, _, _ = _gplogjoint(
                vp, gp, grad_flags, avg_flag, jacobian_flag, False
            )
            varG = varGss = 0
    # print("G:", G)
    # print("Time for gplogjoint: ", time.time() - ts)
    # Entropy term
    ts = time.time()
    if Ns > 0:
        # Monte carlo approximation
        H, dH = entmc_vbmc(vp, Ns, grad_flags, jacobian_flag)
    else:
        # Deterministic approximation via lower bound on the entropy
        H, dH = entlb_vbmc(vp, grad_flags, jacobian_flag)
    # print("Time for entropy: ", time.time() - ts)
    # Negative ELBO and its gradient
    F = -G - H
    if compute_grad:
        dF = -dG - dH
    else:
        dF = None
        dH = None

    # For the moment use zero variance for entropy
    varH = 0
    if compute_var:
        varF = varG + varH
    else:
        varF = 0

    # Negative ELCBO (add confidence bound)
    if beta != 0:
        F += beta * np.sqrt(varF)
        if compute_grad:
            dF += 0.5 * beta * dvarG / np.sqrt(varF)

    # Additional loss for variational parameter bound violation (soft bounds)
    # and for weight size (if optimizing mixture weights)
    # Only done when optimizing the variational parameters, but not when
    # computing the EL(C)BO at each iteration.
    if theta_bnd is not None:
        if compute_grad:
            L, dL = _vp_bound_loss(
                vp, theta, theta_bnd, tol_con=theta_bnd["tol_con"]
            )
            dF += dL
        else:
            L = _vp_bound_loss(
                vp,
                theta,
                theta_bnd,
                tol_con=theta_bnd["tol_con"],
                compute_grad=False,
            )
        F += L

        #  Penalty to reduce weight size.
        if vp.optimize_weights:
            thresh = theta_bnd["weight_threshold"]
            L = (
                np.sum(vp.w * (vp.w < thresh) + thresh * (vp.w >= thresh))
                * theta_bnd["weight_penalty"]
            )

            F += L
            if compute_grad:
                w_grad = theta_bnd["weight_penalty"] * (
                    vp.w.flatten() < thresh
                )
                eta_sum = np.sum(np.exp(vp.eta))
                J_w = (
                    -np.exp(vp.eta).T * np.exp(vp.eta) / eta_sum**2
                ) + np.diag(np.exp(vp.eta.flatten()) / eta_sum)
                w_grad = np.dot(J_w, w_grad)
                dL = np.zeros(dF.shape)
                dL[-vp.K :] = w_grad
                dF += dL

    # Missing port: way to return stuff here is not that good,
    #               though it works currently.
    if separate_K:
        return F, dF, G, H, varF, dH, varGss, varG, varH, I_sk, J_sjk
    return F, dF, G, H, varF


@partial(
    jax.jit,
    static_argnames=["grad_var", "compute_var", "separate_K"],
)
def sgpr_gplogjoint_value(
    vp_params, gp_post, grad_var=False, compute_var=False, separate_K=False
):
    mu = vp_params["mu"]  # [D, K]
    sigma = vp_params["sigma"]
    lambd = vp_params["lambd"]
    w = vp_params["w"]

    mu = mu.T  # [K, D]
    sigma = jnp.sqrt((lambd**2 * sigma**2).T)  # [K, D]

    # gp_post = gp.post_cache
    res = gp_post.quad_mixture(mu, sigma, w, compute_var, separate_K)

    F, F_var = res[:2]

    I, J = None, None
    if separate_K:
        I, J = res[-2:]

    if grad_var:
        value = F_var
    else:
        value = F
    return value, (F_var, I, J)


@partial(
    jax.jit,
    static_argnames=["grad_var", "compute_var", "separate_K"],
)
@partial(jax.value_and_grad, has_aux=True)
def sgpr_gplogjoint(
    vp_params, gp_post, grad_var=False, compute_var=False, separate_K=False
):
    return sgpr_gplogjoint_value(
        vp_params, gp_post, grad_var=False, compute_var=False, separate_K=False
    )


@dispatch
def _gplogjoint(
    vp: VariationalPosterior,
    gp: jgp.HeteroskedasticSGPR,
    grad_flags: Union[bool, tuple],
    avg_flag: bool = True,
    jacobian_flag: bool = True,
    compute_var: bool = False,
    separate_K: bool = False,
):
    # In VBMC we are using unconstrained params for optimization
    assert jacobian_flag

    avg_flag = True  # No hyperparameter samples for SGPR

    if np.isscalar(grad_flags):
        if grad_flags:
            grad_flags = (True, True, True, True)
        else:
            grad_flags = (False, False, False, False)

    compute_vargrad = compute_var and np.any(grad_flags)
    assert not compute_vargrad  #  if true, need to uncomment stop_gradient in jaxgp's implementation of quad_mixture
    # if compute_vargrad and compute_var != 2:
    #     raise NotImplementedError(
    #         "Computation of gradient of log joint variance is currently "
    #         "available only for diagonal approximation of the variance."
    #     )

    D = vp.D
    K = vp.K
    N = gp.X.shape[0]
    mu = vp.mu.copy()  # [D, K]
    assert mu.shape == (D, K)
    sigma = vp.sigma.copy()  # [1, K]
    assert sigma.shape == (1, K)
    lambd = vp.lambd.copy().reshape(-1, 1)  # [D, 1]
    assert lambd.shape == (D, 1)

    w = vp.w.copy()[0, :]  # [K,]
    assert w.shape == (K,)
    Ns = 1
    if hasattr(vp, "delta") and vp.delta is not None and np.any(vp.delta > 0):
        # TODO: add smoothing by passing delta to posterior.quad_mixture if needed
        raise ValueError("Smoothing is not supported and tested yet for SGPR.")

    vp_params = {"mu": mu, "sigma": sigma, "lambd": lambd, "w": w}
    if np.any(grad_flags):
        (F, (varF, I, J)), dF = sgpr_gplogjoint(
            vp_params,
            gp.post_cache,
            grad_var=False,
            compute_var=compute_var,
            separate_K=separate_K,
        )
    else:
        F, (varF, I, J) = sgpr_gplogjoint_value(
            vp_params,
            gp.post_cache,
            grad_var=False,
            compute_var=compute_var,
            separate_K=separate_K,
        )
    if compute_vargrad:
        (varF, (_, I, J)), dvarF = sgpr_gplogjoint(
            vp_params,
            gp.post_cache,
            grad_var=True,
            compute_var=compute_var,
            separate_K=separate_K,
        )
    else:
        dvarF = None

    ## Convert to numpy for vbmc
    # Store contribution to the jog joint separately for each component?
    if separate_K:
        I_sk = np.reshape(I, (1, K))
        if compute_var:
            J_sjk = np.reshape(J, (1, K, K))

    F = np.array(F)[..., None]
    if compute_var:
        varF = np.array(varF)

    if grad_flags[0]:
        mu_grad = np.array(dF["mu"])  # [D, K]
        mu_grad = mu_grad[..., None]  # [D, K, 1]
    if grad_flags[1]:
        sigma_grad = np.array(dF["sigma"]).T  # [K, 1]
    if grad_flags[2]:
        lambd_grad = np.array(dF["lambd"])  # [D, 1]
    if grad_flags[3]:
        w_grad = np.array(dF["w"])  # [K]
        w_grad = w_grad[..., None]  # [K, 1]

    if compute_vargrad:
        mu_vargrad = np.array(dvarF["mu"])  # [D, K]
        mu_vargrad = mu_vargrad[..., None]  # [D, K, 1]
        sigma_vargrad = np.array(dvarF["sigma"]).T  # [K, 1]
        lambd_vargrad = np.array(dvarF["lambd"])  # [D, 1]
        w_vargrad = np.array(dvarF["w"])  # [K]
        w_vargrad = w_vargrad[..., None]  # [K, 1]

    if np.any(grad_flags):
        grad_list = []
        if grad_flags[0]:
            mu_grad = np.reshape(mu_grad, (D * K, Ns), order="F")
            grad_list.append(mu_grad)

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            sigma_grad *= np.reshape(sigma, (-1, 1))
            grad_list.append(sigma_grad)

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            lambd_grad *= lambd
            grad_list.append(lambd_grad)

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            eta_sum = np.sum(np.exp(vp.eta))
            J_w = (
                -np.exp(vp.eta).T * np.exp(vp.eta) / eta_sum**2
                + np.diag(np.exp(vp.eta.flatten())) / eta_sum
            )
            w_grad = np.dot(J_w, w_grad)
            grad_list.append(w_grad)

        dF = np.concatenate(grad_list, axis=0)
    else:
        dF = None

    if compute_vargrad:
        # TODO: compute vargrad is untested
        vargrad_list = []
        if grad_flags[0]:
            mu_vargrad = np.reshape(mu_vargrad, (D * K, Ns))
            vargrad_list.append(mu_vargrad)

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            sigma_vargrad *= np.reshape(sigma_vargrad, (-1, 1))
            vargrad_list.append(sigma_vargrad)

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            lambd_vargrad *= lambd
            vargrad_list.append(lambd_vargrad)

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            w_vargrad = np.dot(J_w, w_vargrad)
            vargrad_list.append(w_vargrad)

        dvarF = np.concatenate(grad_list, axis=0).squeeze()
    else:
        dvarF = None

    if Ns == 1:
        F = F[0]
        if np.any(grad_flags):
            dF = dF[:, 0]

    # Correct for numerical error
    if compute_var:
        varF = np.maximum(varF, np.spacing(1))
    else:
        varF = None

    varss = 0
    if separate_K:
        return F, dF, varF, dvarF, varss, I_sk, J_sjk
    return F, dF, varF, dvarF, varss


@dispatch
def _gplogjoint(
    vp: VariationalPosterior,
    gp: gpr.GP,
    grad_flags: Union[bool, tuple],
    avg_flag: bool = True,
    jacobian_flag: bool = True,
    compute_var: bool = False,
    separate_K: bool = False,
):
    """
    Expected variational log joint probability via GP approximation.

    Parameters
    ==========
    vp : VariationalPosterior
        Variational posterior.
    gp : GP
        Gaussian process from optimization.
    grad_flags : Union[bool, tuple]
        Flags on which gradients to compute. If a boolean then this
        sets all flags to the boolean value, and if a 4-tuple then
        each entry specifies which gradients to compute, in order
        mu, sigma, lambd, w.
    avg_flag : bool, defaults to True
        Whether to average over multiple GP hyperparameters if provided.
    jacobian_flag : bool, defaults to True
        Whether variational parameters are transformed.
    compute_var : bool, defaults to False
        Whether to compute variance.
    separate_K : bool, defaults to False
        Whether to return expected log joint per component.

    Returns
    =======
    F : object
        The expected variational log joint probability.
    dF : np.ndarray
        The gradient.
    varF : np.ndarray, optional
        The variance.
    dvarF : np.ndarray, optional
        The gradient of the variance.
    varss : float
        To be written by Luigi. TODO: ?
    I_sk : np.ndarray, optional
        To be written by Luigi.
    J_sjk : np.ndarray, optional
        To be written by Luigi.
    """
    if np.isscalar(grad_flags):
        if grad_flags:
            grad_flags = (True, True, True, True)
        else:
            grad_flags = (False, False, False, False)

    compute_vargrad = compute_var and np.any(grad_flags)
    if compute_vargrad and compute_var != 2:
        raise NotImplementedError(
            "Computation of gradient of log joint variance is currently "
            "available only for diagonal approximation of the variance."
        )

    D = vp.D
    K = vp.K
    N = gp.X.shape[0]
    mu = vp.mu.copy()
    sigma = vp.sigma.copy()
    lambd = vp.lambd.copy().reshape(-1, 1)

    w = vp.w.copy()[0, :]
    Ns = len(gp.posteriors)

    # TODO: once we get more mean function add a check here
    # if all(gp.meanfun ~= [0,1,4,6,8,10,12,14,16,18,20,22])
    #     error('gplogjoint:UnsupportedMeanFun', ...
    #     'Log joint computation currently only supports zero, constant,
    #     negative quadratic, negative quadratic (fixed/isotropic),
    #     negative quadratic-only, or squared exponential mean functions.');
    # end

    # Which mean function is being used?
    quadratic_meanfun = isinstance(
        gp.mean, gpr.mean_functions.NegativeQuadratic
    )

    F = np.zeros((Ns,))
    # Check which gradients are computed
    if grad_flags[0]:
        mu_grad = np.zeros((D, K, Ns))
    if grad_flags[1]:
        sigma_grad = np.zeros((K, Ns))
    if grad_flags[2]:
        lambd_grad = np.zeros((D, Ns))
    if grad_flags[3]:
        w_grad = np.zeros((K, Ns))
    if compute_var:
        varF = np.zeros((Ns,))
    # Compute gradient of variance?
    if compute_vargrad:
        # TODO: compute vargrad is untested
        if grad_flags[0]:
            mu_vargrad = np.zeros((D, K, Ns))
        if grad_flags[1]:
            sigma_vargrad = np.zeros((K, Ns))
        if grad_flags[2]:
            lambd_vargrad = np.zeros((D, Ns))
        if grad_flags[3]:
            w_vargrad = np.zeros((K, Ns))

    # Store contribution to the jog joint separately for each component?
    if separate_K:
        I_sk = np.zeros((Ns, K))
        if compute_var:
            J_sjk = np.zeros((Ns, K, K))

    if vp.delta is None:
        delta = 0
    else:
        delta = vp.delta.copy().T

    Xt = np.zeros((D, N, K))
    for k in range(0, K):
        Xt[:, :, k] = np.reshape(mu[:, k], (-1, 1)) - gp.X.T

    # Number of GP hyperparameters
    cov_N = gp.covariance.hyperparameter_count(D)
    # mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    # Loop over hyperparameter samples.
    # Missing port: below loop does not have code related to mean functions
    #               we haven't implemented in gpyreg
    for s in range(0, Ns):
        hyp = gp.posteriors[s].hyp

        # Extract GP hyperparameters from hyperparameter array.
        ell = np.exp(hyp[0:D]).reshape(-1, 1)
        ln_sf2 = 2 * hyp[D]
        sum_lnell = np.sum(hyp[0:D])

        # GP mean function hyperparameters
        if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
            m0 = 0
        else:
            m0 = hyp[cov_N + noise_N]

        if quadratic_meanfun:
            xm = hyp[cov_N + noise_N + 1 : cov_N + noise_N + D + 1].reshape(
                -1, 1
            )
            omega = np.exp(hyp[cov_N + noise_N + D + 1 :]).reshape(-1, 1)

        # GP posterior parameters
        alpha = gp.posteriors[s].alpha
        L = gp.posteriors[s].L
        L_chol = gp.posteriors[s].L_chol
        sn2_eff = 1 / gp.posteriors[s].sW[0] ** 2

        for k in range(0, K):
            tau_k = np.sqrt(sigma[:, k] ** 2 * lambd**2 + ell**2 + delta**2)
            lnnf_k = (
                ln_sf2 + sum_lnell - np.sum(np.log(tau_k), axis=0)
            )  # Covariance normalization factor
            delta_k = Xt[:, :, k] / tau_k
            z_k = np.exp(lnnf_k - 0.5 * np.sum(delta_k**2, axis=0))
            I_k = np.dot(z_k, alpha) + m0

            if quadratic_meanfun:
                nu_k = -0.5 * np.sum(
                    1
                    / omega**2
                    * (
                        mu[:, k : k + 1] ** 2
                        + sigma[:, k] ** 2 * lambd**2
                        - 2 * mu[:, k : k + 1] * xm
                        + xm**2
                        + delta**2
                    ),
                    axis=0,
                )
                I_k += nu_k
            F[s] += w[k] * I_k

            if separate_K:
                I_sk[s, k] = I_k

            if grad_flags[0]:
                dz_dmu = -(delta_k / tau_k) * z_k
                mu_grad[:, k, s : s + 1] = w[k] * np.dot(dz_dmu, alpha)
                if quadratic_meanfun:
                    mu_grad[:, k, s : s + 1] -= (
                        w[k] / omega**2 * (mu[:, k : k + 1] - xm)
                    )

            if grad_flags[1]:
                dz_dsigma = (
                    np.sum((lambd / tau_k) ** 2 * (delta_k**2 - 1), axis=0)
                    * sigma[:, k]
                    * z_k
                )
                sigma_grad[k, s] = w[k] * np.dot(dz_dsigma, alpha)
                if quadratic_meanfun:
                    sigma_grad[k, s] -= (
                        w[k]
                        * sigma[:, k]
                        * np.sum(1 / omega**2 * lambd**2, axis=0)
                    )

            if grad_flags[2]:
                dz_dlambd = (
                    (sigma[:, k] / tau_k) ** 2
                    * (delta_k**2 - 1)
                    * (lambd * z_k)
                )
                lambd_grad[:, s : s + 1] += w[k] * np.dot(dz_dlambd, alpha)
                if quadratic_meanfun:
                    lambd_grad[:, s : s + 1] -= (
                        w[k] * sigma[:, k] ** 2 / omega**2 * lambd
                    )

            if grad_flags[3]:
                w_grad[k, s] = I_k

            if compute_var == 2:
                # Missing port: compute_var == 2 skipped since it is not used
                assert False
            elif compute_var:
                for j in range(0, k + 1):
                    tau_j = np.sqrt(
                        sigma[:, j] ** 2 * lambd**2 + ell**2 + delta**2
                    )
                    lnnf_j = ln_sf2 + sum_lnell - np.sum(np.log(tau_j), axis=0)
                    delta_j = (mu[:, j : j + 1] - gp.X.T) / tau_j
                    z_j = np.exp(lnnf_j - 0.5 * np.sum(delta_j**2, axis=0))

                    tau_jk = np.sqrt(
                        (sigma[:, j] ** 2 + sigma[:, k] ** 2) * lambd**2
                        + ell**2
                        + 2 * delta**2
                    )
                    lnnf_jk = ln_sf2 + sum_lnell - np.sum(np.log(tau_jk))
                    delta_jk = (mu[:, j : j + 1] - mu[:, k : k + 1]) / tau_jk

                    J_jk = np.exp(lnnf_jk - 0.5 * np.sum(delta_jk**2, axis=0))
                    if L_chol:
                        J_jk -= np.dot(
                            z_k,
                            sp.linalg.solve_triangular(
                                L,
                                sp.linalg.solve_triangular(
                                    L, z_j, trans=1, check_finite=False
                                ),
                                trans=0,
                                check_finite=False,
                            )
                            / sn2_eff,
                        )
                    else:
                        J_jk += np.dot(z_k, np.dot(L, z_j.T))

                    # Off-diagonal elements are symmetric (count twice)
                    if j == k:
                        varF[s] += w[k] ** 2 * np.maximum(np.spacing(1), J_jk)
                        if separate_K:
                            J_sjk[s, k, k] = J_jk
                    else:
                        varF[s] += 2 * w[j] * w[k] * J_jk
                        if separate_K:
                            J_sjk[s, j, k] = J_jk
                            J_sjk[s, k, j] = J_jk

    # Correct for numerical error
    if compute_var:
        varF = np.maximum(varF, np.spacing(1))
    else:
        varF = None

    if np.any(grad_flags):
        grad_list = []
        if grad_flags[0]:
            mu_grad = np.reshape(mu_grad, (D * K, Ns), order="F")
            grad_list.append(mu_grad)

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            sigma_grad *= np.reshape(sigma, (-1, 1))
            grad_list.append(sigma_grad)

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            lambd_grad *= lambd
            grad_list.append(lambd_grad)

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            eta_sum = np.sum(np.exp(vp.eta))
            J_w = (
                -np.exp(vp.eta).T * np.exp(vp.eta) / eta_sum**2
                + np.diag(np.exp(vp.eta.flatten())) / eta_sum
            )
            w_grad = np.dot(J_w, w_grad)
            grad_list.append(w_grad)

        dF = np.concatenate(grad_list, axis=0)
    else:
        dF = None

    if compute_vargrad:
        # TODO: compute vargrad is untested
        vargrad_list = []
        if grad_flags[0]:
            mu_vargrad = np.reshape(mu_vargrad, (D * K, Ns))
            vargrad_list.append(mu_vargrad)

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            sigma_vargrad *= np.reshape(sigma_vargrad, (-1, 1))
            vargrad_list.append(sigma_vargrad)

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            lambd_vargrad *= lambd
            vargrad_list.append(lambd_vargrad)

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            w_vargrad = np.dot(J_w, w_vargrad)
            vargrad_list.append(w_vargrad)

        dvarF = np.concatenate(grad_list, axis=0)
    else:
        dvarF = None

    # Average multiple hyperparameter samples
    varss = 0
    if Ns > 1 and avg_flag:
        F_bar = np.sum(F) / Ns
        if compute_var:
            # Estimated variance of the samples
            varFss = np.sum((F - F_bar) ** 2) / (Ns - 1)
            # Variability due to sampling
            varss = varFss + np.std(varF, ddof=1)
            varF = np.sum(varF, axis=0) / Ns + varFss
        if compute_vargrad:
            # TODO: compute vargrad is untested
            dvv = 2 * np.sum(F * dF, axis=1) / (Ns - 1) - 2 * F_bar * np.sum(
                dF, axis=1
            ) / (Ns - 1)
            dvarF = np.sum(dvarF, axis=1) / Ns + dvv
        F = F_bar
        if np.any(grad_flags):
            dF = np.sum(dF, axis=1) / Ns

    # In case of separate samples but only one sample simplify
    # expressions slightly.
    # TODO: what other parts need to be fixed like this?
    if Ns == 1:
        F = F[0]
        if np.any(grad_flags):
            dF = dF[:, 0]

    if separate_K:
        return F, dF, varF, dvarF, varss, I_sk, J_sjk
    return F, dF, varF, dvarF, varss
