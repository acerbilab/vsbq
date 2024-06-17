from typing import Optional, Union

import gpyreg as gpr
import jax
import jax.numpy as jnp
import jaxgp as jgp
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy as sp
import torch
import torch.nn as nn
from einops import rearrange
from plum import dispatch
from pyvbmc.entropy import entmc_vbmc
from pyvbmc.stats import get_hpd
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.iteration_history import IterationHistory
from pyvbmc.vbmc.minimize_adam import minimize_adam
from pyvbmc.vbmc.options import Options


def compute_elbo(target_model, vp, ns_reparamization=None, ns_entropy=1000):
    F, dF, varF, dvarF, varss, I_sk, J_sjk = _gplogjoint(
        vp,
        target_model,
        False,
        separate_K=True,
        ns_reparamization=ns_reparamization,
    )
    H, dH = entmc_vbmc(vp, ns_entropy)
    return F + H, F, H


def nn_logjoint_value(
    vp_params,
    model,
    compute_grad=True,
    grad_var=False,
    compute_var=False,
    separate_K=False,
    ns_reparamization=None,
):
    if ns_reparamization is None:
        ns_reparamization = 10
    # print(Ns)
    assert not compute_var, "Not implemented"
    # assert not separate_K, "Not implemented"
    assert not grad_var, "Not implemented"

    mu = vp_params["mu"]  # [D, K]
    sigma = vp_params["sigma"]
    lambd = vp_params["lambd"]
    w = vp_params["w"]

    mu = mu.T  # [K, D]
    D = mu.shape[1]
    device = model.device
    if compute_grad:
        require_grad = True
    else:
        require_grad = False
    mu = torch.tensor(
        mu, device=device, requires_grad=require_grad, dtype=torch.float
    )
    lambd = torch.tensor(
        lambd, device=device, requires_grad=require_grad, dtype=torch.float
    )
    sigma = torch.tensor(
        sigma, device=device, requires_grad=require_grad, dtype=torch.float
    )
    w = torch.tensor(
        w, device=device, requires_grad=require_grad, dtype=torch.float
    )
    K = mu.shape[0]
    sigma_gaussian = torch.sqrt((lambd**2 * sigma**2).T)  # [K, D]
    z = torch.randn((ns_reparamization, K, D)).to(device)
    # z = torch.Tensor([0.3, 0.8])[None, None, :].to(device)
    samples = z * sigma_gaussian + mu
    samples = rearrange(samples, "n k d -> (n k) d")
    log_density_preds = model.predict(samples)
    log_density_preds = rearrange(
        log_density_preds, "(n k)-> n k", n=ns_reparamization
    )
    I = log_density_preds.mean(0)
    F = (w * I).sum()
    dF = None
    if compute_grad:
        F.backward()
        dF = {
            "mu": mu.grad.cpu().numpy().T,  # [D, K]
            "lambd": lambd.grad.cpu().numpy(),
            "sigma": sigma.grad.cpu().numpy(),
            "w": w.grad.cpu().numpy(),
        }
    F_var, J = None, None
    value = F.item()
    return (value, (F_var, I, J)), dF


@dispatch
def _gplogjoint(
    vp: VariationalPosterior,
    gp: nn.Module,
    grad_flags: Union[bool, tuple],
    avg_flag: bool = True,
    jacobian_flag: bool = True,
    compute_var: bool = False,
    separate_K: bool = False,
    ns_reparamization: Optional[int] = None,
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
    assert not compute_vargrad
    assert not compute_var

    D = vp.D
    K = vp.K
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
    (F, (varF, I, J)), dF = nn_logjoint_value(
        vp_params,
        gp,
        compute_grad=np.any(grad_flags),
        grad_var=False,
        compute_var=compute_var,
        separate_K=separate_K,
        ns_reparamization=ns_reparamization,
    )

    if compute_vargrad:
        raise NotImplementedError("compute_vargrad is not supported for SGPR")
    else:
        dvarF = None

    ## Convert to numpy for vbmc
    # Store contribution to the jog joint separately for each component?
    if separate_K:
        I_sk = np.reshape(I.detach().cpu().numpy(), (1, K))
        J_sjk = None
        # if compute_var:
        #     J_sjk = np.reshape(J, (1, K, K))

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
