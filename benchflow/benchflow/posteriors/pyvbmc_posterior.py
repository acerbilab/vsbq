import numpy as np
from tqdm import tqdm

from benchflow.posteriors.posterior import Posterior, is_json_dumpable_dict
from benchflow.utilities.cfg import cfg_to_seed, combine_seeds


class PyVBMCPosterior(Posterior):
    """Class for accessing inferred ``pyvbmc`` posteriors.

    Attributes
    ----------
    vbmc : pyvbmc.vbmc.VBMC
        The ``VBMC`` object used to execute the inference.
    final_vp : pyvbmc.variational_posterior.VariationalPosterior
        The final (best) ``VariationalPosterior`` computed by VBMC.
    final_elbo : float
        An estimate of the final ELBO for the returned ``vp``.
    final_elbo_sd : float
        The standard deviation of the estimate of the ELBO. Note that this
        standard deviation is *not* representative of the error between the
        ``elbo`` and the true log marginal likelihood.
    final_success_flag : bool
        ``final_success_flag`` is ``True`` if the inference reached stability within
        the provided budget of function evaluations, suggesting convergence.
        If ``False``, the returned solution has not stabilized and should
        not be trusted.
    results : dict
        A result dictionary that will be save to results.json file.
    additional_result_dict : dict
        A dictionary with additional information about the VBMC run.
    history : [Posterior]
        A list of ``PyVBMCPosterior`` objects for each iteration of the
        algorithm.
    metrics : dict
        A dictionary containing the computed metrics. Keys are metrics names
        (e.g.  "c2st") and values are lists of computed metrics (one for each
        ``Posterior`` object in ``self.history``, and one for the final
        ``Posterior``)
    cfg : omegaconf.DictConfig
        The original ``hydra`` config describing the run.
    task : benchflow.task.Task
        The target ``Task`` for inference.
    """

    def __init__(
        self,
        cfg,
        task,
        final_vp,
        final_elbo,
        final_elbo_sd,
        final_success_flag=None,
        result_json=None,
        additional_result_dict=None,
        vbmc=None,
        iteration=None,
    ):
        """Initialize a ``PyVBMCPosterior``.

        Designed to accept the unpacked results of ``vbmc.optimize()`` (and
        optionally the ``vbmc`` object itself) as final arguments:
        .. highlight:: python
        .. code-block:: python

            result = vbmc.optimize()
            posterior = PyVBMCPosterior(cfg, task, \*result, vbmc=vbmc)

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            The original ``hydra`` config describing the run.
        task : benchflow.task.Task
            The target ``Task`` for inference.
        final_vp : pyvbmc.variational_posterior.VariationalPosterior
            The final (best) ``VariationalPosterior`` computed by VBMC.
        final_elbo : float
            An estimate of the final ELBO for the returned ``vp``.
        final_elbo_sd : float
            The standard deviation of the estimate of the ELBO. Note that this
            standard deviation is *not* representative of the error between the
            ``elbo`` and the true log marginal likelihood.
        final_success_flag : bool
            ``final_success_flag`` is ``True`` if the inference reached stability within
            the provided budget of function evaluations, suggesting convergence.
            If ``False``, the returned solution has not stabilized and should
            not be trusted.
        result_json: dict
            A result dictionary that will be save to results.json file.
        additional_result_dict : dict
            A dictionary with additional information about the VBMC run.
        history : [Posterior]
            A list of ``PyVBMCPosterior`` objects for each iteration of the
            algorithm.
        metrics : dict
            A dictionary containing the computed metrics. Keys are metrics names
            (e.g.  "c2st") and values are lists of computed metrics (one for each
            ``Posterior`` object in ``self.history``, and one for the final
            ``Posterior``)
        vbmc : pyvbmc.vbmc.VBMC
            The ``VBMC`` object used to execute the inference.
        """
        self.cfg = cfg
        self.task = task
        self.vbmc = vbmc
        self.iteration = iteration
        self.final_vp = final_vp
        self.final_elbo = final_elbo
        self.final_elbo_sd = final_elbo_sd
        self.final_success_flag = final_success_flag
        self.additional_result_dict = additional_result_dict
        if result_json is not None:
            is_json_dumpable_dict(result_json)
            self.results = result_json
        else:
            self.results = {}
        self.history = []
        self.metrics = {}

    def sample(self, n_samples=10000):
        """Draw samples from the final variational posterior.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : np.array
            The posterior samples, shape ``(n_samples, D)`` where ``D`` is the
            task dimension.
        """
        return self.final_vp.sample(n_samples)[0]

    def get_lml_estimate(self):
        """Get the estimated log marginal likelihood (LML)."""
        return self.final_elbo

    def get_lml_sd(self):
        """Get the standard deviation of the estimated LML."""
        return self.final_elbo_sd

    def post_process(self, boost=False):
        """Post-process intermediate results from previous iterations.

        For each intermediate ``PyVBMCPosterior`` in ``self.history``,
        completes the actions that ``pyvbmc`` would have taken if that
        iteration were the final one.
        """
        for i in tqdm(range(len(self.vbmc.iteration_history.get("vp")))):
            # np.random.seed(combine_seeds(i, cfg_to_seed(self.cfg)))
            # stable_flag = self.vbmc.iteration_history["stable"][i]
            # self.vbmc.iteration_history["stable"][i] = False
            vp, elbo, elbo_sd, idx_best = self.vbmc.determine_best_vp(i)
            if boost:
                vp2, elbo, elbo_sd, changed_flag = self.vbmc.finalboost(
                    vp, self.vbmc.iteration_history["gp"][idx_best]
                )
            else:
                vp2 = vp
            self.history.append(
                PyVBMCPosterior(
                    self.cfg,
                    self.task,
                    vp2,
                    elbo,
                    elbo_sd,
                    vbmc=self.vbmc,
                    iteration=i,
                )
            )
            # self.vbmc.iteration_history["stable"][i] = stable_flag

    def fun_evals(self):
        if self.iteration is None:  # Final iteration, get current func. count
            return self.vbmc.function_logger.func_count
        else:  # Get previous function count
            return self.vbmc.iteration_history["func_count"][self.iteration]

    def idx_best(self):
        if self.iteration is None:
            __, __, __, idx_best = self.vbmc.determine_best_vp()
            return idx_best
        else:
            __, __, __, idx_best = self.vbmc.determine_best_vp(self.iteration)
            return idx_best
