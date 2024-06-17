import numpy as np

from benchflow.function_logger import ParameterTransformer, log_function

from .task import Task


class ToyBimodal(Task):
    def __init__(
        self,
        cfg=None,
        transform_to_unconstrained_coordinates=False,
    ):
        """Initialize posteriordb model.

        Parameters
        ----------
        cfg : omegaconf.DictConfig, optional
            The section of the Hydra config specifying the task information.
        """
        self.D = 2

        # Initialize general attributes:
        super().__init__(cfg=cfg, D=self.D)

        # Load parameter upper/lower bounds
        plb = np.array([[-1.0, -1.0]])
        pub = np.array([[1.0, 1.0]])
        lb = np.full((1, self.D), -np.inf)
        ub = np.full((1, self.D), np.inf)

        # transform to unconstrained coordinates
        if transform_to_unconstrained_coordinates:
            self.transform = ParameterTransformer(self.D, lb, ub, plb, pub)
        else:
            self.transform = ParameterTransformer(self.D)  # Identity
        self.lb = self.transform(lb)
        self.ub = self.transform(ub)
        self.plb = self.transform(plb)
        self.pub = self.transform(pub)

        self.posterior_log_Z = 6.165761171767828
        self.mcmc_info.update(
            {  # Hints for MCMC reference sampling
                "multimodal": True,  # (Potentially) multimodal posterior
                "multiprocessing": True,  # Use multiprocessing, if possible
            }
        )

    @log_function
    def log_joint(self, theta: np.ndarray):
        # Transform unconstrained variables to original space
        x_orig = self.transform.inverse(theta)
        dy = self.transform.log_abs_det_jacobian(theta)
        x_orig = np.atleast_2d(x_orig)
        dy = np.atleast_1d(dy)

        rs = 0.1
        kappa = 8
        r = np.sqrt(np.sum(x_orig**2, 1))
        inds = ~(r == 0)
        log_joint = np.full(x_orig.shape[0], np.nan)

        if np.sum(inds) != 0:
            log_joint[inds] = (
                -0.5 * ((r[inds] - 1 / np.sqrt(2)) / rs) ** 2
                + np.log(
                    np.exp(kappa * x_orig[inds, 0] / r[inds]) / 3
                    + 2 * np.exp(-kappa * x_orig[inds, 0] / r[inds]) / 3
                )
                + dy[inds]
            )
        if np.sum(~inds) != 0:
            # Undefined at origin, set to a low value -25
            log_joint[~inds] = -25 + dy[~inds]

        return log_joint[:, None]

    def log_likelihood(self, theta):
        return self.log_joint(theta)

    def log_prior(self, theta):
        return 0


if __name__ == "__main__":
    from tqdm import tqdm

    cfg = {}
    task = ToyBimodal(cfg)
    samples = np.genfromtxt(
        "./cache/partial_mcmc/bimodal/init_samples_x_1000.csv"
    )
    lps = []
    for i in tqdm(range(samples.shape[0])):
        # for i in range(3):
        lp = task.log_joint(samples[i : i + 1])
        lps.append(lp)
    lps = np.concatenate(lps)
    np.savetxt(
        "./cache/partial_mcmc/bimodal/init_samples_y_1000.csv",
        lps,
    )
