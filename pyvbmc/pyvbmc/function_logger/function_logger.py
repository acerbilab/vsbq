import copy

import numpy as np

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.timer import Timer


class FunctionLogger:
    """
    Class that evaluates a function and caches its values.

    Parameters
    ----------
    fun : callable
        The function to be logged.
        `fun` must take a vector input and return a scalar value and,
        optionally, the (estimated) SD of the returned value (if the
        function fun is stochastic).
    D : int
        The number of dimensions that the function takes as input.
    noise_flag : bool
        Whether the function fun is stochastic or not.
    uncertainty_handling_level : {0, 1, 2}
        The uncertainty handling level which can be one of
        (0: none; 1: unknown noise level; 2: user-provided noise).
    cache_size : int, optional
        The initial size of caching table (default 500).
    parameter_transformer : ParameterTransformer, optional
        A ParameterTransformer is required to transform the parameters
        between constrained and unconstrained space, by default None.
    """

    def __init__(
        self,
        fun,
        D: int,
        noise_flag: bool,
        uncertainty_handling_level: int,
        cache_size: int = 500,
        parameter_transformer: ParameterTransformer = None,
    ):
        self.fun = fun
        self.D: int = D
        self.noise_flag: bool = noise_flag
        self.uncertainty_handling_level: int = uncertainty_handling_level
        self.transform_parameters = parameter_transformer is not None
        self.parameter_transformer = parameter_transformer

        self.func_count: int = 0
        self.cache_count: int = 0
        self.X_orig = np.full([cache_size, self.D], np.nan)
        self.y_orig = np.full([cache_size, 1], np.nan)
        self.X = np.full([cache_size, self.D], np.nan)
        self.y = np.full([cache_size, 1], np.nan)
        self.ymax = np.nan
        self.nevals = np.full([cache_size, 1], 0)

        if self.noise_flag:
            self.S = np.full([cache_size, 1], np.nan)

        self.Xn: int = -1  # Last filled entry
        # Use 1D array since this is a boolean mask.
        self.X_flag = np.full((cache_size,), False, dtype=bool)
        self.y_max = float("-Inf")
        self.fun_evaltime = np.full([cache_size, 1], np.nan)
        self.total_fun_evaltime = 0

    def __call__(self, x: np.ndarray):
        """
        Evaluates the function FUN at x and caches values.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function will be evaluated. The shape of x
            should be (1, D) or (D,).

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """

        timer = Timer()
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        try:
            timer.start_timer("funtime")
            if self.noise_flag and self.uncertainty_handling_level == 2:
                fval_orig, fsd = self.fun(x_orig)
            else:
                fval_orig = self.fun(x_orig)
                if self.noise_flag:
                    raise NotImplementedError
                else:
                    fsd = None
            if isinstance(fval_orig, np.ndarray):
                # fval_orig can only be an array with size 1
                fval_orig = fval_orig.item()
            timer.stop_timer("funtime")

        except Exception as err:
            err.args += (
                "FunctionLogger:FuncError "
                + "Error in executing the logged function"
                + "with input: "
                + str(x_orig),
            )
            raise

        # if fval is an array with only one element, extract that element
        if not np.isscalar(fval_orig) and np.size(fval_orig) == 1:
            fval_orig = np.array(fval_orig).flat[0]

        # Check function value
        if np.any(
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd)
            or not np.isfinite(fsd)
            or not np.isreal(fsd)
            or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD: {})"""
            raise ValueError(error_message.format(str(fsd)))

        # record timer stats
        funtime = timer.get_duration("funtime")

        self.func_count += 1
        fval, idx = self._record(x_orig, x, fval_orig, fsd, funtime)

        # optimstate.N = self.Xn
        # optimstate.Neff = np.sum(self.nevals[self.X_flag])
        # optimState.totalfunevaltime = optimState.totalfunevaltime + t;
        return fval, fsd, idx

    def add(
        self,
        x: np.ndarray,
        fval_orig: float,
        fsd: float = None,
        fun_evaltime=np.nan,
    ):
        """
        Add an previously evaluated function sample to the function cache.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function has been evaluated. The shape of x
            should be (1, D) or (D,).
        fval_orig : float
            The result of the evaluation of the function.
        fsd : float, optional
            The (estimated) SD of the returned value (if heteroskedastic noise
            handling is on) of the evaluation of the function, by default None.
        fun_evaltime : float
            The duration of the time it took to evaluate the function,
            by default np.nan.

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        if self.noise_flag:
            assert fsd is not None
        else:
            fsd = None

        # Check function value
        if (
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd)
            or not np.isfinite(fsd)
            or not np.isreal(fsd)
            or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD:{})"""
            raise ValueError(error_message.format(str(fsd)))

        self.cache_count += 1
        fval, idx = self._record(x_orig, x, fval_orig, fsd, fun_evaltime)
        return fval, fsd, idx

    def initialize(self, X_orig, X, y_orig, y, S=None):
        """
        Initialize the function logger with a set of function evaluations.

        Parameters
        ----------
        X_orig : np.ndarray
            The points at which the function has been evaluated
            (in original space).
        X : np.ndarray
            The points at which the function has been evaluated
            (in transformed space).
        y_orig : np.ndarray
            The results of the evaluation of the function.
        y : np.ndarray
            The results of the evaluation of the function.
        S : np.ndarray, optional
            The (estimated) SD of the returned value (if heteroskedastic noise
            handling is on) of the evaluation of the function, by default None.
        """
        assert X_orig.shape == X.shape
        assert X_orig.shape[0] == y_orig.size
        assert y_orig.shape == y.shape
        y_orig = y_orig.reshape(-1, 1)
        y = y.reshape(-1, 1)

        self.X_orig = X_orig
        self.X = X
        self.y_orig = y_orig
        self.y = y
        self.Xn = X_orig.shape[0] - 1
        self.X_flag = np.full((X_orig.shape[0],), True, dtype=bool)
        self.nevals = np.full([X_orig.shape[0], 1], 1)
        self.fun_evaltime = np.full([X_orig.shape[0], 1], np.nan)
        if self.noise_flag:
            assert X_orig.shape[0] == S.size
            S = S.reshape(-1, 1)
            self.S = S

    def finalize(self):
        """
        Remove unused caching entries.
        """
        self.X_orig = self.X_orig[: self.Xn + 1]
        self.y_orig = self.y_orig[: self.Xn + 1]

        # in the original matlab version X and Y get deleted
        self.X = self.X[: self.Xn + 1]
        self.y = self.y[: self.Xn + 1]

        if self.noise_flag:
            self.S = self.S[: self.Xn + 1]
        self.X_flag = self.X_flag[: self.Xn + 1]
        self.fun_evaltime = self.fun_evaltime[: self.Xn + 1]

    def _expand_arrays(self, resize_amount: int = None):
        """
        A private function to extend the rows of the object attribute arrays.

        Parameters
        ----------
        resize_amount : int, optional
            The number of additional rows, by default expand current table
            by 50%.
        """

        if resize_amount is None:
            resize_amount = int(np.max((np.ceil(self.Xn / 2), 1)))

        self.X_orig = np.append(
            self.X_orig, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y_orig = np.append(
            self.y_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.X = np.append(
            self.X, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y = np.append(self.y, np.full([resize_amount, 1], np.nan), axis=0)

        if self.noise_flag:
            self.S = np.append(
                self.S, np.full([resize_amount, 1], np.nan), axis=0
            )
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount,), False, dtype=bool)
        )
        self.fun_evaltime = np.append(
            self.fun_evaltime, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.nevals = np.append(
            self.nevals, np.full([resize_amount, 1], 0), axis=0
        )

    def _record(
        self,
        x_orig: float,
        x: float,
        fval_orig: float,
        fsd: float,
        fun_evaltime: float,
    ):
        """
        A private method to save function values to class attributes.

        Parameters
        ----------
        x_orig : float
            The point at which the function has been evaluated
            (in original space).
        x : float
            The point at which the function has been evaluated
            (in transformed space).
        fval_orig : float
            The result of the evaluation.
        fsd : float
            The (estimated) SD of the returned value
            (if heteroskedastic noise handling is on).
        fun_evaltime : float
            The duration of the time it took to evaluate the function.

        Returns
        -------
        fval : float
            The result of the evaluation.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if there is more than one match for a duplicate entry.
        """
        duplicate_flag = self.X == x
        if np.any(duplicate_flag.all(axis=1)):
            if np.sum(duplicate_flag.all(axis=1)) > 1:
                raise ValueError("More than one match for duplicate entry.")
            idx = np.argwhere(duplicate_flag)[0, 0]
            N = self.nevals[idx]
            if fsd is not None:
                tau_n = 1 / self.S[idx] ** 2
                tau_1 = 1 / fsd**2
                self.y_orig[idx] = (
                    tau_n * self.y_orig[idx] + tau_1 * fval_orig
                ) / (tau_n + tau_1)
                self.S[idx] = 1 / np.sqrt(tau_n + tau_1)
            else:
                self.y_orig[idx] = (N * self.y_orig[idx] + fval_orig) / (N + 1)

            fval = self.y_orig[
                idx
            ].item()  # item() to convert to scalar, or else it's an array and may be wrongly modified in place.
            if self.transform_parameters:
                fval += self.parameter_transformer.log_abs_det_jacobian(x)
            self.y[idx] = fval
            self.fun_evaltime[idx] = (
                N * self.fun_evaltime[idx] + fun_evaltime
            ) / (N + 1)
            self.nevals[idx] += 1
            return fval, idx
        else:
            self.Xn += 1
            if self.Xn > self.X_orig.shape[0] - 1:
                self._expand_arrays()

            # record function time
            if not np.isnan(fun_evaltime):
                self.fun_evaltime[self.Xn] = fun_evaltime
                self.total_fun_evaltime += fun_evaltime

            self.X_orig[self.Xn] = x_orig
            self.X[self.Xn] = x
            self.y_orig[self.Xn] = fval_orig
            fval = fval_orig
            if self.transform_parameters:
                fval += self.parameter_transformer.log_abs_det_jacobian(
                    np.reshape(x, (1, x.shape[0]))
                )[0]
            self.y[self.Xn] = fval
            if fsd is not None:
                self.S[self.Xn] = fsd
            self.X_flag[self.Xn] = True
            self.nevals[self.Xn] += 1
            self.ymax = np.nanmax(self.y[self.X_flag])
            return fval, self.Xn

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Avoid infinite recursion in deepcopy
        memo[id(self)] = result
        # Copy class properties:
        for k, v in self.__dict__.items():
            if k == "fun":  # Avoid deepcopy of log-joint function
                # (interferes with benchmark logging)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def delete(self, count):
        assert count >= 0
        assert self.Xn + 1 > count
        self.Xn -= count
        self.X_flag[self.Xn + 1 :] = False
