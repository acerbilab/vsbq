"""Some utils (formatting, Timer, handle_0D_1D_input) taken from pyvbmc.
"""

import logging
import time
from functools import wraps
from textwrap import indent

import numpy as np


class Timer:
    """
    A small Timer class.
    """

    def __init__(self):
        """
        Initialize a new timer.
        """
        self._start_times = {}
        self._durations = {}

    def start_timer(self, name: str):
        """
        Start the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            self._start_times[name] = time.time()

    def stop_timer(self, name: str):
        """
        Stop the specified timer.

        If this is the first call to ``stop_timer(name)`` for string ``name``,
        record the duration. Otherwise, add to the duration.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """

        if name in self._start_times:
            end_time = time.time()
            if self._durations.get(name) is not None:
                self._durations[name] += end_time - self._start_times[name]
            else:
                self._durations[name] = end_time - self._start_times[name]
            self._start_times.pop(name)
        else:
            logging.getLogger("timer").warning(
                "Timer start not found for key '%s'.", name
            )

    def get_duration(self, name: str):
        """
        Return the duration of the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer which time should be returned.

        Returns
        -------
        duration : float
            The duration of the timer or None when the timer is not existing.
        """
        time_ = self._durations.get(name)
        if time_ is None:
            logging.getLogger("timer").warning(
                "Timer duration not found for key '%s'.", name
            )
        return time_

    def reset(self):
        """
        Reset the timer be emptying the durations and start times.
        """
        self._durations = {}
        self._start_times = {}

    def __repr__(self, arr_size_thresh=10, expand=True):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        return full_repr(
            self, "Timer", expand=expand, arr_size_thresh=arr_size_thresh
        )


def summarize(
    obj,
    arr_size_thresh=10,
    precision=4,
):
    """Construct a string summary of an object.

    If the object is an array and the array is small enough, print the full
    array and type. Otherwise, just print the size and type. If the passed
    object is not an array, get a short representation with get_repr().

    Parameters
    ----------
    obj : any
        The object to summarize.
    arr_size_thresh : float, optional
        If ``obj`` is an array whose product of dimensions is less than
        ``arr_size_thresh``, print the full array. Otherwise print only the
        shape. Default `10`.
    precision : int, optional
        The number of decimal points to use when printing float values within
        arrays. Default `4`.

    Returns
    -------
    string : str
        The string summarizing the object/array.
    """
    if not isinstance(obj, np.ndarray):
        # Stringify anything that's not an array.
        string = get_repr(obj)
    elif np.prod(obj.shape) < arr_size_thresh:
        # Array is small: print the full (but abbreviated precision) array on
        # one line.
        array_string = np.array2string(
            obj, precision=precision, suppress_small=True, separator=", "
        )
        if "\n" in array_string:  # Print multi-line arrays on new line
            array_string = indent("\n" + array_string, "    ")
        string = f"{array_string} : {type(obj).__name__}"
    else:
        # Array is large: print only the shape of the array.
        string = f"{obj.shape} {type(obj).__name__}"

    return string


def format_dict(d, **kwargs):
    """Pretty-print a dictionary.

    Summarize possible array values with ``summarize()``.

    Parameters
    ----------
    d : dict or None
        The dictionary to format.
    kwargs : dict, optional
        The keywords for summarizing child objects, forwarded to
        ``summarize()``.

    Returns
    -------
    string : str
        The formatted dictionary, as a string.
    """
    if d is None:
        string = "None"
    else:
        string = "{\n"
        body = ""
        for key, val in d.items():
            if isinstance(key, str):  # Enclose string keys in quotes
                body += repr(key)
            else:
                body += str(key)
            if type(val) == dict:  # Recursively format nested dictionaries
                val_string = f": {format_dict(val, **kwargs)}"
            else:  # Format possible array values
                val_string = f": {summarize(val, **kwargs)}"
            body += val_string + ",\n"
        body = indent(body, "    ")
        string += body
        string += "}"

    return string


def get_repr(obj, expand=False, full=False, **kwargs):
    """Get a (possibly abbreviated) string representation of an object.

    Parameters
    ----------
    obj : any
        the object to represent.
    expand : bool, optional
        If ``expand`` is `False`, then describe the object's complex child
        attributes by their name and memory location. Otherwise, recursively
        expand the child attributes into their own representations, passing
        along the appropriate keyword arguments. Default `False`.
    full : bool, optional
        If ``full`` is `False`, print only the relevant object attributes.
        Otherwise print all attributes. If ``expand`` is also `True`, then the
        children will follow this behavior. Default `False`.

    Returns
    -------
    string : str
        The string representation of ``obj``.
    """
    if expand:  # Expand child elements
        if type(obj) == dict:
            return format_dict(obj, **kwargs)
        elif type(obj) == np.ndarray:
            return summarize(obj, **kwargs)
        else:
            try:
                return obj.__repr__(expand=True, full=full)
            except TypeError:  # keyword error
                return repr(obj)
    else:  # Don't expand, only return short representation
        if hasattr(obj, "_short_repr"):  # Custom short representation
            return obj._short_repr()
        elif type(obj) == dict:  # Just return type and memory location
            return object.__repr__(obj)
        elif type(obj) == np.ndarray:  # Summarize numpy arrays
            return summarize(obj, **kwargs)
        else:
            return repr(obj)  # If all else fails, return usual __repr__()


def full_repr(obj, title, order=None, exclude=None, **kwargs):
    """Get a complete string representation of an object.

    Prints the names and a summary of their values. Attributes listed in
    ``order`` are printed first, in the order they appear, then all remaining
    attributes are printed (in sorted order, if possible).

    Parameters
    ----------
    obj : any
        The object to represent.
    title : string
        The title to print (e.g. class name).
    order : list, optional
        The order of selected attributes, to print first. Default ``[]``.
    kwargs : dict, optional
        The keyword arguments for printing the objects child attributes,
        forwarded to ``get_repr()``.
    """
    body = []
    if order is None:
        order = []
    if exclude is None:
        exclude = []
    # Print select attributes first
    for key in order:
        if "." in key:  # Handle request to print e.g. 'vp.K'
            sub_obj = obj
            for subkey in key.split("."):
                sub_obj = getattr(sub_obj, subkey, None)
        else:  # Just get the attribute
            sub_obj = getattr(obj, key, None)
        body.append(f"self.{key} = {get_repr(sub_obj, **kwargs)}")

    # Print all remaining attributes
    try:
        for key, val in sorted(obj.__dict__.items()):
            if key not in order and key not in exclude:
                body.append(f"self.{key} = {get_repr(val, **kwargs)}")
    except TypeError:  # Keys cannot be sorted
        for key, val in obj.__dict__.items():
            if key not in order and key not in exclude:
                body.append(f"self.{key} = {get_repr(val, **kwargs)}")

    body = ",\n".join(body)
    return title + ":\n" + indent(body, "    ")


def handle_0D_1D_input(
    patched_kwargs: list, patched_argpos: list, return_scalar=False
):
    """
    A decorator that handles 0D, 1D inputs and transforms them to 2D.

    Parameters
    ----------
    kwarg : list of str
        The names of the keyword arguments that should be handeled.
    argpos : list of int
        The positions of the arguments that should be handeled.
    return_scalar : bool, optional
        If the input is 1D the function should return a scalar,
        by default False.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            for idx, patched_kwarg in enumerate(patched_kwargs):
                if patched_kwarg in kwargs:
                    # for keyword arguments
                    input_dims = np.ndim(kwargs.get(patched_kwarg))
                    kwargs[patched_kwarg] = np.atleast_2d(
                        kwargs.get(patched_kwarg)
                    )

                elif len(args) > patched_argpos[idx]:
                    # for positional arguments
                    arg_list = list(args)
                    input_dims = np.ndim(args[patched_argpos[idx]])
                    arg_list[patched_argpos[idx]] = np.atleast_2d(
                        args[patched_argpos[idx]]
                    )
                    args = tuple(arg_list)

            res = function(self, *args, **kwargs)

            # return value 1D or scalar when boolean set
            if input_dims == 1:
                # handle functions with multiple return values
                if type(res) is tuple:
                    returnvalues = list(res)
                    returnvalues = [o.ravel() for o in returnvalues]
                    if return_scalar:
                        returnvalues = [o[0] for o in returnvalues]
                    return tuple(returnvalues)

                elif return_scalar and np.ndim(res) != 0:
                    return res.ravel()[0]
                elif np.ndim(res) != 0:
                    return res.ravel()

            return res

        return wrapper

    return decorator
