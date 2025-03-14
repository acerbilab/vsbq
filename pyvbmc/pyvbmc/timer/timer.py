import time


class Timer:
    """
    A small Timer class used in the context of VBMC.
    """

    def __init__(self):
        """
        Initialize a new timer.
        """
        self._start_times = dict()
        self._durations = dict()

    def start_timer(self, name: str):
        """
        Start the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            if name in self._durations:
                self._durations.pop(name)
            self._start_times[name] = time.time()

    def stop_timer(self, name: str):
        """
        Stop the specified timer

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            raise ValueError(
                f"timer {name} was not started or has already been stopped."
            )

        end_time = time.time()
        self._durations[name] = end_time - self._start_times[name]
        self._start_times.pop(name)

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
        return self._durations.get(name)
