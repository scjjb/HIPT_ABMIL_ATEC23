from typing import Dict, Optional
from collections import defaultdict, deque
import numpy as np

from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper


@PublicAPI
class TrialPlateauStopper(Stopper):
    """Early stop single trials when they reached a plateau.

    This is an adjusted version of the default RayTune TrialPlateauStopper
    which also ensures loss is not increasing
    """
    def __init__(
        self,
        metric: str,
        std: float = 0.01,
        num_results: int = 4,
        grace_period: int = 4,
        metric_threshold: Optional[float] = None,
        mode: Optional[str] = None,
    ):


        self._metric = metric
        self._mode = mode

        self._std = std
        self._mean = float("inf")
        self._num_results = num_results
        self._grace_period = grace_period
        self._metric_threshold = metric_threshold

        if self._metric_threshold:
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"When specifying a `metric_threshold`, the `mode` "
                    f"argument has to be one of [min, max]. "
                    f"Got: {mode}"
                )

        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)
        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and metric_result < self._metric_threshold:
                return False

        # Calculate stdev, mean of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
        except Exception:
            current_std = float("inf")

        try:
            current_mean = np.mean(self._trial_results[trial_id])
        except Exception:
            current_mean = float("inf")

        # If stdev is lower than threshold or mean is increasing, stop early.
        if (current_std < self._std) or (current_mean>self._mean):
            return True
        else:
            self._mean = current_mean
            return False

    def stop_all(self):
        return False

