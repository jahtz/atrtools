# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


# Forked from https://github.com/bertsky/ocrd_cis/blob/5cf22f5baa093ffaf0049e3c9756094116273598/
class NumpyUtils:
    @staticmethod
    def norm_max(a) -> np.ndarray:
        """
        Normalize the array by its maximum value.
        Args:
            a: Input array.
        Returns:
            Normalized array.
        """
        norm = np.amax(a)  # or nanmax?
        if norm:
            return a / norm
        else:
            return a  # or 0?
    
    @staticmethod
    def odd(num: float) -> int:
        """
        Return the next odd integer for the given number.
        Args:
            num: Input number.
        Returns:
            Next odd integer.
        """
        return int(num) + int((num + 1) % 2)
    
    @staticmethod
    def find(condition):
        """
        Return the indices where ravel(condition) is true.
        Args:
            condition: Input condition array.
        Returns:
            Indices where the condition is True.
        """
        (res,) = np.nonzero(np.ravel(condition))
        return res
    