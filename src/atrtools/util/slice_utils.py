# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


# Forked from https://github.com/bertsky/ocrd_cis/blob/5cf22f5baa093ffaf0049e3c9756094116273598/
class slu:
    """
    Utilities for tuples of slices, treating them like rectangles
    """
    @staticmethod
    def dim0(s: tuple[slice, ...]) -> int:
        """Dimension of the slice list for dimension 0."""
        return s[0].stop - s[0].start

    @staticmethod
    def area(a: tuple[slice, ...]) -> int:
        """Return the area of the slice list (ignores anything past a[:2])."""
        return np.prod([max(x.stop - x.start, 0) for x in a[:2]])

    @staticmethod
    def aspect(a: tuple[slice, ...]) -> float:
        """Compute the aspect ratio of the slice list"""
        return slu.height(a) * 1.0 / slu.width(a)
    
    @staticmethod
    def width(s: tuple[slice, ...]) -> int:
        """Width of the slice list."""
        return s[1].stop - s[1].start

    @staticmethod
    def height(s: tuple[slice, ...]) -> int:
        """Height of the slice list."""
        return s[0].stop - s[0].start

    @staticmethod
    def dims(s: tuple[slice, ...]) -> tuple[int, ...]:
        """List of dimensions of the slice list."""
        return tuple([x.stop - x.start for x in s])

    @staticmethod
    def box(r0: int, r1: int, c0: int, c1: int) -> tuple[slice, slice]:
        """Create a bounding box from coordinates."""
        return (slice(r0, r1), slice(c0, c1))

    @staticmethod
    def intersect(u: tuple[slice, ...] | None, v: tuple[slice, ...] | None) -> tuple[slice, ...] | None:
        """Compute the intersection of the two slice lists."""
        if u is None:
            return v
        if v is None:
            return u
        return tuple([slice(max(u[i].start, v[i].start), min(u[i].stop, v[i].stop)) for i in range(len(u))])

    @staticmethod
    def union(u: tuple[slice, ...] | None, v: tuple[slice, ...] | None) -> tuple[slice, ...] | None:
        """Compute the union of the two slice lists."""
        if u is None:
            return v
        if v is None:
            return u
        return tuple([slice(min(u[i].start, v[i].start), max(u[i].stop, v[i].stop)) for i in range(len(u))])
    
    @staticmethod
    def volume(a: tuple[slice, ...]) -> int:
        """Return the volume (area) of the slice list."""
        return np.prod([max(x.stop - x.start, 0) for x in a])

    @staticmethod
    def empty(a: tuple[slice, ...]) -> bool:
        """Test whether the slice is empty."""
        return a is None or slu.volume(a) == 0

    @staticmethod
    def xcenter(s: tuple[slice, ...]) -> float:
        """Compute the x-center of the slice list."""
        return np.mean([s[1].stop, s[1].start])

    @staticmethod
    def xoverlap(u: tuple[slice, ...], v: tuple[slice, ...]) -> int:
        """Compute the overlap in the x-direction."""
        return max(0, min(u[1].stop, v[1].stop) - max(u[1].start, v[1].start))

    @staticmethod
    def xoverlaps(u: tuple[slice, ...], v: tuple[slice, ...]) -> bool:
        """Check if the slices overlap in the x-direction."""
        return u[1].stop >= v[1].start and v[1].stop >= u[1].start

    @staticmethod
    def xoverlap_rel(u: tuple[slice, ...], v: tuple[slice, ...]) -> float:
        """Compute the relative overlap in the x-direction."""
        return slu.xoverlap(u, v) * 1.0 / max(1, slu.width(u), slu.width(v))

    @staticmethod
    def ycenter(s: tuple[slice, ...]) -> float:
        """Compute the y-center of the slice list."""
        return np.mean([s[0].stop, s[0].start])

    @staticmethod
    def ycenter_in(u: tuple[slice, ...], v: tuple[slice, ...]) -> bool:
        """Check if the y-center of u is within v."""
        y = slu.ycenter(u)
        return y >= v[0].start and y <= v[0].stop

    @staticmethod
    def yoverlaps(u: tuple[slice, ...], v: tuple[slice, ...]) -> bool:
        """Check if the slices overlap in the y-direction."""
        return u[0].stop >= v[0].start and v[0].stop >= u[0].start
