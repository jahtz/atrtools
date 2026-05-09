# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from pypxml import PageElement, PageType


class pageu:
    """
    Collection of add-ons to the pypxml package
    """
    @staticmethod
    def points_to_polygon(points: str) -> np.ndarray:
        """Convert a PAGE-XML points string to a numpy array."""
        return np.array([tuple(map(int, xy.split(','))) for xy in points.split()], dtype=np.int32)
    
    @staticmethod
    def element_to_polygon(element: PageElement) -> np.ndarray | None:
        """Extract coordinates from a PageElement and return a numpy array."""
        coords: PageElement | None = element.find(pagetype=PageType.Coords)
        if coords is None or coords['points'] is None:
            return None
        return pageu.points_to_polygon(coords['points'])
    
    @staticmethod
    def polygon_to_points(points: np.ndarray) -> str:
        """Convert a numpy array to a PAGE-XML points string."""
        return ' '.join(f'{int(x)},{int(y)}' for x, y in points)
