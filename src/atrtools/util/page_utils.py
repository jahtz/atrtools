# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import ceil

import numpy as np
from pypxml import PageElement, PageType
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union


class pageu:
    """
    Collection of add-ons to the pypxml package
    """
    @staticmethod
    def points_to_polygon(points: str) -> np.ndarray:
        """Convert a PAGE-XML points string to a numpy array"""
        return np.array([tuple(map(int, xy.split(','))) for xy in points.split()], dtype=np.int32)
    
    @staticmethod
    def element_to_polygon(element: PageElement) -> np.ndarray | None:
        """Extract coordinates from a PageElement and return a numpy array"""
        if element.pagetype != PageType.Coords:
            element: PageElement | None = element.find(pagetype=PageType.Coords)
        if element is None or element['points'] is None:
            return None
        return pageu.points_to_polygon(element['points'])
    
    @staticmethod
    def polygon_to_points(points: np.ndarray) -> str:
        """Convert a numpy array to a PAGE-XML points string"""
        return ' '.join(f'{int(max(0, x))},{int(max(0, y))}' for x, y in points)
    
    @staticmethod
    def shift_polygon(
        poly: np.ndarray, 
        x_offset: int, 
        y_offset: int, 
        upper: tuple[int, int] | None = None,
        lower: tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Shift a polygon by given x and y offsets, optionally clipping to bounds.
        Args:
            poly: Polygon coordinates as a 2D array of shape (N, 2).
            x_offset: X-axis offset.
            y_offset: Y-axis offset.
            upper: Upper bounds for clipping.
            lower: Lower bounds for clipping.
        Returns:
            Shifted and optionally clipped polygon.
        """
        shifted = poly + [x_offset, y_offset]
        if upper is not None: 
            shifted = np.clip(shifted, list(lower), list(upper))
        return shifted
    
    @staticmethod
    def merge_polygons(polygons: list[Polygon], buffer: int = 1) -> Polygon | None:
        """
        Merge a list of polygons, connecting them if they do not intersect.
        Args:
            polygons: Polygons to merge.
            buffer: Buffer size in pixels for connecting multiple polygons.
        Returns:
            A single polygon or None if the result is not a Polygon.
        """
        merged_geom = unary_union(polygons)
        if merged_geom.geom_type == 'Polygon':
            return merged_geom
        else:
            polygons = list(merged_geom.geoms)
            if not polygons:
                return None
            centroids = np.array([p.centroid.coords[0] for p in polygons])
            dist_matrix = distance_matrix(centroids, centroids)
            np.fill_diagonal(dist_matrix, np.inf)

            graph = csr_matrix(dist_matrix)
            mst = minimum_spanning_tree(graph)
            mst_edges = list(zip(*mst.nonzero()))

            connections = [LineString([centroids[i], centroids[j]]).buffer(ceil(buffer / 2)) for i, j in mst_edges]
            combined_geometry = unary_union(polygons + connections)

            if combined_geometry.geom_type == 'Polygon':
                return combined_geometry
            elif combined_geometry.geom_type == 'GeometryCollection':
                all_exteriors = []
                for poly in combined_geometry.geoms:
                    all_exteriors.extend(poly.exterior.coords)
                return Polygon(all_exteriors)
            else:
                return None
    