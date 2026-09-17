# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from pypxml import PageElement, PageType, PageXML
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from shapely.affinity import translate
from shapely.geometry import Polygon

from atrtools.utils import ClickUtils, ImageUtils, PageUtils


logger: logging.Logger = logging.getLogger(__name__)

ESTIMATE_MAX = [PageType.TextRegion]
ESTIMATE_MIN = [PageType.SeparatorRegion]


class RegionShrink:
    def __init__(
        self,
        padding: int = 5,
        smoothing: float = 1.0,
        mode: Literal['merge', 'largest'] = 'merge',
        bbox: list[tuple[PageType, str | None]] | None = None,
        exclude: list[tuple[PageType, str | None]] | None = None,
        threads: int = 1,
    ) -> None:
        """
        Args:
            padding: Padding between region borders and its content in pixels. Defaults to 5.
            smoothing: Smoothing, calculated as the factor of the average glyph size. Prevents regions cutting between 
                       text. Defaults to 1.0.
            mode: Shrinking mode to use for regions. "merge" merges all resulting polygons of each region after 
                  shrinking. "largest" keeps only the largest resulting polygon of each region after shrinking. 
                  Defaults to 'merge'.
            bbox: Draw a minimal bounding box for a specific region after shrinking. Defaults to [].
            exclude: Exclude a specific region from shrinking. Defaults to [].
            threads: Number of threads for parallel region computation. Defaults to 1.
        """
        self.padding = padding
        self.smoothing = smoothing
        self.mode = mode
        self.bbox = [pt.value if t is None else f'{pt.value}.{t}' for pt, t in bbox] if bbox is not None else []
        self.exclude = [pt.value if t is None else f'{pt.value}.{t}' for pt, t in exclude] if exclude is not None else []
        self.threads = max(1, threads)
        
    def process(self, img: PathLike | np.ndarray, xml: PathLike | PageXML) -> PageXML:
        if isinstance(img, PathLike):
            img: np.ndarray | None = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f'Could not open image at {img!s}')
        img: np.ndarray = ~img
        if len(np.unique(img)) > 2:
            raise ValueError('Image is not binary')
        
        self.shape: tuple[int, int] = img.shape
        if len(self.shape) != 2 or not self.shape[0] or not self.shape[1]:
            raise ValueError(f'Invalid image shape: {self.shape}')
        logger.info(f'Image dimensions: {self.shape[0]}x{self.shape[1]}')
        
        if isinstance(xml, PathLike):
            xml: PageXML = PageXML.open(str(xml))
        
        regions: list[PageElement] = []
        for r in xml.regions:
            if r.pagetype.value not in self.exclude and f'{r.pagetype.value}.{r["type"]!s}' not in self.exclude:
                regions.append(r)
        
        if self.threads == 1:
            for region in regions:
                self._process_region(img, region)
        else:
            with ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='lineseg') as executor:
                for region in regions:
                    executor.submit(self._process_region, img, region)
        return xml
    
    def _process_region(self, img: np.ndarray, region: PageElement) -> None:
        logger.debug(f'Processing {region.pagetype.value} {region["id"]}')
        coords_element: PageElement | None = region.find(pagetype=PageType.Coords)
        if coords_element is None or coords_element['points'] is None:
            logger.warning(f'Could not find coordinates in {region.pagetype.value} {region["id"]}')
            return
        region_polygon: np.ndarray | None = PageUtils.points_to_polygon(coords_element['points'])
            
        logger.debug('Mask region')
        region_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(region_mask, [region_polygon], (255, 255, 255))
        im_masked = img & region_mask
        
        logger.debug('Calculating region contour scales')
        w_scale, h_scale = ImageUtils.estimate_page_scale(im_masked)
        if region.pagetype in ESTIMATE_MAX:
            scale = max(1, max(w_scale, h_scale))
        elif region.pagetype in ESTIMATE_MIN:
            scale = max(1, min(w_scale, h_scale))
        else:
            scale = 25
        logger.debug(f'Estimated scale: {scale}')
        
        logger.debug('Pad image')
        padd_size = ceil(scale * self.smoothing + self.padding)
        img_padded = cv2.copyMakeBorder(
            im_masked, padd_size, padd_size, padd_size, padd_size, 
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        
        logger.debug('Dilate text considering the median contour scale')
        dilation_kernel = np.ones((scale, scale), np.uint8)
        im_dilated = cv2.dilate(
            src=img_padded, 
            kernel=dilation_kernel, 
            iterations=2
        )
        
        logger.debug('Close gaps between symbols using provided smoothing factor')
        smooth_kernel = np.ones(
            shape=(round(scale * self.smoothing), round(scale * self.smoothing)), 
            dtype=np.uint8
        )
        im_smoothed = cv2.morphologyEx(
            src=im_dilated, 
            op=cv2.MORPH_CLOSE, 
            kernel=smooth_kernel
        )
        
        logger.debug('Calculting content contours')
        contours = cv2.findContours(im_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not sum([cv2.contourArea(contour) for contour in contours]):
            logger.warning(f'Sum of resulting polygons equals 0 in {region.pagetype.value} {region["id"]}')
            return
        
        logger.debug('Converting found contours to polygons')
        if self.bbox is not None and (region.pagetype.value in self.bbox or f'{region.pagetype.value}.{region["type"]!s}' in self.bbox):
                contours = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(contours)
                polygon = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
                polygon = polygon.buffer(self.padding - scale, join_style='mitre')
                if polygon.area > 0:
                    polygon = translate(polygon, -padd_size, -padd_size)
                    coords_element['points'] = PageUtils.polygon_to_points(np.array(polygon.exterior.coords, dtype=np.int32))
                else:
                    logger.warning(f'Area of resulting bounding box equals 0 in {region.pagetype.value} {region["id"]}')
                return
        
        if self.mode == 'largest':
            polygon = Polygon([tuple(pt[0] for pt in contours[0])])
            polygon = polygon.buffer(self.padding - scale, join_style='mitre')
            if polygon.area > 0:
                polygon = translate(polygon, -padd_size, -padd_size)
                coords_element['points'] = PageUtils.polygon_to_points(np.array(polygon.exterior.coords, dtype=np.int32))
            else:
                logger.warning(f'Area of resulting polygon equals 0 in {region.pagetype.value} {region["id"]}')
        
        elif self.mode == 'merge':
            polygons = [Polygon([tuple(pt[0]) for pt in contour]) for contour in contours if len(contour) > 3]  # create shapely polygons from contours
            polygons = [poly.buffer(self.padding - scale, join_style='mitre') for poly in polygons]  # add padding to final polygons
            if not polygons:
                logger.warning(f'No valid polygons remaining in {region.pagetype.value} {region["id"]}')
                return
            logger.debug(f'Merging {len(polygons)} resulting polygons')
            polygon = PageUtils.merge_polygons(polygons, max(3, self.padding))
            if polygon is None:  # if merging fails, keep original polygon
                logger.warning(f'No remaining polygons after merging in {region.pagetype.value} {region["id"]}')
                return
            polygon = translate(polygon, -padd_size, -padd_size)
            coords_element['points'] = PageUtils.polygon_to_points(np.array(polygon.exterior.coords, dtype=np.int32))


@click.command('regionshrink', short_help='Shrink existing regions.')
@click.help_option('-h', '--help', hidden=True)
@click.argument('xml', type=click.Path(), callback=ClickUtils.glob, nargs=-1)
@click.option(
    '-i', '--image',
    help='Full suffix of the image files to be used. Defaults to the first image found with "<filename>.*png".',
    type=click.STRING
)
@click.option(
    '-o', '--output',
    help='Output directory for generated PAGE-XML files. If omitted, the input file will be overwritten.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-p', '--padding',
    help='Padding between region borders and its content in pixels.',
    type=click.INT, 
    default=5, 
    show_default=True
)
@click.option(
    '-s', '--smoothing',
    help='Smoothing, calculated as the factor of the average glyph size. Prevents regions cutting between text.',
    type=click.FLOAT, 
    default=1.0, 
    show_default=True
)
@click.option(
    '-m', '--mode',
    help='Shrinking mode to use for regions. "merge" merges all resulting polygons of each region after shrinking. '
         '"largest" keeps only the largest resulting polygon of each region after shrinking.',
    type=click.Choice(['merge', 'largest']), 
    default='merge', 
    show_default=True
)
@click.option(
    '-b', '--bbox', 'bbox',
    help='Draw a minimal bounding box for a specific region after shrinking. '
         'Should be of format "PageType" or "PageType.subtype". Multiple regions can be specified. '
         'Examples: "-b ImageRegion", "-b TextRegion.paragraph"',
    callback=ClickUtils.pagetype,
    metavar='PageType',
    type=click.STRING,
    multiple=True
)
@click.option(
    '-e', '--exclude', 'exclude',
    help='Exclude a specific region from shrinking. Should be of format "PageType" or "PageType.subtype". '
         'Multiple excludes can be specified. Examples: "-e ImageRegion", "-e TextRegion.paragraph"',
    callback=ClickUtils.pagetype,
    metavar='PageType',
    type=click.STRING,
    multiple=True,
)
@click.option(
    '-t', '--threads',
    help='Number of threads for concurrent region processing.',
    type=click.IntRange(1),
    default=1,
    show_default=True
)
def cli(
    xml: list[Path],
    image: str | None,
    output: Path | None,
    padding: int,
    smoothing: float,
    mode: Literal['merge', 'largest'],
    bbox: list[tuple[PageType, str | None]],
    exclude: list[tuple[PageType, str | None]],
    threads: int,
    **kwargs
) -> None:
    """
    Shrink existing regions to their contents in PAGE-XML files.
    
    Only binary image inputs are supported!
    
    XML: One or more PAGE-XML paths. Use glob patterns in quotes to process multiple files.
    """
    if not xml:
        raise click.BadArgumentUsage('No input PAGE-XML files found')
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    
    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        load_task = progress.add_task('Loading module', total=None)
        pairs: list[tuple[Path, Path]] = []
        for xml_fp in xml:
            if image:
                img_fp = xml_fp.parent / (xml_fp.name.split('.')[0] + image)
            else:
                img_fp = next(xml_fp.parent.glob(f'{xml_fp.name.split('.')[0]}.*png'), None)
            if img_fp is None or not img_fp.exists():
                logger.error(f'No matching image found for PAGE-XML: {xml}')
                continue
            else:
                logger.debug(f'Found image {img_fp} for PAGE-XML {xml_fp}')
                pairs.append((xml_fp, img_fp))
        if not pairs:
            raise click.BadArgumentUsage('No files to process')
        
        shrinker = RegionShrink(padding, smoothing, mode, bbox, exclude, threads)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(pairs))
        for xml_fp, img_fp in pairs:
            progress.update(task, description='/'.join(xml_fp.parts[-4:]))
            logger.info(f'Processing: {xml_fp} and {img_fp}')
            try:
                out_dir: Path = output or xml_fp.parent
                out_path: Path = out_dir / f'{xml_fp.name.split(".")[0]}.xml'
                shrinker.process(img_fp, xml_fp).save(out_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f'Processing failed for {xml_fp}: {exc}')
            progress.advance(task)
        progress.update(task, description='Done')
