# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from .callback import glob_callback
from .util import find_image

logger: logging.Logger = logging.getLogger(__name__)


@click.command('lineseg', short_help='Compute baselines and polygons for existing TextRegions')
@click.help_option('--help', hidden=True)
@click.argument(
    'xmls', 
    type=click.Path(), 
    callback=glob_callback, 
    nargs=-1, 
    required=True
)
@click.option(
    '-i', '--image', 'image_suffix',
    help='Full suffix of the image files to be used. If not set, the suffix is derived from the PAGE-XML files.',
    type=click.STRING
)
@click.option(
    '-o', '--output',
    help='Output directory for generated PAGE-XML files. If omitted, the input file will be overwritten.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-s', '--spread',
    help='Distance in points (pt) from the foreground to project text line (or text region) labels into the '
         'background for polygonal contours; If zero, project half a scale/capheight.',
    type=click.FloatRange(0.0),
    default=0.0,
    show_default=True,
)
@click.option(
    '-t', '--threads',
    help='Number of threads for concurrent region processing',
    type=click.IntRange(1),
    default=1,
    show_default=True
)
def lineseg(
    xmls: list[Path],
    image_suffix: str | None = None,
    output: Path | None = None,
    spread: float = 2.4, 
    threads: int = 1
) -> None:
    """
    Compute baselines and polygons for existing TextRegions in PAGE-XML files.
    
    Only binary image inputs are supported!

    XMLs: One or more PAGE-XML paths. Use glob patterns in quotes to process multiple files.
    """
    if not xmls:
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
        from .. import LineSegmenter
        
        pairs: list[tuple[Path, Path]] = []
        for xml in xmls:
            img = find_image(xml, image_suffix)
            if img is None:
                logger.error(f'No matching image found for PAGE-XML: {xml}')
            else:
                logger.debug(f'Found image {img} for PAGE-XML {xml}')
                pairs.append((xml, img))
        
        if not pairs:
            raise click.BadArgumentUsage('No files to process')
        
        segmenter = LineSegmenter(spread, threads)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(pairs))
        for xml, img in pairs:
            progress.update(task, description='/'.join(xml.parts[-4:]))
            logger.info(f'Processing: {xml} and {img}')
            try:
                out_dir: Path = output or xml.parent
                out_path: Path = out_dir / f'{xml.name.split(".")[0]}.xml'
                segmenter.process(img, xml).save(out_path)
            except Exception as exc:
                logger.error(f'Processing failed for {xml}: {exc}')
            progress.advance(task)
        progress.update(task, description='Done')
