# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging

import click

from .line_segmentation import line_segmentation, LineSegmentation
from .region_shrink import region_shrink, RegionShrink
from ..util import clicku


__all__ = ['LineSegmentation', 'RegionShrink']
logger = logging.getLogger(__name__)


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.pass_context
@click.version_option(version('atrtools'), '--version', prog_name='ATRtools')
@click.help_option('--help')
@click.option(
     '-v', '--verbose', 'verbosity',
     help='Set the verbosity level. Default: ERROR. Use -v for WARNING, -vv for INFO, -vvv for DEBUG.', 
     count=True
)
def main(ctx, verbosity: int, *args, **kwargs) -> None:
    """
    Collection of useful tools for ATR and PAGE-XML
    """
    clicku.setup_logging(verbosity)
    
main.add_command(line_segmentation)
main.add_command(region_shrink)
