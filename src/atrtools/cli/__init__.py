# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging

import click
from rich.logging import RichHandler

from .lineseg import lineseg


logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(level: int = 0) -> None:
    logging.basicConfig(
        level=max(10, 40 - (10 * level)),
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )
    logging.getLogger('pypxml').setLevel(max(30, 40 - (10 * level)))
    logger.info(f'Logging verbosity set to {logging.getLevelName(logger.getEffectiveLevel())}')


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.version_option(version('atrtools'), '--version', prog_name='ATRtools')
@click.help_option('--help')
@click.pass_context
@click.option(
     '-v', '--verbose', 'verbosity',
     help='Set the verbosity level. Default: ERROR. Use -v for WARNING, -vv for INFO, -vvv for DEBUG.', 
     count=True
)
def main(ctx, verbosity: int = 0, *args, **kwargs) -> None:
    """
    Collection of useful tools for ATR and PAGE-XML
    """
    setup_logging(verbosity)
    
main.add_command(lineseg)
