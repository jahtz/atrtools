# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version

import click

from atrtools.linesegmentation import cli as linesegmentation_cli
from atrtools.regionshrink import cli as regionshrink_cli
from atrtools.utils import ClickUtils


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.pass_context
@click.version_option(version('ATRtools'), '--version', prog_name='ATRtools', hidden=True)
@click.help_option('-h', '--help', hidden=True)
@click.option(
     '-v', '--verbose', 'verbosity',
     help='Set the verbosity level. Default: ERROR. Use -v for WARNING, -vv for INFO, -vvv for DEBUG.', 
     count=True
)
def cli(ctx: click.Context, verbosity: int, **kwargs) -> None:
    """
    Collection of useful tools for ATR and PAGE-XML
    """
    ClickUtils.setup_logging(verbosity)


cli.add_command(linesegmentation_cli)
cli.add_command(regionshrink_cli)
