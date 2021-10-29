import click

from .aws_eb import h1st_aws_eb


@click.group(name='h1st',
             cls=click.Group,
             commands={'aws-eb': h1st_aws_eb},
             invoke_without_command=False,
             no_args_is_help=True,
             subcommand_metavar='H1ST_SUB_COMMAND',
             chain=False,
             help='H1st CLI >>>',
             epilog='^^^ H1st CLI',
             short_help='H1st CLI',
             options_metavar='[OPTIONS]',
             add_help_option=True,
             hidden=False,
             deprecated=False)
def h1st():
    """Trigger H1st from CLI."""
