import os
from pathlib import Path
import shutil
from typing import Optional

import click

from ..run_command import run_command


_H1ST_AWS_EB_CLI_UTIL_DIR_PATH = Path(__file__).parent
_EB_EXTENSIONS_DIR_NAME = '.ebextensions'
_EB_IGNORE_FILE_NAME = '.ebignore'
_PLATFORM_DIR_NAME = '.platform'


@click.command(name='init',
               cls=click.Command,
               context_settings=None,
               help=('H1st AWS Elastic Beanstalk CLI: '
                     'Initialize Configuration >>>'),
               epilog=('^^^ H1st AWS Elastic Beanstalk CLI: '
                       'Initialize Configuration'),
               short_help='H1st AWS-EB Init',
               options_metavar='',
               add_help_option=True,
               hidden=False,
               deprecated=False)
def init():
    """Initialize H1st AWS Elastic Beanstalk Configuration."""
    os.system(command='eb init')


@click.command(name='deploy',
               cls=click.Command,
               context_settings=None,
               help='H1st AWS Elastic Beanstalk CLI: Deploy >>>',
               epilog='^^^ H1st AWS Elastic Beanstalk CLI: Deploy',
               short_help='H1st AWS-EB Deploy',
               options_metavar='[OPTIONS]',
               add_help_option=True,
               hidden=False,
               deprecated=False)
@click.argument('aws_eb_env_name',
                cls=click.Argument,
                required=False,
                type=str,
                default=None,
                callback=None,
                nargs=None,
                metavar='AWS_EB_ENV_NAME',
                expose_value=True,
                is_eager=False,
                envvar=None,
                autocompletion=None)
@click.option('--asgi',
              cls=click.Option,
              show_default=True,
              prompt=False,
              confirmation_prompt=False,
              hide_input=False,
              is_flag=False,
              # flag_value=...,
              multiple=False,
              count=False,
              allow_from_autoenv=False,
              type=str,
              help=('ASGI Server (daphne, hypercorn, uvicorn) to use '
                    '[default: None]'),
              show_choices=True,
              default=None,
              required=False,
              callback=None,
              nargs=None,
              metavar='ASGI',
              expose_value=True,
              is_eager=False,
              envvar=None)
@click.option('--create',
              cls=click.Option,
              show_default=True,
              prompt=False,
              confirmation_prompt=False,
              hide_input=False,
              is_flag=True,
              flag_value=True,
              multiple=False,
              count=False,
              allow_from_autoenv=False,
              type=bool,
              help='whether to create a new AWS Elastic Beanstalk environment',
              show_choices=True,
              default=False,
              required=False,
              callback=None,
              nargs=None,
              metavar='CREATE',
              expose_value=True,
              is_eager=False,
              envvar=None)
def deploy(aws_eb_env_name: Optional[str] = None,
           asgi: Optional[str] = None,
           create: bool = False):
    """Deploy H1st onto AWS Elastic Beanstalk."""
    assert not os.path.exists(path=_EB_EXTENSIONS_DIR_NAME)
    shutil.copytree(
        src=_H1ST_AWS_EB_CLI_UTIL_DIR_PATH / _EB_EXTENSIONS_DIR_NAME,
        dst=_EB_EXTENSIONS_DIR_NAME,
        symlinks=False,
        ignore=None,
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False)
    assert os.path.isdir(_EB_EXTENSIONS_DIR_NAME)

    _eb_ignore_exists = os.path.exists(path=_EB_IGNORE_FILE_NAME)

    if _eb_ignore_exists:
        assert os.path.isfile(_EB_IGNORE_FILE_NAME)

    else:
        shutil.copyfile(
            src=_H1ST_AWS_EB_CLI_UTIL_DIR_PATH / _EB_IGNORE_FILE_NAME,
            dst=_EB_IGNORE_FILE_NAME)
        assert os.path.isfile(_EB_IGNORE_FILE_NAME)

    assert not os.path.exists(path=_PLATFORM_DIR_NAME)
    shutil.copytree(
        src=_H1ST_AWS_EB_CLI_UTIL_DIR_PATH / _PLATFORM_DIR_NAME,
        dst=_PLATFORM_DIR_NAME,
        symlinks=False,
        ignore=None,
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False)
    assert os.path.isdir(_PLATFORM_DIR_NAME)

    profile = input('AWS IAM Profile: ')
    if not profile:
        profile = 'default'

    if create:
        region = input('AWS Region: ')
        vpc = input('AWS VPC: ')
        subnets = input('AWS Subnets: ')
        instance_type = input('AWS EC2 Instance Type: ')
        assert region and vpc and subnets and instance_type

        run_command(
            command=(f'eb create --region {region} --vpc.id {vpc}'
                     f' --vpc.dbsubnets {subnets} --vpc.ec2subnets {subnets}'
                     f' --vpc.elbsubnets {subnets} --vpc.elbpublic'
                     ' --vpc.publicip'
                     f' --instance_type {instance_type}'
                     f' --profile {profile}'
                     f" {aws_eb_env_name if aws_eb_env_name else ''}"),
            copy_standard_files=True,
            asgi=asgi)

    else:
        run_command(
            command=(f'eb deploy --profile {profile}'
                     f" {aws_eb_env_name if aws_eb_env_name else ''}"),
            copy_standard_files=True,
            asgi=asgi)

    shutil.rmtree(
        path=_EB_EXTENSIONS_DIR_NAME,
        ignore_errors=False,
        onerror=None)
    assert not os.path.exists(path=_EB_EXTENSIONS_DIR_NAME)

    if not _eb_ignore_exists:
        os.remove(_EB_IGNORE_FILE_NAME)
        assert not os.path.exists(path=_EB_IGNORE_FILE_NAME)

    shutil.rmtree(
        path=_PLATFORM_DIR_NAME,
        ignore_errors=False,
        onerror=None)
    assert not os.path.exists(path=_PLATFORM_DIR_NAME)


@click.group(name='aws-eb',
             cls=click.Group,
             commands={'init': init, 'deploy': deploy},
             invoke_without_command=False,
             no_args_is_help=True,
             subcommand_metavar='H1ST_AWS_EB_SUB_COMMAND',
             chain=False,
             help='H1st AWS Elastic Beanstalk CLI >>>',
             epilog='^^^ H1st AWS Elastic Beanstalk CLI',
             short_help='H1st AWS-EB CLI',
             options_metavar='[OPTIONS]',
             add_help_option=True,
             hidden=False,
             deprecated=False)
def h1st_aws_eb():
    """Trigger H1st AWS Elastic Beanstalk from CLI."""
