"""
Copyright (c) 2019-2019-2020 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


__author__ = "Alexandros Papangelis"

from plato.controller import basic_controller, sgui_controller
from plato.domain import create_domain_sqlite_db
from plato.utilities.parser import run_data_parser

import click

"""
This is the main entry point to Plato Research dialogue System.

"""


@click.group()
def entry_point():
    pass


@click.command()
@click.option('--config',
              default='')
@click.option('--test',
              default=False)
def run(config, test):
    if test:
        basic_controller.run(None, True)

    elif config:
        basic_controller.run(config)

    else:
        print_usage()


@click.command()
@click.option('--config',
              default='')
def gui(config):
    if config:
        sgui_controller.run(config)

    else:
        print_usage()


@click.command()
@click.option('--config',
              default='')
def domain(config):
    if config:
        create_domain_sqlite_db.run(config)

    else:
        print_usage()


@click.command()
@click.option('--config',
              default='')
def parse(config):
    if config:
        run_data_parser.run(config)

    else:
        print_usage()


def print_usage():
    print("Usage:\n"
          "plato run --config <path_to_config.yaml>\n"
          "plato run --test True\n"
          "plato gui --config <path_to_config.yaml>\n"
          "plato domain --config <path_to_config.yaml>\n"
          "plato parse --config <path_to_data>\n"
          "\n"
          "Remember, Plato RDS runs with Python 3.6")


entry_point.add_command(run)
entry_point.add_command(gui)
entry_point.add_command(domain)
entry_point.add_command(parse)

if __name__ == '__main__':
    """
    This is an alternative entry point to Plato RDS.

    Usage:

    python run_plato_rds.py -config <path_to_config.yaml>

    (testing mode)
    python run_plato_rds.py --test

    Remember, Plato RDS runs with Python 3.6
    """
    run()
