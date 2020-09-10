"""
Copyright (c) 2019-2020 Uber Technologies, Inc.

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

from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent

import yaml
import sys
import os


"""
This script runs the data parser specified in the configuration file.
"""


def run(config):
    """
    This script will create a data-specific parser, and run it.
    """

    if not config:
        raise AttributeError('Please provide a path to the data. For'
                             'example: <PATH...>/DSTC2/dstc2_traindev/data/')

    if config[-5:] == '.yaml':
        config = check_file_path(config)

        with open(config, 'r') as file:
            args = yaml.load(file, Loader=yaml.Loader)

    else:
        raise ValueError('Unacceptable config file type for data parser.')

    if 'package' not in args:
        raise AttributeError('Please provide a "package" argument for '
                             'data parser!')

    if 'class' not in args:
        raise AttributeError('Please provide a "class" argument for '
                             'data parser!')

    if 'arguments' not in args:
        print(f'Warning! Data Parser {args["package"]}.'
              f'{args["class"]} called without arguments!')

        args['arguments'] = {}

    parser = ConversationalGenericAgent.load_module(args['package'],
                                                    args['class'],
                                                    args['arguments'])

    parser.initialize(**args['arguments'])

    print('Parsing...')

    parser.parse_data()

    print('Parsing complete.')

    # Save plato experience logs
    parser.save(f'logs/')

    print('Logs saved.')


def check_file_path(path):
    if os.path.isfile(path):
        return path

    else:
        # Else look for the config file in the example folder
        import plato

        # __file__ points to __init__.py, which is 11 characters but we
        # want the root path only.
        plato_path = "/".join(plato.__file__.split("/")[:-1])[:-6] + '/'
        new_config_path = \
            plato_path + 'example/config/parser/' + path

        if os.path.isfile(new_config_path):
            return new_config_path

        else:
            raise ValueError(f'Configuration file {path} '
                             f'not found!')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--config':
        run(check_file_path(sys.argv[2]))

    else:
        raise ValueError('Please provide a configuration file:\n'
                         'python run_data_parser.py --config <PATH>')
