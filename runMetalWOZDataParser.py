"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"


import sys

from Data import Parse_MetalWOZ

"""
This script runs the MetalWOZ Data Parser.
"""


if __name__ == '__main__':
    """
    This script will create a MetalWOZ-specific data parser, and run it.
    """

    if len(sys.argv) < 3:
        raise AttributeError('Please provide a path to the MetalWOZ data.')

    if sys.argv[1] == '-data_path':
        data_path = sys.argv[2]

    else:
        raise TypeError(f'Incorrect option: {sys.argv[1]}')

    parser = Parse_MetalWOZ.Parser()

    if len(sys.argv) > 2:
        if sys.argv[1] == '-data':
            data_path = sys.argv[2]

    args = {'path': data_path}

    parser.initialize(**args)

    print('Parsing {0}'.format(args['path']))

    parser.parse_data()

    print('Data parsing complete.')

    # Save data
    parser.save('Logs')

