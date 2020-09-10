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

from plato.utilities.parser.data_parser import DataParser
import json
import csv

"""
This Parser will read MetalWOZ txt files and parse them into CSV.
"""


class Parser(DataParser):
    def __init__(self, args):
        super(Parser, self).__init__()

        self.data_path = None

    def initialize(self, **kwargs):
        """
        Initialize the internal structures of the data parser.

        :param kwargs:
        :return:
        """

        if 'data_path' in kwargs:
            self.data_path = kwargs['data_path']

    def parse_data(self):
        """
        Parse the data and generate Plato dialogue Experience logs.

        :return:
        """

        if not self.data_path:
            raise ValueError('Parse_MetalWOZ: No data_path provided')

        with open(self.data_path, "r") as data_file, \
                open('data/metalwoz.csv', 'w') as parsed_data_file:
            line = data_file.readline()
            csv_writer = csv.writer(parsed_data_file, delimiter=',')

            # Write header
            csv_writer.writerow(['user', 'system'])

            while line:
                dialogue = json.loads(line)

                if dialogue['turns']:
                    # Write first turn explicitly, since in the dialogues the
                    # system has the initiative.
                    csv_writer.writerow(['hi',
                                         dialogue['turns'][0]])

                for t in range(1, len(dialogue['turns']), 2):
                    if t+1 < len(dialogue['turns']):
                        csv_writer.writerow([dialogue['turns'][t],
                                             dialogue['turns'][t+1]])
                    else:
                        # Write last turn
                        csv_writer.writerow([dialogue['turns'][t],
                                             ''])

                line = data_file.readline()

        print('MetalWOZ parser Reading done.')

    def save(self, path):
        """
        Save the experience

        :param path: path to save the experience to
        :return:
        """

        pass
