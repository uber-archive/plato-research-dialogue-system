"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from Data.DataParser import DataParser
import json
import csv

"""
This Parser will read MetalWOZ txt files and parse them into CSV.
"""


class Parser(DataParser):
    def __int__(self):
        self.data_path = None

    def initialize(self, **kwargs):
        """
        Initialize the internal structures of the data parser.

        :param kwargs:
        :return:
        """

        if 'path' in kwargs:
            self.data_path = kwargs['path']

    def parse_data(self):
        """
        Parse the data and generate Plato Dialogue Experience Logs.

        :return:
        """

        if not self.data_path:
            raise ValueError('Parse_MetalWOZ: No data_path provided')

        with open(self.data_path, "r") as data_file, \
                open('Data/data/metalwoz.csv', 'w') as parsed_data_file:
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

        print('Data Reading done.')

    def save(self, path):
        """
        Save the experience

        :param path: path to save the experience to
        :return:
        """

        pass
