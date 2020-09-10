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

import os.path
import json

"""
domain is a class that loads ontology files (in .json format) into Plato.
"""


class Ontology:
    """
    Define the ontology, i.e. the general schema that dictates the dialogue as
    well as DB / KB.
    """

    def __init__(self, filename):
        """
        Initialize the internal structures of the domain
        :param filename: path to load the ontolgoy from
        """

        self.ontology_file_name = None
        self.ontology = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.ontology_file_name = filename
                self.load_ontology()

            else:
                raise FileNotFoundError('domain file %s not found'
                                        % filename)
        else:
            raise ValueError('Unacceptable value for ontology file name: %s '
                             % filename)

    def load_ontology(self):
        """
        Loads the ontology file
        :return: nothing
        """

        with open(self.ontology_file_name) as ont_file:
            self.ontology = json.load(ont_file)
