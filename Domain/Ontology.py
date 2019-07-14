"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

import os.path
import json

"""
Domain is a class that loads ontology files (in .json format) into Plato.
"""


class Ontology:
    """
    Define the ontology, i.e. the general schema that dictates the dialogue as
    well as DB / KB.
    """

    def __init__(self, filename):
        """
        Initialize the internal structures of the Domain
        :param filename: path to load the ontolgoy from
        """

        self.ontology_file_name = None
        self.ontology = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.ontology_file_name = filename
                self.load_ontology()

            else:
                raise FileNotFoundError('Domain file %s not found'
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
