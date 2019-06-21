'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''
'''
# TODO Add documentation
'''

__author__ = "Alexandros Papangelis"

import os.path
import json


class Ontology:
    '''
    Define the ontology, i.e. the general schema that dictates the dialogue as well as DB / KB.
    '''

    def __init__(self, filename):
        self.ontology_file_name = None
        self.ontology = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.ontology_file_name = filename
                self.load_ontology()

            else:
                raise FileNotFoundError('Ontology file %s not found' % filename)
        else:
            raise ValueError('Unacceptable value for ontology file name: %s ' % filename)

    def load_ontology(self):
        # TODO This is just for proof of concept
        with open(self.ontology_file_name) as ont_file:
            self.ontology = json.load(ont_file)

        # TODO: Move such prints to logger
        # print('Ontology loaded.')
