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

from .. import Policy
from Ontology import Ontology
import tensorflow as tf


class DQN_Policy(Policy.Policy):

    def __init__(self, ontology):
        super(DQN_Policy, self).__init__()

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

    def initialize(self, **kwargs):
        pass

    def next_action(self, state):
        pass

    def train(self, dialogues):
        pass

    def restart(self, args):
        pass

    def save(self, path=None):
        pass

    def load(self, path):
        pass

