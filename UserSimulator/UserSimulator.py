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

from abc import abstractmethod
from ConversationalAgent.ConversationalModule import ConversationalModule


class UserSimulator(ConversationalModule):
    '''
    Abstract class to define the interface for user simulators.
    '''

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def receive_input(self, inpt):
        pass

    @abstractmethod
    def respond(self):
        pass

    def generate_output(self, args=None):
        return self.respond()

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def at_terminal_state(self):
        pass
