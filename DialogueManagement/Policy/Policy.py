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
Abstract Policy Class
'''

__author__ = "Alexandros Papangelis"

from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def initialize(self, **kwargs):
        '''
        Initialize anything that should not be in __init__

        :return: Nothing
        '''

        pass

    @abstractmethod
    def restart(self, **kwargs):
        '''
        Re-initialize relevant parameters / variables at the beginning of each dialogue.

        :return:
        '''

        pass

    @abstractmethod
    def next_action(self, state):
        pass

    @abstractmethod
    def train(self, dialogues):
        pass

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path):
        pass
