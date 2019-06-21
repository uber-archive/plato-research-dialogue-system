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

from abc import ABC, abstractmethod


class ConversationalAgent(ABC):
    '''
    Abstract class defining what it means to be a Conversational Agent
    '''

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def start_dialogue(self, **kwargs):
        pass

    @abstractmethod
    def continue_dialogue(self, **kwargs):
        pass

    @abstractmethod
    def end_dialogue(self):
        pass

    @abstractmethod
    def terminated(self):
        pass

