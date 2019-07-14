"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from abc import ABC, abstractmethod

"""
DialoguePolicy is the abstract parent class of all policies and defines the 
interface that each DialoguePolicy derived class should
adhere to.
"""


class DialoguePolicy(ABC):

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        pass

    @abstractmethod
    def restart(self, **kwargs):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return:
        """

        pass

    @abstractmethod
    def next_action(self, state):
        """
        Consult the internal model and produce the agent's response, given
        the current state

        :param state: the current dialogue state
        :return:
        """
        pass

    @abstractmethod
    def train(self, dialogues):
        """
        Train the policy's internal model

        :param dialogues: the dialogue experience
        :return:
        """
        pass

    @abstractmethod
    def save(self, path=None):
        """
        Save the internal model to the path provided (or to a default one)

        :param path: the path to save the model to
        :return:
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the model from the path provided

        :param path: the path to load the model from
        :return:
        """
        pass
