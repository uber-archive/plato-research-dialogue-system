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
ConversationalAgent is the parent abstract class of all 
Conversational Agents. It defines the interface that the 
Controller will use.
"""


class ConversationalAgent(ABC):
    """
    Abstract class defining what it means to be a Conversational Agent
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize internal structures of a Conversational Agent

        :return: nothing
        """
        pass

    @abstractmethod
    def start_dialogue(self, **kwargs):
        """
        Reset or initialize internal structures at the beginning of the
        dialogue. May issue first utterance if this agent has the initiative.

        :param kwargs:
        :return:

        """
        pass

    @abstractmethod
    def continue_dialogue(self, **kwargs):
        """
        Perform one dialogue turn.
        :param kwargs:
        :return: this agent's output (dialogue acts, text, speech, statistics)
        """

        pass

    @abstractmethod
    def end_dialogue(self):
        """
        End the current dialogue and train
        :return: nothing
        """

        pass

    @abstractmethod
    def terminated(self):
        """
        Check if this agent is at a terminal state.
        :return: True or False
        """

        pass
