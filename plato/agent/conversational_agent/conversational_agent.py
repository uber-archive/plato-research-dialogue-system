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

from abc import ABC, abstractmethod

"""
agent is the parent abstract class of all 
Conversational Agents. It defines the interface that the 
controller will use.
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
