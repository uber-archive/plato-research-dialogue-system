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

from abc import abstractmethod
from plato.agent.component.conversational_module import ConversationalModule

"""
UserSimulator is the abstract parent class for all User Simulator classes and 
defines the interface that should be followed.
"""


class UserSimulator(ConversationalModule):
    """
    Abstract class to define the interface for user simulators.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialise the User Simulator. Here we initialize structures that
        we need throughout the life of the User Simulator.
        """
        super(UserSimulator, self).__init__()

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the User Simulator. Here we initialize structures that
        need to be reset after each dialogue.

        :param kwargs: arguments necessary for initialization
        :return:
        """
        pass

    @abstractmethod
    def receive_input(self, inpt):
        """
        Handles the input.

        :param inpt: the input received
        :return: optional
        """
        pass

    @abstractmethod
    def respond(self):
        """
        Generates (or simply returns) the User Simulator's response

        :return: the generated output
        """
        pass

    def generate_output(self, args=None):
        """
        This is the generic function used to generate output.

        :param args: input arguments
        :return: the User Simulator's generated output
        """
        return self.respond()

    @abstractmethod
    def train(self, data):
        """
        Train the User Simulator

        :param data: dialogue experience
        :return: nothing
        """
        pass

    @abstractmethod
    def save(self, path=None):
        """
        Save trained models

        :param path: path to save the models to
        :return: nothing
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load pre-trained models

        :param path: path to load models from
        :return: nothing
        """

        pass

    @abstractmethod
    def at_terminal_state(self):
        """
        Checks if the User Simulator is in a terminal state
        :return: True or False
        """

        pass
