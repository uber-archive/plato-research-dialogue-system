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

from plato.agent.component.conversational_module \
    import ConversationalModule
from abc import abstractmethod

"""
DialoguePolicy is the abstract parent class of all policies and defines the 
interface that each dialogue policy derived class should adhere to.
"""


class DialoguePolicy(ConversationalModule):

    def __init__(self):
        """
        Initialize the internal structures of the dialogue policy
        """
        super(DialoguePolicy, self).__init__()

    @abstractmethod
    def initialize(self, args):
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

    # From the ConversationalModule interface
    def generate_output(self, args=None):
        """

        :param args:
        :return:
        """

        # Unpack args
        if isinstance(args, dict):
            if 'args' in args:
                args = args['args']

            else:
                raise ValueError('DialoguePolicy: unacceptable input!')

        return self.next_action(args)

    @abstractmethod
    def train(self, dialogues):
        """
        Train the dialogue policy's internal model

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
