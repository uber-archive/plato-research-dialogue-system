"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from plato.agent.component.conversational_module \
    import ConversationalModule
from abc import abstractmethod

"""
The DialogueStateTracker is the abstract parent class of all state trackers. 
It defines the interface that should be followed and owns all dialogue State 
updates. The Dialogue State Tracker should be the source of truth for the 
current dialogue State.
"""


class DialogueStateTracker(ConversationalModule):

    def __init__(self):
        """
        Initialize the internal structures of the dialogue state tracker
        """
        super(DialogueStateTracker, self).__init__()

    @abstractmethod
    def initialize(self, args):
        """
        Initialize structures at the beginning of each dialogue
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
            if args == {}:
                args = []

            elif 'args' in args:
                args = args['args']

            else:
                raise ValueError(f'DialogueStateTracker: unacceptable input:'
                                 f'{args}')

        return self.update_state(args)

    @abstractmethod
    def update_state(self, inpt):
        """
        Update the current dialogue state given the input

        :param inpt: input to the model
        :return:
        """
        pass

    @abstractmethod
    def train(self, data):
        """
        Train the internal model for model-based Dialogue State Trackers
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def save(self):
        """
        Train the internal model

        :return:
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the internal model from the given path

        :param path: path to the model
        :return:
        """
        pass
