"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from ConversationalAgent.ConversationalModule import ConversationalModule


class GenericTextInputHelper(ConversationalModule):
    """
    This class is a helper for listening to text input for Generic agents.
    """

    def __init__(self, args):
        super(GenericTextInputHelper, self).__init__()

    def initialize(self, args):
        pass

    def receive_input(self, args):
        pass

    def generate_output(self, args=None):

        # Listen for input
        utterance = input('USER > ')

        return utterance

    def train(self, dialogue_episodes):
        pass

    def load(self, path):
        pass

    def save(self):
        pass
