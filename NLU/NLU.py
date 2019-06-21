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
from ConversationalAgent.ConversationalModule import ConversationalModule, ConversationalFrame


class NLU(ConversationalModule):

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def initialize(self, args):
        pass

    def receive_input(self, args):
        # Pass the input - useful for maintaining the flow in ConversationalAgentGeneric
        return args

    @abstractmethod
    def process_input(self, utterance, dialogue_state=None):
        pass

    def generate_output(self, args=None):
        if not args:
            print('WARNING! NLU called without args!')
            return {}

        if isinstance(args, ConversationalFrame):
            args = args.content

        if 'args' in args:
            args = {'utterance': args['args']}

        if 'utterance' not in args:
            print('WARNING! NLU called without utterance!')
            return {}

        dialogue_state = None
        if 'dialogue_state' in args:
            dialogue_state = args['dialogue_state']

        return self.process_input(args['utterance'], dialogue_state)

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path):
        pass
