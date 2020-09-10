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
from plato.agent.component.conversational_module \
    import ConversationalModule, ConversationalFrame

"""
NLU is the abstract parent class for all nlu classes and defines the interface 
that should be followed.
"""


class NLU(ConversationalModule):

    def __init__(self):
        """
        Initialize the internal structures of the nlu
        """
        super(NLU, self).__init__()

    @abstractmethod
    def initialize(self, args):
        """
        Initialize internal structures that need to be reset at the beginning
        of each dialogue

        :param args: dictionary containing initialization arguments
        :return:
        """
        pass

    def receive_input(self, args):
        """
        Function to process input used by the Generic Agents.

        :param args: dictionary containing the input
        :return: the arguments as a default
        """
        # Pass the input - useful for maintaining the flow in
        # ConversationalAgentGeneric
        return args

    @abstractmethod
    def process_input(self, utterance, dialogue_state=None):
        """
        nlu-specific function to process input utterances

        :param utterance: a string, the utterance to be processed
        :param dialogue_state: the current dialogue state
        :return: a list of recognised dialogue acts
        """
        pass

    def generate_output(self, args=None):
        """
        Generate output, used by the Generic Agent.

        :param args:
        :return:
        """
        if not args:
            print('WARNING! nlu.generate_output called without args!')
            return ConversationalFrame([])

        if isinstance(args, ConversationalFrame):
            args = args.content

        if 'args' in args:
            args = {'utterance': args['args']}

        if 'utterance' not in args:
            print('WARNING! nlu.generate_output called without utterance!')
            return ConversationalFrame([])

        dialogue_state = None
        if 'dialogue_state' in args:
            dialogue_state = args['dialogue_state']

        return self.process_input(args['utterance'], dialogue_state)

    @abstractmethod
    def train(self, data):
        """
        Train the nlu

        :param data: dialogue experience
        :return:
        """
        pass

    @abstractmethod
    def save(self, path=None):
        """
        Save trained models into the provided path. Use a default path if no
        path is provided.

        :param path: path to save models into
        :return: nothing
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load trained models from the provided path

        :param path: path to load models from
        :return: nothing
        """
        pass
