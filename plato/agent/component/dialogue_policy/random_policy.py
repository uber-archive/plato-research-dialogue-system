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

from plato.domain.ontology import Ontology
from plato.agent.component.dialogue_policy import dialogue_policy
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator

import random

"""
RandomPolicy is a random walk system policy.
"""


class RandomPolicy(dialogue_policy.DialoguePolicy):

    def __init__(self, args):
        """
        Load the ontology.

        :param args: contain the domain ontology
        """
        super(RandomPolicy, self).__init__()

        if 'ontology' in args:
            ontology = args['ontology']
        else:
            raise ValueError('No ontology provided for RandomPolicy!')

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.intents = ['welcomemsg', 'inform', 'request', 'hello', 'bye',
                        'repeat', 'offer']

    def initialize(self, args):
        """
        Nothing to do here

        :param args:
        :return:
        """
        pass

    def next_action(self, dialogue_state):
        """
        Generate a response given which conditions are met by the current
        dialogue state.

        :param dialogue_state:
        :return:
        """
        # Check for terminal state
        if dialogue_state.is_terminal_state:
            return [DialogueAct('bye', [DialogueActItem('', Operator.EQ, '')])]

        # Select intent
        intent = random.choice(self.intents)

        dact = DialogueAct(intent, [])

        # Select slot
        if intent in ['inform', 'request']:

            if intent == 'inform':
                # The Dialogue Manager will fill the slot's value
                slot = random.choice(
                    self.ontology.ontology['requestable']
                )
            else:
                slot = \
                    random.choice(self.ontology.ontology['system_requestable'])

            dact.params = [DialogueActItem(slot, Operator.EQ, '')]

        return [dact]

    def train(self, data):
        """
        Nothing to do here.

        :param data:
        :return:
        """
        pass

    def restart(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def save(self, path=None):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass

    def load(self, path):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass
