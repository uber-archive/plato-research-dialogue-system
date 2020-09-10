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

from plato.agent.component.nlu.ludwig_nlu import LudwigNLU
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.domain.ontology import Ontology
import pandas as pd
import string

"""
camrest_nlu is an implementation of nlu for the Cambridge Restaurants domain, 
using a Ludwig model.
"""


class CamRestNLU(LudwigNLU):
    def __init__(self, args):
        """
        Load ontology and database, Ludwig nlu model, and create the static
        iob tag lists, punctuation, and patterns.

        :param args:
        """
        super(CamRestNLU, self).__init__(args)

        self.ontology = None
        self.database = None

        if 'ontology' not in args:
            raise AttributeError('camrest_nlu: Please provide an ontology!')

        ontology = args['ontology']

        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.iob_tag_list = []
        self.dontcare_pattern = []

        self.punctuation_remover = str.maketrans('', '', string.punctuation)

        self.TRAIN_ONLINE = False
        if 'train_online' in args:
            self.TRAIN_ONLINE = bool(args['train_online'])

        self.iob_tag_list = \
            ['B-inform-' +
             slot for slot in self.ontology.ontology['requestable']] + \
            ['I-inform-' +
             slot for slot in self.ontology.ontology['requestable']]

        self.dontcare_pattern = ['anything', 'any', 'i do not care',
                                 'i dont care', 'dont care', 'dontcare',
                                 'it does not matter', 'it doesnt matter',
                                 'does not matter', 'doesnt matter']

    def initialize(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def process_input(self, utterance, dialogue_state=None):
        """
        Query the Ludwig model with the given utterance to obtain predictions
        for IOB tags and intent. Then parse the results and package them into
        a list of dialogue Acts.

        :param utterance: a string, the utterance to be recognised
        :param dialogue_state: the current dialogue state, if available
        :return: a list of dialogue acts containing the recognised intents
        """
        # Pre-process utterance
        utterance = utterance.rstrip().lower()
        utterance = utterance.translate(self.punctuation_remover)

        # Warning: Make sure the same tokenizer that was used to train the
        # model is used during prediction
        result = \
            self.model.predict(
                pd.DataFrame(
                    data={'transcript': [utterance]}), return_type=dict)

        dacts = []

        # Only keep the first act for now
        last_sys_act = dialogue_state.last_sys_acts[0] \
            if dialogue_state and dialogue_state.last_sys_acts else None

        utterance_parts = utterance.split(' ')
        iob_tags = [tag for tag in result['iob']['predictions'][0]]

        for intent in result['intent']['predictions'][0]:
            intent_parts = intent.split('_')
            intent = intent_parts[0]

            if intent == 'request' and len(intent_parts) > 1:
                dacts.append(
                    DialogueAct(
                        intent,
                        [DialogueActItem(intent_parts[1], Operator.EQ, '')]))

            elif intent == 'dontcare' and len(intent_parts) > 1:
                if intent_parts[1] == 'this':
                    if dialogue_state and last_sys_act and last_sys_act.params:
                        dacts.append(
                            DialogueAct(
                                'inform',
                                [DialogueActItem(
                                    last_sys_act.params[0].slot,
                                    Operator.EQ,
                                    'dontcare')]))

                else:
                    dacts.append(
                        DialogueAct(
                            'inform',
                            [DialogueActItem(
                                intent_parts[1],
                                Operator.EQ,
                                'dontcare')]))

            # Exclude acts that take slot-value arguments
            # (will be handled below)
            elif intent not in ['inform', 'confirm', 'offer']:
                dacts.append(DialogueAct(intent, []))

        # If no act was recognised
        if not dacts:
            # Search for tags
            intent = ''
            slot = ''
            value = ''

            # iob_tags is a fixed length but the utterance may be shorter.
            for t in range(len(utterance_parts)):
                # If however we encounter input longer than the IOB tags,
                # we can't handle it.
                if t >= len(iob_tags):
                    print('Warning! camrest_nlu cannot handle such a long '
                          'sequence. Returning partial result.')
                    break

                if iob_tags[t][0] == 'B':
                    # Case where we have B-slot1 I-slot1 I-slot1 B-slot2 ...
                    if value:
                        # Correct offer / inform mis-prediction
                        if slot == 'name':
                            dacts.append(
                                DialogueAct(
                                    'offer',
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        value)]))
                        else:
                            dacts.append(
                                DialogueAct(
                                    'inform',
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        value)]))

                    else:
                        tag_parts = iob_tags[t].split('-')
                        intent = tag_parts[1]
                        slot = tag_parts[2]
                        value = utterance_parts[t]

                elif iob_tags[t][0] == 'I':
                    # Case where nlu doesn't work perfectly well and the first
                    # tag is I-... instead of B-...
                    if not value or not slot or not intent:
                        tag_parts = iob_tags[t].split('-')
                        intent = tag_parts[1]
                        slot = tag_parts[2]
                        value = utterance_parts[t]

                    value += ' ' + utterance_parts[t]

                elif iob_tags[t] == 'O' and value:
                    if slot == 'name':
                        dacts.append(
                            DialogueAct(
                                'offer', [
                                    DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        value)]))
                    else:
                        dacts.append(
                            DialogueAct(
                                'inform',
                                [DialogueActItem(
                                    slot,
                                    Operator.EQ,
                                    value)]))

                    # Reset intent, slot, and value
                    intent = ''
                    slot = ''
                    value = ''

            # Special case for when a tag is at the end of the utterance
            if value and intent:
                # Save the recognised slot value
                dacts.append(
                    DialogueAct(
                        intent,
                        [DialogueActItem(
                            slot,
                            Operator.EQ,
                            value)]))

        # If still no act is recognised
        if not dacts:
            print('WARNING! camrest_nlu did not understand slots or values '
                  'for utterance: {0}\n'.format(utterance))

        return dacts

    def train(self, data):
        """
        Pass to train online.

        :param data: dialogue experience
        :return: nothing
        """
        if self.TRAIN_ONLINE:
            self.train_online(data)

    def train_online(self, data):
        """
        Train the model.

        :param data: dialogue experience
        :return: nothing
        """
        self.model.train_online(pd.DataFrame(data={'transcript': [data]}))

    def save(self, path=None):
        """
        Saves the Ludwig model.

        :param path: path to save the model to
        :return:
        """
        super(CamRestNLU, self).save(path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        super(CamRestNLU, self).load(model_path)
