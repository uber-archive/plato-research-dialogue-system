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

from NLU.LudwigNLU import LudwigNLU
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Ontology.Ontology import Ontology
import pandas as pd
import string


class CamRestNLU(LudwigNLU):
    def __init__(self, args):
        super(CamRestNLU, self).__init__(args)

        self.ontology = None
        self.database = None

        if 'ontology' not in args:
            raise AttributeError('DummyNLU: Please provide ontology!')

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

        self.iob_tag_list = ['B-inform-' + slot for slot in self.ontology.ontology['requestable']] + \
                            ['I-inform-' + slot for slot in self.ontology.ontology['requestable']]

        self.dontcare_pattern = ['anything', 'any', 'i do not care', 'i dont care', 'dont care', 'dontcare',
                                 'it does not matter', 'it doesnt matter', 'does not matter', 'doesnt matter']

    def initialize(self, args):
        pass

    def process_input(self, utterance, dialogue_state=None):
        # Pre-process utterance
        utterance = utterance.rstrip().lower()
        utterance = utterance.translate(self.punctuation_remover)

        # Warning: Make sure the same tokenizer that was used to train the model is used during prediction
        result = self.model.predict(pd.DataFrame(data={'transcript': [utterance]}), return_type=dict)

        dacts = []

        # Only keep the first act for now
        last_sys_act = dialogue_state.last_sys_acts[0] if dialogue_state and dialogue_state.last_sys_acts else None

        utterance_parts = utterance.split(' ')
        iob_tags = [tag for tag in result['iob']['predictions'][0]]

        for intent in result['intent']['predictions'][0]:
            intent_parts = intent.split('_')
            intent = intent_parts[0]

            # First check if the user doesn't care
            # if last_sys_act and last_sys_act.intent in ['request', 'expl-conf'] and last_sys_act.params:
            #     dontcare_found = False
            #
            #     for p in self.dontcare_pattern:
            #         # Look for exact matches here only (i.e. user just says 'i don't care')
            #         if p == utterance:
            #             dacts.append(DialogueAct('inform', [DialogueActItem(last_sys_act.params[0].slot, Operator.EQ, 'dontcare')]))
            #             dontcare_found = True
            #
            #             break
            #
            #     if dontcare_found:
            #         continue

            if intent == 'request' and len(intent_parts) > 1:
                dacts.append(DialogueAct(intent, [DialogueActItem(intent_parts[1], Operator.EQ, '')]))

            elif intent == 'dontcare' and len(intent_parts) > 1:
                if intent_parts[1] == 'this':
                    if dialogue_state and last_sys_act and last_sys_act.params:
                        dacts.append(DialogueAct('inform', [DialogueActItem(last_sys_act.params[0].slot, Operator.EQ, 'dontcare')]))

                else:
                    dacts.append(DialogueAct('inform', [DialogueActItem(intent_parts[1], Operator.EQ, 'dontcare')]))

            # Exclude acts that take slot-value arguments (will be handled below)
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
                # If however we encounter input longer than the IOB tags, we can't handle it.
                if t >= len(iob_tags):
                    print('Warning! CamRestNLU cannot handle such a long sequence. Returning partial result.')
                    break

                if iob_tags[t][0] == 'B':
                    # Case where we have B-slot1 I-slot1 I-slot1 B-slot2 ...
                    if value:
                        # Correct offer / inform mis-prediction
                        if slot == 'name':
                            dacts.append(DialogueAct('offer', [DialogueActItem(slot, Operator.EQ, value)]))
                        else:
                            dacts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, value)]))

                    else:
                        tag_parts = iob_tags[t].split('-')
                        intent = tag_parts[1]
                        slot = tag_parts[2]
                        value = utterance_parts[t]

                elif iob_tags[t][0] == 'I':
                    # Case where NLU doesn't work perfectly well and the first tag is I-... instead of B-...
                    if not value or not slot or not intent:
                        tag_parts = iob_tags[t].split('-')
                        intent = tag_parts[1]
                        slot = tag_parts[2]
                        value = utterance_parts[t]

                    value += ' ' + utterance_parts[t]

                elif iob_tags[t] == 'O' and value:
                    if slot == 'name':
                        dacts.append(DialogueAct('offer', [DialogueActItem(slot, Operator.EQ, value)]))
                    else:
                        dacts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, value)]))

                    # Reset intent, slot, and value
                    intent = ''
                    slot = ''
                    value = ''

            # Special case for when a tag is at the end of the utterance
            if value and intent:
                # Save the recognised slot value
                dacts.append(DialogueAct(intent, [DialogueActItem(slot, Operator.EQ, value)]))

        # If still no act is recognised
        if not dacts:
            print('WARNING! CamRestNLU did not understand slots or values for utterance: {0}\n IOB: {1}\n'.format(utterance, iob_tags))
            # dacts.append(DialogueAct('UNK', []))

            # Attempt to recover
            # print('Attempting to recover...')
            # dacts += self.backup_nlu.process_input(utterance, [last_sys_act])

        return dacts

    def train(self, data):
        if self.TRAIN_ONLINE:
            self.train_online(data)

    def train_online(self, data):
        self.model.train_online(pd.DataFrame(data={'transcript': [data]}))

    def save(self, path=None):
        pass

    def load(self, path):
        pass

