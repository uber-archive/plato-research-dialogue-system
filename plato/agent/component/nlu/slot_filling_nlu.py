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

from plato.agent.component.nlu.nlu import NLU
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.domain.ontology import Ontology
from plato.domain.database import DataBase, SQLDataBase, JSONDataBase

import string
import re

"""
SlotFillingNLU is a basic implementation of nlu designed to work for 
Slot-Filling applications. The purpose of this class is to provide a quick way 
of running Conversational Agents, sanity checks, and to aid debugging.
"""


class SlotFillingNLU(NLU):
    def __init__(self, args):
        """
        Load the ontology and database, create some patterns, and preprocess
        the database so that we avoid some computations at runtime.

        :param args:
        """
        super(SlotFillingNLU, self).__init__()

        self.ontology = None
        self.database = None
        self.requestable_only_slots = None
        self.slot_values = None

        if 'ontology' not in args:
            raise AttributeError('SlotFillingNLU: Please provide ontology!')
        if 'database' not in args:
            raise AttributeError('SlotFillingNLU: Please provide database!')

        ontology = args['ontology']
        database = args['database']

        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        if database:
            if isinstance(database, DataBase):
                self.database = database

            elif isinstance(database, str):
                if database[-3:] == '.db':
                    self.database = SQLDataBase(database)
                elif database[-5:] == '.json':
                    self.database = JSONDataBase(database)
                else:
                    raise ValueError('Unacceptable database type %s '
                                     % database)
            else:
                raise ValueError('Unacceptable database type %s ' % database)

        # In order to work for simulated users, we need access to possible
        # values of requestable slots
        cursor = self.database.SQL_connection.cursor()

        print('SlotFillingNLU: Preprocessing Database... '
              '(do not use SlotFillingNLU with large databases!)')

        # Get table name
        db_result = cursor.execute("select * from sqlite_master "
                                   "where type = 'table';").fetchall()
        if db_result and db_result[0] and db_result[0][1]:
            db_table_name = db_result[0][1]

            self.slot_values = {}

            # Get all entries in the database
            all_items = cursor.execute("select * from " +
                                       db_table_name + ";").fetchall()

            i = 0

            for item in all_items:
                # Get column names
                slot_names = [i[0] for i in cursor.description]

                result = dict(zip(slot_names, item))

                for slot in result:
                    if slot in ['id', 'signature', 'description']:
                        continue

                    if slot not in self.slot_values:
                        self.slot_values[slot] = []

                    if result[slot] not in self.slot_values[slot]:
                        self.slot_values[slot].append(result[slot])

                i += 1
                if i % 2000 == 0:
                    print(f'{float(i/len(all_items))*100}% done')

            print('SlotFillingNLU: Done!')
        else:
            raise ValueError(
                'dialogue Manager cannot specify Table Name from database '
                '{0}'.format(self.database.db_file_name))

        # For this SlotFillingNLU create a list of requestable-only to reduce
        # computational load
        self.requestable_only_slots = \
            [slot for slot in self.ontology.ontology['requestable']
             if slot not in self.ontology.ontology['informable']] + ['name']

        self.bye_pattern = ['bye', 'goodbye', 'exit', 'quit', 'stop']

        self.hi_pattern = ['hi', 'hello']

        self.welcome_pattern = ['welcome', 'how may i help']

        self.deny_pattern = ['no']

        self.negate_pattern = ['is not']

        self.confirm_pattern = ['so is']

        self.repeat_pattern = ['repeat']

        self.ack_pattern = ['ok']

        self.restart_pattern = ['start over']

        self.affirm_pattern = ['yes']

        self.thankyou_pattern = ['thank you']

        self.reqmore_pattern = ['tell me more']

        self.expl_conf_pattern = ['alright']

        self.reqalts_pattern = ['anything else']

        self.select_pattern = ['you prefer']

        self.dontcare_pattern = ['anything', 'any', 'i do not care',
                                 'i dont care', 'dont care', 'dontcare',
                                 'it does not matter', 'it doesnt matter',
                                 'does not matter', 'doesnt matter']

        self.request_pattern = ['what', 'which', 'where', 'how', 'would']

        self.cant_help_pattern = ['can not help', 'cannot help', 'cant help']

        punctuation = string.punctuation.replace('$', '')
        punctuation = punctuation.replace('_', '')
        punctuation = punctuation.replace('.', '')
        punctuation = punctuation.replace('&', '')
        punctuation = punctuation.replace('-', '')
        punctuation += '.'
        self.punctuation_remover = str.maketrans('', '', punctuation)

    def initialize(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def process_input(self, utterance, dialogue_state=None):
        """
        Process the utterance and see if any intent pattern matches.

        :param utterance: a string, the utterance to be recognised
        :param dialogue_state: the current dialogue state, if available
        :return: a list of recognised dialogue acts
        """
        dacts = []
        dact = DialogueAct('UNK', [])

        # TODO: Remove this once nlg is updated
        utterance = utterance.replace('<PAD>', '')

        if not utterance:
            return [dact]

        last_sys_act = \
            dialogue_state.last_sys_acts[0] \
            if dialogue_state and dialogue_state.last_sys_acts else None

        utterance = utterance.rstrip().lower()
        utterance = utterance.translate(self.punctuation_remover)

        # Replace synonyms
        utterance = utterance.replace('location', 'area')
        utterance = utterance.replace('part of town', 'area')
        utterance = utterance.replace('center', 'centre')
        utterance = utterance.replace('cheaply', 'cheap')
        utterance = utterance.replace('moderately', 'moderate')
        utterance = utterance.replace('expensively', 'expensive')
        utterance = utterance.replace('address', 'addr')
        utterance = utterance.replace('telephone', 'phone')
        utterance = utterance.replace('postal code', 'postcode')
        utterance = utterance.replace('post code', 'postcode')
        utterance = utterance.replace('zip code', 'postcode')
        utterance = utterance.replace('price range', 'pricerange')

        # First check if the user doesn't care
        if last_sys_act and last_sys_act.intent in ['request', 'expl-conf']:
            for p in self.dontcare_pattern:
                # Look for exact matches here only (i.e. user just says
                # 'i don't care')
                if p == utterance:
                    dact.intent = 'inform'
                    dact.params.append(
                        DialogueActItem(
                            last_sys_act.params[0].slot,
                            Operator.EQ,
                            'dontcare'))

                    return [dact]

        # Look for slot keyword and corresponding value
        words = utterance.split(' ')

        for p in self.ack_pattern:
            if p == utterance:
                dact.intent = 'ack'
                break

        for p in self.deny_pattern:
            if p == utterance:
                dact.intent = 'deny'
                break

        for p in self.affirm_pattern:
            if p == utterance:
                dact.intent = 'affirm'
                break

        # Check for dialogue ending
        for p in self.bye_pattern:
            match = re.search(r'\b{0}\b'.format(p), utterance)
            if match:
                dact.intent = 'bye'
                break

        # Search for 'welcome' first because it may contain 'hello'
        if dact.intent == 'UNK':
            for p in self.welcome_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'welcomemsg'
                    break

        if dact.intent == 'UNK':
            for p in self.hi_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'hello'
                    break

        if dact.intent == 'UNK':
            for p in self.reqalts_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'reqalts'
                    break

        if dact.intent == 'UNK':
            for p in self.reqmore_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'reqmore'
                    break

        if dact.intent == 'UNK':
            for p in self.repeat_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'repeat'
                    break

        if dact.intent == 'UNK':
            for p in self.restart_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'restart'
                    break

        if dact.intent == 'UNK':
            for p in self.thankyou_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'thankyou'
                    break

        if dact.intent == 'UNK':
            for p in self.request_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'request'
                    break

        if dact.intent == 'UNK':
            for p in self.select_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'select'
                    break

        if dact.intent == 'UNK':
            for p in self.confirm_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'confirm'
                    break

        if dact.intent == 'UNK':
            for p in self.expl_conf_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'expl-conf'
                    break

        if dact.intent == 'UNK':
            for p in self.cant_help_pattern:
                match = re.search(r'\b{0}\b'.format(p), utterance)
                if match:
                    dact.intent = 'canthelp'
                    dact.params = []
                    return [dact]

        if dact.intent == 'UNK':
            dact.intent = 'inform'

            # Check if there is no information about the slot
            if 'no info' in utterance:
                # Search for a slot name in the utterance
                for slot in self.ontology.ontology['requestable']:
                    if slot in utterance:
                        dact.params.append(
                            DialogueActItem(slot, Operator.EQ, 'no info'))
                        return [dact]

                # Else try to grab slot name from the other agent's
                # previous act
                if last_sys_act and \
                        last_sys_act.intent in ['request', 'expl-conf']:
                    dact.params.append(
                        DialogueActItem(
                            last_sys_act.params[0].slot,
                            Operator.EQ,
                            'dontcare'))

                    return [dact]

                # Else do nothing, and see if anything matches below

        if dact.intent in ['inform', 'request']:
            for word in words:
                # Check for requests. Requests for informable slots are
                # captured below
                if word in self.requestable_only_slots:
                    if dact.intent == 'request':
                        dact.params.append(
                            DialogueActItem(word, Operator.EQ, ''))
                        break

                    elif word != 'name':
                        if 'is' not in utterance and 'its' not in utterance:
                            dact.intent = 'request'
                            dact.params.append(
                                DialogueActItem(word, Operator.EQ, ''))
                            break

                        # For any other kind of intent, we have no way of
                        # determining the slot's value, since such
                        # information is not in the ontology.

                # Check for informs (most intensive)
                if word in self.ontology.ontology['informable']:

                    # If a request intent has already been recognized,
                    # do not search for slot values
                    if dact.intent == 'request':
                        dact.params.append(
                            DialogueActItem(word, Operator.EQ, ''))
                        break

                    found = False

                    for p in self.ontology.ontology['informable'][word]:
                        match = re.search(r'\b{0}\b'.format(p), utterance)
                        if match:
                            if word == 'name':
                                dact.intent = 'offer'
                            else:
                                dact.intent = 'inform'

                            dact.params.append(
                                DialogueActItem(word, Operator.EQ, p))
                            found = True
                            break

                    if not found:
                        # Search for dontcare (e.g. I want any area)
                        for p in self.dontcare_pattern:
                            match = re.search(r'\b{0}\b'.format(p), utterance)
                            if match:
                                dact.intent = 'inform'
                                dact.params.append(
                                    DialogueActItem(
                                        word,
                                        Operator.EQ,
                                        'dontcare'))

                                return [dact]

                        dact.intent = 'request'
                        dact.params.append(
                            DialogueActItem(word, Operator.EQ, ''))

        # If nothing was recognised, do an even more brute-force search
        if dact.intent in ['UNK', 'inform'] and not dact.params:
            slot_vals = self.ontology.ontology['informable']
            if self.slot_values:
                slot_vals = self.slot_values

            for slot in slot_vals:
                for value in slot_vals[slot]:
                    if value and \
                            value.lower().translate(self.punctuation_remover) \
                            in utterance:
                        if slot == 'name':
                            dact.intent = 'offer'

                        di = DialogueActItem(slot, Operator.EQ, value)

                        if di not in dact.params:
                            dact.params.append(di)

        # Check if something has been missed (e.g. utterance is dont care and
        # there's no previous sys act)
        if dact.intent == 'inform':
            # Check to see that all slots have an identified value
            if dact.params:
                for dact_item in dact.params:
                    if not dact_item.slot or not dact_item.value:
                        dact.params.remove(dact_item)

                        if not dact.params:
                            dact.intent = 'UNK'
                            break

                    # Else, break up the inform into several acts
                    elif dact_item.slot == 'name':
                        dacts.append(DialogueAct('offer', [dact_item]))
                    else:
                        dacts.append(DialogueAct('inform', [dact_item]))
            else:
                # Try to see if the utterance refers to a slot in the
                # requestable ones (e.g. 'I prefer any phone')
                for slot in self.ontology.ontology['requestable']:
                    if slot in utterance:
                        # We can only handle 'dontcare' kind of values here,
                        # as we do not know values of req. slots.
                        for p in self.dontcare_pattern:
                            match = re.search(r'\b{0}\b'.format(p), utterance)
                            if match:
                                dact.params = \
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        'dontcare')]

                dact.intent = 'UNK'

        else:
            dacts.append(dact)

        return dacts

    def train(self, data):
        """
        Nothing to train.

        :param data:
        :return:
        """
        pass

    def save(self, path=None):
        """
        Nothing to save.

        :param path:
        :return:
        """
        pass

    def load(self, path):
        """
        Nothing to load.

        :param path:
        :return:
        """
        pass
