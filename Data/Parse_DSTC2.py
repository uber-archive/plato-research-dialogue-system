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


from Data.DataParser import DataParser
from Ontology.Ontology import Ontology
from Ontology.DataBase import DataBase
from NLU.CamRestNLU import CamRestNLU
from NLU.DummyNLU import DummyNLU
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from DialogueStateTracker.DialogueStateTracker import DummyStateTracker
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder
from DialogueManagement.Policy.ReinforcementLearning.RewardFunction import SlotFillingReward

from copy import deepcopy

import os
import re
import csv
import json
import math
import pickle


class Parser(DataParser):
    def __init__(self):
        self.nlu = None
        self.path = None
        self.ontology = None
        self.database = None
        self.DSTrackerSys = None
        self.DSTrackerUsr = None
        self.NLUSys = None
        self.NLUUsr = None
        self.DStateSys = None
        self.prev_sys_act = None
        self.reward = SlotFillingReward()

        self.dstc2_acts = ['repeat', 'canthelp', 'affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye',
                           'reqmore', 'hello', 'welcomemsg', 'expl-conf', 'select', 'offer', 'reqalts',
                           'confirm-domain', 'confirm']

        self.TRAIN_LANG_2_LANG = False

        # Counts of user responses to system actions - {system_act: {total_turns: int, user_act: frequency}}
        self.user_policy_reactive = {}

        # Frequency-based user policy
        self.user_policy = {}

        # Counts of system responses to system actions given the state
        #   {encoding(DStateSys, user_act): {total_turns: int, sys_act: frequency}}
        self.system_policy_reactive = {}

        # Frequency-based system policy
        self.system_policy = {}

        # Keep self.goals
        self.goals = []

    def initialize(self, **kwargs):
        path = ''
        if 'path' in kwargs:
            path = kwargs['path']

        if 'ontology' in kwargs and 'database' in kwargs:
            ontology_path = kwargs['ontology']
            self.ontology = Ontology(ontology_path)

            database_path = kwargs['database']
            self.database = DataBase(database_path)

            if self.TRAIN_LANG_2_LANG:
                # NLU for System in order to collect data to train Language 2 Language Agent
                # (must be trained in a previous pass)
                self.NLUSys = CamRestNLU('Models/CamRestNLU/System/model')
                self.NLUSys.initialize(self.ontology)

                # NLU for User in order to collect data to train Language 2 Language Agent
                # (must be trained in a previous pass)
                self.NLUUsr = CamRestNLU('Models/CamRestNLU/User/model')
                self.NLUUsr.initialize(self.ontology)

            # Track the dialogue state from the system's perspective
            dst_args = dict(zip(['ontology', 'database', 'domain'], [self.ontology, self.database, 'CamRest']))
            self.DSTrackerSys = DummyStateTracker(dst_args)
            
            # Track the dialogue state from the user's perspective
            self.DSTrackerUsr = DummyStateTracker(dst_args)

            self.informable_slots = deepcopy(list(self.ontology.ontology['informable'].keys()))
            self.requestable_slots = deepcopy(self.ontology.ontology['requestable'] + ['this'])
            self.system_requestable_slots = deepcopy(self.ontology.ontology['system_requestable'])

            self.NActions = 4 + len(self.system_requestable_slots) + len(self.requestable_slots)
            self.agent_role = 'system'

        if isinstance(path, str):
            if os.path.exists(os.path.dirname(path)):
                self.path = path

            else:
                raise FileNotFoundError('Invalid data path %s ' % path)
        else:
            raise ValueError('Unacceptable value for data path: %s ' % path)

        model_path = 'Models/CamRestNLU/model'
        metadata_path = 'Data/data/NLU_data.json'

        # self.nlu = CamRestNLU(model_path=model_path, metadata_path=metadata_path)
        nlu_args = dict(zip(['ontology', 'database', 'domain'], [self.ontology, self.database, 'CamRest']))
        self.nlu = DummyNLU(nlu_args)
        self.nlu.initialize(self.ontology)

        self.dontcare_patterns = ['anything', 'any', 'i do not care', 'i dont care', 'dont care',
                                  'it does not matter', 'it doesnt matter', 'does not matter', 'doesnt matter']

        self.bye_patterns = ['bye']

        self.recorder_sys = DialogueEpisodeRecorder()
        self.recorder_usr = DialogueEpisodeRecorder()

    def delexicalize(self, utterance, usr_acts, sys_acts):
        # Do not track 'bye' utterances as those were generated by taking into account the goal the user had at time
        # of data collection - which is not reflected in randomly sampled goals.
        for pattern in self.bye_patterns:
            if pattern in utterance:
                return 'UNK'

        delex_utterance = deepcopy(utterance)

        # Replace synonyms (we are only generating NLG templates here)
        delex_utterance = delex_utterance.replace('center', 'centre')
        delex_utterance = delex_utterance.replace('cheaply', 'cheap')
        delex_utterance = delex_utterance.replace('moderately', 'moderate')
        delex_utterance = delex_utterance.replace('expensively', 'expensive')

        for dc in self.dontcare_patterns:
            delex_utterance = delex_utterance.replace(dc, 'dontcare')

        # Look for a request (there may also be implicit or explicit confirms)
        sys_req_slot = ''
        for sa in sys_acts:
            if sa['act'] == 'request':
                sys_req_slot = sa['slots'][0][1]
                break

        # Replace each slot value with token
        for usr_act in usr_acts:
            for slot in usr_act['slots']:
                if slot[0] == 'this':
                    if sys_req_slot:
                        slot[0] = sys_req_slot
                    else:
                        continue

                if len(slot) > 1:
                    if usr_act['act'] == 'request':
                        if slot[1] == 'addr':
                            delex_utterance = delex_utterance.replace('address', '<' + slot[1].upper() + '>')

                        elif slot[1] == 'phone':
                            delex_utterance = delex_utterance.replace('phone number', '<' + slot[1].upper() + '>')
                            delex_utterance = delex_utterance.replace('phone', '<' + slot[1].upper() + '>')

                        elif slot[1] == 'postcode':
                            delex_utterance = delex_utterance.replace('postcode', '<' + slot[1].upper() + '>')
                            delex_utterance = delex_utterance.replace('post code', '<' + slot[1].upper() + '>')
                            delex_utterance = delex_utterance.replace('postal code', '<' + slot[1].upper() + '>')
                            delex_utterance = delex_utterance.replace('zip code', '<' + slot[1].upper() + '>')

                        else:
                            delex_utterance = delex_utterance.replace(slot[1], '<' + slot[1].upper() + '>')
                    else:
                        delex_utterance = delex_utterance.replace(slot[1], '<' + slot[0].upper() + '>')

        return delex_utterance if '<' in delex_utterance else 'UNK'
        # return delex_utterance

    def delexicalizeNLG(self, transcript, dacts):
        delex_transcript = deepcopy(transcript).lower()
        delex_dacts = ''

        for dact in dacts:

            act = dact['act']
            slots = dact['slots']

            delex_dacts += 'act_' + act + ' '
            if slots:
                for slot in slots:
                    s = slot[0]
                    v = ''

                    if len(slot) > 1:
                        v = slot[1]

                    if s == 'slot':
                        delex_dacts += s

                        if v:
                            delex_dacts += '_' + v + ' '

                    else:

                        # Deal with some special cases
                        if v == 'dontcare':
                            delex_transcript = delex_transcript.replace('it doesnt matter', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('doesnt matter', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('do not care', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('dont care', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('any kind', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('any thing', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('any type', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('any', '<' + s + '>')
                        elif v == 'seafood':
                            delex_transcript = delex_transcript.replace('sea food', '<' + s + '>')
                        elif v == 'moderate':
                            delex_transcript = delex_transcript.replace('moderately', '<' + s + '>')
                        elif v == 'centre':
                            delex_transcript = delex_transcript.replace('center', '<' + s + '>')
                        elif v == 'asian oriental':
                            delex_transcript = delex_transcript.replace('asian oriental', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('oriental', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('asian', '<' + s + '>')
                        elif v == 'north american':
                            delex_transcript = delex_transcript.replace('north american', '<' + s + '>')
                            delex_transcript = delex_transcript.replace('american', '<' + s + '>')

                        if s == 'postcode':
                            idx = delex_transcript.find(v)
                            if idx >= 0:
                                delex_transcript = delex_transcript.replace(delex_transcript[idx:], '<' + s + '>')

                        delex_transcript = delex_transcript.replace(str(v), '<' + s + '>')

                        delex_dacts += '<' + s + '> '

        return [delex_transcript.strip(), delex_dacts.strip()]

    def delexicalizeNLU(self, utterance, sys_act):
        delex_utterance = deepcopy(utterance)

        # Replace synonyms (we are only generating NLG templates here)
        delex_utterance = delex_utterance.replace('center', 'centre')
        delex_utterance = delex_utterance.replace('cheaply', 'cheap')
        delex_utterance = delex_utterance.replace('moderately', 'moderate')
        delex_utterance = delex_utterance.replace('expensively', 'expensive')

        sys_dact = None
        if sys_act:
            if self.prev_sys_act:
                # If there is a slot value
                if self.prev_sys_act['slots'] and self.prev_sys_act['slots'][0]:
                    if self.prev_sys_act['slots'][0][0] != 'slot':
                        sys_dact = DialogueAct(self.prev_sys_act['act'], [DialogueActItem(self.prev_sys_act['slots'][0][0], Operator.EQ, self.prev_sys_act['slots'][0][1])])
                    else:
                        sys_dact = DialogueAct(self.prev_sys_act['act'], [DialogueActItem(self.prev_sys_act['slots'][0][1], Operator.EQ, '')])

        dacts = self.nlu.process_input(utterance, sys_dact)

        # If the utterance cannot be parsed, skip it
        if dacts[0].intent == 'UNK':
            return 'UNK'

        # TODO: This is a quick and dirty way to do this. Revisit soon!

        # Search and replace each value
        for dact in dacts:
            # For inform dacts swap slot values
            if dact.intent == 'inform':
                for item in dact.params:
                    if item.value:
                        delex_utterance = delex_utterance.replace(item.value, '<' + item.slot.upper() + '>')

                    else:
                        if not sys_dact or sys_dact.intent not in ['request', 'impl-conf', 'expl-conf'] or not sys_dact.params:
                            return 'UNK'

                        for dc in self.dontcare_patterns:
                            delex_utterance = delex_utterance.replace(dc, '<' + sys_dact.params[0].slot.upper() + '>')

            # For request dacts swap slot names
            elif dact.intent == 'request':
                for item in dact.params:
                    if item.slot == 'addr':
                        delex_utterance = delex_utterance.replace('address', '<' + item.slot.upper() + '>')

                    elif item.slot == 'phone':
                        delex_utterance = delex_utterance.replace('phone number', '<' + item.slot.upper() + '>')
                        delex_utterance = delex_utterance.replace('phone', '<' + item.slot.upper() + '>')

                    elif item.slot == 'postcode':
                        delex_utterance = delex_utterance.replace('postcode', '<' + item.slot.upper() + '>')
                        delex_utterance = delex_utterance.replace('post code', '<' + item.slot.upper() + '>')
                        delex_utterance = delex_utterance.replace('postal code', '<' + item.slot.upper() + '>')
                        delex_utterance = delex_utterance.replace('zip code', '<' + item.slot.upper() + '>')

                    else:
                        delex_utterance = delex_utterance.replace(item.slot, '<' + item.slot.upper() + '>')

        # Reject any utterances that make it here undelexicalized
        return delex_utterance if '<' in delex_utterance else 'UNK'

    def BIO_tag(self, dialogue_acts, transcript, mode):
        if mode == 'sys' or mode == 'system':
            acts_with_slots = {'inform', 'deny', 'confirm'}
        elif mode == 'usr' or mode == 'user':
            acts_with_slots = {'canthelp', 'select', 'canthelp.exception',
                               'impl-conf',
                               'offer', 'inform', 'expl-conf'}
        else:
            acts_with_slots = {'canthelp', 'select', 'canthelp.exception',
                               'impl-conf',
                               'offer', 'inform', 'expl-conf'}

        curr_intents = set()

        for act in dialogue_acts:
            curr_act = act['act']
            if curr_act == 'request':
                curr_act = '{}_{}'.format(curr_act, act['slots'][0][1])

            # Alex: Correction for 'dontcare'
            if curr_act == 'inform' and act['slots'][0][1] == 'dontcare':
                curr_act = 'dontcare_{}'.format(act['slots'][0][0])

            curr_intents.add(curr_act)

        # word2tag = {}
        firstword2split = {}
        split2tag = {}
        for act in dialogue_acts:
            curr_act = act['act']
            if curr_act in acts_with_slots and len(act['slots']) > 0:
                for curr_slot in act['slots']:
                    slot_name = curr_slot[0]
                    slot_value_split = curr_slot[1].split()

                    splits = firstword2split.get(slot_value_split[0].lower(), [])
                    splits.append(slot_value_split)
                    firstword2split[slot_value_split[0]] = splits

                    split2tag[tuple(slot_value_split)] = '{}-{}'.format(curr_act, slot_name)

        transcript_split = transcript.split()
        iob_tags = []
        len_transcript = len(transcript_split)
        i = 0
        while i < len_transcript:
            word = transcript_split[i].lower()

            if word in firstword2split:
                splits = firstword2split[word]

                for split in splits:

                    full_split_matches = True
                    for j in range(len(split)):
                        if i + j < len(transcript_split):
                            if split[j].lower() != transcript_split[i + j].lower():
                                full_split_matches = False
                                break
                        else:
                            break

                    if full_split_matches:
                        tag = split2tag[tuple(split)]
                        for k in range(len(split)):
                            if k == 0:
                                iob_tags.append('{}-{}'.format('B', tag))
                            else:
                                iob_tags.append('{}-{}'.format('I', tag))

                        i += len(split)
                        break

                    else:
                        i += 1

            else:
                iob_tags.append('O')
                i += 1

        return curr_intents, iob_tags

    def parse_data(self, data_filepath='Data/data/', ontology=None, database=None):
        # Get state encoding length
        temp_dstate = SlotFillingDialogueState({'slots': self.ontology.ontology['system_requestable']})
        temp_dstate.initialize()

        for (dirpath, dirnames, filenames) in os.walk(self.path):
            if not filenames or filenames[0] == '.DS_Store':
                continue

            print('Parsing files at %s' % dirpath)

            # Open files
            with open(dirpath + '/label.json') as label_file, open(dirpath + '/log.json') as log_file:
                label = json.load(label_file)
                log = json.load(log_file)

                prev_usr_act_slot = ''

                # Initialize the dialogue states
                self.DSTrackerSys.initialize()
                DStateSys = deepcopy(self.DSTrackerSys.get_state())
                DStateSys_prev = deepcopy(DStateSys)

                self.DSTrackerUsr.initialize()

                sys_dacts = []
                usr_dacts = []

                prev_usr_act_slots = ''

                # Update user dialogue state with goal information
                goal = Goal()

                constr = {}
                req = {}

                for c in label['task-information']['goal']['constraints']:
                    constr[c[0]] = DialogueActItem(c[0], Operator.EQ, c[1])

                goal.constraints = constr

                for r in label['task-information']['goal']['request-slots']:
                    req[r] = DialogueActItem(r, Operator.EQ, [])

                goal.requests = req

                # Save goal
                self.goals.append(goal)

                self.DSTrackerUsr.update_goal(goal)

                DStateUsr = deepcopy(self.DSTrackerUsr.get_state())
                DStateUsr_prev = deepcopy(DStateUsr)

                sys_turn = {}
                user_turn = {}
                prev_sys_input = ''

                # Parse each dialogue turn
                for t in range(len(label['turns'])):
                    # The system has the initiative and always starts first
                    sys_turn = log['turns'][t]
                    user_turn = label['turns'][t]

                    sys_acts = []
                    sys_slots = {'area': False,
                                 'food': False,
                                 'pricerange': False,
                                 'addr': False,
                                 'name': False,
                                 'phone': False,
                                 'postcode': False}

                    delex_utterance = self.delexicalize(user_turn['transcription'], user_turn['semantics']['json'], sys_turn['output']['dialog-acts'])

                    usr_dacts = []

                    # Get all semantic acts
                    for udact in user_turn['semantics']['json']:
                        # TODO: THIS OVERRIDES PREVIOUS ACTS
                        user_act = udact['act']
                        user_dact = DialogueAct(user_act, [])

                        if user_act == 'bye':
                            user_terminating = True
                            udact['slots'] = [['slot']]

                        elif user_act == 'request':
                            requested = udact['slots'][0][1]

                        # For each slot-value pair
                        for slot in udact['slots']:
                            for sdact in sys_turn['output']['dialog-acts']:
                                sys_act = sdact['act']

                                dact_items = []

                                if sys_act not in sys_acts:
                                    sys_acts.append(sys_act)

                                sys_act_slot = sys_act
                                if sdact['slots']:
                                    if sdact['act'] == 'request':
                                        sys_act_slot += '_' + sdact['slots'][0][1]
                                        ss = sdact['slots'][0][1]
                                        dact_items.append(DialogueActItem(ss, Operator.EQ, ''))
                                    else:
                                        sys_act_slot += '_' + sdact['slots'][0][0]
                                        ss = sdact['slots'][0][0]
                                        dact_items.append(DialogueActItem(ss, Operator.EQ, sdact['slots'][0][1]))

                                    if ss:
                                        sys_slots[ss] = True

                                # Retrieve user act slot
                                if user_act == 'request':
                                    usr_act_slot = user_act + '_' + slot[1]
                                    user_dact.params.append(DialogueActItem(slot[1], Operator.EQ, ''))
                                elif user_act == 'bye':
                                    # Add underscore for consistent parsing later
                                    usr_act_slot = user_act + '_'
                                else:
                                    usr_act_slot = user_act + '_' + slot[0]
                                    user_dact.params.append(DialogueActItem(slot[0], Operator.EQ, slot[1]))

                                # Reactive version of user policy - just reacts to system actions
                                if sys_act_slot not in self.user_policy_reactive:
                                    self.user_policy_reactive[sys_act_slot] = {}
                                    self.user_policy_reactive[sys_act_slot]['total_turns'] = 0
                                    self.user_policy_reactive[sys_act_slot]['dacts'] = {}
                                    self.user_policy_reactive[sys_act_slot]['responses'] = {}

                                if usr_act_slot not in self.user_policy_reactive[sys_act_slot]['dacts']:
                                    self.user_policy_reactive[sys_act_slot]['dacts'][usr_act_slot] = 1
                                    if delex_utterance != 'UNK':
                                        self.user_policy_reactive[sys_act_slot]['responses'][delex_utterance] = 1
                                else:
                                    self.user_policy_reactive[sys_act_slot]['dacts'][usr_act_slot] += 1

                                if delex_utterance != 'UNK':
                                    if delex_utterance not in self.user_policy_reactive[sys_act_slot]['responses']:
                                        self.user_policy_reactive[sys_act_slot]['responses'][delex_utterance] = 1
                                    else:
                                        self.user_policy_reactive[sys_act_slot]['responses'][delex_utterance] += 1

                                self.user_policy_reactive[sys_act_slot]['total_turns'] += 1

                        usr_dacts.append(user_dact)

                    # Update system's policy. Here we use the previous dialogue state, which is where the sys
                    # act was taken from.
                    state_enc_sys = self.encode_state(DStateSys, agent_role='system')

                    # Collapse state encoding to a number
                    # dstate_idx = 0
                    # for t in state_enc_sys:
                    #     dstate_idx = (dstate_idx << 1) | t
                    dstate_idx = ''.join([str(bit) for bit in state_enc_sys])

                    sys_dacts = []

                    if 'output' in sys_turn and 'dialog-acts' in sys_turn['output'] and sys_turn['output']['dialog-acts']:
                        sys_act_slots = ''
                        for sda in sys_turn['output']['dialog-acts']:
                            sys_act_slot = sda['act']
                            sys_dact = DialogueAct(sys_act_slot, [])

                            if sda['slots']:
                                if sys_act_slot == 'request':
                                    sys_act_slot += '_' + sda['slots'][0][1]
                                    sys_dact.params.append(DialogueActItem(sda['slots'][0][1], Operator.EQ, ''))
                                else:
                                    sys_act_slot += '_' + sda['slots'][0][0]
                                    sys_dact.params.append(
                                        DialogueActItem(sda['slots'][0][0], Operator.EQ, sda['slots'][0][1]))

                            sys_dacts.append(sys_dact)
                            sys_act_slots += sys_act_slot + ';'

                        # Trim last ;
                        if sys_act_slots:
                            sys_act_slots = sys_act_slots[:-1]

                        if dstate_idx not in self.system_policy:
                            self.system_policy[dstate_idx] = {}
                            self.system_policy[dstate_idx]['total_turns'] = 0
                            self.system_policy[dstate_idx]['dacts'] = {}
                            self.system_policy[dstate_idx]['responses'] = {}

                        if sys_act_slots not in self.system_policy[dstate_idx]['dacts']:
                            self.system_policy[dstate_idx]['dacts'][sys_act_slots] = 1
                        else:
                            self.system_policy[dstate_idx]['dacts'][sys_act_slots] += 1

                        self.system_policy[dstate_idx]['total_turns'] += 1

                        for prev_usr_act_slot in prev_usr_act_slots.split(';'):
                            if prev_usr_act_slot not in self.system_policy_reactive:
                                self.system_policy_reactive[prev_usr_act_slot] = {}
                                self.system_policy_reactive[prev_usr_act_slot]['total_turns'] = 0
                                self.system_policy_reactive[prev_usr_act_slot]['dacts'] = {}
                                self.system_policy_reactive[prev_usr_act_slot]['responses'] = {}

                            if sys_act_slots not in self.system_policy_reactive[prev_usr_act_slot]['dacts']:
                                self.system_policy_reactive[prev_usr_act_slot]['dacts'][sys_act_slots] = 1
                            else:
                                self.system_policy_reactive[prev_usr_act_slot]['dacts'][sys_act_slots] += 1

                            self.system_policy_reactive[prev_usr_act_slot]['total_turns'] += 1

                    if self.TRAIN_LANG_2_LANG:
                        usr_dacts = self.NLUSys.process_input(user_turn['transcription'], DStateSys)
                        sys_dacts = self.NLUUsr.process_input(sys_turn['output']['transcript'], DStateUsr)

                    # print('\nSystem Dacts: ' + '\n'.join([str(s) for s in sys_dacts]))
                    # print('\nUser Dacts: ' + '\n'.join([str(s) for s in usr_dacts]))

                    # Track the system's dialogue state. This will be relevant in the next turn.
                    # Encode DStateSys
                    DStateSys_prev = deepcopy(DStateSys)
                    self.DSTrackerSys.update_state(usr_dacts)
                    DStateSys = deepcopy(self.DSTrackerSys.update_state_db(self.db_lookup()))


                    # Track the user's dialogue state. This is relevant in the present turn.
                    DStateUsr_prev = deepcopy(DStateUsr)

                    # For Supervised agent this seems to help as it keeps track of the slots filled
                    # self.DSTrackerUsr.update_state(usr_dacts)

                    self.DSTrackerUsr.update_state(sys_dacts)
                    self.DSTrackerUsr.update_state_db(sys_acts=sys_dacts)
                    # self.DSTrackerUsr.update_state_sysact(usr_dacts)
                    DStateUsr = deepcopy(self.DSTrackerUsr.get_state())

                    # Encode DStateSys
                    state_enc_usr = self.encode_state(DStateUsr, agent_role='user')

                    # Collapse state encoding to a number
                    # dstate_idx = 0
                    # for t in state_enc_usr:
                    #     dstate_idx = (dstate_idx << 1) | t
                    dstate_idx = ''.join([str(bit) for bit in state_enc_usr])

                    # Agent to agent version of user policy - based on state

                    # Note: It may be duplicate effort to re-iterate over the user acts. For now I let it be
                    # for clarity / debugging.

                    # Disregard empty user actions
                    if usr_dacts:
                        usr_act_slots = ''
                        for ud in usr_dacts:
                            usr_act_slot = ud.intent

                            if ud.params:
                                usr_act_slot += '_' + ud.params[0].slot

                            usr_act_slots += usr_act_slot + ';'

                        # Trim last ;
                        usr_act_slots = usr_act_slots[:-1]

                        if dstate_idx not in self.user_policy:
                            self.user_policy[dstate_idx] = {}
                            self.user_policy[dstate_idx]['total_turns'] = 0
                            self.user_policy[dstate_idx]['dacts'] = {}
                            self.user_policy[dstate_idx]['responses'] = {}

                        if usr_act_slots not in self.user_policy[dstate_idx]['dacts']:
                            self.user_policy[dstate_idx]['dacts'][usr_act_slots] = 1
                        else:
                            self.user_policy[dstate_idx]['dacts'][usr_act_slots] += 1

                        self.user_policy[dstate_idx]['total_turns'] += 1

                        prev_usr_act_slots = usr_act_slots

                    if not sys_acts:
                        continue

                    # Record experience

                    if bool(label['task-information']['feedback']['success']):
                        # reward, success = self.reward.calculate(DStateSys_prev, sys_dacts,
                        #                                         goal=DStateSys_prev.user_goal,
                        #                                         agent_role='system')

                        # Hack for Supervised policies that cannot handle multiple actions
                        if prev_usr_act_slot and 'request' in prev_usr_act_slot and len(sys_dacts) > 1 and\
                                sys_dacts[0].intent == 'offer':
                            ssdacts = deepcopy(sys_dacts[1:])
                        else:
                            ssdacts = deepcopy(sys_dacts)

                        self.recorder_sys.record(DStateSys_prev, DStateSys, ssdacts,
                                                 input_utterance=prev_sys_input,
                                                 output_utterance=sys_turn['output']['transcript'],
                                                 # int(label['task-information']['feedback']['success']),
                                                 # bool(label['task-information']['feedback']['success'])
                                                 # reward=reward,
                                                 # success=success
                                                 reward=-1,
                                                 success=False,
                                                 custom=str(sys_turn['output']['dialog-acts'])
                                                 )

                        # reward, success = self.reward.calculate(DStateUsr_prev, usr_dacts,
                        #                                         goal=DStateUsr_prev.user_goal,
                        #                                         agent_role='user')

                        self.recorder_usr.record(DStateUsr_prev, DStateUsr, deepcopy(usr_dacts),
                                                 input_utterance=sys_turn['output']['transcript'],
                                                 output_utterance=user_turn['transcription'],
                                                 # int(label['task-information']['feedback']['success']),
                                                 # bool(label['task-information']['feedback']['success'])
                                                 # reward=reward,
                                                 # success=success
                                                 reward=-1,
                                                 success=False,
                                                 custom=str(user_turn['semantics']['json'])
                                                 )

                        prev_sys_input = sys_turn['input']['live']['asr-hyps'][0]['asr-hyp']

                # Record final experience (end of dialogue) - prev & current states will be the same here.
                if bool(label['task-information']['feedback']['success']):
                    # reward, success = self.reward.calculate(DStateSys_prev, sys_dacts,
                    #                                         goal=DStateSys_prev.user_goal,
                    #                                         force_terminal=True,
                    #                                         agent_role='system')

                    # Hack for Supervised policies that cannot handle multiple actions
                    if prev_usr_act_slot and 'request' in prev_usr_act_slot and len(sys_dacts) > 1 and \
                            sys_dacts[0].intent == 'offer':
                        ssdacts = deepcopy(sys_dacts[1:])
                    else:
                        ssdacts = deepcopy(sys_dacts)

                    self.recorder_sys.record(DStateSys_prev, DStateSys, ssdacts,
                                             20 if label['task-information']['feedback']['success'] else -20,
                                             bool(label['task-information']['feedback']['success']),
                                             input_utterance=prev_sys_input,
                                             output_utterance='',
                                             # reward=reward,
                                             # success=success,
                                             force_terminate=True)

                    # reward, success = self.reward.calculate(DStateUsr_prev, usr_dacts,
                    #                                         goal=DStateUsr_prev.user_goal,
                    #                                         force_terminal=True,
                    #                                         agent_role='user')

                    self.recorder_usr.record(DStateUsr_prev, DStateUsr, deepcopy(usr_dacts),
                                             20 if label['task-information']['feedback']['success'] else -20,
                                             bool(label['task-information']['feedback']['success']),
                                             input_utterance=sys_turn['output']['transcript'],
                                             output_utterance=user_turn['transcription'],
                                             # reward=reward,
                                             # success=success,
                                             force_terminate=True)

        # Save data for LU and LG
        print('\n\nProcessing NLU and NLG files...\n')

        # System's NLG and User's NLU
        with open(data_filepath+'DSTC2_NLG_sys.csv', 'a') as sys_nlg_file, \
                open(data_filepath + 'DSTC2_NLU_usr.csv', 'a') as usr_nlu_file:
            sys_nlg_writer = csv.writer(sys_nlg_file, delimiter=',')

            # Write header
            sys_nlg_writer.writerow(['dialog-acts_str', 'transcript', 'dialog-acts_delex', 'transcript_delex'])

            usr_nlu_writer = csv.writer(usr_nlu_file, delimiter=',')

            # Write header
            usr_nlu_writer.writerow(['transcript', 'intent', 'iob'])

            for dialogue in self.recorder_sys.dialogues:
                for sys_turn in dialogue:
                    utterance = sys_turn['output_utterance']
                    dact_str = sys_turn['custom']

                    if not utterance or not dact_str:
                        continue

                    try:
                        dialogue_acts = json.loads(dact_str.replace("'", '"'))

                    except:
                        tmp = dact_str.replace("\'", '"')
                        tmp = re.sub(r'([a-z])"([a-z])', r"\1'\2", tmp)
                        dialogue_acts = json.loads(tmp)

                    intent, iob_tags = self.BIO_tag(dialogue_acts, utterance, 'usr')
                    [delex_transcript, delex_dacts] = self.delexicalizeNLG(utterance, dialogue_acts)

                    sys_nlg_writer.writerow([dact_str, utterance, delex_dacts, delex_transcript])
                    usr_nlu_writer.writerow([utterance, ' '.join(intent), ' '.join(iob_tags)])

                    # Special cases for System NLG
                    # If there is act_offer <name> in delex_dacts add another pair with act_offer <name> removed
                    if "act_offer <name> " in delex_dacts:
                        delex_dacts = delex_dacts.replace('act_offer <name> ', '')
                        delex_transcript = delex_transcript.replace('of <name> ', '')
                        delex_transcript = delex_transcript.replace('<name> ', 'it ')

                        sys_nlg_writer.writerow([dact_str, utterance, delex_dacts, delex_transcript])

                # Another special case for System NLG: System side has no bye()
                sys_nlg_writer.writerow(["[{'slots': [], 'act': 'bye'}]", 'good bye', 'act_bye', 'good bye'])

        # System's NLU and User's NLG
        with open(data_filepath + 'DSTC2_NLU_sys.csv', 'a') as sys_nlu_file, \
                open(data_filepath + 'DSTC2_NLG_usr.csv', 'a') as usr_nlg_file:
            sys_nlu_writer = csv.writer(sys_nlu_file, delimiter=',')

            # Write header
            sys_nlu_writer.writerow(['dialog-acts_str', 'transcript', 'dialog-acts_delex', 'transcript_delex'])

            usr_nlg_writer = csv.writer(usr_nlg_file, delimiter=',')

            # Write header
            usr_nlg_writer.writerow(['transcript', 'intent', 'iob'])

            for dialogue in self.recorder_usr.dialogues:
                for usr_turn in dialogue:
                    utterance = usr_turn['output_utterance']
                    dact_str = usr_turn['custom']

                    if not utterance or not dact_str:
                        continue

                    try:
                        dialogue_acts = json.loads(dact_str.replace("'", '"'))

                    except:
                        tmp = dact_str.replace("\'", '"')
                        tmp = re.sub(r'([a-z])"([a-z])', r"\1'\2", tmp)
                        dialogue_acts = json.loads(tmp)

                    intent, iob_tags = self.BIO_tag(dialogue_acts, utterance, 'sys')
                    [delex_transcript, delex_dacts] = self.delexicalizeNLG(utterance, dialogue_acts)

                    usr_nlg_writer.writerow([dact_str, utterance, delex_dacts, delex_transcript])
                    sys_nlu_writer.writerow([utterance, ' '.join(intent), ' '.join(iob_tags)])

        print('Done!\n')

        # Normalize frequencies for user policy
        for sa in self.user_policy_reactive:
            for ua in self.user_policy_reactive[sa]['dacts']:
                self.user_policy_reactive[sa]['dacts'][ua] /= self.user_policy_reactive[sa]['total_turns']

            for ur in self.user_policy_reactive[sa]['responses']:
                self.user_policy_reactive[sa]['responses'][ur] /= self.user_policy_reactive[sa]['total_turns']

        # Normalize frequencies for system policy
        for ua in self.system_policy_reactive:
            for sa in self.system_policy_reactive[ua]['dacts']:
                self.system_policy_reactive[ua]['dacts'][sa] /= self.system_policy_reactive[ua]['total_turns']

            for sr in self.system_policy_reactive[ua]['responses']:
                self.system_policy_reactive[ua]['responses'][sr] /= self.system_policy_reactive[ua]['total_turns']

        # Normalize frequencies for user calculated policy
        for state in self.user_policy:
            for ua in self.user_policy[state]['dacts']:
                self.user_policy[state]['dacts'][ua] /= self.user_policy[state]['total_turns']

        # Normalize frequencies for system policy
        for state in self.system_policy:
            for sa in self.system_policy[state]['dacts']:
                self.system_policy[state]['dacts'][sa] /= self.system_policy[state]['total_turns']

            # for ur in self.system_policy[sa]['responses']:
            #     self.system_policy[sa]['responses'][ur] /= self.system_policy[sa]['total_turns']

    def db_lookup(self):
        # TODO: Add check to assert if each slot in DStateSys.slots_filled actually exists in the schema.

        DStateSys = self.DSTrackerSys.get_state()

        # Query the database
        cursor = self.database.SQL_connection.cursor()
        sql_command = " SELECT * FROM CamRestaurants "

        args = ''
        prev_arg = False
        for slot in DStateSys.slots_filled:
            if DStateSys.slots_filled[slot] and DStateSys.slots_filled[slot] != 'dontcare':
                if prev_arg:
                    args += " AND "

                args += slot + " = \"" + DStateSys.slots_filled[slot] + "\""
                prev_arg = True

        if args:
            sql_command += " WHERE " + args + ";"

        cursor.execute(sql_command)
        db_result = cursor.fetchall()

        if db_result:
            # Get the slot names
            slot_names = [i[0] for i in cursor.description]
            result = []
            for db_item in db_result:
                result.append(dict(zip(slot_names, db_item)))

            # Calculate entropy of requestable slot values in results
            entropies = dict.fromkeys(self.ontology.ontology['system_requestable'])
            value_probabilities = {}

            # Count the values
            for req_slot in self.ontology.ontology['system_requestable']:
                value_probabilities[req_slot] = {}

                for db_item in result:
                    if db_item[req_slot] not in value_probabilities[req_slot]:
                        value_probabilities[req_slot][db_item[req_slot]] = 1
                    else:
                        value_probabilities[req_slot][db_item[req_slot]] += 1

            # Calculate probabilities
            for slot in value_probabilities:
                for value in value_probabilities[slot]:
                    value_probabilities[slot][value] /= len(result)

            # Calculate entropies
            for slot in entropies:
                entropies[slot] = 0

                if slot in value_probabilities:
                    for value in value_probabilities[slot]:
                        entropies[slot] += value_probabilities[slot][value] * math.log(value_probabilities[slot][value])

                entropies[slot] = -entropies[slot]

            return result, entropies

        # Failed to retrieve anything
        print('Warning! Database call retrieved zero results.')
        return ['empty'], {}

    def save(self, path):
        # Save data
        self.recorder_sys.save(path+'/DSTC2_system')
        self.recorder_sys.save(path+'/DSTC2_user')

        # Pickle the self.user_policy_reactive and the responses
        obj = {'policy': self.user_policy_reactive}

        with open('Models/UserSimulator/user_policy_reactive.pkl', 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        # Pickle the self.system_policy_reactive and the responses
        obj = {'policy': self.system_policy_reactive}

        with open('Models/CamRestPolicy/system_policy_reactive.pkl', 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        # Pickle the self.user_policy
        obj = {'policy': self.user_policy}

        with open('Models/UserSimulator/user_policy.pkl', 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        # Pickle the self.system_policy and the responses
        obj = {'policy': self.system_policy}

        with open('Models/CamRestPolicy/sys_policy.pkl', 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        # Pickle self.goals
        obj = {'goals': self.goals}

        with open('Models/UserSimulator/goals.pkl', 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    # These encodings are the same that the SupervisedPolicy uses, for performance comparison.
    def encode_state(self, state, agent_role='system'):
        '''
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state encoding
        '''

        temp = []

        temp.append(int(state.is_terminal_state))

        temp.append(1) if state.system_made_offer else temp.append(0)

        # If the agent plays the role of the user it needs access to its own goal
        if agent_role == 'user':
            # The user agent needs to know which constraints and requests need to be communicated and which of them
            # actually have.
            if state.user_goal:
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints and state.user_goal.constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                # Put these features separately from the above
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.actual_constraints and state.user_goal.actual_constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                for r in self.requestable_slots:
                    if r in state.user_goal.requests:  # and state.user_goal.requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

                # Put these features separately from the above
                for r in self.requestable_slots:
                    if r in state.user_goal.actual_requests and state.user_goal.actual_requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

            else:
                temp += [0] * 2 * (len(self.informable_slots) - 1 + len(self.requestable_slots))

        if agent_role == 'system':
            for value in state.slots_filled.values():
                # This contains the requested slot
                temp.append(1) if value else temp.append(0)

            for r in self.requestable_slots:
                temp.append(1) if r == state.requested_slot else temp.append(0)

        return temp

    def encode_action(self, actions, system=True):
        '''

        :param actions: The list of actions to encode
        :param system: If the actions were taken by a 'system' or a 'user' agent
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Parse DSTC2 action encoding called with empty actions list (returning 0).')
            return 0

        action = actions[0]

        if self.dstc2_acts and action.intent in self.dstc2_acts:
            return self.dstc2_acts.index(action.intent)

        if action.intent == 'request':
            if system:
                return len(self.dstc2_acts) + self.system_requestable_slots.index(action.params[0].slot)
            else:
                return len(self.dstc2_acts) + self.requestable_slots.index(action.params[0].slot)

        if action.intent == 'inform':
            if system:
                return len(self.dstc2_acts) + len(self.system_requestable_slots) + self.requestable_slots.index(action.params[0].slot)
            else:
                return len(self.dstc2_acts) + len(self.requestable_slots) + self.requestable_slots.index(action.params[0].slot)

        # Default fall-back action
        print('Parse DSTC2 ({0}) action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return 0

    def encode_action_dstc(self, actions, system=True):
        if not actions:
            print('WARNING: Parse DSTC2 action encoding called with empty actions list (returning 0).')
            return 0

        action = actions[0]

        if self.dstc2_acts and action.intent in self.dstc2_acts:
            return self.dstc2_acts.index(action.intent)

        if action.intent == 'request':
            if system and action.params[0].slot in self.system_requestable_slots:
                return len(self.dstc2_acts) + self.system_requestable_slots.index(action.params[0].slot)

            elif action.params[0].slot in self.requestable_slots:
                return len(self.dstc2_acts) + self.requestable_slots.index(action.params[0].slot)

        if action.intent == 'inform' and action.params[0].slot in self.requestable_slots:
            if system:
                return len(self.dstc2_acts) + len(self.system_requestable_slots) + self.requestable_slots.index(action.params[0].slot)

            else:
                return len(self.dstc2_acts) + len(self.requestable_slots) + self.requestable_slots.index(action.params[0].slot)

        # Default fall-back action
        print('Parse DSTC2 action encoder warning: Selecting default action (unable to encode: {0})!'.format(action))
        return 0
