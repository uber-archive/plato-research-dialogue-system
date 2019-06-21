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

from DialogueManagement.Policy import Policy
from DialogueManagement.Policy.HandcraftedPolicy import HandcraftedPolicy
from Ontology import Ontology, DataBase
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState, DummyDialogueState
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from copy import deepcopy

import pickle
import numpy as np
import random
import os


class Calculated_Policy(Policy.Policy):

    def __init__(self, ontology, database, agent_id=0, agent_role='system', domain=None):
        super(Calculated_Policy, self).__init__()

        self.agent_id = agent_id
        self.agent_role = agent_role

        # True for greedy, False for stochastic
        self.IS_GREEDY_POLICY = True

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Calculated Policy: Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase.DataBase):
            self.database = database
        else:
            raise ValueError('Calculated Policy: Unacceptable database type %s ' % database)

        self.policy_path = None

        self.policy = None

        # Extract lists of slots that are frequently used
        self.informable_slots = deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = deepcopy(self.ontology.ontology['requestable'] + ['this'])  # , 'signature'])
        self.system_requestable_slots = deepcopy(self.ontology.ontology['system_requestable'])

        self.dstc2_acts = None

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions.
            # Does not include inform and request that are modelled together with their arguments
            self.dstc2_acts = ['offer', 'canthelp', 'affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye', 'reqmore',
                               'hello', 'welcomemsg', 'expl-conf', 'select', 'repeat', 'reqalts', 'confirm-domain',
                               'confirm']
        else:
            # Try to identify number of state features
            if domain in ['CamRest', 'SFH', 'SlotFilling']:
                DState = SlotFillingDialogueState({'slots': self.ontology.ontology['system_requestable']})

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled together with their arguments
                    self.dstc2_acts = ['offer', 'canthelp', 'affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye',
                                       'reqmore', 'hello', 'welcomemsg', 'expl-conf', 'select', 'repeat', 'reqalts',
                                       'confirm-domain', 'confirm']

            else:
                print('Warning! Domain has not been defined. Using Dummy Dialogue State')
                DState = DummyDialogueState({'slots': self.ontology.ontology['system_requestable']})

            DState.initialize()

        if self.dstc2_acts:
            self.NActions = len(self.dstc2_acts) + len(self.requestable_slots)

            if self.agent_role == 'system':
                self.NActions += len(self.system_requestable_slots)

            elif self.agent_role == 'user':
                self.NActions += len(self.requestable_slots)
        else:
            if self.agent_role == 'system':
                self.NActions = 4 + len(self.system_requestable_slots) + len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = 3 + 2 * len(self.requestable_slots)

    def initialize(self, **kwargs):
        if 'policy_path' in kwargs:
            self.policy_path = kwargs['policy_path']

            # Re-load policy since policy_path may have changed
            self.load()

    def restart(self, args):
        pass

    def next_action(self, state):
        sys_acts = []

        state_enc = self.encode_state(state)

        if state not in self.policy:
            # TODO: Reactive policy. Fix this properly.
            state_enc = ''
            if state.user_acts:
                for sa in state.user_acts:
                    state_enc = sa.intent

                    # This is due to the DM rules and to be fair to the other policies
                    if sa.intent == 'offer':
                        state_enc += '_name'

                    elif sa.params:
                        state_enc += '_' + sa.params[0].slot

                    state_enc += ';'

                state_enc = state_enc[:-1]

        if state_enc in self.policy:
            sys_actions = list(self.policy[state_enc]['dacts'].keys())
            probs = [self.policy[state_enc]['dacts'][i] for i in sys_actions]

            sys_act_slots = deepcopy(random.choices(sys_actions, weights=probs)[0]).split(';')

            for sys_act_slot in sys_act_slots:
                if not sys_act_slot:
                    # Skip empty sys_act_slot strings (e.g. case where there is ; at the end: inform_food;inform_area;)
                    continue

                sys_act = DialogueAct('UNK')
                sys_act_slot_parts = sys_act_slot.split('_')

                sys_act.intent = sys_act_slot_parts[0]

                if len(sys_act_slot_parts) > 1:
                    sys_act.params = [DialogueActItem(sys_act_slot_parts[1], Operator.EQ, '')]

                if sys_act.intent == 'offer':
                    sys_act.params = []

                elif sys_act.intent == 'canthelp.exception':
                    sys_act.intent = 'canthelp'

                sys_acts.append(sys_act)

        else:
            print(f'Warning! {self.agent_role} Calculated Policy: state not found, selecting random action.')
            sys_act = DialogueAct('UNK')

            if self.agent_role == 'system':
                sys_act.intent = random.choice(['welcomemsg', 'inform', 'request'])
            elif self.agent_role == 'user':
                sys_act.intent = random.choice(['hello', 'inform', 'request'])
            else:
                sys_act.intent = random.choice(['bye', 'inform', 'request'])

            sys_acts.append(sys_act)

        return sys_acts

    def train(self, dialogues):
        pass

    def encode_state(self, state):
        '''
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state encoding
        '''

        temp = []

        temp.append(int(state.is_terminal_state))

        temp.append(1) if state.system_made_offer else temp.append(0)

        # If the agent plays the role of the user it needs access to its own goal
        if self.agent_role == 'user':
            # The user agent needs to know which constraints and requests need to be communicated and which of them
            # actually have.
            if state.user_goal:
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints and state.user_goal.constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                        if c in state.user_goal.actual_constraints and state.user_goal.actual_constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                for r in self.requestable_slots:
                    if r in state.user_goal.requests:  # and state.user_goal.requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

                    if r in state.user_goal.actual_requests and state.user_goal.actual_requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

            else:
                temp += [0] * 2 * (len(self.informable_slots) - 1 + len(self.requestable_slots))

        if self.agent_role == 'system':
            for value in state.slots_filled.values():
                # This contains the requested slot
                temp.append(1) if value else temp.append(0)

            for r in self.requestable_slots:
                temp.append(1) if r == state.requested_slot else temp.append(0)

        # Encode state
        state_enc = 0
        for t in temp:
            state_enc = (state_enc << 1) | t

        return state_enc

    def encode_action(self, actions, system=True):
        '''

        :param actions:
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Calculated Policy action encoding called with empty actions list (returning 0).')
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
        print('Calculated ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return 0

    def decode_action(self, action_enc, system=True):
        '''

        :param action_enc:
        :return:
        '''

        if action_enc < len(self.dstc2_acts):
            return self.dstc2_acts[action_enc]

        if system:
            if action_enc < len(self.dstc2_acts) + len(self.system_requestable_slots):
                return [DialogueAct('request', [DialogueActItem(self.system_requestable_slots[action_enc-len(self.dstc2_acts)], Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts) + len(self.system_requestable_slots) + len(self.requestable_slots):
                index = action_enc - len(self.dstc2_acts) - len(self.system_requestable_slots)
                return [DialogueAct('inform', [DialogueActItem(self.requestable_slots[index], Operator.EQ, '')])]

        else:
            if action_enc < len(self.dstc2_acts) + len(self.requestable_slots):
                return [DialogueAct('request', [DialogueActItem(self.requestable_slots[action_enc-len(self.dstc2_acts)], Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts) + 2 * len(self.requestable_slots):
                return [DialogueAct('inform', [DialogueActItem(self.requestable_slots[action_enc-len(self.dstc2_acts)-len(self.requestable_slots)], Operator.EQ, '')])]

        # Default fall-back action
        print('Calculated Policy ({0}) policy action decoder warning: Selecting default action (index: {1})!'.format(self.agent_role, action_enc))
        return [DialogueAct('bye', [])]

    def save(self, path=None):
        pass

    def load(self, path=None):
        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'Models/CamRestPolicy/sys_policy.pkl' + self.agent_role + '_' + str(self.agent_id)

        self.policy = None
        if isinstance(pol_path, str):
            if os.path.isfile(pol_path):
                with open(pol_path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'policy' in obj:
                        self.policy = obj['policy']

                    print(f'Calculated Policy {self.agent_role} policy loaded.')

            else:
                print(f'Warning! {self.agent_role} Policy file {pol_path} not found')
        else:
            print(f'Warning! Unacceptable value for policy file name: {pol_path}')



