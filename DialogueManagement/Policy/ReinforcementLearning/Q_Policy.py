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

from .. import Policy, HandcraftedPolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from Ontology.Ontology import Ontology
from Ontology.DataBase import DataBase
from copy import deepcopy

import pickle
import random
import pprint
import os.path


class Q_Policy(Policy.Policy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system', domain=None, alpha=0.95, epsilon=0.95,
                 gamma=0.15, alpha_decay=0.995, epsilon_decay=0.995):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.is_training = False
        self.IS_GREEDY_POLICY = True

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase):
            self.database = database
        else:
            raise ValueError('WoLF PHC Policy: Unacceptable database type %s ' % database)

        self.Q = {}

        self.pp = pprint.PrettyPrinter(width=160)     # For debug!

        if self.agent_role == 'system':
            self.warmup_policy = HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = dict(zip(['ontology', 'database'], [self.ontology, self.database]))
            self.warmup_simulator = AgendaBasedUS(usim_args)

        # Extract lists of slots that are frequently used
        self.informable_slots = deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = deepcopy(self.ontology.ontology['system_requestable'])

        self.dstc2_acts = None

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['repeat', 'canthelp', 'affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye',
                               'reqmore', 'hello', 'welcomemsg', 'expl-conf', 'select', 'offer', 'reqalts',
                               'confirm-domain', 'confirm']
        else:
            # Try to identify number of state features
            if domain in ['SlotFilling', 'CamRest']:

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled together with their arguments
                    # self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm', 'deny', 'ack', 'bye', 'reqmore', 'welcomemsg',
                    #                        'expl-conf', 'select', 'repeat', 'confirm-domain', 'confirm']
                    self.dstc2_acts_sys = ['offer', 'welcomemsg']

                    # Does not include inform and request that are modelled together with their arguments
                    # self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye', 'reqmore', 'hello',
                    #                        'expl-conf', 'repeat', 'reqalts', 'restart', 'confirm']
                    self.dstc2_acts_usr = ['hello', 'bye']

                    if self.agent_role == 'system':
                        self.dstc2_acts = self.dstc2_acts_sys

                    elif self.agent_role == 'user':
                        self.dstc2_acts = self.dstc2_acts_usr

                    self.NActions = len(self.dstc2_acts) + len(self.requestable_slots)

                    if self.agent_role == 'system':
                        self.NActions += len(self.system_requestable_slots)
                    else:
                        self.NActions += len(self.requestable_slots)

    def initialize(self, **kwargs):
        '''
        Initialize anything that should not be in __init__

        :return: Nothing
        '''

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

        if 'agent_role' in kwargs:
            self.agent_role = kwargs['agent_role']

    def restart(self, args):
        '''
        Re-initialize relevant parameters / variables at the beginning of each dialogue.

        :return:
        '''

        pass

    def next_action(self, state):
        state_enc = self.encode_state(state)

        if state_enc not in self.Q or (self.is_training and random.random() < self.epsilon):

            if random.random() < 0.5:
                print('---: Selecting warmup action.')

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)
                else:
                    self.warmup_simulator.receive_input(state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()

            else:
                # Return a random action
                print('---: Selecting random action')
                return self.decode_action(random.choice(range(0, self.NActions)), self.agent_role == 'system')

        if self.IS_GREEDY_POLICY:
            # Return action with maximum Q value from the given state
            sys_acts = self.decode_action(max(self.Q[state_enc], key=self.Q[state_enc].get), self.agent_role == 'system')
        # else:
        #     sys_acts = self.decode_action(random.choices(range(0, self.NActions), self.Q[state_enc])[0],
        #                                   self.agent_role == 'system')

        return sys_acts

    def encode_state_old(self, state):
        '''
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state ID
        '''

        temp = []

        temp += [int(b) for b in format(state.turn, '06b')]

        for value in state.slots_filled.values():
            # This contains the requested slot
            temp.append(1) if value else temp.append(0)

        for slot in self.ontology.ontology['requestable']:
            temp.append(1) if slot == state.requested_slot else temp.append(0)

        temp.append(int(state.is_terminal_state))

        # temp.append(1) if state.item_in_focus else temp.append(0)
        # If the agent is a system, then this shows what the top db result is.
        # If the agent is a user, then this shows what information the system has provided
        if state.item_in_focus:
            for slot in self.ontology.ontology['requestable']:
                if slot in state.item_in_focus and state.item_in_focus[slot]:
                    temp.append(1)
                else:
                    temp.append(0)
        else:
            temp += [0] * len(self.ontology.ontology['requestable'])

        if state.db_matches_ratio >= 0:
            temp += [int(b) for b in format(int(round(state.db_matches_ratio, 2) * 100), '07b')]
        else:
            # If the number is negative (should not happen in general) there will be a minus sign
            temp += [int(b) for b in format(int(round(state.db_matches_ratio, 2) * 100), '07b')[1:]]

        temp.append(1) if state.system_made_offer else temp.append(0)

        if state.user_acts:
            temp += [int(b) for b in format(self.encode_action(state.user_acts, False), '05b')]
        else:
            temp += [0, 0, 0, 0, 0]

        if state.last_sys_acts:
            temp += [int(b) for b in format(self.encode_action([state.last_sys_acts[0]]), '04b')]
        else:
            temp += [0, 0, 0, 0]

        # If the agent plays the role of the user it needs access to its own goal
        if state.user_goal:
            for c in self.ontology.ontology['informable']:
                if c in state.user_goal.constraints and state.user_goal.constraints[c].value:
                    temp.append(1)
                else:
                    temp.append(0)

            for r in self.ontology.ontology['requestable']:
                if r in state.user_goal.requests and state.user_goal.requests[r].value:
                    temp.append(1)
                else:
                    temp.append(0)
        else:
            # Just for symmetry, for all other roles append zeros
            temp += [0] * (len(self.ontology.ontology['informable']) + len(self.ontology.ontology['requestable']))

        # Encode state
        state_enc = 0
        for t in temp:
            state_enc = (state_enc << 1) | t

        return state_enc

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
            temp.append(int(state.is_terminal()))
            temp.append(int(state.system_made_offer))

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
        :param system;
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Supervised Policy action encoding called with empty actions list (returning -1).')
            return -1

        action = actions[0]

        slot = None
        if action.params and action.params[0].slot:
            slot = action.params[0].slot

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if slot:
                if action.intent == 'request' and slot in self.system_requestable_slots:
                    return len(self.dstc2_acts_sys) + self.system_requestable_slots.index(slot)

                if action.intent == 'inform' and slot in self.requestable_slots:
                    return len(self.dstc2_acts_sys) + len(self.system_requestable_slots) + self.requestable_slots.index(slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if slot:
                if action.intent == 'request' and slot in self.requestable_slots:
                    return len(self.dstc2_acts_usr) + self.requestable_slots.index(slot)

                if action.intent == 'inform' and slot in self.requestable_slots:
                    return len(self.dstc2_acts_usr) + len(self.requestable_slots) + self.requestable_slots.index(slot)

        # Default fall-back action
        print('Q-Learning ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return -1

    def decode_action(self, action_enc, system=True):
        '''

        :param action_enc:
        :param system:
        :return:
        '''

        if system:
            if action_enc < len(self.dstc2_acts_sys):
                return [DialogueAct(self.dstc2_acts_sys[action_enc], [])]

            if action_enc < len(self.dstc2_acts_sys) + len(self.system_requestable_slots):
                return [DialogueAct('request',
                                    [DialogueActItem(
                                        self.system_requestable_slots[action_enc - len(self.dstc2_acts_sys)],
                                        Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts_sys) + len(self.system_requestable_slots) + len(self.requestable_slots):
                index = action_enc - len(self.dstc2_acts_sys) - len(self.system_requestable_slots)
                return [DialogueAct('inform', [DialogueActItem(self.requestable_slots[index], Operator.EQ, '')])]

        else:
            if action_enc < len(self.dstc2_acts_usr):
                return [DialogueAct(self.dstc2_acts_usr[action_enc], [])]

            if action_enc < len(self.dstc2_acts_usr) + len(self.requestable_slots):
                return [DialogueAct('request',
                                    [DialogueActItem(self.requestable_slots[action_enc - len(self.dstc2_acts_usr)],
                                                     Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts_usr) + 2 * len(self.requestable_slots):
                return [DialogueAct('inform',
                                    [DialogueActItem(self.requestable_slots[action_enc - len(self.dstc2_acts_usr) - len(
                                        self.requestable_slots)], Operator.EQ, '')])]

    def train(self, dialogues):
        '''
        Train the model using SARSA.

        :param dialogues: a list dialogues, which is a list of dialogue turns (state - action - reward triplets).
        :return:
        '''

        for dialogue in dialogues:
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                new_state_enc = self.encode_state(turn['new_state'])
                action_enc = self.encode_action(turn['action'])

                if action_enc < 0:
                    continue

                if state_enc not in self.Q:
                    self.Q[state_enc] = {}

                if action_enc not in self.Q[state_enc]:
                    self.Q[state_enc][action_enc] = 0#random.gauss(0, 1)

                maxQ = 0#random.gauss(0, 1)
                if new_state_enc in self.Q:
                    maxQ = max(self.Q[new_state_enc].values())

                self.Q[state_enc][action_enc] += self.alpha * (turn['reward'] + self.gamma * maxQ - self.Q[state_enc][action_enc])

        # Decay learning rate
        if self.alpha > 0.001:
            self.alpha *= self.alpha_decay

        # Decay exploration rate
        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_decay

        print('Q-Learning: [alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def save(self, path=None):
        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'Models/Policies/q_policy.pkl'
            print('No policy file name provided. Using default: {0}'.format(path))

        obj = {'Q': self.Q,
               'a': self.alpha,
               'e': self.epsilon,
               'g': self.gamma}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        if not path:
            print('No policy loaded.')
            return

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'Q' in obj:
                        self.Q = obj['Q']
                    if 'a' in obj:
                        self.alpha = obj['a']
                    if 'e' in obj:
                        self.epsilon = obj['e']
                    if 'g' in obj:
                        self.gamma = obj['g']

                    print('Q Policy loaded from {0}.'.format(path))

            else:
                print('Warning! Q Policy file %s not found' % path)
        else:
            print('Warning! Unacceptable value for Q policy file name: %s ' % path)
