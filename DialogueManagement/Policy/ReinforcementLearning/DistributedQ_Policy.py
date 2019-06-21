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
from Ontology.Ontology import Ontology

import pickle
import random
import pprint
import os.path
import calendar
import time


class DistributedQ_Policy(Policy.Policy):
    def __init__(self, ontology, agent_id=0, agent_role='system', domain=None):
        self.alpha = 0.95
        self.gamma = 0.95
        self.epsilon = 0.95

        self.is_training = False

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.Q = {}

        self.pp = pprint.PrettyPrinter(width=160)     # For debug!

        self.warmup_policy = HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        if self.agent_role == 'system':
            self.NActions = 3 + len(self.ontology.ontology['system_requestable']) + len(self.ontology.ontology['requestable'])

        elif self.agent_role == 'user':
            self.NActions = 3 + 2 * len(self.ontology.ontology['requestable'])

        random.seed(calendar.timegm(time.gmtime()))

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
            print('---: Selecting random action.')

            # Return a random action
            return self.decode_action(random.choice(range(0, self.NActions)))
            # if self.agent_role == 'user':
            #     return self.decode_action(random.choice(range(0, self.NActions)))
            # else:
            #     return self.warmup_policy.next_action(state)

        # Return action with maximum Q value from the given state
        sys_acts = self.decode_action(max(self.Q[state_enc], key=self.Q[state_enc].get))

        # TODO Loop over all returned sys acts here
        if sys_acts[0].intent == 'inform' and state.item_in_focus:
            dactitems = []

            for item in state.item_in_focus:
                dactitems.append(DialogueActItem(item, Operator.EQ, state.item_in_focus[item]))

        return sys_acts

    def encode_state(self, state):
        '''
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state encoding
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

        # if state.db_matches_ratio >= 0:
        #     temp += [int(b) for b in format(int(round(state.db_matches_ratio, 2) * 100), '07b')]
        # else:
        #     # If the number is negative (should not happen in general) there will be a minus sign
        #     temp += [int(b) for b in format(int(round(state.db_matches_ratio, 2) * 100), '07b')[1:]]

        if state.db_matches_ratio <= 0.25:
            temp += [1, 0, 0, 0]
        elif state.db_matches_ratio <= 0.5:
            temp += [0, 1, 0, 0]
        elif state.db_matches_ratio <= 0.75:
            temp += [0, 0, 1, 0]
        else:
            temp += [0, 0, 0, 1]

        temp.append(1) if state.system_made_offer else temp.append(0)

        if state.user_acts:
            # If this agent is the system then "user" is a user (hopefully).
            # If this agent is a user then "user" is a system.
            temp += [int(b) for b in format(self.encode_action(state.user_acts, not self.agent_role == 'system'), '05b')]
        else:
            temp += [0, 0, 0, 0, 0]

        if state.last_sys_acts:
            temp += [int(b) for b in format(self.encode_action([state.last_sys_acts[0]], self.agent_role == 'system'), '05b')]
        else:
            temp += [0, 0, 0, 0, 0]

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

    def encode_action(self, actions, system=True):
        '''

        :param actions:
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Distributed Q Policy action encoding called with empty actions list (returning 0).')
            return 0

        action = actions[0]

        if action.intent == 'bye':
            return 0

        if action.intent == 'offer':
            return 1

        if action.intent == 'request':
            if system:
                return 2 + self.ontology.ontology['system_requestable'].index(action.params[0].slot)
            else:
                return 2 + self.ontology.ontology['requestable'].index(action.params[0].slot)

        if action.intent == 'inform':
            if system:
                return 2 + len(self.ontology.ontology['system_requestable']) + \
                       self.ontology.ontology['requestable'].index(action.params[0].slot)
            else:
                return 2 + len(self.ontology.ontology['requestable']) + \
                       self.ontology.ontology['requestable'].index(action.params[0].slot)

        if action.intent == 'hello':
            if system:
                return 2 + len(self.ontology.ontology['system_requestable']) + len(self.ontology.ontology['requestable'])
            else:
                return 2 + 2 * len(self.ontology.ontology['requestable'])

        # Default fall-back action
        print('Distributed Q ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return 0

    def decode_action(self, action_enc, system=True):
        '''

        :param action_enc:
        :return:
        '''

        if action_enc == 0:
            return [DialogueAct('bye', [])]

        if action_enc == 1:
            return [DialogueAct('offer', [])]

        if system:
            if action_enc < 2 + len(self.ontology.ontology['system_requestable']):
                return [DialogueAct('request', [DialogueActItem(self.ontology.ontology['system_requestable'][action_enc-2], Operator.EQ, '')])]

            if action_enc < 2 + len(self.ontology.ontology['system_requestable']) + len(self.ontology.ontology['requestable']):
                index = action_enc - 2 - len(self.ontology.ontology['system_requestable'])
                return [DialogueAct('inform',
                                    [DialogueActItem(self.ontology.ontology['requestable'][index], Operator.EQ, '')])]

            if action_enc == 2 + len(self.ontology.ontology['system_requestable']) + len(self.ontology.ontology['requestable']):
                return [DialogueAct('hello', [])]

        else:
            if action_enc < 2 + len(self.ontology.ontology['requestable']):
                return [DialogueAct('request', [DialogueActItem(self.ontology.ontology['requestable'][action_enc-2], Operator.EQ, '')])]

            if action_enc < 2 + 2 * len(self.ontology.ontology['requestable']):
                return [DialogueAct('inform', [DialogueActItem(self.ontology.ontology['requestable'][action_enc-2-len(self.ontology.ontology['requestable'])], Operator.EQ, '')])]

            if action_enc == 2 + 2 * len(self.ontology.ontology['requestable']):
                return [DialogueAct('hello', [])]

        # Default fall-back action
        print('Distributed Q Policy ({0}) policy action decoder warning: Selecting default action (index: {1})!'.format(self.agent_role, action_enc))
        return [DialogueAct('bye', [])]

    def train(self, dialogues):
        '''
        Train the model using SARSA.

        :param dialogues: a list dialogues, which is a list of dialogue turns (state - action - reward triplets).
        :return:
        '''

        for dialogue in dialogues:
            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                new_state_enc = self.encode_state(turn['new_state'])
                action_enc = self.encode_action(turn['action'])

                if state_enc not in self.Q:
                    self.Q[state_enc] = {}

                if action_enc not in self.Q[state_enc]:
                    self.Q[state_enc][action_enc] = 0#random.gauss(0, 1)

                maxQ = random.gauss(0, 1)
                if new_state_enc in self.Q:
                    maxQ = max(self.Q[new_state_enc].values())

                delta = turn['reward'] + self.gamma * maxQ - self.Q[state_enc][action_enc]

                # Only update Q values that lead to an increase in Q
                if delta > self.Q[state_enc][action_enc]:
                    self.Q[state_enc][action_enc] += self.alpha * delta

            # Decay learning rate after each episode
            if self.alpha > 0.05:
                self.alpha *= 0.9995

            # Decay exploration rate after each episode
            if self.epsilon > 0.05:
                self.epsilon *= 0.9995

            # print('[alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

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
