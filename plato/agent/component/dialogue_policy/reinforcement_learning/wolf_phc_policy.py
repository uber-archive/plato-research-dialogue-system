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

from .. import dialogue_policy, slot_filling_policy
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.domain.ontology import Ontology
from plato.domain.database import DataBase
from plato.agent.component.user_simulator.agenda_based_user_simulator.\
    agenda_based_us import AgendaBasedUS

from copy import deepcopy

import pickle
import random
import pprint
import os.path
import numpy as np

"""
WoLF_PHC_Policy implements a Win or Lose Fast dialogue_policy Hill Climbing 
dialogue dialogue_policy learning algorithm, designed for multi-agent systems.
"""


class WoLFPHCPolicy(dialogue_policy.DialoguePolicy):
    def __init__(self, args):
        """
        Initialize parameters and internal structures

        :param args: dictionary containing the dialogue_policy's settings
        """

        super(WoLFPHCPolicy, self).__init__()

        self.ontology = None
        if 'ontology' in args:
            ontology = args['ontology']

            if isinstance(ontology, Ontology):
                self.ontology = ontology
            elif isinstance(ontology, str):
                self.ontology = Ontology(ontology)
            else:
                raise ValueError('WoLFPHCPolicy Unacceptable '
                                 'ontology type %s ' % ontology)
        else:
            raise ValueError('WoLFPHCPolicy: No ontology provided')

        self.database = None
        if 'database' in args:
            database = args['database']

            if isinstance(database, DataBase):
                self.database = database
            elif isinstance(database, str):
                self.database = DataBase(database)
            else:
                raise ValueError('WoLFPHCPolicy: Unacceptable '
                                 'database type %s ' % database)
        else:
            raise ValueError('WoLFPHCPolicy: No database provided')

        self.agent_role = \
            args['agent_role'] if 'agent_role' in args else 'system'

        self.alpha = args['alpha'] if 'alpha' in args else 0.2
        self.gamma = args['gamma'] if 'gamma' in args else 0.95
        self.epsilon = args['epsilon'] if 'epsilon' in args else 0.95
        self.alpha_decay_rate = \
            args['alpha_decay'] if 'alpha_decay' in args else 0.995
        self.exploration_decay_rate = \
            args['epsilon_decay'] if 'epsilon_decay' in args else 0.9995

        self.IS_GREEDY_POLICY = False

        # TODO: Put these as arguments in the config
        self.d_win = 0.0025
        self.d_lose = 0.01

        self.is_training = False

        self.Q = {}
        self.pi = {}
        self.mean_pi = {}
        self.state_counter = {}

        self.pp = pprint.PrettyPrinter(width=160)     # For debug!

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert dialogue_policy here
            self.warmup_policy = \
                slot_filling_policy.HandcraftedPolicy({
                    'ontology': self.ontology})

        elif self.agent_role == 'user':
            usim_args = dict(
                zip(['ontology', 'database'], [self.ontology, self.database]))
            # Put your user expert dialogue_policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)

        # Sub-case for CamRest
        self.dstc2_acts_sys = self.dstc2_acts_usr = None

        # Plato does not use action masks (rules to define which
        # actions are valid from each state) and so training can
        # be harder. This becomes easier if we have a smaller
        # action set.

        # Does not include inform and request that are modelled together with
        # their arguments
        self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm', 'deny', 'ack',
                               'bye', 'reqmore', 'welcomemsg', 'expl-conf',
                               'select', 'repeat', 'confirm-domain', 'confirm']

        # Does not include inform and request that are modelled together with
        # their arguments
        self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack', 'thankyou',
                               'bye', 'reqmore', 'hello', 'expl-conf',
                               'repeat', 'reqalts', 'restart', 'confirm']

        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        if self.dstc2_acts_sys:
            if self.agent_role == 'system':
                # self.NActions = 5
                # self.NOtherActions = 4
                self.NActions = \
                    len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)

                self.NOtherActions = \
                    len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)

            elif self.agent_role == 'user':
                # self.NActions = 4
                # self.NOtherActions = 5
                self.NActions = \
                    len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots) +\
                    len(self.system_requestable_slots)

                self.NOtherActions = len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)
        else:
            if self.agent_role == 'system':
                self.NActions = \
                    5 + len(self.ontology.ontology['system_requestable']) + \
                    len(self.ontology.ontology['requestable'])
                self.NOtherActions = \
                    4 + 2 * len(self.ontology.ontology['requestable'])

            elif self.agent_role == 'user':
                self.NActions = \
                    4 + 2 * len(self.ontology.ontology['requestable'])
                self.NOtherActions = \
                    5 + len(self.ontology.ontology['system_requestable']) + \
                    len(self.ontology.ontology['requestable'])

        self.statistics = {'supervised_turns': 0, 'total_turns': 0}

    def initialize(self, args):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if 'train' in args:
            self.is_training = bool(args['train'])

            if 'learning_rate' in args:
                self.alpha = float(args['learning_rate'])

            if 'learning_decay_rate' in args:
                self.alpha_decay_rate = float(args['learning_decay_rate'])

            if 'exploration_rate' in args:
                self.epsilon = float(args['exploration_rate'])

            if 'exploration_decay_rate' in args:
                self.exploration_decay_rate = \
                    float(args['exploration_decay_rate'])

            if 'gamma' in args:
                self.gamma = float(args['gamma'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in args:
                    self.warmup_simulator.initialize({args['goal']})
                else:
                    print('WARNING ! No goal provided for WoLF PHC policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

    def restart(self, args):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return: nothing
        """

        if self.agent_role == 'user' and self.warmup_simulator:
            if 'goal' in args:
                self.warmup_simulator.initialize(args)
            else:
                print('WARNING! No goal provided for WoLF PHC policy user '
                      'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the dialogue_policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        state_enc = self.encode_state(state)
        self.statistics['total_turns'] += 1

        if state_enc not in self.pi or \
                (self.is_training and random.random() < self.epsilon):
            if not self.is_training:
                if not self.pi:
                    print(f'\nWARNING! WoLF-PHC pi is empty '
                          f'({self.agent_role}). Did you load the correct '
                          f'file?\n')
                else:
                    print(f'\nWARNING! WoLF-PHC state not found in policy '
                          f'pi ({self.agent_role}).\n')

            if random.random() < 0.35:
                print('--- {0}: Selecting warmup action.'
                      .format(self.agent_role))
                self.statistics['supervised_turns'] += 1

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)

                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()
            else:
                print(
                    '--- {0}: Selecting random action.'.format(self.agent_role)
                )
                return self.decode_action(
                    random.choice(range(0, self.NActions)),
                    self.agent_role == 'system')

        if self.IS_GREEDY_POLICY:
            # Get greedy action
            max_pi = max(self.pi[state_enc][:-1])  # Do not consider 'UNK'
            maxima = \
                [i for i, j in enumerate(self.pi[state_enc]) if j == max_pi]

            # Break ties randomly
            if maxima:
                sys_acts = \
                    self.decode_action(random.choice(maxima),
                                       self.agent_role == 'system')
            else:
                print('--- {0}: Warning! No maximum value identified for '
                      'dialogue policy. Selecting random action.'
                      .format(self.agent_role))

                return self.decode_action(
                    random.choice(range(0, self.NActions)),
                    self.agent_role == 'system')
        else:
            # Sample next action
            sys_acts = \
                self.decode_action(
                    random.choices(range(len(self.pi[state_enc])),
                                   self.pi[state_enc])[0],
                    self.agent_role == 'system')

        return sys_acts

    def encode_state(self, state):
        """
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state encoding
        """

        temp = [int(state.is_terminal_state)]

        temp.append(1) if state.system_made_offer else temp.append(0)

        if self.agent_role == 'user':
            # The user agent needs to know which constraints and requests
            # need to be communicated and which of them
            # actually have.
            if state.user_goal:
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints and \
                                state.user_goal.constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                        if c in state.user_goal.actual_constraints and \
                                state.user_goal.actual_constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                for r in self.requestable_slots:
                    if r in state.user_goal.requests:
                        temp.append(1)
                    else:
                        temp.append(0)

                    if r in state.user_goal.actual_requests and \
                            state.user_goal.actual_requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

            else:
                temp += \
                    [0] * 2*(len(self.informable_slots)-1 +
                             len(self.requestable_slots))

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
        """
        Encode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be encoding another agent's action
        (e.g. a system encoding the previous user act).

        :param actions: actions to be encoded
        :param system: whether the role whose action we are encoding is a
                       'system'
        :return: the encoded action
        """

        # TODO: Handle multiple actions
        if not actions:
            print('WARNING: WoLF-PHC dialogue_policy action encoding called '
                  'with empty actions list (returning -1).')
            return -1

        action = actions[0]

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if action.intent == 'request':
                if action.params[0].slot not in self.system_requestable_slots:
                    return -1

                return len(self.dstc2_acts_sys) + \
                       self.system_requestable_slots.index(
                           action.params[0].slot)

            if action.intent == 'inform':
                if action.params[0].slot not in self.requestable_slots:
                    return -1

                return len(self.dstc2_acts_sys) + \
                       len(self.system_requestable_slots) + \
                       self.requestable_slots.index(action.params[0].slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if action.intent == 'request':
                if action.params[0].slot not in self.requestable_slots:
                    return -1

                return len(self.dstc2_acts_usr) + \
                       self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                if action.params[0].slot not in self.system_requestable_slots:
                    return -1

                return len(self.dstc2_acts_usr) + \
                       len(self.requestable_slots) + \
                       self.system_requestable_slots.index(
                           action.params[0].slot)

        if (self.agent_role == 'system') == system:
            print('WoLF-PHC ({0}) policy action encoder warning: Selecting '
                  'default action (unable to encode: {1})!'
                  .format(self.agent_role, action))
        else:
            print('WoLF-PHC ({0}) policy action encoder warning: Selecting '
                  'default action (unable to encode other agent action: {1})!'
                  .format(self.agent_role, action))

        return -1

    def decode_action(self, action_enc, system=True):
        """
        Decode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be decoding another agent's action
        (e.g. a system decoding the previous user act).

        :param action_enc: action encoding to be decoded
        :param system: whether the role whose action we are decoding is a
                       'system'
        :return: the decoded action
        """

        if system:
            if action_enc < len(self.dstc2_acts_sys):
                return [DialogueAct(self.dstc2_acts_sys[action_enc], [])]
            
            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots):
                return [DialogueAct(
                    'request',
                    [DialogueActItem(
                        self.system_requestable_slots[
                            action_enc-len(self.dstc2_acts_sys)],
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots) + \
                    len(self.requestable_slots):
                index = \
                    action_enc - len(self.dstc2_acts_sys) - \
                    len(self.system_requestable_slots)
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.requestable_slots[index],
                        Operator.EQ,
                        '')])]

        else:
            if action_enc < len(self.dstc2_acts_usr):
                return [DialogueAct(self.dstc2_acts_usr[action_enc], [])]
            
            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots):
                return [DialogueAct(
                    'request',
                    [DialogueActItem(
                        self.requestable_slots[
                            action_enc-len(self.dstc2_acts_usr)],
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots):
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.system_requestable_slots[
                            action_enc-len(self.dstc2_acts_usr) -
                            len(self.requestable_slots)],
                        Operator.EQ,
                        '')])]

        # Default fall-back action
        print('WoLF-PHC dialogue_policy ({0}) policy action decoder warning: '
              'Selecting repeat() action (index: {1})!'
              .format(self.agent_role, action_enc))
        return [DialogueAct('repeat', [])]

    def train(self, dialogues):
        """
        Train the model using WoLF-PHC.

        :param dialogues: a list dialogues, which is a list of dialogue turns
                         (state, action, reward triplets).
        :return:
        """

        if not self.is_training:
            return

        for dialogue in dialogues:
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                new_state_enc = self.encode_state(turn['new_state'])

                role = self.agent_role
                if 'role' in turn:
                    role = turn['role']

                action_enc = \
                    self.encode_action(
                        turn['action'],
                        role == 'system')

                # Skip unrecognised actions
                if action_enc < 0 or turn['action'][0].intent == 'bye':
                    continue

                if state_enc not in self.Q:
                    self.Q[state_enc] = [0] * self.NActions

                if new_state_enc not in self.Q:
                    self.Q[new_state_enc] = [0] * self.NActions

                if state_enc not in self.pi:
                    self.pi[state_enc] = \
                        [float(1/self.NActions)] * self.NActions

                if state_enc not in self.mean_pi:
                    self.mean_pi[state_enc] = \
                        [float(1/self.NActions)] * self.NActions

                if state_enc not in self.state_counter:
                    self.state_counter[state_enc] = 1
                else:
                    self.state_counter[state_enc] += 1

                # Update Q
                self.Q[state_enc][action_enc] = \
                    ((1 - self.alpha) * self.Q[state_enc][action_enc]) + \
                    self.alpha * (
                            turn['reward'] +
                            (self.gamma * np.max(self.Q[new_state_enc])))

                # Update mean dialogue_policy estimate
                for a in range(self.NActions):
                    self.mean_pi[state_enc][a] = \
                        self.mean_pi[state_enc][a] + \
                        ((1.0 / self.state_counter[state_enc]) *
                         (self.pi[state_enc][a] - self.mean_pi[state_enc][a]))

                # Determine delta
                sum_policy = 0.0
                sum_mean_policy = 0.0

                for a in range(self.NActions):
                    sum_policy = sum_policy + (self.pi[state_enc][a] *
                                               self.Q[state_enc][a])
                    sum_mean_policy = \
                        sum_mean_policy + \
                        (self.mean_pi[state_enc][a] * self.Q[state_enc][a])

                if sum_policy > sum_mean_policy:
                    delta = self.d_win
                else:
                    delta = self.d_lose

                # Update dialogue_policy estimate
                max_q_idx = np.argmax(self.Q[state_enc])

                d_plus = delta
                d_minus = ((-1.0) * d_plus) / (self.NActions - 1.0)

                for a in range(self.NActions):
                    if a == max_q_idx:
                        self.pi[state_enc][a] = \
                            min(1.0, self.pi[state_enc][a] + d_plus)
                    else:
                        self.pi[state_enc][a] = \
                            max(0.0, self.pi[state_enc][a] + d_minus)

                # Constrain pi to a legal probability distribution
                sum_pi = sum(self.pi[state_enc])
                for a in range(self.NActions):
                    self.pi[state_enc][a] /= sum_pi

        # Decay learning rate after each episode
        if self.alpha > 0.001:
            self.alpha *= self.alpha_decay_rate

        # Decay exploration rate after each episode
        if self.epsilon > 0.25:
            self.epsilon *= self.exploration_decay_rate

        print('[alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def save(self, path=None):
        """
        Saves the dialogue_policy model to the path provided

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'models/policies/wolf_phc_policy.pkl'
            print('No dialogue_policy file name provided. Using default: {0}'
                  .format(path))

        # If the directory does not exist, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        obj = {'Q': self.Q,
               'pi': self.pi,
               'mean_pi': self.mean_pi,
               'state_counter': self.state_counter,
               'a': self.alpha,
               'e': self.epsilon,
               'g': self.gamma}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        if self.statistics['total_turns'] > 0:
            print('DEBUG > {0} WoLF PHC dialogue_policy supervision ratio: {1}'
                  .format(self.agent_role,
                          float(
                              self.statistics['supervised_turns'] /
                              self.statistics['total_turns'])))

        print(f'DEBUG > {self.agent_role} WoLF PHC policy state space '
              f'size: {len(self.pi)}')

    def load(self, path=None):
        """
        Load the dialogue_policy model from the path provided

        :param path: path to load the model from
        :return:
        """

        if not path:
            print('No dialogue_policy loaded.')
            return

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'Q' in obj:
                        self.Q = obj['Q']
                    if 'pi' in obj:
                        self.pi = obj['pi']
                    if 'mean_pi' in obj:
                        self.mean_pi = obj['mean_pi']
                    if 'state_counter' in obj:
                        self.state_counter = obj['state_counter']
                    if 'a' in obj:
                        self.alpha = obj['a']
                    if 'e' in obj:
                        self.epsilon = obj['e']
                    if 'g' in obj:
                        self.gamma = obj['g']

                    print('WoLF-PHC dialogue_policy loaded from {0}.'
                          .format(path))

            else:
                print('Warning! WoLF-PHC dialogue_policy file %s not found'
                      % path)
        else:
            print('Warning! Unacceptable value for WoLF-PHC policy file name:'
                  ' %s ' % path)
