"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from .. import DialoguePolicy, HandcraftedPolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS

from copy import deepcopy

import pickle
import random
import pprint
import os.path
import numpy as np

"""
WoLF_PHC_Policy implements a Win or Lose Fast DialoguePolicy Hill Climbing 
dialogue policy learning algorithm, designed for multi-agent systems.
"""


class WoLFPHCPolicy(DialoguePolicy.DialoguePolicy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 alpha=0.25, gamma=0.95, epsilon=0.25,
                 alpha_decay=0.9995, epsilon_decay=0.995):
        """
        Initialize parameters and internal structures

        :param ontology: the domain's ontology
        :param database: the domain's database
        :param agent_id: the agent's id
        :param agent_role: the agent's role
        :param alpha: the learning rate
        :param gamma: the discount rate
        :param epsilon: the exploration rate
        :param alpha_decay: the learning rate discount rate
        :param epsilon_decay: the exploration rate discount rate
        """

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay

        self.IS_GREEDY_POLICY = False

        # TODO: Put these as arguments in the config
        self.d_win = 0.0025
        self.d_lose = 0.01

        self.is_training = False

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
            raise ValueError('WoLF PHC DialoguePolicy: Unacceptable database '
                             'type %s ' % database)

        self.Q = {}
        self.pi = {}
        self.mean_pi = {}
        self.state_counter = {}

        self.pp = pprint.PrettyPrinter(width=160)     # For debug!

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert policy here
            self.warmup_policy = \
                HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = dict(
                zip(['ontology', 'database'], [self.ontology, self.database]))
            # Put your user expert policy here
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

    def initialize(self, **kwargs):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

            if 'learning_rate' in kwargs:
                self.alpha = float(kwargs['learning_rate'])

            if 'learning_decay_rate' in kwargs:
                self.alpha_decay = float(kwargs['learning_decay_rate'])

            if 'exploration_rate' in kwargs:
                self.epsilon = float(kwargs['exploration_rate'])

            if 'exploration_decay_rate' in kwargs:
                self.epsilon_decay = float(kwargs['exploration_decay_rate'])

            if 'gamma' in kwargs:
                self.gamma = float(kwargs['gamma'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in kwargs:
                    self.warmup_simulator.initialize({kwargs['goal']})
                else:
                    print('WARNING ! No goal provided for Supervised policy '
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
                print('WARNING! No goal provided for Supervised policy user '
                      'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the policy to produce the agent's response

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

            if random.random() < 0.5:
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
                      'policy. Selecting random action.'
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
            print('WARNING: WoLF-PHC DialoguePolicy action encoding called '
                  'with empty actions list (returning -1).')
            return -1

        action = actions[0]

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_sys) + \
                       self.system_requestable_slots.index(
                           action.params[0].slot)
    
            if action.intent == 'inform':
                return len(self.dstc2_acts_sys) + \
                       len(self.system_requestable_slots) + \
                       self.requestable_slots.index(action.params[0].slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_usr) + \
                       self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
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
        print('WoLF-PHC DialoguePolicy ({0}) policy action decoder warning: '
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
                action_enc = \
                    self.encode_action(
                        turn['action'],
                        self.agent_role == 'system')

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

                # Update mean policy estimate
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

                # Update policy estimate
                max_Q_idx = np.argmax(self.Q[state_enc])

                d_plus = delta
                d_minus = ((-1.0) * d_plus) / (self.NActions - 1.0)

                for a in range(self.NActions):
                    if a == max_Q_idx:
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
            self.alpha *= self.alpha_decay

        # Decay exploration rate after each episode
        if self.epsilon > 0.25:
            self.epsilon *= self.epsilon_decay

        print('[alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def save(self, path=None):
        """
        Saves the policy model to the path provided

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'Models/Policies/wolf_phc_policy.pkl'
            print('No policy file name provided. Using default: {0}'
                  .format(path))

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
            print('DEBUG > {0} WoLF PHC DialoguePolicy supervision ratio: {1}'
                  .format(self.agent_role,
                          float(
                              self.statistics['supervised_turns'] /
                              self.statistics['total_turns'])))

        print(f'DEBUG > {self.agent_role} WoLF PHC DialoguePolicy state space '
              f'size: {len(self.pi)}')

    def load(self, path=None):
        """
        Load the policy model from the path provided

        :param path: path to load the model from
        :return:
        """

        if not path:
            print('No policy loaded.')
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

                    print('WoLF-PHC DialoguePolicy loaded from {0}.'
                          .format(path))

            else:
                print('Warning! WoLF-PHC DialoguePolicy file %s not found'
                      % path)
        else:
            print('Warning! Unacceptable value for WoLF-PHC policy file name:'
                  ' %s ' % path)
