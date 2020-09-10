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

from scipy.optimize import linprog

from copy import deepcopy

import pickle
import random
import pprint
import os.path
import numpy as np

"""
MinimaxQ_Policy implements the MiniMaxQ algorithm for dialogue policy learning 
in multi-agent environments.
"""


class MinimaxQPolicy(dialogue_policy.DialoguePolicy):
    def __init__(self, args):
        """
        Initialize parameters and internal structures

        :param args: the policy's arguments
        """

        super(MinimaxQPolicy, self).__init__()

        self.ontology = None
        if 'ontology' in args:
            ontology = args['ontology']

            if isinstance(ontology, Ontology):
                self.ontology = ontology
            else:
                raise ValueError('MinimaxQPolicy dialogue_policy Unacceptable '
                                 'ontology type %s ' % ontology)
        else:
            raise ValueError('MinimaxQPolicy dialogue_policy: No ontology '
                             'provided')

        self.database = None
        if 'database' in args:
            database = args['database']

            if isinstance(database, DataBase):
                self.database = database
            else:
                raise ValueError('MinimaxQPolicy policy: Unacceptable '
                                 'database type %s ' % database)
        else:
            raise ValueError('MinimaxQPolicy policy: No database '
                             'provided')

        self.agent_id = args['agent_id'] if 'agent_id' in args else 0
        self.agent_role = \
            args['agent_role'] if 'agent_role' in args else 'system'

        self.alpha = args['alpha'] if 'alpha' in args else 0.2
        self.gamma = args['gamma'] if 'gamma' in args else 0.95
        self.epsilon = args['epsilon'] if 'epsilon' in args else 0.95
        self.alpha_decay = \
            args['alpha_decay'] if 'alpha_decay' in args else 0.995
        self.epsilon_decay = \
            args['epsilon_decay'] if 'epsilon_decay' in args else 0.9995

        self.is_training = False

        self.Q = {}
        self.V = {}
        self.pi = {}

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
            usim_args = \
                dict(
                    zip(['ontology', 'database'],
                        [self.ontology, self.database]))
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
                self.NActions = \
                    len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)

                self.NOtherActions = \
                    len(self.dstc2_acts_usr) + \
                    2 * len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = \
                    len(self.dstc2_acts_usr) + \
                    2 * len(self.requestable_slots)

                self.NOtherActions = \
                    len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)
        else:
            if self.agent_role == 'system':
                self.NActions = \
                    5 + \
                    len(self.ontology.ontology['system_requestable']) + \
                    len(self.ontology.ontology['requestable'])

                self.NOtherActions = \
                    4 + 2 * len(self.ontology.ontology['requestable'])

            elif self.agent_role == 'user':
                self.NActions = \
                    4 + 2 * len(self.ontology.ontology['requestable'])
                self.NOtherActions = \
                    5 + len(self.ontology.ontology['system_requestable']) + \
                    len(self.ontology.ontology['requestable'])

    def initialize(self, args):
        """
        Initialize internal parameters

        :return: Nothing
        """

        if 'is_training' in args:
            self.is_training = bool(args['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in args:
                    self.warmup_simulator.initialize({args['goal']})
                else:
                    print('WARNING ! No goal provided for Minimax Q policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

    def restart(self, args):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return:
        """

        if self.agent_role == 'user' and self.warmup_simulator:
            if 'goal' in args:
                self.warmup_simulator.initialize(args)
            else:
                print('WARNING! No goal provided for Minimax Q policy user '
                      'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the dialogue_policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        state_enc = self.encode_state(state)

        if state_enc not in self.pi or \
                (self.is_training and random.random() < self.epsilon):
            if not self.is_training:
                if not self.pi:
                    print(f'\nWARNING! Minimax Q {self.agent_role} matrix is '
                          f'empty. Did you load the correct file?\n')
                else:
                    print(f'\nWARNING! Minimax Q {self.agent_role} state not '
                          f'found in Q matrix.\n')

            if random.random() < 0.5:
                print('--- {0}: Selecting warmup action.'
                      .format(self.agent_role))

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)

                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()

            else:
                print('--- {0}: Selecting random action.'
                      .format(self.agent_role))
                return self.decode_action(
                    random.choice(
                        range(0, self.NActions)),
                    self.agent_role == 'system')

        # Return best action
        max_pi = max(self.pi[state_enc])
        maxima = [i for i, j in enumerate(self.pi[state_enc]) if j == max_pi]

        # Break ties randomly
        if maxima:
            sys_acts = \
                self.decode_action(
                    random.choice(maxima), self.agent_role == 'system')
        else:
            print('--- {0}: Warning! No maximum value identified for policy. '
                  'Selecting random action.'.format(self.agent_role))
            return self.decode_action(
                random.choice(range(0, self.NActions)),
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

        # If the agent plays the role of the user it needs access to its own
        # goal
        if self.agent_role == 'user':
            # The user agent needs to know which constraints and requests need
            # to be communicated and which of them
            # actually have.
            if state.user_goal:
                found_unanswered_constr = False
                found_unanswered_req = False

                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints and \
                                c not in state.user_goal.actual_constraints:
                            found_unanswered_constr = True
                            break

                for r in self.requestable_slots:
                    if r in state.user_goal.requests and \
                            not state.user_goal.requests[r].value:
                        found_unanswered_req = True
                        break

                temp += \
                    [int(found_unanswered_constr), int(found_unanswered_req)]
            else:
                temp += [0, 0]

        if self.agent_role == 'system':
            temp.append(int(state.is_terminal()))
            temp.append(int(state.system_made_offer))

            for value in state.slots_filled.values():
                # This contains the requested slot
                temp.append(1) if value else temp.append(0)

            for r in self.requestable_slots:
                temp.append(1) if r == state.requested_slot else temp.append(0)

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
            print('WARNING: MinimaxQ policy action encoding called '
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
                       self.requestable_slots.index(
                           action.params[0].slot)
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
            print(
                'MinimaxQ ({0}) policy action encoder warning: Selecting '
                'default action (unable to encode: {1})!'.format(
                    self.agent_role, action))
        else:
            print(
                'MinimaxQ ({0}) policy action encoder warning: Selecting '
                'default action (unable to encode other agent action: '
                '{1})!'.format(self.agent_role, action))

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
                            action_enc - len(self.dstc2_acts_sys)],
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
                            action_enc - len(self.dstc2_acts_usr)],
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots):
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.system_requestable_slots[
                            action_enc - len(self.dstc2_acts_usr) -
                            len(self.requestable_slots)],
                        Operator.EQ,
                        '')])]

        # Default fall-back action
        print(
            'MinimaxQ dialogue_policy ({0}) policy action decoder warning: '
            'Selecting repeat() action '
            '(index: {1})!'.format(self.agent_role, action_enc))
        return [DialogueAct('repeat', [])]

    def train(self, dialogues):
        """
        Train the model using MinimaxQ.

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
                other_action_enc = \
                    self.encode_action(
                        turn['state'].user_acts,
                        self.agent_role != 'system')

                if action_enc < 0 or other_action_enc < 0 or \
                        turn['action'][0].intent == 'bye':
                    continue

                if state_enc not in self.Q:
                    self.Q[state_enc] = []

                    for oa in range(self.NOtherActions):
                        self.Q[state_enc].append([])

                        for a in range(self.NActions):
                            self.Q[state_enc][oa].append(1)

                if state_enc not in self.pi:
                    self.pi[state_enc] = float(1/self.NActions)

                if action_enc not in self.Q[state_enc][other_action_enc]:
                    self.Q[state_enc][other_action_enc][action_enc] = 0

                if new_state_enc not in self.V:
                    self.V[new_state_enc] = 0

                if new_state_enc not in self.pi:
                    self.pi[new_state_enc] = float(1/self.NActions)

                delta = turn['reward'] + self.gamma * self.V[new_state_enc]

                # Only update Q values (actor) that lead to an increase in Q
                # if delta > self.Q[state_enc][other_action_enc][action_enc]:
                self.Q[state_enc][other_action_enc][action_enc] += \
                    self.alpha * delta

                # Update V (critic)
                self.V[state_enc] = self.maxmin(state_enc)

        # Decay learning rate after each episode
        if self.alpha > 0.001:
            self.alpha *= self.alpha_decay

        # Decay exploration rate after each episode
        if self.epsilon > 0.25:
            self.epsilon *= self.epsilon_decay

        print('MiniMaxQ [alpha: {0}, epsilon: {1}]'
              .format(self.alpha, self.epsilon))

    def maxmin(self, state_enc, retry=False):
        """
        Solve the maxmin problem

        :param state_enc: the encoding to the state
        :param retry:
        :return:
        """

        c = np.zeros(self.NActions + 1)
        c[0] = -1
        A_ub = np.ones((self.NOtherActions, self.NActions + 1))
        A_ub[:, 1:] = -np.asarray(self.Q[state_enc])
        b_ub = np.zeros(self.NOtherActions)
        A_eq = np.ones((1, self.NActions + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.NActions

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                      b_eq=b_eq, bounds=bounds)

        if res.success:
            self.pi[state_enc] = res.x[1:]
        elif not retry:
            return self.maxmin(state_enc, retry=True)
        else:
            print("Alert : %s" % res.message)
            if state_enc in self.V:
                return self.V[state_enc]
            else:
                print('Warning, state not in V, returning 0.')
                return 0

        return res.x[0]

    def save(self, path=None):
        """
        Save the model in the path provided

        :param path: path to dave the model to
        :return: nothing
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'models/policies/minimax_q_policy.pkl'
            print('No dialogue_policy file name provided. Using default: {0}'
                  .format(path))

        # If the directory does not exist, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        obj = {'Q': self.Q,
               'V': self.V,
               'pi': self.pi,
               'a': self.alpha,
               'e': self.epsilon,
               'g': self.gamma}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Load the model from the path provided

        :param path: path to load the model from
        :return: nothing
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
                    if 'V' in obj:
                        self.V = obj['V']
                    if 'pi' in obj:
                        self.pi = obj['pi']
                    if 'a' in obj:
                        self.alpha = obj['a']
                    if 'e' in obj:
                        self.epsilon = obj['e']
                    if 'g' in obj:
                        self.gamma = obj['g']

                    print('Q dialogue_policy loaded from {0}.'.format(path))

            else:
                print('Warning! Q dialogue_policy file %s not found' % path)
        else:
            print('Warning! Unacceptable value for Q policy file name: %s '
                  % path)
