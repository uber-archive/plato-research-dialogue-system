"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from .. import DialoguePolicy
from DialogueManagement.DialoguePolicy.HandcraftedPolicy import \
    HandcraftedPolicy
from Domain import Ontology, DataBase
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from copy import deepcopy

import numpy as np
import random
import os
import pickle

"""
ReinforcePolicy implements the REINFORCE algorithm for dialogue policy 
learning.
"""


class ReinforcePolicy(DialoguePolicy.DialoguePolicy):

    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 domain=None, alpha=0.2, epsilon=0.95,
                 gamma=0.95, alpha_decay=0.995, epsilon_decay=0.9995):
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

        super(ReinforcePolicy, self).__init__()

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.IS_GREEDY = False

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase.DataBase):
            self.database = database
        else:
            raise ValueError('Reinforce DialoguePolicy: Unacceptable '
                             'database type %s ' % database)

        self.policy_path = None

        self.weights = None
        self.sess = None

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay_rate = alpha_decay
        self.exploration_decay_rate = epsilon_decay

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert policy here
            self.warmup_policy = HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = \
                dict(
                    zip(['ontology', 'database'],
                        [self.ontology, self.database]))
            # Put your user expert policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)

        self.tf_scope = "policy_" + self.agent_role + '_' + str(self.agent_id)

        # Default value
        self.is_training = True

        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['inform', 'offer', 'request', 'canthelp',
                               'affirm', 'negate', 'deny', 'ack', 'thankyou',
                               'bye', 'reqmore', 'hello', 'welcomemsg',
                               'expl-conf', 'select', 'repeat', 'reqalts',
                               'confirm-domain', 'confirm']
        else:
            # Try to identify number of state features
            if domain in ['CamRest', 'SFH', 'SlotFilling']:
                d_state = \
                    SlotFillingDialogueState(
                        {'slots': self.system_requestable_slots})

                # Plato does not use action masks (rules to define which
                # actions are valid from each state) and so training can
                # be harder. This becomes easier if we have a smaller
                # action set.

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled
                    # together with their arguments
                    self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm',
                                           'deny', 'ack', 'bye',
                                           'reqmore', 'welcomemsg',
                                           'expl-conf', 'select', 'repeat',
                                           'confirm-domain', 'confirm']

                    # Does not include inform and request that are modelled
                    # together with their arguments
                    self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack',
                                           'thankyou', 'bye',
                                           'reqmore', 'hello', 'expl-conf',
                                           'repeat', 'reqalts', 'restart',
                                           'confirm']

            else:
                print('Warning! Domain has not been defined. Using '
                      'Slot-Filling Dialogue State')
                d_state = \
                    SlotFillingDialogueState({'slots': self.informable_slots})

            d_state.initialize()
            self.NStateFeatures = len(self.encode_state(d_state))

            print('Reinforce DialoguePolicy {0} automatically determined '
                  'number of state features: {1}'
                  .format(self.agent_role, self.NStateFeatures))

        if domain == 'CamRest' and self.dstc2_acts_sys:
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
                    3 + len(self.system_requestable_slots) + \
                    len(self.requestable_slots)

                self.NOtherActions = \
                    2 + len(self.requestable_slots) +\
                    len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = \
                    2 + len(self.requestable_slots) + \
                    len(self.requestable_slots)

                self.NOtherActions = \
                    3 + len(self.system_requestable_slots) + \
                    len(self.requestable_slots)

        print('Reinforce {0} DialoguePolicy Number of Actions: {1}'
              .format(self.agent_role, self.NActions))

    def initialize(self, **kwargs):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in kwargs:
                    self.warmup_simulator.initialize({kwargs['goal']})
                else:
                    print('WARNING ! No goal provided for Reinforce policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

        if 'policy_path' in kwargs:
            self.policy_path = kwargs['policy_path']

        if 'learning_rate' in kwargs:
            self.alpha = kwargs['learning_rate']
            
        if 'learning_decay_rate' in kwargs:
            self.alpha_decay_rate = kwargs['learning_decay_rate']

        if 'discount_factor' in kwargs:
            self.gamma = kwargs['discount_factor']

        if 'exploration_rate' in kwargs:
            self.alpha = kwargs['exploration_rate']

        if 'exploration_decay_rate' in kwargs:
            self.exploration_decay_rate = kwargs['exploration_decay_rate']

        if self.weights is None:
            self.weights = np.random.rand(self.NStateFeatures, self.NActions)

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
                print('WARNING! No goal provided for Reinforce policy user '
                      'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        if self.is_training and random.random() < self.epsilon:
            if random.random() < 0.75:
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
                    self.agent_role == "system")

        # Probabilistic policy: Sample from action wrt probabilities
        probs = self.calculate_policy(self.encode_state(state))

        if any(np.isnan(probs)):
            print('WARNING! NAN detected in action probabilities! Selecting '
                  'random action.')
            return self.decode_action(
                random.choice(range(0, self.NActions)),
                self.agent_role == "system")

        if self.IS_GREEDY:
            # Get greedy action
            max_pi = max(probs)
            maxima = [i for i, j in enumerate(probs) if j == max_pi]

            # Break ties randomly
            if maxima:
                sys_acts = \
                    self.decode_action(
                        random.choice(maxima), self.agent_role == 'system')
            else:
                print(
                    f'--- {self.agent_role}: Warning! No maximum value '
                    f'identified for policy. Selecting random action.')
                return self.decode_action(
                    random.choice(
                        range(0, self.NActions)),
                    self.agent_role == 'system')
        else:
            # Pick from top 3 actions
            top_3 = np.argsort(-probs)[0:2]
            sys_acts = \
                self.decode_action(
                    random.choices(
                        top_3, probs[top_3])[0], self.agent_role == 'system')

        return sys_acts

    @staticmethod
    def softmax(x):
        """
        Calculates the softmax of x

        :param x: a number
        :return: the softmax of the number
        """
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    @staticmethod
    def softmax_gradient(x):
        """
        Calculates the gradient of the softmax

        :param x: a number
        :return: the gradient of the softmax
        """
        x = np.asarray(x)
        x_reshaped = x.reshape(-1, 1)
        return np.diagflat(x_reshaped) - np.dot(x_reshaped, x_reshaped.T)

    def calculate_policy(self, state):
        """
        Calculates the probabilities for each action from the given state

        :param state: the current dialogue state
        :return: probabilities of actions
        """
        dot_prod = np.dot(state, self.weights)
        exp_dot_prod = np.exp(dot_prod)
        return exp_dot_prod / np.sum(exp_dot_prod)

    def train(self, dialogues):
        """
        Train the policy network

        :param dialogues: dialogue experience
        :return: nothing
        """
        # If called by accident
        if not self.is_training:
            return

        for dialogue in dialogues:
            discount = self.gamma

            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            rewards = [t['reward'] for t in dialogue]
            norm_rewards = \
                (rewards - np.mean(rewards)) / (np.std(rewards) + 0.000001)

            for (t, turn) in enumerate(dialogue):
                act_enc = self.encode_action(turn['action'],
                                             self.agent_role == 'system')
                if act_enc < 0:
                    continue

                state_enc = self.encode_state(turn['state'])

                if len(state_enc) != self.NStateFeatures:
                    raise ValueError(f'Reinforce DialoguePolicy '
                                     f'{self.agent_role} mismatch in state'
                                     f'dimensions: State Features: '
                                     f'{self.NStateFeatures} != State '
                                     f'Encoding Length: {len(state_enc)}')

                # Calculate the gradients

                # Call policy again to retrieve the probability of the
                # action taken
                probabilities = self.calculate_policy(state_enc)

                softmax_deriv = self.softmax_gradient(probabilities)[act_enc]
                log_policy_grad = softmax_deriv / probabilities[act_enc]
                gradient = \
                    np.asarray(
                        state_enc)[None, :].transpose().dot(
                        log_policy_grad[None, :])
                gradient = np.clip(gradient, -1.0, 1.0)

                # Train policy
                self.weights += \
                    self.alpha * gradient * norm_rewards[t] * discount
                self.weights = np.clip(self.weights, -1, 1)

                discount *= self.gamma

        if self.alpha > 0.01:
            self.alpha *= self.alpha_decay_rate

        if self.epsilon > 0.5:
            self.epsilon *= self.exploration_decay_rate

        print(f'REINFORCE train, alpha: {self.alpha}, epsilon: {self.epsilon}')

    def encode_state(self, state):
        """
        Encodes the dialogue state into a vector.

        :param state: the state to encode
        :return: int - a unique state encoding
        """

        temp = [int(state.is_terminal_state), int(state.system_made_offer)]

        if self.agent_role == 'user':
            # The user agent needs to know which constraints and requests
            # need to be communicated and which of them
            # actually have.
            if state.user_goal:
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints:
                            temp.append(1)
                        else:
                            temp.append(0)

                for c in self.informable_slots:
                    if c != 'name':
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

                for r in self.requestable_slots:

                    if r in state.user_goal.actual_requests and \
                            state.user_goal.actual_requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

            else:
                temp += [0] * 2 * (len(self.informable_slots) - 1 +
                                   len(self.requestable_slots))

        if self.agent_role == 'system':
            for value in state.slots_filled.values():
                # This contains the requested slot
                temp.append(1) if value else temp.append(0)

            for r in self.requestable_slots:
                temp.append(1) if r == state.requested_slot else temp.append(0)

        return temp

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
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Reinforce DialoguePolicy action encoding called '
                  'with empty actions list (returning 0).')
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
                       self.requestable_slots.index(action.params[0].slot)

        # Default fall-back action
        print('Reinforce ({0}) policy action encoder warning: Selecting '
              'default action (unable to encode: {1})!'
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
                            action_enc - len(self.dstc2_acts_sys)],
                        Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots) + \
                    len(self.requestable_slots):
                index = action_enc - len(self.dstc2_acts_sys) - \
                        len(self.system_requestable_slots)
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.requestable_slots[index], Operator.EQ, '')])]

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
                    2 * len(self.requestable_slots):
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.requestable_slots[
                            action_enc - len(self.dstc2_acts_usr) -
                            len(self.requestable_slots)],
                        Operator.EQ,
                        '')])]

        # Default fall-back action
        print('Reinforce DialoguePolicy ({0}) policy action decoder warning: '
              'Selecting default action (index: {1})!'
              .format(self.agent_role, action_enc))
        return [DialogueAct('bye', [])]

    def save(self, path=None):
        """
        Saves the policy model to the provided path

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'Models/Policies/reinforce.pkl'
            print('No policy file name provided. Using default: {0}'
                  .format(path))

        obj = {'weights': self.weights,
               'alpha': self.alpha,
               'alpha_decay_rate': self.alpha_decay_rate,
               'epsilon': self.epsilon,
               'exploration_decay_rate': self.exploration_decay_rate}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        """
        Load the policy model from the provided path

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

                    if 'weights' in obj:
                        self.weights = obj['weights']

                    if 'alpha' in obj:
                        self.alpha = obj['alpha']

                    if 'alpha_decay_rate' in obj:
                        self.alpha_decay_rate = obj['alpha_decay_rate']

                    if 'epsilon' in obj:
                        self.epsilon = obj['epsilon']

                    if 'exploration_decay_rate' in obj:
                        self.exploration_decay_rate = \
                            obj['exploration_decay_rate']

                    print('Reinforce DialoguePolicy loaded from {0}.'
                          .format(path))

            else:
                print('Warning! Reinforce DialoguePolicy file %s not found'
                      % path)
        else:
            print('Warning! Unacceptable value for Reinforce DialoguePolicy '
                  'file name: %s ' % path)
