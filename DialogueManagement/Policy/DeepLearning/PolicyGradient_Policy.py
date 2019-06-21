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

from .. import Policy
from DialogueManagement.Policy.HandcraftedPolicy import HandcraftedPolicy
from Ontology import Ontology, DataBase
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState, DummyDialogueState
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from copy import deepcopy

import tensorflow as tf
import numpy as np
import random
import os


class PolicyGradient_Policy(Policy.Policy):

    def __init__(self, ontology, database, agent_id=0, agent_role='system', domain=None, alpha=0.2, epsilon=0.95,
                 gamma=0.95, alpha_decay=0.995, epsilon_decay=0.9995):
        super(PolicyGradient_Policy, self).__init__()

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase.DataBase):
            self.database = database
        else:
            raise ValueError('Supervised Policy: Unacceptable database type %s ' % database)

        self.policy_path = None

        self.policy_grad = None
        self.value_grad = None
        self.sess = None

        self.policy_alpha = 0.00001
        self.value_alpha = 0.001
        self.epsilon = epsilon
        self.gamma = gamma

        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            self.warmup_policy = HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = dict(zip(['ontology', 'database'], [self.ontology, self.database]))
            self.warmup_simulator = AgendaBasedUS(usim_args)

        self.tf_scope = "policy_" + self.agent_role + '_' + str(self.agent_id)

        # Default value
        self.is_training = True

        # Extract lists of slots that are frequently used
        self.informable_slots = deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = deepcopy(self.ontology.ontology['requestable'] + ['this', 'signature'])
        self.system_requestable_slots = deepcopy(self.ontology.ontology['system_requestable'])

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['inform', 'offer', 'request', 'canthelp', 'affirm', 'negate', 'deny', 'ack', 'thankyou',
                               'bye', 'reqmore', 'hello', 'welcomemsg', 'expl-conf', 'select', 'repeat', 'reqalts',
                               'confirm-domain', 'confirm']
        else:
            # Try to identify number of state features
            if domain in ['CamRest', 'SFH', 'SlotFilling']:
                DState = SlotFillingDialogueState({'slots': self.informable_slots})

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled together with their arguments
                    self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm', 'deny', 'ack', 'bye',
                                           'reqmore', 'welcomemsg', 'expl-conf', 'select', 'repeat',
                                           'confirm-domain', 'confirm']

                    # Does not include inform and request that are modelled together with their arguments
                    self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye',
                                           'reqmore', 'hello', 'expl-conf', 'repeat', 'reqalts', 'restart',
                                           'confirm']

            else:
                print('Warning! Domain has not been defined. Using Dummy Dialogue State')
                DState = DummyDialogueState({'slots': self.informable_slots})

            DState.initialize()
            self.NStateFeatures = len(self.encode_state(DState))

            print('Policy Gradient Policy {0} automatically determined number of state features: {1}'.format(
                self.agent_role, self.NStateFeatures))

        if domain == 'CamRest' and self.dstc2_acts_sys:
            if self.agent_role == 'system':
                self.NActions = len(self.dstc2_acts_sys) + len(self.requestable_slots) + len(self.system_requestable_slots)
                self.NOtherActions = len(self.dstc2_acts_usr) + 2 * len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = len(self.dstc2_acts_usr) + 2 * len(self.requestable_slots)
                self.NOtherActions = len(self.dstc2_acts_sys) + len(self.requestable_slots) + len(self.system_requestable_slots)
        else:
            if self.agent_role == 'system':
                self.NActions = 3 + len(self.system_requestable_slots) + len(self.requestable_slots)
                self.NOtherActions = 2 + len(self.requestable_slots) + len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = 2 + len(self.requestable_slots) + len(self.requestable_slots)
                self.NOtherActions = 3 + len(self.system_requestable_slots) + len(self.requestable_slots)

        print('Policy Gradient {0} Policy Number of Actions: {1}'.format(self.agent_role, self.NActions))

        self.tf_saver = None

    def initialize(self, **kwargs):
        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in kwargs:
                    self.warmup_simulator.initialize({kwargs['goal']})
                else:
                    print('WARNING ! No goal provided for Supervised policy user simulator @ initialize')
                    self.warmup_simulator.initialize({})

        if 'policy_path' in kwargs:
            self.policy_path = kwargs['policy_path']
        
        if 'policy_alpha' in kwargs:
            self.policy_alpha = kwargs['policy_alpha']
            
        if 'value_alpha' in kwargs:
            self.value_alpha = kwargs['value_alpha']

        if self.sess is None:
            self.policy_grad = self.policy_gradient()
            self.value_grad = self.value_gradient()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self.tf_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      scope=self.tf_scope))

    def restart(self, args):
        if self.agent_role == 'user' and self.warmup_simulator:
            if 'goal' in args:
                self.warmup_simulator.initialize(args)
            else:
                print('WARNING! No goal provided for Supervised policy user simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        if self.is_training and random.random() < self.epsilon:
            print('---: Selecting warmup action.')

            if random.random() < 0.5:
                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)
                else:
                    self.warmup_simulator.receive_input(state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()

            else:
                # Return a random action
                return self.decode_action(random.choice(range(0, self.NActions)))

        pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = self.policy_grad

        obs_vector = np.expand_dims(self.encode_state(state), axis=0)

        probs = self.sess.run(pl_calculated, feed_dict={pl_state: obs_vector})

        # if self.is_training:
        # Probabilistic policy: Sample from action wrt probabilities
        if any(np.isnan(probs[0])):
            print('WARNING! NAN detected in action probabilities! Selecting random action.')
            return self.decode_action(random.choice(range(0, self.NActions)))

        sys_acts = self.decode_action(random.choices(range(self.NActions), probs[0])[0], self.agent_role == 'system')

        # else:
        #     # Greedy policy: Return action with maximum value from the given state
        #     sys_acts = self.decode_action(np.argmax(probs), self.agent_role == 'system')

        return sys_acts

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def policy_gradient(self):
        with tf.variable_scope(self.tf_scope):
            params = tf.get_variable("policy_parameters", [self.NStateFeatures, self.NActions])
            state = tf.placeholder("float", [None, self.NStateFeatures])
            actions = tf.placeholder("float", [None, self.NActions])
            advantages = tf.placeholder("float", [None, 1])

            linear = tf.matmul(state, params)
            probabilities = tf.nn.softmax(linear)
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * advantages

            loss = -tf.reduce_sum(eligibility)
            optimizer = tf.train.AdamOptimizer(self.policy_alpha).minimize(loss)

            return probabilities, state, actions, advantages, optimizer

    def value_gradient(self):
        with tf.variable_scope(self.tf_scope):
            state = tf.placeholder("float", [None, self.NStateFeatures])
            newvals = tf.placeholder("float", [None, 1])

            w1 = tf.get_variable("w1", [self.NStateFeatures, self.NStateFeatures])
            b1 = tf.get_variable("b1", [self.NStateFeatures])
            h1 = tf.nn.relu(tf.matmul(state, w1) + b1)

            w2 = tf.get_variable("w2", [self.NStateFeatures, self.NActions])
            b2 = tf.get_variable("b2", [self.NActions])
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

            w3 = tf.get_variable("w3", [self.NActions, 1])
            b3 = tf.get_variable("b3", [1])

            calculated = tf.matmul(h2, w3) + b3
            diffs = calculated - newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(self.value_alpha).minimize(loss)

            return calculated, state, newvals, optimizer, loss

    def train(self, dialogues):
        # If called by accident
        if not self.is_training:
            return

        pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = self.policy_grad
        vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = self.value_grad

        states = []
        actions = []
        advantages = []
        update_vals = []

        for dialogue in dialogues:
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for index, turn in enumerate(dialogue):
                act_enc = self.encode_action(turn['action'], self.agent_role == 'system')

                # TODO: Check the rest of the encodings and issue errors or warinings (e.g. len(state_enc) != self.NStateFeatures)
                if act_enc < 0:
                    continue

                obs = self.encode_state(turn['new_state'])
                states.append(self.encode_state(turn['state']))
                action = np.zeros(self.NActions)
                action[act_enc] = 1
                actions.append(action)

                # calculate discounted monte-carlo return
                future_reward = 0
                future_transitions = len(dialogue) - index
                decrease = 1

                for index2 in range(future_transitions):
                    future_reward += dialogue[(index2) + index]['reward'] * decrease
                    decrease = decrease * self.gamma

                obs_vector = np.expand_dims(obs, axis=0)
                currentval = self.sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

                # advantage: how much better was this action than normal
                advantages.append(future_reward - currentval)

                # update the value function towards new return
                update_vals.append(future_reward)

        # update value function
        update_vals_vector = np.expand_dims(update_vals, axis=1)
        self.sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

        advantages_vector = np.expand_dims(advantages, axis=1)
        self.sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

        # print('Epsilon: {0}', self.epsilon)

        if self.epsilon > 0.05:
            self.epsilon *= 0.995

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

        return temp

    def encode_action(self, actions, system=True):
        '''

        :param actions:
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Policy Gradient Policy action encoding called with empty actions list (returning 0).')
            return -1

        action = actions[0]

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_sys) + self.system_requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                return len(self.dstc2_acts_sys) + len(self.system_requestable_slots) + self.requestable_slots.index(action.params[0].slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_usr) + self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                return len(self.dstc2_acts_usr) + len(self.requestable_slots) + self.requestable_slots.index(action.params[0].slot)

        # Default fall-back action
        print('Policy Gradient ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return -1

    def decode_action(self, action_enc, system=True):
        '''

        :param action_enc:
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

        # Default fall-back action
        print('Policy Gradient Policy ({0}) policy action decoder warning: Selecting default action (index: {1})!'.format(self.agent_role, action_enc))
        return [DialogueAct('repeat', [])]

    def save(self, path=None):
        # Don't save if not training
        if not self.is_training:
            return

        print('DEBUG: {0} Exploration is: {1}, p learning rate is: {2}, v learning rate is: {3}'.format(self.agent_role, self.epsilon, self.policy_alpha, self.value_alpha))

        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'Models/Policies/pg_policy_' + str(self.agent_id)

        if self.sess is not None and self.is_training:
            save_path = self.tf_saver.save(self.sess, pol_path)
            print('PolicyGradient model saved at: %s' % save_path)

    def load(self, path=None):
        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'Models/Policies/pg_policy_' + str(self.agent_id)

        if os.path.isfile(pol_path + '.meta'):
            self.policy_grad = self.policy_gradient()
            self.value_grad = self.value_gradient()
            self.sess = tf.InteractiveSession()

            self.tf_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      scope=self.tf_scope))
            self.tf_saver.restore(self.sess, pol_path)

            print('PolicyGradient model loaded from {0}.'.format(pol_path))

        else:
            print('Warning: No Policy Gradient policy loaded.')


