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

from .. import dialogue_policy
from plato.agent.component.dialogue_policy.slot_filling_policy\
    import HandcraftedPolicy
from plato.domain.ontology import Ontology
from plato.domain.database import DataBase
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.dialogue.state import SlotFillingDialogueState
from plato.agent.component.user_simulator.\
    agenda_based_user_simulator.agenda_based_us import AgendaBasedUS
from copy import deepcopy

import tensorflow as tf
import numpy as np
import random
import os

"""
SupervisedPolicy implements a tensorflow-based supervised policy, tuned for 
DSTC2.
"""


class SupervisedPolicy(dialogue_policy.DialoguePolicy):

    def __init__(self, args):
        """
        Initialize parameters and internal structures

        :param args: dictionary containing the policy's arguments
        """
        super(SupervisedPolicy, self).__init__()

        self.ontology = None
        if 'ontology' in args:
            ontology = args['ontology']

            if isinstance(ontology, Ontology):
                self.ontology = ontology
            else:
                raise ValueError('SupervisedPolicy Unacceptable '
                                 'ontology type %s ' % ontology)
        else:
            raise ValueError('SupervisedPolicy: No ontology provided')

        self.database = None
        if 'database' in args:
            database = args['database']

            if isinstance(database, DataBase):
                self.database = database
            else:
                raise ValueError('SupervisedPolicy: Unacceptable '
                                 'database type %s ' % database)
        else:
            raise ValueError('SupervisedPolicy: No database provided')

        self.agent_id = args['agent_id'] if 'agent_id' in args else 0
        self.agent_role = args['agent_role'] \
            if 'agent_role' in args else 'system'
        domain = args['domain'] if 'domain' in args else None

        # True for greedy, False for stochastic
        self.IS_GREEDY_POLICY = False

        self.policy_path = None

        self.policy_net = None
        self.tf_scope = "policy_" + self.agent_role + '_' + str(self.agent_id)
        self.sess = None

        # The system and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        # Default value
        self.is_training = True

        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'] +
                     ['this', 'signature'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        self.dstc2_acts = None

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['repeat', 'canthelp', 'affirm', 'negate',
                               'deny', 'ack', 'thankyou', 'bye',
                               'reqmore', 'hello', 'welcomemsg', 'expl-conf',
                               'select', 'offer', 'reqalts',
                               'confirm-domain', 'confirm']
        else:
            # Try to identify number of state features
            if domain in ['SlotFilling', 'CamRest']:
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
                                           'deny', 'ack', 'bye', 'reqmore',
                                           'welcomemsg', 'expl-conf', 'select',
                                           'repeat', 'confirm-domain',
                                           'confirm']

                    # Does not include inform and request that are modelled
                    # together with their arguments
                    self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack',
                                           'thankyou', 'bye', 'reqmore',
                                           'hello', 'expl-conf', 'repeat',
                                           'reqalts', 'restart', 'confirm']

                    if self.agent_role == 'system':
                        self.dstc2_acts = self.dstc2_acts_sys

                    elif self.agent_role == 'user':
                        self.dstc2_acts = self.dstc2_acts_usr

            else:
                print('Warning! domain has not been defined. Using '
                      'Slot-Filling dialogue State')
                d_state = \
                    SlotFillingDialogueState({'slots': self.informable_slots})

            d_state.initialize()
            self.NStateFeatures = len(self.encode_state(d_state))
            print('Supervised dialogue policy automatically determined number '
                  'of state features: {0}'
                  .format(self.NStateFeatures))

        if domain == 'CamRest':
            self.NActions = len(self.dstc2_acts) + len(self.requestable_slots)

            if self.agent_role == 'system':
                self.NActions += len(self.system_requestable_slots)
            else:
                self.NActions += len(self.requestable_slots)
        else:
            self.NActions = 5

        self.policy_alpha = 0.05

        self.tf_saver = None

    def initialize(self, args):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if self.agent_role == 'system':
            # Put your system expert dialogue policy here
            self.warmup_policy = HandcraftedPolicy({'ontology': self.ontology})

        elif self.agent_role == 'user':
            usim_args = \
                dict(
                    zip(['ontology', 'database'],
                        [self.ontology, self.database]))
            # Put your user expert dialogue policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)

        if 'is_training' in args:
            self.is_training = bool(args['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in args:
                    self.warmup_simulator.initialize({args['goal']})
                else:
                    print('WARNING ! No goal provided for Supervised policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

        if 'policy_path' in args:
            self.policy_path = args['policy_path']

        if 'learning_rate' in args:
            self.policy_alpha = args['learning_rate']

        if self.sess is None:
            self.policy_net = self.feed_forward_net_init()
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

            self.tf_saver = \
                tf.train.Saver(var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope))

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
                print('WARNING! No goal provided for Supervised policy user '
                      'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the dialogue policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        if self.is_training:
            # This is a Supervised dialogue policy, so no exploration here.

            if self.agent_role == 'system':
                return self.warmup_policy.next_action(state)
            else:
                self.warmup_simulator.receive_input(
                    state.user_acts, state.user_goal)
                return self.warmup_simulator.generate_output()

        pl_calculated, pl_state, pl_newvals, pl_optimizer, pl_loss = \
            self.policy_net

        obs_vector = np.expand_dims(self.encode_state(state), axis=0)

        probs = self.sess.run(pl_calculated, feed_dict={pl_state: obs_vector})

        if self.IS_GREEDY_POLICY:
            # Greedy policy: Return action with maximum value from the given
            # state
            sys_acts = \
                self.decode_action(
                    np.argmax(probs), self.agent_role == 'system')

        else:
            # Stochastic dialogue policy: Sample action wrt Q values
            if any(np.isnan(probs[0])):
                print('WARNING! Supervised dialogue policy: NAN detected in a'
                      'ction probabilities! Selecting random action.')
                return self.decode_action(
                    random.choice(range(0, self.NActions)),
                    self.agent_role == 'system')

            # Make sure weights are positive
            min_p = min(probs[0])

            if min_p < 0:
                positive_weights = [p + abs(min_p) for p in probs[0]]
            else:
                positive_weights = probs[0]

            # Normalize weights
            positive_weights /= sum(positive_weights)

            sys_acts = \
                self.decode_action(
                    random.choices(
                        [a for a in range(self.NActions)],
                        weights=positive_weights)[0],
                    self.agent_role == 'system')

        return sys_acts

    def feed_forward_net_init(self):
        """
        Initialize the feed forward network.

        :return: some useful variables
        """
        self.tf_scope = "policy_" + self.agent_role + '_' + str(self.agent_id)

        with tf.variable_scope(self.tf_scope):
            state = tf.placeholder("float", [None, self.NStateFeatures])
            newvals = tf.placeholder("float", [None, self.NActions])

            w1 = \
                tf.get_variable("w1",
                                [self.NStateFeatures, self.NStateFeatures])
            b1 = tf.get_variable("b1", [self.NStateFeatures])
            h1 = tf.nn.sigmoid(tf.matmul(state, w1) + b1)

            w2 = \
                tf.get_variable("w2",
                                [self.NStateFeatures, self.NStateFeatures])
            b2 = tf.get_variable("b2", [self.NStateFeatures])
            h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

            w3 = tf.get_variable("w3", [self.NStateFeatures, self.NActions])
            b3 = tf.get_variable("b3", [self.NActions])

            calculated = tf.nn.softmax(tf.matmul(h2, w3) + b3)

            diffs = calculated - newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = \
                tf.train.AdamOptimizer(self.policy_alpha).minimize(loss)

            return calculated, state, newvals, optimizer, loss

    def train(self, dialogues):
        """
        Train the neural net dialogue policy model

        :param dialogues: dialogue experience
        :return: nothing
        """

        # If called by accident
        if not self.is_training:
            return

        pl_calculated, pl_state, pl_newvals, pl_optimizer, pl_loss =\
            self.policy_net

        states = []
        actions = []

        for dialogue in dialogues:
            for index, turn in enumerate(dialogue):
                act_enc = \
                    self.encode_action(turn['action'],
                                       self.agent_role == 'system')
                if act_enc > -1:
                    states.append(self.encode_state(turn['state']))
                    action = np.zeros(self.NActions)
                    action[act_enc] = 1
                    actions.append(action)

        # Train dialogue policy
        self.sess.run(
            pl_optimizer,
            feed_dict={pl_state: states, pl_newvals: actions})

    def encode_state(self, state):
        """
        Encodes the dialogue state into a vector.

        :param state: the state to encode
        :return: int - a unique state encoding
        """

        temp = [int(state.is_terminal_state)]

        temp.append(1) if state.system_made_offer else temp.append(0)

        # If the agent plays the role of the user it needs access to its own
        # goal
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

        if not actions:
            print('WARNING: Supervised dialogue policy action encoding called '
                  'with empty actions list (returning -1).')
            return -1

        action = actions[0]

        slot = None
        if action.params and action.params[0].slot:
            slot = action.params[0].slot

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if slot:
                if action.intent == 'request' and \
                        slot in self.system_requestable_slots:
                    return len(self.dstc2_acts_sys) + \
                           self.system_requestable_slots.index(slot)

                if action.intent == 'inform' and \
                        slot in self.requestable_slots:
                    return len(self.dstc2_acts_sys) + \
                           len(self.system_requestable_slots) + \
                           self.requestable_slots.index(slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if slot:
                if action.intent == 'request' and \
                        slot in self.requestable_slots:
                    return len(self.dstc2_acts_usr) + \
                           self.requestable_slots.index(slot)

                if action.intent == 'inform' and \
                        slot in self.requestable_slots:
                    return len(self.dstc2_acts_usr) + \
                           len(self.requestable_slots) + \
                           self.requestable_slots.index(slot)

        # Default fall-back action
        print('Supervised ({0}) policy action encoder warning: Selecting '
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
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots) +\
                    len(self.requestable_slots):
                index = action_enc - \
                        len(self.dstc2_acts_sys) - \
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

    def save(self, path=None):
        """
        Saves the policy model to the provided path

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'models/policies/supervised_policy_' + \
                       self.agent_role + '_' + str(self.agent_id)

        # If the directory does not exist, create it
        if not os.path.exists(os.path.dirname(pol_path)):
            os.makedirs(os.path.dirname(pol_path), exist_ok=True)

        if self.sess is not None and self.is_training:
            save_path = self.tf_saver.save(self.sess, pol_path)
            print('Supervised policy model saved at: %s' % save_path)

    def load(self, path):
        """
        Load the policy model from the provided path

        :param path: path to load the model from
        :return:
        """

        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'models/policies/supervised_policy_' + \
                       self.agent_role + '_' + str(self.agent_id)

        if os.path.isfile(pol_path + '.meta'):
            self.policy_net = self.feed_forward_net_init()
            self.sess = tf.InteractiveSession()

            self.tf_saver = \
                tf.train.Saver(
                    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=self.tf_scope))

            self.tf_saver.restore(self.sess, pol_path)

            print('Supervised policy model loaded from {0}.'
                  .format(pol_path))

        else:
            print('WARNING! Supervised policy cannot load policy '
                  'model from {0}!'.format(pol_path))
