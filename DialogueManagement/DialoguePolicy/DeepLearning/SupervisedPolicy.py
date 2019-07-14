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

import tensorflow as tf
import numpy as np
import random
import os

"""
SupervisedPolicy implements a tensorflow-based supervised policy, tweaked for 
DSTC2.
"""


class SupervisedPolicy(DialoguePolicy.DialoguePolicy):

    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 domain=None):
        """
        Initialize parameters and internal structures

        :param ontology: the domain's ontology
        :param database: the domain's database
        :param agent_id: the agent's id
        :param agent_role: the agent's role
        :param domain: the dialogue's domain
        """
        super(SupervisedPolicy, self).__init__()

        self.agent_id = agent_id
        self.agent_role = agent_role

        # True for greedy, False for stochastic
        self.IS_GREEDY_POLICY = False

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Supervised DialoguePolicy: Unacceptable '
                             'ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase.DataBase):
            self.database = database
        else:
            raise ValueError('Supervised DialoguePolicy: Unacceptable '
                             'database type %s ' % database)

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
                DState = \
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
                print('Warning! Domain has not been defined. Using '
                      'Slot-Filling Dialogue State')
                DState = \
                    SlotFillingDialogueState({'slots': self.informable_slots})

            DState.initialize()
            self.NStateFeatures = len(self.encode_state(DState))
            print('Supervised DialoguePolicy automatically determined number '
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

    def initialize(self, **kwargs):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

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

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in kwargs:
                    self.warmup_simulator.initialize({kwargs['goal']})
                else:
                    print('WARNING ! No goal provided for Supervised policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

        if 'policy_path' in kwargs:
            self.policy_path = kwargs['policy_path']

        if 'learning_rate' in kwargs:
            self.policy_alpha = kwargs['learning_rate']

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
        Consults the policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        if self.is_training:
            # This is a Supervised DialoguePolicy, so no exploration here.

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
            # Stochastic policy: Sample action wrt Q values
            if any(np.isnan(probs[0])):
                print('WARNING! Supervised DialoguePolicy: NAN detected in a'
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
        Train the neural net policy model

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

        # Train policy
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
            print('WARNING: Supervised DialoguePolicy action encoding called '
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

        print('DEBUG: {0} learning rate is: {1}'
              .format(self.agent_role, self.policy_alpha))

        pol_path = path

        if not pol_path:
            pol_path = self.policy_path

        if not pol_path:
            pol_path = 'Models/Policies/supervised_policy_' + \
                       self.agent_role + '_' + str(self.agent_id)

        if self.sess is not None and self.is_training:
            save_path = self.tf_saver.save(self.sess, pol_path)
            print('Supervised DialoguePolicy model saved at: %s' % save_path)

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
            pol_path = 'Models/Policies/supervised_policy_' + \
                       self.agent_role + '_' + str(self.agent_id)

        if os.path.isfile(pol_path + '.meta'):
            self.policy_net = self.feed_forward_net_init()
            self.sess = tf.InteractiveSession()

            self.tf_saver = \
                tf.train.Saver(
                    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=self.tf_scope))

            self.tf_saver.restore(self.sess, pol_path)

            print('Supervised DialoguePolicy model loaded from {0}.'
                  .format(pol_path))

        else:
            print('WARNING! Supervised DialoguePolicy cannot load policy '
                  'model from {0}!'.format(pol_path))
