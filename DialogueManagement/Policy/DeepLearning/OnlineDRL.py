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

from DialogueManagement.Policy.Policy import Policy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from ludwig.api import LudwigModel
import os.path
import pandas as pd
import numpy as np
import random

from copy import deepcopy


class OnlineDRL(Policy):
    def __init__(self, model_path):
        super(OnlineDRL, self).__init__()

        self.epsilon = 0.995
        self.is_training = False

        self.informable_slots = None
        self.requestable_slots = None
        self.system_requestable_slots = None

        self.NActions = None
        self.state_encoding_length = None

        self.agent_role = 'system'
        self.policy_model = None

        self.load(model_path)

        # Sub-case for CamRest
        self.dstc2_acts_sys = self.dstc2_acts_usr = None

        # Does not include inform and request that are modelled together with their arguments
        self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm', 'deny', 'ack', 'bye',
                               'reqmore', 'welcomemsg', 'expl-conf', 'select', 'repeat',
                               'confirm-domain', 'confirm']

        # Does not include inform and request that are modelled together with their arguments
        self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack', 'thankyou', 'bye',
                               'reqmore', 'hello', 'expl-conf', 'repeat', 'reqalts', 'restart',
                               'confirm']

    def __del__(self):
        if self.policy_model:
            self.policy_model.close()

    def initialize(self, **kwargs):
        '''
        Initialize anything that should not be in __init__

        :return: Nothing
        '''

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

        if 'ontology' in kwargs:
            ontology = kwargs['ontology']

            self.informable_slots = deepcopy(list(ontology.ontology['informable'].keys()))
            self.requestable_slots = deepcopy(ontology.ontology['requestable'] + ['this'])
            self.system_requestable_slots = deepcopy(ontology.ontology['system_requestable'])

            self.NActions = 4 + len(self.system_requestable_slots) + len(self.requestable_slots)
            self.agent_role = 'system'

            # Get state encoding length
            temp_dstate = SlotFillingDialogueState({'slots': ontology.ontology['system_requestable']})
            temp_dstate.initialize()
            self.state_encoding_length = len(self.encode_state(temp_dstate))

            # Extract lists of slots that are frequently used
            self.informable_slots = deepcopy(list(ontology.ontology['informable'].keys()))
            self.requestable_slots = deepcopy(ontology.ontology['requestable'])
            self.system_requestable_slots = deepcopy(ontology.ontology['system_requestable'])

            if self.dstc2_acts_sys:
                if self.agent_role == 'system':
                    self.NActions = len(self.dstc2_acts_sys) + len(self.requestable_slots) + len(
                        self.system_requestable_slots)
                    self.NOtherActions = len(self.dstc2_acts_usr) + 2 * len(self.requestable_slots)

                elif self.agent_role == 'user':
                    self.NActions = len(self.dstc2_acts_usr) + 2 * len(self.requestable_slots)
                    self.NOtherActions = len(self.dstc2_acts_sys) + len(self.requestable_slots) + len(
                        self.system_requestable_slots)
            else:
                if self.agent_role == 'system':
                    self.NActions = 5 + len(ontology.ontology['system_requestable']) + len(ontology.ontology['requestable'])
                    self.NOtherActions = 4 + 2 * len(ontology.ontology['requestable'])

                elif self.agent_role == 'user':
                    self.NActions = 4 + 2 * len(ontology.ontology['requestable'])
                    self.NOtherActions = 5 + len(ontology.ontology['system_requestable']) + len(ontology.ontology['requestable'])

    def restart(self, args):
        '''
        Re-initialize relevant parameters / variables at the beginning of each dialogue.

        :return:
        '''

        pass

    def next_action(self, state):
        if random.random() < self.epsilon and self.is_training:
            print(f'Online DRL {self.agent_role} taking random exploratory action.')
            action_enc = random.choice(range(0, self.NActions))

        else:
            result = self.policy_model.predict(
                pd.DataFrame(data={'state': [' '.join([str(s) for s in self.encode_state(state)])]}), return_type=dict)

            if not result['sys_act']['probability'][0]:
                print('Warning! Online DRL: No prediction from policy model!')
                print(result)

            action_enc = result['sys_act']['predictions'][0][np.argmax(result['sys_act']['probability'][0])]

            if action_enc == '<UNK>':
                action_enc = random.randint(0, self.NActions)

            else:
                action_enc = int(action_enc)

        return self.decode_action(action_enc)

    def train(self, dialogues):
        for dialogue in dialogues:
            for turn in dialogue:
                # Only pick examples with positive reward -- assuming the goal advancement reward was used.
                if turn['reward'] < 0:
                    continue

                state_enc = self.encode_state(turn['state'])
                act_enc_int = self.encode_action(turn['action'], self.agent_role == 'system')
                action_enc = np.zeros(self.NActions)
                action_enc[act_enc_int] = 1

                input_state = ' '.join([str(s) for s in state_enc])
                target_sys_act = ' '.join([str(a) for a in action_enc])

                self.policy_model.train_online(data_dict={'state': [input_state], 'sys_act': [target_sys_act]})

        # Decay exploration rate
        if self.is_training and self.epsilon > 0.15:
            self.epsilon *= 0.995

    def save(self, path=None):
        if isinstance(path, str):
            if os.path.isdir(path):
                self.policy_model.save(path)

            else:
                raise FileNotFoundError('Online DRL Policy model directory {0} not found'.format(path))
        else:
            raise ValueError('Online DRL Policy: Unacceptable value for model file name: {0} '.format(path))

    def load(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                self.policy_model = LudwigModel.load(path)

            else:
                raise FileNotFoundError('Online DRL Policy model directory {0} not found'.format(path))
        else:
            raise ValueError('Online DRL Policy: Unacceptable value for model file name: {0} '.format(path))

    # These encodings are the same that the SupervisedPolicy uses, for performance comparison.
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

        temp.append(int(state.is_terminal_state))

        # If the agent is a system, then this shows what the top db result is.
        # If the agent is a user, then this shows what information the system has provided
        # if self.agent_role == 'system':
        if state.item_in_focus:
            state_filled_info = []
            requested_slot = []

            for slot in self.requestable_slots:
                if slot in state.item_in_focus and state.item_in_focus[slot]:
                    state_filled_info.append(1)
                else:
                    state_filled_info.append(0)

                requested_slot.append(1) if slot == state.requested_slot else requested_slot.append(0)

            temp += state_filled_info + requested_slot

            if self.agent_role == 'system':
                if state.system_requestable_slot_entropies:
                    max_entr = max(state.system_requestable_slot_entropies.values())
                    temp += [1 if state.system_requestable_slot_entropies[s] == max_entr else 0 for s in
                             state.system_requestable_slot_entropies]
                else:
                    temp += [0] * len(self.system_requestable_slots)

        elif self.agent_role == 'system':
            temp += [0] * (2 * len(self.requestable_slots) + len(self.system_requestable_slots))

        elif self.agent_role == 'user':
            temp += [0] * (2 * len(self.requestable_slots))

        temp.append(1) if state.system_made_offer else temp.append(0)

        if state.user_acts:
            # If this agent is the system then "user" is a user (hopefully).
            # If this agent is a user then "user" is a system.
            temp += [int(b) for b in format(self.encode_action(state.user_acts, self.agent_role != 'system'), '06b')]
        else:
            temp += [0] * 6

        if state.last_sys_acts:
            temp += [int(b) for b in format(self.encode_action([state.last_sys_acts[0]], self.agent_role == 'system'), '06b')]
        else:
            temp += [0] * 6

        # If the agent plays the role of the user it needs access to its own goal
        if state.user_goal:
            for c in self.informable_slots:
                if c in state.user_goal.constraints and state.user_goal.constraints[c].value:
                    temp.append(1)
                else:
                    temp.append(0)

            for r in self.requestable_slots:
                if r in state.user_goal.requests and state.user_goal.requests[r].value:
                    temp.append(1)
                else:
                    temp.append(0)

            for r in self.requestable_slots:
                if r in state.user_goal.actual_requests and state.user_goal.actual_requests[r].value:
                    temp.append(1)
                else:
                    temp.append(0)
        else:
            temp += [0] * (len(self.informable_slots) + 2 * len(self.requestable_slots))

        return temp

    def encode_action(self, actions, system=True):
        '''

        :param actions: the actions to be encoded
        :param system: if the actions were / will be performed by a system agent or a user agent
        :return: integer representing actions encoding
        '''

        # TODO: Handle multiple actions
        if not actions:
            print('WARNING: Online DRL Policy action encoding called with empty actions list (returning 0).')
            return 0

        action = actions[0]

        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if action.intent == 'request':
                if system:
                    return len(self.dstc2_acts_sys) + self.system_requestable_slots.index(action.params[0].slot)
                else:
                    return len(self.dstc2_acts_sys) + self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                if system:
                    return len(self.dstc2_acts_sys) + len(self.system_requestable_slots) + self.requestable_slots.index(
                        action.params[0].slot)
                else:
                    return len(self.dstc2_acts_sys) + len(self.requestable_slots) + self.requestable_slots.index(
                        action.params[0].slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if action.intent == 'request':
                if system:
                    return len(self.dstc2_acts_usr) + self.system_requestable_slots.index(action.params[0].slot)
                else:
                    return len(self.dstc2_acts_usr) + self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                if system:
                    return len(self.dstc2_acts_usr) + len(self.system_requestable_slots) + self.requestable_slots.index(
                        action.params[0].slot)
                else:
                    return len(self.dstc2_acts_usr) + len(self.requestable_slots) + self.requestable_slots.index(
                        action.params[0].slot)

        # Default fall-back action
        if (self.agent_role == 'system') == system:
            print(
                'Online DRL ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(
                    self.agent_role, action))
        else:
            print(
                'Online DRL ({0}) policy action encoder warning: Selecting default action (unable to encode other agent action: {1})!'.format(
                    self.agent_role, action))

        return self.NActions - 1

    def decode_action(self, action_enc, system=True):
        '''

        :param action_enc: the actions to be decoded
        :param system: if the actions were / will be performed by a system agent or a user agent
        :return: List of DialogueAct representing decoded actions
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
        print('Online DRL Policy ({0}) policy action decoder warning: Selecting UNK() action (index: {1})!'.format(
            self.agent_role, action_enc))
        return [DialogueAct('repeat', [])]
