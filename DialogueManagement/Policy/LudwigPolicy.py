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

from copy import deepcopy


class LudwigPolicy(Policy):
    def __init__(self, model_path):
        super(LudwigPolicy, self).__init__()

        self.model = None

        self.informable_slots = None
        self.requestable_slots = None
        self.system_requestable_slots = None

        self.NActions = None
        self.state_encoding_length = None

        self.agent_role = 'system'
        self.policy_model = None

        self.load(model_path)

    def initialize(self, **kwargs):
        '''
        Initialize anything that should not be in __init__

        :return: Nothing
        '''

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

    def restart(self, args):
        '''
        Re-initialize relevant parameters / variables at the beginning of each dialogue.

        :return:
        '''

        pass

    def next_action(self, state):
        data_dic = {'act': ['na'],
                    'inform_area': ['na'],
                    'inform_food': ['na'],
                    'inform_pricerange': ['na'],
                    'request_area': [False],
                    'request_food': [False],
                    'request_pricerange': [False],
                    'request_addr': [False],
                    'request_name': [False],
                    'request_phone': [False],
                    'request_postcode': [False],
                    'ds_user_terminating': [state.is_terminal()]
                    }

        state_cols = ['state_col' + str(i) for i in range(0, self.state_encoding_length)]

        state_enc = self.encode_state(state)

        data_dic.update({k: v for (k, v) in zip(state_cols, state_enc)})

        if state.user_acts:
            # Take first act only
            data_dic['act'] = state.user_acts[0].intent

            if state.user_acts[0].params and state.user_acts[0].params[0].slot:
                if state.user_acts[0].intent == 'inform' and state.user_acts[0].params[0].value:
                    if state.user_acts[0].params[0].slot == 'area':

                        data_dic['inform_area'] = state.user_acts[0].params[0].value

                    if state.user_acts[0].params[0].slot == 'food':
                        data_dic['inform_food'] = state.user_acts[0].params[0].value

                    if state.user_acts[0].params[0].slot == 'pricerange':
                        data_dic['inform_pricerange'] = state.user_acts[0].params[0].value

                elif state.user_acts[0].intent == 'request':
                    data_dic['request_area'] = [state.user_acts[0].params[0].slot == 'area']
                    data_dic['request_addr'] = [state.user_acts[0].params[0].slot == 'addr']
                    data_dic['request_food'] = [state.user_acts[0].params[0].slot == 'food']
                    data_dic['request_name'] = [state.user_acts[0].params[0].slot == 'name']
                    data_dic['request_pricerange'] = [state.user_acts[0].params[0].slot == 'pricerange']
                    data_dic['request_phone'] = [state.user_acts[0].params[0].slot == 'phone']
                    data_dic['request_postcode'] = [state.user_acts[0].params[0].slot == 'postcode']

        result = self.policy_model.predict(pd.DataFrame(data=data_dic), return_type=dict)

        dact = DialogueAct(result['system_act']['predictions'][0], [])

        dactitems = []

        if dact.intent in ['inform, offer'] and state.item_in_focus:
            if result['sys_area']['predictions'][0]:
                dactitems.append(DialogueActItem('area', Operator.EQ, state.item_in_focus['area']))
            if result['sys_food']['predictions'][0]:
                dactitems.append(DialogueActItem('food', Operator.EQ, state.item_in_focus['food']))
            if result['sys_pricerange']['predictions'][0]:
                dactitems.append(DialogueActItem('pricerange', Operator.EQ, state.item_in_focus['pricerange']))
            if result['sys_addr']['predictions'][0]:
                dactitems.append(DialogueActItem('addr', Operator.EQ, state.item_in_focus['addr']))
            if result['sys_name']['predictions'][0]:
                dactitems.append(DialogueActItem('name', Operator.EQ, state.item_in_focus['name']))
                # Make sure no inform(name) goes out
                dact.intent = 'offer'
            if result['sys_phone']['predictions'][0]:
                dactitems.append(DialogueActItem('phone', Operator.EQ, state.item_in_focus['phone']))
            if result['sys_postcode']['predictions'][0]:
                dactitems.append(DialogueActItem('postcode', Operator.EQ, state.item_in_focus['postcode']))

        elif dact.intent == 'request':
            # TODO: Get the slots from the ontology and loop over them
            if result['sys_area']['predictions'][0]:
                dactitems.append(DialogueActItem('area', Operator.EQ, ''))
            if result['sys_food']['predictions'][0]:
                dactitems.append(DialogueActItem('food', Operator.EQ, ''))
            if result['sys_pricerange']['predictions'][0]:
                dactitems.append(DialogueActItem('pricerange', Operator.EQ, ''))
            if result['sys_addr']['predictions'][0]:
                dactitems.append(DialogueActItem('addr', Operator.EQ, ''))
            if result['sys_name']['predictions'][0]:
                dactitems.append(DialogueActItem('name', Operator.EQ, ''))
            if result['sys_phone']['predictions'][0]:
                dactitems.append(DialogueActItem('phone', Operator.EQ, ''))
            if result['sys_postcode']['predictions'][0]:
                dactitems.append(DialogueActItem('postcode', Operator.EQ, ''))

        elif dact.intent != 'bye' and state.slots_filled['area'] and state.slots_filled['food'] and state.slots_filled['pricerange']:
            dact.intent = 'db_lookup'

        elif dact.intent == 'na' and not state.slots_filled['area'] and not state.slots_filled['food'] and not state.slots_filled['pricerange']:
            dact.intent = 'welcomemsg'

        dact.params = dactitems

        return [dact]

    def train(self, dialogues):
        pass

    def save(self, path=None):
        pass

    def load(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                print('Loading Ludwig Policy model...')
                self.model = LudwigModel.load(path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig Policy model directory {0} not found'.format(path))
        else:
            raise ValueError('Ludwig Policy: Unacceptable value for model file name: {0} '.format(path))

    # These endocings are the same that the SupervisedPolicy uses, for performance comparison.
    def encode_state(self, state):
        '''
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state encoding
        '''

        temp = []

        # temp += [int(b) for b in format(state.turn, '06b')]

        for value in state.slots_filled.values():
            temp.append(1) if value else temp.append(0)

        temp.append(int(state.is_terminal_state))

        # If the agent is a system, then this shows what the top db result is.
        # If the agent is a user, then this shows what information the system has provided
        if self.agent_role == 'system':
            state_filled_info = []
            requested_slot = []

            for slot in self.requestable_slots:
                if state.item_in_focus and slot in state.item_in_focus and state.item_in_focus[slot]:
                    state_filled_info.append(1)
                else:
                    state_filled_info.append(0)

                requested_slot.append(1) if slot == state.requested_slot else requested_slot.append(0)

            temp += state_filled_info + requested_slot
            # temp += requested_slot

        #     if self.agent_role == 'system':
        #         if state.system_requestable_slot_entropies:
        #             max_entr = max(state.system_requestable_slot_entropies.values())
        #             temp += [1 if state.system_requestable_slot_entropies[s] == max_entr else 0 for s in state.system_requestable_slot_entropies]
        #         else:
        #             temp += [0] * len(self.system_requestable_slots)
        #
        # elif self.agent_role == 'system':
        #     temp += [0] * (2 * len(self.requestable_slots) + len(self.system_requestable_slots))
        #
        elif self.agent_role == 'user':
            temp += [0] * (2 * len(self.requestable_slots))

        temp.append(1) if state.system_made_offer else temp.append(0)

        # if state.user_acts:
        #     # If this agent is the system then "user" is a user (hopefully).
        #     # If this agent is a user then "user" is a system.
        #     temp += [int(b) for b in format(self.encode_action(state.user_acts, self.agent_role != 'system'), '05b')]
        # else:
        #     temp += [0, 0, 0, 0, 0]
        #
        # if state.last_sys_acts:
        #     temp += [int(b) for b in
        #              format(self.encode_action([state.last_sys_acts[0]], self.agent_role == 'system'), '05b')]
        # else:
        #     temp += [0, 0, 0, 0, 0]
        #
        # # If the agent plays the role of the user it needs access to its own goal
        # if state.user_goal:
        #     for c in self.informable_slots:
        #         if c in state.user_goal.constraints and state.user_goal.constraints[c].value:
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        #
        #     for r in self.requestable_slots:
        #         if r in state.user_goal.requests and state.user_goal.requests[r].value:
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        #
        #     for r in self.requestable_slots:
        #         if r in state.user_goal.actual_requests and state.user_goal.actual_requests[r].value:
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        # else:
        #     temp += [0] * (len(self.informable_slots) + 2 * len(self.requestable_slots))

        return temp

    def encode_action(self, actions, system=True):
        '''

        :param actions:
        :return:
        '''

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Supervised Policy action encoding called with empty actions list (returning 0).')
            return 0

        action = actions[0]

        if action.intent == 'bye':
            return 0

        if action.intent == 'offer':
            return 1

        if action.intent == 'request' and action.params:
            if system:
                return 2 + self.system_requestable_slots.index(action.params[0].slot)
            else:
                return 2 + self.requestable_slots.index(action.params[0].slot)

        if action.intent == 'inform' and action.params:
            if system:
                return 2 + len(self.system_requestable_slots) + self.requestable_slots.index(action.params[0].slot)
            else:
                return 2 + len(self.requestable_slots) + self.requestable_slots.index(action.params[0].slot)

        if action.intent in ['hello', 'welcomemsg']:
            if system:
                return 2 + len(self.system_requestable_slots) + len(self.requestable_slots)
            else:
                return 2 + 2 * len(self.requestable_slots)

        if action.intent == 'canthelp' and system:
            return self.NActions - 1

        # Default fall-back action
        print('Supervised ({0}) policy action encoder warning: Selecting default action (unable to encode: {1})!'.format(self.agent_role, action))
        return 0
