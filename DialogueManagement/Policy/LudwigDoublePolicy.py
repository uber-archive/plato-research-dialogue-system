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
from ludwig.api import LudwigModel
import os.path
import pandas as pd


class LudwigDoublePolicy(Policy):
    def __init__(self, policy_model_path, policy_metadata_path, slot_model_path, slot_medatada_path):
        super(LudwigDoublePolicy, self).__init__()

        self.model = None

        if isinstance(policy_model_path, str) and isinstance(policy_metadata_path, str):
            if os.path.isdir(policy_model_path) and os.path.isfile(policy_metadata_path):
                self.policy_model = LudwigModel.load(policy_model_path, policy_metadata_path)
                self.slot_model = LudwigModel.load(slot_model_path, slot_medatada_path)

            else:
                raise FileNotFoundError(
                    'Model directory {0} or metadata file {1} not found'.format(policy_model_path, policy_metadata_path))
        else:
            raise ValueError(
                'Unacceptable value for model file name: {0} or metadata file name: {1} '.format(policy_model_path,
                                                                                                 policy_metadata_path))

    def initialize(self, **kwargs):
        '''
        Initialize anything that should not be in __init__

        :return: Nothing
        '''

        pass

    def restart(self, args):
        '''
        Re-initialize relevant parameters / variables at the beginning of each dialogue.

        :return:
        '''

        pass

    def next_action(self, state):
        data_dic = {'prev_ds_area': ['na'],
                    'prev_ds_food': ['na'],
                    'prev_ds_pricerange': ['na'],
                    'prev_ds_requested': ['na'],
                    # 'prev_ds_user_terminating': [False]}
                    'db_matches_ratio': [1.0],
                    'act': ['na'],
                    'inform_area': ['na'],
                    'inform_food': ['na'],
                    'inform_pricerange': ['na'],
                    'request_area': ['na'],
                    'request_food': ['na'],
                    'request_pricerange': ['na'],
                    'request_addr': ['na'],
                    'request_name': ['na'],
                    'request_phone': ['na'],
                    'request_postcode': ['na']
                    }

        if state.slots_filled['area']:
            data_dic['prev_ds_area'] = [state.slots_filled['area']]

        if state.slots_filled['food']:
            data_dic['prev_ds_food'] = [state.slots_filled['food']]

        if state.slots_filled['pricerange']:
            data_dic['prev_ds_pricerange'] = [state.slots_filled['pricerange']]

        if state.slots_filled['requested']:
            data_dic['prev_ds_requested'] = [state.slots_filled['requested']]

        # data_dic['ds_user_terminating'] = [state.user_terminating]
        data_dic['db_matches_ratio'] = [state.db_matches_ratio]

        if state.user_acts:
            data_dic['act'] = state.user_acts[0].intent

        sys_act_result = self.policy_model.predict(pd.DataFrame(data=data_dic))

        dact = DialogueAct(sys_act_result['system_act_predictions'][0], [])
        
        data_dic['system_act'] = dact.intent
        
        sys_slot_result = self.slot_model.predict(pd.DataFrame(data=data_dic))

        dactitems = []

        if dact.intent in ['inform, offer'] and state.item_in_focus:
            if sys_slot_result['sys_area_predictions'][0]:
                dactitems.append(DialogueActItem('area', Operator.EQ, state.item_in_focus['area']))
            if sys_slot_result['sys_food_predictions'][0]:
                dactitems.append(DialogueActItem('food', Operator.EQ, state.item_in_focus['food']))
            if sys_slot_result['sys_pricerange_predictions'][0]:
                dactitems.append(DialogueActItem('pricerange', Operator.EQ, state.item_in_focus['pricerange']))
            if sys_slot_result['sys_addr_predictions'][0]:
                dactitems.append(DialogueActItem('addr', Operator.EQ, state.item_in_focus['addr']))
            if sys_slot_result['sys_name_predictions'][0]:
                dactitems.append(DialogueActItem('name', Operator.EQ, state.item_in_focus['name']))
            if sys_slot_result['sys_phone_predictions'][0]:
                dactitems.append(DialogueActItem('phone', Operator.EQ, state.item_in_focus['phone']))
            if sys_slot_result['sys_postcode_predictions'][0]:
                dactitems.append(DialogueActItem('postcode', Operator.EQ, state.item_in_focus['postcode']))

        elif dact.intent == 'request':
            # TODO: Get the slots from the ontology and loop over them
            if sys_slot_result['sys_area_predictions'][0]:
                dactitems.append(DialogueActItem('area', Operator.EQ, ''))
            if sys_slot_result['sys_food_predictions'][0]:
                dactitems.append(DialogueActItem('food', Operator.EQ, ''))
            if sys_slot_result['sys_pricerange_predictions'][0]:
                dactitems.append(DialogueActItem('pricerange', Operator.EQ, ''))
            if sys_slot_result['sys_addr_predictions'][0]:
                dactitems.append(DialogueActItem('addr', Operator.EQ, ''))
            if sys_slot_result['sys_name_predictions'][0]:
                dactitems.append(DialogueActItem('name', Operator.EQ, ''))
            if sys_slot_result['sys_phone_predictions'][0]:
                dactitems.append(DialogueActItem('phone', Operator.EQ, ''))
            if sys_slot_result['sys_postcode_predictions'][0]:
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

    def load(self, path=None):
        pass
