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

from DialogueStateTracker.LudwigDST import LudwigDST
from Dialogue.State import SlotFillingDialogueState
from copy import deepcopy
import pandas as pd


class CamRestDST(LudwigDST):
    def __init__(self, args):
        super(CamRestDST, self).__init__(args)

        self.DState = SlotFillingDialogueState([])

    def initialize(self):
        super(CamRestDST, self).initialize()
        self.DState.initialize()

    def update_state(self, dacts):
        self.DState.user_acts = deepcopy(dacts)

        input_data = {'transcription': '',
                      'prev_ds_area': ['na'],
                      'prev_ds_food': ['na'],
                      'prev_ds_pricerange': ['na'],
                      'prev_ds_requested': ['na'],
                      'prev_ds_user_terminating': [False],
                      'act': ['na'],
                      'inform_area': ['na'],
                      'inform_food': ['na'],
                      'inform_pricerange': ['na'],
                      'request_area': [False],
                      'request_food': [False],
                      'request_pricerange': [False],
                      'request_addr': [False],
                      'request_name': [False],
                      'request_phone': [False],
                      'request_postcode': [False]}

        if self.DState.slots_filled['area']:
            input_data['prev_ds_area'] = [self.DState.slots_filled['area']]

        if self.DState.slots_filled['food']:
            input_data['prev_ds_food'] = [self.DState.slots_filled['food']]

        if self.DState.slots_filled['pricerange']:
            input_data['prev_ds_pricerange'] = [self.DState.slots_filled['pricerange']]

        if self.DState.slots_filled['requested']:
            input_data['prev_ds_requested'] = [self.DState.slots_filled['requested']]

        if self.DState.is_terminal_state:
            input_data['prev_ds_user_terminating'] = [self.DState.is_terminal_state]

        for dact in dacts:
            input_data['act'] = dact.intent

            if dact.intent in ['inform', 'request']:
                for item in dact.params:
                    input_data[dact.intent + '_' + item.slot] = [item.value]

            # Warning: Make sure the same tokenizer that was used to train the model is used during prediction
            result = self.model.predict(pd.DataFrame(data=input_data))

            self.DState.slots_filled['area'] = result['ds_area_predictions'][0] if result['ds_area_predictions'][0] != 'na' else ''
            self.DState.slots_filled['food'] = result['ds_food_predictions'][0] if result['ds_food_predictions'][0] != 'na' else ''
            self.DState.slots_filled['pricerange'] = result['ds_pricerange_predictions'][0] if result['ds_pricerange_predictions'][0] != 'na' else ''
            self.DState.requested_slot = result['ds_requested_predictions'][0] if result['ds_requested_predictions'][0] != 'na' else ''
            self.DState.is_terminal_state = result['ds_user_terminating_predictions'][0]

        self.DState.turn += 1

        return self.DState

    def update_state_db(self, db_result):
        if db_result:
            self.DState.db_matches_ratio = len(db_result)
            self.DState.item_in_focus = db_result[0]

        return self.DState

    def update_state_sysact(self, sys_acts):
        for sys_act in sys_acts:
            if sys_act.intent == 'offer':
                self.DState.system_made_offer = True

            # Reset the request if the system asks for more information, assuming that any previously offered item
            # is now invalid.
            elif sys_act.intent == 'request':
                self.DState.system_made_offer = False

    def get_state(self):
        return self.DState

    def train(self):
        pass

    def save(self):
        pass

    def load(self, path):
        pass