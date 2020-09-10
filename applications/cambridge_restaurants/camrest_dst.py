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

from plato.agent.component.dialogue_state_tracker.ludwig_dst import LudwigDST
from plato.dialogue.state import SlotFillingDialogueState
from copy import deepcopy
import pandas as pd

"""
CamRestDST implements Dialogue State Tracking for the Cambridge Restaurants 
domain, using a Ludwig model.
"""


class CamRestDST(LudwigDST):
    def __init__(self, args):
        """
        Initialize the Ludwig model and the dialogue state.

        :param args: a dictionary containing the path to the Ludwig model
        """
        super(CamRestDST, self).__init__(args)

        self.DState = SlotFillingDialogueState([])

    def initialize(self, args):
        """
        Initialize the dialogue state

        :return: nothing
        """

        super(CamRestDST, self).initialize(args)
        self.DState.initialize()

    def update_state(self, dacts):
        """
        Updates the current dialogue state given the input dialogue acts. This
        function will process the input, query the ludwig model, and retrieve
        the updated dialogue state.

        :param dacts: the input dialogue acts (usually nlu output)
        :return: the current dialogue state
        """

        self.DState.user_acts = deepcopy(dacts)

        dst_in = dict()

        dst_in['dst_prev_area'] = 'none'
        if self.DState.slots_filled['area']:
            dst_in['dst_prev_area'] = self.DState.slots_filled['area']

        dst_in['dst_prev_food'] = 'none'
        if self.DState.slots_filled['food']:
            dst_in['dst_prev_food'] = self.DState.slots_filled['food']

        dst_in['dst_prev_pricerange'] = 'none'
        if self.DState.slots_filled['pricerange']:
            dst_in['dst_prev_pricerange'] = \
                self.DState.slots_filled['pricerange']

        dst_in['nlu_intent'] = 'none'
        dst_in['req_slot'] = 'none'
        dst_in['inf_area_value'] = 'none'
        dst_in['inf_food_value'] = 'none'
        dst_in['inf_pricerange_value'] = 'none'

        intents = set()

        for da in dacts:
            intents.add(da.intent)

            for p in da.params:
                if da.intent == 'inform':
                    if p.slot == 'area':
                        dst_in['inf_area_value'] = p.value

                    elif p.slot == 'food':
                        dst_in['inf_food_value'] = p.value

                    elif p.slot == 'pricerange':
                        dst_in['inf_pricerange_value'] = p.value

                elif da.intent == 'request':
                    dst_in['req_slot'] = p.slot

        dst_in['nlu_intent'] = ' '.join(intents)

        input_data = [dst_in]

        # Warning: Make sure the same tokenizer that was used to train
        # the model is used during prediction
        result = self.model.predict(pd.DataFrame(data=input_data))

        area_prediction = result['dst_area_predictions'][0]
        if area_prediction == 'none':
            area_prediction = None

        self.DState.slots_filled['area'] = area_prediction

        food_prediction = result['dst_food_predictions'][0]
        if food_prediction == 'none':
            food_prediction = None

        self.DState.slots_filled['food'] = food_prediction

        pricerange_prediction = result['dst_pricerange_predictions'][0]
        if pricerange_prediction == 'none':
            pricerange_prediction = None

        self.DState.slots_filled['pricerange'] = pricerange_prediction

        req_slot_prediction = result['dst_req_slot_predictions'][0]
        if req_slot_prediction == 'none':
            req_slot_prediction = None

        self.DState.requested_slot = req_slot_prediction
        self.DState.is_terminal_state = 'bye' in intents

        self.DState.turn += 1

        return self.DState

    def update_state_db(self, db_result, sys_req_slot_entropies=None):
        """
        Updates the current database results in the dialogue state.

        :param db_result: a dictionary containing the database query results
        :param sys_req_slot_entropies: entropy values for each slot
        :return:
        """

        if db_result:
            self.DState.db_matches_ratio = len(db_result)
            self.DState.item_in_focus = db_result[0]

        if sys_req_slot_entropies:
            self.DState.system_requestable_slot_entropies = \
                deepcopy(sys_req_slot_entropies)

        return self.DState

    def update_state_sysact(self, sys_acts):
        """
        Updates the state, given the last system act. This is
        useful as we may want to update parts of the state given nlu output
        and then update again once the system produces a response.

        :param sys_acts:
        :return:
        """

        for sys_act in sys_acts:
            if sys_act.intent == 'offer':
                self.DState.system_made_offer = True

            # Reset the request if the system asks for more information,
            # assuming that any previously offered item
            # is now invalid.
            elif sys_act.intent == 'request':
                self.DState.system_made_offer = False

    def get_state(self):
        """
        Returns the current dialogue state

        :return: the current dialogue state
        """
        return self.DState

    def train(self, dialogue_episodes):
        """
        Not implemented.

        We can use Ludwig's API to train the model online (i.e. for a single
        dialogue).

        :param dialogue_episodes: dialogue experience
        :return:
        """
        pass

    def save(self, model_path=None):
        """
        Saves the Ludwig model.

        :param model_path: path to save the model to
        :return:
        """
        if model_path:
            super(CamRestDST, self).save(model_path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        super(CamRestDST, self).load(model_path)
