"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from DialogueStateTracker.LudwigDST import LudwigDST
from Dialogue.State import SlotFillingDialogueState
from copy import deepcopy
import pandas as pd

"""
CamRestDST implements Dialogue State Tracking for the Cambridge Restaurants 
domain, using a Ludwig model.
"""


class CamRestLudwigDST(LudwigDST):
    def __init__(self, args):
        """
        Initialize the Ludwig model and the dialogue state.

        :param args: a dictionary containing the path to the Ludwig model
        """
        super(CamRestLudwigDST, self).__init__(args)

        self.DState = SlotFillingDialogueState([])

    def initialize(self):
        """
        Initialize the dialogue state

        :return: nothing
        """

        super(CamRestLudwigDST, self).initialize()
        self.DState.initialize()

    def update_state(self, dacts):
        """
        Updates the current dialogue state given the input dialogue acts. This
        function will process the input, query the ludwig model, and retrieve
        the updated dialogue state.

        :param dacts: the input dialogue acts (usually NLU output)
        :return: the current dialogue state
        """

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
            input_data['prev_ds_pricerange'] = \
                [self.DState.slots_filled['pricerange']]

        if self.DState.slots_filled['requested']:
            input_data['prev_ds_requested'] = \
                [self.DState.slots_filled['requested']]

        if self.DState.is_terminal_state:
            input_data['prev_ds_user_terminating'] = \
                [self.DState.is_terminal_state]

        for dact in dacts:
            input_data['act'] = dact.intent

            if dact.intent in ['inform', 'request']:
                for item in dact.params:
                    input_data[dact.intent + '_' + item.slot] = [item.value]

            # Warning: Make sure the same tokenizer that was used to train
            # the model is used during prediction
            result = self.model.predict(pd.DataFrame(data=input_data))

            self.DState.slots_filled['area'] = \
                result['ds_area_predictions'][0] \
                if result['ds_area_predictions'][0] != 'na' else ''

            self.DState.slots_filled['food'] = \
                result['ds_food_predictions'][0] \
                if result['ds_food_predictions'][0] != 'na' else ''

            self.DState.slots_filled['pricerange'] = \
                result['ds_pricerange_predictions'][0] \
                if result['ds_pricerange_predictions'][0] != 'na' else ''

            self.DState.requested_slot = \
                result['ds_requested_predictions'][0] \
                if result['ds_requested_predictions'][0] != 'na' else ''

            self.DState.is_terminal_state = \
                result['ds_user_terminating_predictions'][0]

        self.DState.turn += 1

        return self.DState

    def update_state_db(self, db_result):
        """
        Updates the current database results in the dialogue state.

        :param db_result: a dictionary containing the database query results
        :return:
        """

        if db_result:
            self.DState.db_matches_ratio = len(db_result)
            self.DState.item_in_focus = db_result[0]

        return self.DState

    def update_state_sysact(self, sys_acts):
        """
        Updates the state, given the last system act. This is
        useful as we may want to update parts of the state given NLU output
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

        :param data: dialogue experience
        :return:
        """
        pass

    def save(self, model_path=None):
        """
        Saves the Ludwig model.

        :param model_path: path to save the model to
        :return:
        """
        super(CamRestLudwigDST, self).save(model_path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        super(CamRestLudwigDST, self).load(model_path)
