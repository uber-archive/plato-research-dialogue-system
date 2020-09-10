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

import unittest
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.dialogue.state import SlotFillingDialogueState
from plato.agent.component.nlu.slot_filling_nlu import SlotFillingNLU
from plato.agent.component.nlg.slot_filling_nlg import SlotFillingNLG
from plato.controller import basic_controller

'''
The unit_test class will run through some unit tests for Plato's basic functionalities.
'''


class unit_test(unittest.TestCase):
    def test_dialogue_act_item(self):
        self.assertEqual(
            str(DialogueActItem('slot', Operator.EQ, 'value')),
            'slot = value'
        )

    def test_dialogue_act(self):
        self.assertEqual(
            str(DialogueAct('intent', [DialogueActItem('slot', Operator.EQ, 'value')])),
            'intent(slot = value, )'
        )

    def test_dialogue_state(self):
        self.assertEqual(
            str(SlotFillingDialogueState({
                'slots': ['slot1']
            })),
            'SlotFillingDialogueState\nSlots: {}\nSlot Queries: {}\nRequested Slot: \nsys Made Offer: False\nTurn: 0\n'
        )

    def test_nlu(self):
        self.assertEqual(
            SlotFillingNLU({'ontology': 'example/domains/CamRestaurants-rules.json',
                            'database': 'example/domains/CamRestaurants-dbase.db'}).process_input(
                'looking for an expensive restaurant'
            ),
            [DialogueAct('inform',
                         [DialogueActItem('pricerange', Operator.EQ, 'expensive')])]
        )

    def test_nlg(self):
        self.assertEqual(
            SlotFillingNLG().generate_output({
                'dacts': [DialogueAct('request', [DialogueActItem('pricerange', Operator.EQ, '')])],
                'system': True}
            ),
            'Which price range do you prefer? '
        )


'''
The end_to_end_test class runs Plato on the test configurations defined in example/test. These are the most
comprehensive tests, designed to test the entire pipeline on its various configurations (single agent, multi agent,
end-to-end, with a user simulator, etc.).
'''


class end_to_end_test(unittest.TestCase):
    """
    Returns True if all tests pass, otherwise returns False.
    """
    def test_e2e(self):
        self.assertEqual(basic_controller.run({}, True), True)


'''
This script needs to be called from the root directory. For example:

python test/plato_tests.py
'''
if __name__ == '__main__':
    unittest.main()
