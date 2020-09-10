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

from abc import ABC, abstractmethod
from copy import deepcopy

"""
State models the internal state of a Conversational Agent. It is the abstract 
parent class of any State and it defines the interface that should be adhered 
to. States may be accessed by any stateful module but should only be updated
by the dialogue_state_tracker.
"""


class State(ABC):

    @abstractmethod
    def initialize(self):
        """
        Initialize the state (e.g. at the start of a dialogue)

        :return: nothing
        """

        pass

    @abstractmethod
    def is_terminal(self):
        """
        Check if this state is terminal

        :return: True or False
        """

        pass


class DialogueState(State):
    def __init__(self):
        """
        Initialize the dialogue State
        """
        super(DialogueState, self).__init__()

        self.dialogStateUuid = -1
        self.context = Context()
        self.intents = []
        self.is_terminal_state = False
        self.last_sys_acts = None

    def initialize(self):
        """
        Initialize intents and terminal status
        :return:
        """

        self.intents = []
        self.is_terminal_state = False

    def is_terminal(self):
        """
        Check if this state is terminal

        :return: True or False
        """

        return self.is_terminal_state


class SlotFillingDialogueState(DialogueState):
    def __init__(self, args):
        """
        Initialize the Slot Filling dialogue State internal structures
        :param args:
        """
        super(SlotFillingDialogueState, self).__init__()

        self.slots_filled = {}
        self.slot_queries = {}
        self.system_requestable_slot_entropies = {}

        self.slots = None

        if 'slots' in args:
            self.slots = deepcopy(args['slots'])
        else:
            print('WARNING! SlotFillingDialogueState not provided with slots, '
                  'using default CamRest slots.')
            self.slots = ['area', 'food', 'pricerange']

        self.requested_slot = ''

        self.user_acts = None
        self.system_made_offer = False

        # TODO: Have a list of past items in focus - e.g. current and 2 past
        #  items
        # If the agent is a user, then this structure will store information
        # that the system has provided.
        self.item_in_focus = None
        self.db_result = None

        self.db_matches_ratio = 1.0
        self.last_sys_acts = None
        self.turn = 0
        self.num_dontcare = 0

        # NOTE: This should ONLY be used if an agent plays the role of a user
        self.user_goal = None

    def __str__(self):
        """
        Print the Slot Filling dialogue State

        :return: a string representation of the Slot Filling dialogue State
        """
        ret = 'SlotFillingDialogueState\n'
        ret += 'Slots: ' + str(self.slots_filled) + '\n'
        ret += 'Slot Queries: ' + str(self.slot_queries) + '\n'
        ret += 'Requested Slot: ' + self.requested_slot + '\n'
        ret += 'sys Made Offer: ' + str(self.system_made_offer) + '\n'
        ret += 'Turn: ' + str(self.turn) + '\n'
        return ret

    def initialize(self, args=None):
        """
        Initialize the Slot Filling dialogue State (e.g. at the start of a
        dialogue). Reset filled slots, slot queries, entropies, and other
        structures.

        :param args:
        :return:
        """
        self.slots_filled = dict.fromkeys(self.slots)
        self.slot_queries = {}
        self.system_requestable_slot_entropies = {}
        self.requested_slot = ''

        self.user_acts = None
        self.is_terminal_state = False
        self.system_made_offer = False

        # TODO: Have a list of past items in focus - e.g. current and 2 past
        # items
        # If the agent is a user, then this structure will store information
        # that the system has provided.
        self.item_in_focus = None
        self.db_result = None

        self.db_matches_ratio = 1.0
        self.last_sys_acts = None
        self.turn = 0
        self.num_dontcare = 0

        # NOTE: This is ONLY used if an agent plays the role of the user
        if args and 'goal' in args:
            self.user_goal = deepcopy(args['goal'])
        else:
            self.user_goal = None

    def is_terminal(self):
        """
        Check if this state is terminal

        :return: True or False
        """

        return self.is_terminal_state


class Context:
    """
    Not implemented. Class to hold context.
    """
    def __init__(self):
        self.params = {}
