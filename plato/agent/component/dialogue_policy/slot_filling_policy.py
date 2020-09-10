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

from plato.domain.ontology import Ontology
from plato.agent.component.dialogue_policy import dialogue_policy
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator

from copy import deepcopy

import random

"""
HandcraftedPolicy is a rule-based system policy, developed as a baseline and as
a quick way to perform sanity checks and debug a Conversational Agent. 
It will try to fill unfilled slots, then suggest an item, and answer any 
requests from the user.
"""


class HandcraftedPolicy(dialogue_policy.DialoguePolicy):

    def __init__(self, args):
        """
        Load the ontology.

        :param args: contain the domain ontology
        """
        super(HandcraftedPolicy, self).__init__()

        if 'ontology' in args:
            ontology = args['ontology']
        else:
            raise ValueError('No ontology provided for HandcraftedPolicy!')

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

    def initialize(self, args):
        """
        Nothing to do here

        :param args:
        :return:
        """
        pass

    def next_action(self, dialogue_state):
        """
        Generate a response given which conditions are met by the current
        dialogue state.

        :param dialogue_state:
        :return:
        """
        # Check for terminal state
        if dialogue_state.is_terminal_state:
            return [DialogueAct('bye', [DialogueActItem('', Operator.EQ, '')])]

        # Check if the user has made any requests
        elif dialogue_state.requested_slot:
            if dialogue_state.item_in_focus and \
                    dialogue_state.system_made_offer:
                requested_slot = dialogue_state.requested_slot

                # Reset request as we attempt to address it
                dialogue_state.requested_slot = ''

                value = 'not available'
                if requested_slot in dialogue_state.item_in_focus and \
                        dialogue_state.item_in_focus[requested_slot]:
                    value = dialogue_state.item_in_focus[requested_slot]

                return \
                    [DialogueAct(
                        'inform',
                        [DialogueActItem(requested_slot, Operator.EQ, value)])]

            # Else, if no item is in focus or no offer has been made,
            # ignore the user's request

        # Try to fill slots
        requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        if not hasattr(dialogue_state, 'requestable_slot_entropies') or \
                not dialogue_state.requestable_slot_entropies:
            slot = random.choice(requestable_slots)

            while dialogue_state.slots_filled[slot] and \
                    len(requestable_slots) > 1:
                requestable_slots.remove(slot)
                slot = random.choice(requestable_slots)

        else:
            slot = ''
            slots = \
                [k for k, v in
                 dialogue_state.requestable_slot_entropies.items()
                 if v == max(
                    dialogue_state.requestable_slot_entropies.values())
                 and v > 0 and k in requestable_slots]

            if slots:
                slot = random.choice(slots)

                while dialogue_state.slots_filled[slot] \
                        and dialogue_state.requestable_slot_entropies[
                    slot] > 0 \
                        and len(requestable_slots) > 1:
                    requestable_slots.remove(slot)
                    slots = \
                        [k for k, v in
                         dialogue_state.requestable_slot_entropies.items()
                         if v == max(
                            dialogue_state.requestable_slot_entropies.values())
                         and k in requestable_slots]

                    if slots:
                        slot = random.choice(slots)
                    else:
                        break

        if slot and not dialogue_state.slots_filled[slot]:
            return [DialogueAct(
                'request',
                [DialogueActItem(slot, Operator.EQ, '')])]

        elif dialogue_state.item_in_focus:
            name = dialogue_state.item_in_focus['name'] \
                if 'name' in dialogue_state.item_in_focus \
                else 'unknown'

            dacts = [DialogueAct(
                'offer',
                [DialogueActItem('name', Operator.EQ, name)])]

            for slot in dialogue_state.slots_filled:
                if slot != 'requested' and dialogue_state.slots_filled[slot]:
                    if slot in dialogue_state.item_in_focus:
                        if slot not in ['id', 'name']:
                            dacts.append(
                                DialogueAct(
                                    'inform',
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        dialogue_state.item_in_focus[slot])]))
                    else:
                        dacts.append(DialogueAct(
                            'inform',
                            [DialogueActItem(
                                slot,
                                Operator.EQ,
                                'no info')]))

            return dacts
        else:
            # Fallback action - cannot help!
            # Note: We can have this check (no item in focus) at the beginning,
            # but this would assume that the system
            # queried a database before coming in here.
            return [DialogueAct('canthelp', [])]

    def train(self, data):
        """
        Nothing to do here.

        :param data:
        :return:
        """
        pass

    def restart(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def save(self, path=None):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass

    def load(self, path):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass
