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

from plato.agent.conversational_agent.\
    conversational_generic_agent import ConversationalGenericAgent
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.agent.component.dialogue_state_tracker.\
    slot_filling_dst import SlotFillingDST
from plato.domain.ontology import Ontology
from plato.domain.database import DataBase, SQLDataBase, JSONDataBase

from copy import deepcopy

from plato.agent.component.conversational_module \
    import ConversationalModule

import random
import math

"""
The DialogueManagerGeneric consists of a dialogue_state_tracker and a 
policy. Contrary to DialogueManager, the generic version will load the
dialogue state tracker and policy using the class path and name provided.
The DialogueManagerGeneric handles the decision-making part of the 
Conversational Agent. Given new input (a list of DialogueActs) it will ensure 
that the state is updated properly and will output a list of DialogueActs in 
response, after querying its policy.
"""


class DialogueManagerGeneric(ConversationalModule):
    def __init__(self, args):
        """
        Parses the arguments in the dictionary and initializes the appropriate
        models for dialogue State Tracking and dialogue Policy.

        :param args: the configuration file parsed into a dictionary
        """

        super(DialogueManagerGeneric, self).__init__()

        if 'settings' not in args:
            raise AttributeError(
                'DialogueManagerGeneric: Please provide settings (config)!')
        if 'ontology' not in args:
            raise AttributeError(
                'DialogueManagerGeneric: Please provide ontology!')
        if 'database' not in args:
            raise AttributeError(
                'DialogueManagerGeneric: Please provide database!')
        if 'domain' not in args:
            raise AttributeError(
                'DialogueManagerGeneric: Please provide domain!')

        settings = args['settings']
        ontology = args['ontology']
        database = args['database']
        domain = args['domain']

        agent_id = 0
        if 'agent_id' in args:
            agent_id = int(args['agent_id'])

        agent_role = 'system'
        if 'agent_role' in args:
            agent_role = args['agent_role']

        self.settings = settings

        self.TRAIN_DST = False
        self.TRAIN_POLICY = False

        self.MAX_DB_RESULTS = 10

        self.DSTracker = None
        self.DSTracker_info = {}

        self.policy = None
        self.policy_info = {}

        self.policy_path = None
        self.ontology = None
        self.database = None
        self.domain = None

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.dialogue_counter = 0
        self.CALCULATE_SLOT_ENTROPIES = True

        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        if isinstance(database, DataBase):
            self.database = database

        elif isinstance(database, str):
            if database[-3:] == '.db':
                self.database = SQLDataBase(database)
            elif database[-5:] == '.json':
                self.database = JSONDataBase(database)
            else:
                raise ValueError('Unacceptable database type %s ' % database)

        else:
            raise ValueError('Unacceptable database type %s ' % database)
                
        if args and args['policy']:
            if 'domain' in self.settings['DIALOGUE']:
                self.domain = self.settings['DIALOGUE']['domain']
            else:
                raise ValueError(
                    'domain is not specified in DIALOGUE at config.')

            if 'calculate_slot_entropies' in args:
                self.CALCULATE_SLOT_ENTROPIES = \
                    bool(args['calculate_slot_entropies'])

            if 'package' in args['policy'] and 'class' in args['policy']:
                self.policy_info = args['policy']

                if 'global_arguments' in args['settings']['GENERAL']:
                    if 'arguments' not in self.policy_info:
                        self.policy_info['arguments'] = {}

                    self.policy_info['arguments'].update(
                        args['settings']['GENERAL']['global_arguments']
                    )

                if 'train' in self.policy_info['arguments']:
                    self.TRAIN_POLICY = \
                        bool(self.policy_info['arguments']['train'])

                if 'policy_path' in self.policy_info['arguments']:
                    self.policy_path = \
                        self.policy_info['arguments']['policy_path']

                self.policy_info['arguments']['agent_role'] = self.agent_role

                # Replace ontology and database strings with the actual
                # objects to avoid repetitions (these won't change).
                if 'ontology' in self.policy_info['arguments']:
                    self.policy_info['arguments']['ontology'] = self.ontology

                if 'database' in self.policy_info['arguments']:
                    self.policy_info['arguments']['database'] = self.database

                self.policy = ConversationalGenericAgent.load_module(
                    self.policy_info['package'],
                    self.policy_info['class'],
                    self.policy_info['arguments']
                )

            else:
                raise ValueError('DialogueManagerGeneric: Cannot instantiate'
                                 'dialogue policy!')

        # DST Settings
        if 'DST' in args and args:
            if 'package' in args['DST'] and 'class' in args['DST']:
                self.DSTracker_info['package'] = args['DST']['package']
                self.DSTracker_info['class'] = args['DST']['class']

                self.DSTracker_info['args'] = {}

                if 'global_arguments' in args['settings']['GENERAL']:
                    self.DSTracker_info['args'] = \
                        args['settings']['GENERAL']['global_arguments']

                if 'arguments' in args['DST']:
                    self.DSTracker_info['args']. \
                        update(args['DST']['arguments'])

                self.DSTracker = ConversationalGenericAgent.load_module(
                    self.DSTracker_info['package'],
                    self.DSTracker_info['class'],
                    self.DSTracker_info['args']
                )

            else:
                raise ValueError('DialogueManagerGeneric: Cannot instantiate'
                                 'dialogue state tracker!')

        # Default to dummy DST, if no information is provided
        else:
            dst_args = dict(
                zip(
                    ['ontology', 'database', 'domain'],
                    [self.ontology, self.database, domain]))
            self.DSTracker = SlotFillingDST(dst_args)

        self.training = self.TRAIN_DST or self.TRAIN_POLICY

        self.load('')

    def initialize(self, args):
        """
        Initialize the relevant structures and variables of the dialogue
        Manager.

        :return: Nothing
        """

        self.DSTracker.initialize(self.DSTracker_info['args'])
        policy_args = self.policy_info['arguments']

        if 'goal' in args:
            policy_args.update({'goal': args['goal']})

        self.policy.initialize(policy_args)

        self.dialogue_counter = 0

    def receive_input(self, inpt):
        """
        Receive input and update the dialogue state.

        :return: Nothing
        """

        # Update dialogue state given the new input
        self.DSTracker.update_state(inpt)

        if self.domain and self.domain in ['CamRest', 'SFH', 'SlotFilling']:
            if self.agent_role == 'system':
                # Perform a database lookup
                db_result, sys_req_slot_entropies = self.db_lookup()

                # Update the dialogue state again to include the database
                # results
                self.DSTracker.update_state_db(
                    db_result=db_result,
                    sys_req_slot_entropies=sys_req_slot_entropies)

            else:
                # Update the dialogue state again to include the system actions
                self.DSTracker.update_state_db(db_result=None, sys_acts=inpt)

        return inpt

    def generate_output(self, args=None):
        """
        Consult the current policy to generate a response.

        :return: List of DialogueAct representing the system's output.
        """
        
        d_state = self.DSTracker.get_state()

        sys_acts = self.policy.next_action(d_state)
        # Copy the sys_acts to be able to iterate over all sys_acts while also
        # replacing some acts
        sys_acts_copy = deepcopy(sys_acts)
        new_sys_acts = []

        # Safeguards to support policies that make decisions on intents only
        # (i.e. do not output slots or values)
        for sys_act in sys_acts:
            if sys_act.intent == 'canthelp' and not sys_act.params:
                slots = \
                    [
                        s for s in d_state.slots_filled if
                        d_state.slots_filled[s]
                    ]
                if slots:
                    slot = random.choice(slots)

                    # Remove the empty canthelp
                    sys_acts_copy.remove(sys_act)

                    new_sys_acts.append(
                        DialogueAct(
                            'canthelp',
                            [DialogueActItem(
                                slot,
                                Operator.EQ,
                                d_state.slots_filled[slot])]))

                else:
                    print('DialogueManager Warning! No slot provided by '
                          'policy for canthelp and cannot find a reasonable '
                          'one!')

            if sys_act.intent == 'offer' and not sys_act.params:
                # Remove the empty offer
                sys_acts_copy.remove(sys_act)

                if d_state.item_in_focus:
                    new_sys_acts.append(
                        DialogueAct(
                            'offer',
                            [DialogueActItem(
                                'name',
                                Operator.EQ,
                                d_state.item_in_focus['name'])]))

                    # Only add these slots if no other acts were output
                    # by the DM
                    if len(sys_acts) == 1:
                        for slot in d_state.slots_filled:
                            if slot in d_state.item_in_focus:
                                if slot not in ['id', 'name'] and \
                                        slot != d_state.requested_slot:
                                    new_sys_acts.append(
                                        DialogueAct(
                                            'inform',
                                            [DialogueActItem(
                                                slot,
                                                Operator.EQ,
                                                d_state.item_in_focus[slot])]))
                            else:
                                new_sys_acts.append(
                                    DialogueAct(
                                        'inform',
                                        [DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            'no info')]))

            elif sys_act.intent == 'inform':
                if self.agent_role == 'system':
                    if sys_act.params and sys_act.params[0].value:
                        continue

                    if sys_act.params:
                        slot = sys_act.params[0].slot
                    else:
                        slot = d_state.requested_slot

                    if not slot:
                        slot = random.choice(list(d_state.slots_filled.keys()))

                    if d_state.item_in_focus:
                        if slot not in d_state.item_in_focus or \
                                not d_state.item_in_focus[slot]:
                            new_sys_acts.append(
                                DialogueAct(
                                    'inform',
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        'no info')]))
                        else:
                            if slot == 'name':
                                new_sys_acts.append(
                                    DialogueAct(
                                        'offer',
                                        [DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            d_state.item_in_focus[slot])]))
                            else:
                                new_sys_acts.append(
                                    DialogueAct(
                                        'inform',
                                        [DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            d_state.item_in_focus[slot])]))

                    else:
                        new_sys_acts.append(
                            DialogueAct(
                                'inform',
                                [DialogueActItem(
                                    slot,
                                    Operator.EQ,
                                    'no info')]))

                elif self.agent_role == 'user':
                    if sys_act.params:
                        slot = sys_act.params[0].slot

                        # Do nothing if the slot is already filled
                        if sys_act.params[0].value:
                            continue

                    elif d_state.last_sys_acts and d_state.user_acts and \
                            d_state.user_acts[0].intent == 'request':
                        slot = d_state.user_acts[0].params[0].slot

                    else:
                        slot = \
                            random.choice(
                                list(d_state.user_goal.constraints.keys()))

                    # Populate the inform with a slot from the user goal
                    if d_state.user_goal:
                        # Look for the slot in the user goal
                        if slot in d_state.user_goal.constraints:
                            value = d_state.user_goal.constraints[slot].value
                            print(f'value:{value}')
                        else:
                            value = 'dontcare'

                        new_sys_acts.append(
                            DialogueAct(
                                'inform',
                                [DialogueActItem(
                                    slot,
                                    Operator.EQ,
                                    value)]))

                # Remove the empty inform
                sys_acts_copy.remove(sys_act)

            elif sys_act.intent == 'request':
                # If the policy did not select a slot
                if not sys_act.params:
                    found = False

                    if self.agent_role == 'system':
                        # Select unfilled slot
                        for slot in d_state.slots_filled:
                            if not d_state.slots_filled[slot]:
                                found = True
                                new_sys_acts.append(
                                    DialogueAct(
                                        'request',
                                        [DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            '')]))
                                break

                    elif self.agent_role == 'user':
                        # Select request from goal
                        if d_state.user_goal:
                            for req in d_state.user_goal.requests:
                                if not d_state.user_goal.requests[req].value:
                                    found = True
                                    new_sys_acts.append(
                                        DialogueAct(
                                            'request',
                                            [DialogueActItem(
                                                req,
                                                Operator.EQ,
                                                '')]))
                                    break

                    if not found:
                        # All slots are filled
                        new_sys_acts.append(
                            DialogueAct(
                                'request',
                                [DialogueActItem(
                                    random.choice(
                                        list(
                                            d_state.slots_filled.keys())[:-1]),
                                    Operator.EQ, '')]))

                    # Remove the empty request
                    sys_acts_copy.remove(sys_act)

        # Append unique new sys acts
        for sa in new_sys_acts:
            if sa not in sys_acts_copy:
                sys_acts_copy.append(sa)

        self.DSTracker.update_state_sysact(sys_acts_copy)

        return sys_acts_copy

    def db_lookup(self):
        """
        Perform an SQLite query given the current dialogue state (i.e. given
        which slots have values).

        :return: a dictionary containing the current database results
        """

        # TODO: Add check to assert if each slot in d_state.slots_filled
        # actually exists in the schema.

        d_state = self.DSTracker.get_state()

        # Query the database
        db_result = self.database.db_lookup(d_state)

        if db_result:
            # Calculate entropy of requestable slot values in results -
            # if the flag is off this will be empty
            entropies = \
                dict.fromkeys(self.ontology.ontology['system_requestable'])

            if self.CALCULATE_SLOT_ENTROPIES:
                value_probabilities = {}

                # Count the values
                for req_slot in self.ontology.ontology['system_requestable']:
                    value_probabilities[req_slot] = {}

                    for db_item in db_result:
                        if db_item[req_slot] not in \
                                value_probabilities[req_slot]:
                            value_probabilities[req_slot][
                                db_item[req_slot]] = 1
                        else:
                            value_probabilities[req_slot][
                                db_item[req_slot]] += 1

                # Calculate probabilities
                for slot in value_probabilities:
                    for value in value_probabilities[slot]:
                        value_probabilities[slot][value] /= len(db_result)

                # Calculate entropies
                for slot in entropies:
                    entropies[slot] = 0

                    if slot in value_probabilities:
                        for value in value_probabilities[slot]:
                            entropies[slot] += \
                                value_probabilities[slot][value] * \
                                math.log(value_probabilities[slot][value])

                    entropies[slot] = -entropies[slot]

            return db_result[:self.MAX_DB_RESULTS], entropies

        # Failed to retrieve anything
        return ['empty'], {}

    def restart(self, args):
        """
        Restart the relevant structures or variables, e.g. at the beginning of
        a new dialogue.

        :return: Nothing
        """

        self.DSTracker.initialize(args)
        self.policy.restart(args)
        self.dialogue_counter += 1

    def update_goal(self, goal):
        """
        Update this agent's goal. This is mainly used to propagate the update
        down to the Dialogue State Tracker.

        :param goal: a Goal
        :return: nothing
        """

        if self.DSTracker:
            self.DSTracker.update_goal(goal)
        else:
            print('WARNING: DialogueManagerGeneric goal update failed: '
                  'No dDalogue State Tracker!')

    def get_state(self):
        """
        Get the current dialogue state

        :return: the dialogue state
        """

        return self.DSTracker.get_state()

    def at_terminal_state(self):
        """
        Assess whether the agent is at a terminal state.

        :return: True or False
        """

        return self.DSTracker.get_state().is_terminal()

    def train(self, dialogues):
        """
        Train the policy and dialogue state tracker, if applicable.

        :param dialogues: dialogue experience
        :return: nothing
        """

        if self.TRAIN_POLICY:
            self.policy.train(dialogues)

        if self.TRAIN_DST:
            self.DSTracker.train(dialogues)

    def is_training(self):
        """
        Assess whether there are any trainable components in this dialogue
        Manager.

        :return: True or False
        """

        return self.TRAIN_DST or self.TRAIN_POLICY

    def load(self, path):
        """
        Load models for the dialogue State Tracker and Policy.

        :param path: path to the policy model
        :return: nothing
        """

        self.policy.load(self.policy_path)

    def save(self):
        """
        Save the models.

        :return: nothing
        """

        if self.DSTracker:
            self.DSTracker.save()

        if self.policy:
            self.policy.save(self.policy_path)
