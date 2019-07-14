"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from copy import deepcopy

from UserSimulator.UserSimulator import UserSimulator
from UserSimulator.AgendaBasedUserSimulator.Goal import GoalGenerator
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from Dialogue.Action import DialogueActItem, Operator, DialogueAct

import os.path
import pickle
import random

"""
The DTLUserSimulator (DialogueAct To Language) is a Simulated Usr that follows 
a simple policy model, a dictionary from system Dialogue Acts to Usr Language 
templates.
"""


class DTLUserSimulator(UserSimulator):
    def __init__(self, args):
        """
        Initialise the Usr Simulator. Here we initialize structures that
        we need throughout the life of the DTL Usr Simulator.

        :param args: dictionary containing ontology, database, and policy file
        """
        super(DTLUserSimulator, self).__init__()

        if 'ontology' not in args:
            raise AttributeError('DTLUserSimulator: Please provide ontology!')
        if 'database' not in args:
            raise AttributeError('DTLUserSimulator: Please provide database!')
        if 'policy_file' not in args:
            raise AttributeError('DTLUserSimulator: Please provide policy '
                                 'file!')

        ontology = args['ontology']
        database = args['database']
        policy_file = args['policy_file']

        self.policy = None
        self.load(policy_file)

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase):
            self.database = database
        elif isinstance(database, str):
            self.database = DataBase(database)
        else:
            raise ValueError('Unacceptable database type %s ' % database)

        self.input_system_acts = None
        self.goal = None

        self.goal_generator = GoalGenerator(self.ontology, self.database)
        # TODO: Get patience value from config
        self.patience = 3
        self.curr_patience = self.patience
        self.prev_sys_acts = None

        self.goal_met = False
        self.offer_made = False

    def initialize(self, args):
        """
        Initialize the DTL Usr Simulator at the beginning of each dialogue

        :param args:
        :return: nothing
        """

        self.input_system_acts = []
        self.goal = self.goal_generator.generate()
        self.curr_patience = self.patience
        self.prev_sys_acts = None
        self.goal_met = False
        self.offer_made = False

    def receive_input(self, system_acts):
        """
        Process input received and do some housekeeping.

        :param system_acts: list containing dialogue acts from the system
        :return: nothing
        """

        if self.prev_sys_acts and self.prev_sys_acts == system_acts:
            self.curr_patience -= 1
        else:
            self.curr_patience = self.patience
            self.prev_sys_acts = deepcopy(system_acts)

        self.input_system_acts = deepcopy(system_acts)

        # Check for goal satisfaction
        # Update user goal (in ABUS the state is factored into the goal and
        # the agenda)
        for system_act in system_acts:
            if system_act.intent == 'offer':
                self.offer_made = True

                # Reset past requests
                # TODO: Is this reasonable?
                self.goal.actual_requests = {}

                for item in self.goal.requests:
                    self.goal.requests[item].value = ''

        # Gather all inform or offer params into one dialogue act
        inform_dact = DialogueAct('inform', [])
        for system_act in system_acts:
            if system_act.intent in ['inform', 'offer']:
                inform_dact.params += deepcopy(system_act.params)

        # Check for constraint satisfaction
        if self.offer_made:
            # Check that the venue provided meets the constraints
            meets_constraints = all(
                [i.value == self.goal.constraints[i.slot].value
                 for i in inform_dact.params
                 if i.slot in self.goal.constraints])

            # If it meets the constraints, update the requests
            if meets_constraints:
                for item in inform_dact.params:
                    if item.slot in self.goal.actual_requests:
                        self.goal.actual_requests[item.slot].value = item.value

                        if item.slot in self.goal.requests:
                            self.goal.requests[item.slot].value = item.value

                # Use the true requests for asserting goal is met
                self.goal_met = True
                for slot in self.goal.requests:
                    if not self.goal.requests[slot].value:
                        self.goal_met = False
                        break

    def respond(self):
        """
        Consult the policy to retrieve NLG template and generate the response.
        :return: the DTL Usr Simulator's utterance (response)
        """

        if self.curr_patience <= 0 or self.goal_met:
            return 'bye'

        if not self.input_system_acts:
            # Randomly sample from hello + responses to requests, as there is
            # where informs most likely live.
            sys_act_slot = \
                random.choice([act for act in self.policy if 'request' in act])

            replies = list(self.policy[sys_act_slot]['responses'].keys())
            probs = \
                [self.policy[sys_act_slot]['responses'][i] for i in replies]

            response = deepcopy(random.choices(replies, weights=probs)[0])

            # Replace placeholders with values from goal
            for slot in self.ontology.ontology['informable']:
                if slot.upper() in response:
                    if slot in self.goal.constraints:
                        response = \
                            response.replace(
                                '<' + slot.upper() + '>',
                                self.goal.constraints[slot].value)
                    else:
                        # If there is no constraint, replace with slot 'any'
                        response = \
                            response.replace(
                                '<' + slot.upper() + '>', 'any')

            for slot in self.ontology.ontology['requestable']:
                # This check is necessary to know when to mark this as an
                # actual request
                if slot.upper() in response:
                    response = response.replace('<' + slot.upper() + '>', slot)

                    self.goal.actual_requests[slot] = \
                        DialogueActItem(slot, Operator.EQ, '')

            return random.choice([response, 'hello'])

        response_template = ''

        for system_act in self.input_system_acts:

            # 'bye' doesn't seem to appear in the CamRest data
            if system_act.intent == 'bye':
                response_template += 'thank you, goodbye'

            sys_act_slot = \
                'inform' if system_act.intent == 'offer' else system_act.intent

            if system_act.params and system_act.params[0].slot:
                sys_act_slot += '_' + system_act.params[0].slot

            # Attempt to recover
            if sys_act_slot not in self.policy:
                if sys_act_slot == 'inform_name':
                    sys_act_slot = 'offer_name'

            if sys_act_slot not in self.policy:
                print('Warning! DACT-NLG policy does not know what to do for '
                      '%s' % sys_act_slot)
                # return ''
            else:
                replies = list(self.policy[sys_act_slot]['responses'].keys())
                probs = [
                    self.policy[sys_act_slot]['responses'][i] for i in replies]

                response = deepcopy(random.choices(replies, weights=probs)[0])

                # Replace placeholders with values from goal
                for slot in self.ontology.ontology['informable']:
                    if slot.upper() in response:
                        if slot in self.goal.constraints:
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>',
                                    self.goal.constraints[slot].value)
                        else:
                            # If there is no constraint, replace with
                            # slot 'any'
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>', 'any')

                for slot in self.ontology.ontology['requestable']:
                    # This check is necessary to know when to mark this as an
                    # actual request
                    if slot.upper() in response:
                        if slot == 'addr':
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>', 'address')
                        elif slot == 'postcode':
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>', 'post code')
                        elif slot == 'pricerange':
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>', 'price range')
                        else:
                            response = \
                                response.replace(
                                    '<' + slot.upper() + '>', slot)

                        self.goal.actual_requests[slot] = \
                            DialogueActItem(slot, Operator.EQ, '')
                        
                response_template += response + ' '

        return response_template

    def train(self, data):
        """
        Placeholder for training models

        :param data: dialogue experience
        :return: nothing
        """
        pass

    def save(self, path=None):
        """
        Placeholder for saving models

        :param path: path to save the models to
        :return: nothing
        """
        pass

    def load(self, policy_file):
        """
        Loads policy file.

        :param policy_file: path to the policy file
        :return:
        """
        if isinstance(policy_file, str):
            if os.path.isfile(policy_file):
                with open(policy_file, 'rb') as file:
                    obj = pickle.load(file)

                    if 'policy' in obj:
                        self.policy = obj['policy']

                    print('Dact-to-Language Usr Simulator policy loaded.')

            else:
                print('Warning! Usr DialoguePolicy file %s not found'
                      % policy_file)
        else:
            print('Warning! Unacceptable value for DTL Simulator policy file '
                  'name: %s ' % policy_file)

    def at_terminal_state(self):
        """
        Check if the DTL Usr Simulator is at a terminal state. Since it is
        stateless, it always returns False.

        :return: False
        """
        return False
