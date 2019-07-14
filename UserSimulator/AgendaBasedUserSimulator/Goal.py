"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

import heapq
import math
import os
import pickle
import random

from Dialogue.Action import DialogueActItem, Operator
from Domain import Ontology
from Domain.DataBase import DataBase, SQLDataBase, JSONDataBase

"""
The Goal represents Simulated Usr goals, that are composed of a set of 
constraints and a set of requests. Goals can be simple or complex, depending 
on whether they have subgoals or not.
"""


class Goal:
    def __init__(self):
        """
        Initialize the Goal's internal structures.
        """
        self.constraints = {}       # Dict of <slot, Dialogue Act Item>
        self.requests = {}          # Dict of <slot, Dialogue Act Item>
        self.actual_constraints = {}  # Dict of <slot, Dialogue Act Item>
        self.actual_requests = {}   # Dict of <slot, Dialogue Act Item>

        # To be used in the multi-agent setting primarily (where the user does
        # not have access to the ground truth - item in focus - in the
        # dialogue state).
        self.ground_truth = None

        self.subgoals = []

        self.user_model = None
        
    def __str__(self):
        """
        Generate a string representing the Goal
        :return: a string
        """
        ret = ''
        
        for c in self.constraints:
            ret += \
                f'\t\tConstr({self.constraints[c].slot}=' \
                f'{self.constraints[c].value})\n'
        ret += '\t\t-------------\n'
        for r in self.requests:
            ret += f'\t\tReq({self.requests[r].slot})\n'
        ret += '\t\t-------------\n'
        ret += 'Sub-goals:\n'
        for sg in self.subgoals:
            for c in sg.constraints:
                if not sg.constraints[c].slot:
                    ret += f'Error! No slot for {c}\n'
                if not sg.constraints[c].value:
                    ret += f'Error! No value for {c}\n'
                ret += f'\t\tConstr({sg.constraints[c].slot}=' \
                       f'{sg.constraints[c].value})\n'
            ret += '\t\t--------\n'
        ret += '\n'
        
        return ret


class GoalGenerator:
    def __init__(self, ontology, database, goals_file=None):
        """
        Initializes the internal structures of the Goal Generator and does
        some checks.

        :param ontology: Domain object or path to an Domain
        :param database: Database object or path to a Database
        :param goals_file: path to a file that contains goals, in case we want
                           to sample goals from a pool and not generate
        """
        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
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
        
        self.goals_file = goals_file
        self.goals = None

        if self.goals_file:
            self.load_goals(self.goals_file)

        # Get the slot names from the database
        cursor = self.database.SQL_connection.cursor()

        # Get Table name
        result = \
            cursor.execute(
                "select * from sqlite_master where type = 'table';").fetchall()
        if result and result[0] and result[0][1]:
            self.db_table_name = result[0][1]
        else:
            raise ValueError('Goal Generator cannot specify Table Name from '
                             'database {0}'.format(self.database.db_file_name))

        # Dummy SQL command
        sql_command = "SELECT * FROM " + self.db_table_name + " LIMIT 1;"

        cursor.execute(sql_command)
        self.slot_names = [i[0] for i in cursor.description]

        self.db_row_count = \
            cursor.execute("SELECT COUNT(*) FROM " +
                           self.db_table_name + ";").fetchall()[0][0]

    def generate(self, goal_slot_selection_weights=None):
        """
        Generate a goal

        :param goal_slot_selection_weights: list of weights that bias the
                                            sampling of slots for constraints
        :return: a new goal
        """

        # If a goals file has been provided
        if self.goals:
            return random.choice(self.goals)

        # Randomly pick an item from the database
        cursor = self.database.SQL_connection.cursor()

        sql_command = "SELECT * FROM " + self.db_table_name + \
                      " WHERE ROWID == (" + \
                      str(random.randint(1, self.db_row_count)) + ");"

        cursor.execute(sql_command)
        db_result = cursor.fetchone()

        attempt = 0
        while attempt < 3 and not db_result:
            print('GoalGenerator: Database {0} appears to be empty!'
                  .format(self.database))
            print(f'Trying again (attempt {attempt} out of 3)...')

            sql_command = "SELECT * FROM " + self.db_table_name + \
                          " WHERE ROWID == (" + \
                          str(random.randint(1, self.db_row_count)) + ");"

            cursor.execute(sql_command)
            db_result = cursor.fetchone()

            attempt += 1

        if not db_result:
            raise LookupError('GoalGenerator: Database {0} appears to be '
                              'empty!'.format(self.database))

        result = dict(zip(self.slot_names, db_result))

        # Generate goal
        goal = Goal()

        # TODO: Sample from all available operators, not just '='
        # (where applicable)

        # Sample constraints from informable slots
        if goal_slot_selection_weights:
            # If a list of weights has been provided then do weighted sampling

            # Make sure that all slots have some weight. Flatten the
            # dictionaries in the process
            slot_weights = []
            half_min_weight = min(goal_slot_selection_weights.values()) / 2.0

            for s in self.ontology.ontology['informable']:
                if s in goal_slot_selection_weights:
                    slot_weights.append(goal_slot_selection_weights[s])
                else:
                    slot_weights.append(half_min_weight)

            inf_slots = \
                self.weighted_random_sample_no_replacement(
                    list(self.ontology.ontology['informable'].keys()),
                    slot_weights,
                    random.randint(2,
                                   len(self.ontology.ontology['informable'])))

            inf_slots.reverse()
        else:
            inf_slots = \
                random.sample(
                    list(self.ontology.ontology['informable'].keys()),
                    random.randint(2,
                                   len(self.ontology.ontology['informable'])))

        # Sample requests from requestable slots
        req_slots = random.sample(
            self.ontology.ontology['requestable'],
            random.randint(0, len(self.ontology.ontology['requestable'])))

        # Remove slots for which the user places constraints
        # Note: 'name' may or may not be in inf_slots here, and this is
        # randomness is desirable
        for slot in inf_slots:
            if slot in req_slots:
                req_slots.remove(slot)

        # Never ask for specific name unless it is the only constraint
        # if 'name' in inf_slots and len(inf_slots) > 1:
        if 'name' in inf_slots:
            inf_slots.remove('name')

        # Shuffle informable and requestable slots to create some variety
        # when pushing into the agenda.
        random.shuffle(inf_slots)
        random.shuffle(req_slots)

        for slot in inf_slots:
            # Check that the slot has a value in the retrieved item
            if slot in result and result[slot]:
                goal.constraints[slot] = \
                    DialogueActItem(slot, Operator.EQ, result[slot])

        for slot in req_slots:
            if slot in result:
                goal.requests[slot] = DialogueActItem(slot, Operator.EQ, [])

        return goal

    @staticmethod
    def weighted_random_sample_no_replacement(population, weights,
                                              num_samples):
        """
        Samples num_samples from population given the weights

        :param population: a list of things to sample from
        :param weights: weights that bias the sampling from the population
        :param num_samples: how many objects to sample
        :return: a list containing num_samples sampled objects
        """

        elt = \
            [(math.log(random.random()) / weights[i], i)
             for i in range(len(weights))]
        return [population[i[1]] for i in heapq.nlargest(num_samples, elt)]
    
    # Load goals from a pickle to sample from
    def load_goals(self, path):
        """
        Load goals from a file

        :param path: the path to the file that contains the goals
        :return: nothing
        """
        goals_path = path

        if not goals_path:
            goals_path = self.goals_file

        if not goals_path:
            # Try a default value for the goals file path
            goals_path = 'Models/UserSimulator/goals_file.pkl'

        self.goals = None
        if isinstance(goals_path, str):
            if os.path.isfile(goals_path):
                with open(goals_path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'goals' in obj:
                        self.goals = obj['goals']

                    print(f'Goal Generator: Goals loaded.')

            else:
                print(f'Warning! Goals file {goals_path} not found')
        else:
            print(f'Warning! Unacceptable value for goals file name: '
                  f'{goals_path}')


# This generator will generate goals that have sub-goals
class ComplexGoalGenerator(GoalGenerator):
    def __init__(self, ontology, database, goals_file=None, global_key=None,
                 global_slots=[], local_slots=[]):
        """
        Initialize the internal structures of the Complex Goal Generator. This
        Generator will create goals that contain sub-goals and ensure
        consistency.

        :param ontology: the domain Domain
        :param database: the domain Database
        :param goals_file: a file to sample goals from
        :param global_key: the global key to the database
        :param global_slots: a list of slots that will be sampled to create the
                             global constraints
        :param local_slots: a list of slots that will be sampled to create the
                            sub-goal constraints
        """

        super(ComplexGoalGenerator, self).__init__(
            ontology, database, goals_file)

        self.global_key = global_key
        # TODO: What happens / what does it mean when no global key is
        # provided? Fall back to regular GoalGenerator?

        self.global_slots = global_slots

        self.local_slots = local_slots
        if not self.local_slots:
            self.local_slots = \
                list(self.ontology.ontology['informable'].keys())
            for gs in self.global_slots:
                self.local_slots.remove(gs)

    def generate(self, goal_slot_selection_weights=None):
        """
        Generate a new complex goal

        :param goal_slot_selection_weights: list of weights that bias the
                                            sampling of slots for constraints
        :return: a new goal
        """
        if self.goals:
            return random.sample(self.goals)

        # Randomly pick an item from the database
        cursor = self.database.SQL_connection.cursor()

        sql_command = "SELECT * FROM " + self.db_table_name + \
                      " WHERE ROWID == (" + \
                      str(random.randint(1, self.db_row_count)) + ");"

        cursor.execute(sql_command)
        db_result = cursor.fetchone()

        global_key_value = ''
        global_attempt = 0

        result = []

        while global_attempt < 3 and not global_key_value:
            attempt = 0
            while attempt < 3 and not db_result:
                print('GoalGenerator: Database {0} appears to be empty!'
                      .format(self.database))
                print(f'Trying again (attempt {attempt} out of 3)...')

                sql_command = "SELECT * FROM " + self.db_table_name + \
                              " WHERE ROWID == (" + \
                              str(random.randint(1, self.db_row_count)) + ");"

                cursor.execute(sql_command)
                db_result = cursor.fetchone()

                attempt += 1

            if not db_result:
                raise LookupError('GoalGenerator: Database {0} appears to be '
                                  'empty!'.format(self.database))

            result = dict(zip(self.slot_names, db_result))

            if self.global_key in result:
                global_key_value = result[self.global_key]

            global_attempt += 1

        if not result:
            raise LookupError(f'ComplexGoalGenerator cannot find an item with '
                              f'global key {self.global_key}')

        # Generate goal
        goal = Goal()

        # Sample "global" constraints and requests that all sub-goals share
        global_inf_slots = \
            random.sample(self.global_slots,
                          random.randint(2, len(self.global_slots)))

        # Fetch all items from the database that satisfy the constraints
        # sampled above
        sql_command = "SELECT * FROM " + self.db_table_name + \
                      " WHERE " + self.global_key + " = \"" + \
                      global_key_value + "\" AND "

        for gs in global_inf_slots:
            sql_command += gs + " = \"" + result[gs] + "\" AND "
        # Trim last 'AND '
        sql_command = sql_command[:-4] + ";"

        cursor.execute(sql_command)
        db_results = cursor.fetchall()

        results = [dict(zip(self.slot_names, dbr)) for dbr in db_results]

        for slot in global_inf_slots:
            # Check that the slot has a value in the retrieved item
            if slot in result and result[slot]:
                if result[slot] not in ['None', 'Other']:
                    goal.constraints[slot] = \
                        DialogueActItem(slot, Operator.EQ, result[slot])

        # Sample requests
        global_req_slots = \
            random.sample(self.global_slots,
                          random.randint(0, len(self.global_slots)))
        print('DBG: Global Req Sampled')

        # Remove slots for which the user places constraints
        # Note: 'name' may or may not be in inf_slots here, and this is
        # randomness is desirable
        for slot in global_inf_slots:
            if slot in global_req_slots:
                global_req_slots.remove(slot)

        for slot in global_req_slots:
            if slot in result:
                goal.requests[slot] = DialogueActItem(slot, Operator.EQ, [])

        # Sample number of sub-goals
        num_sub_goals = \
            random.choices(range(1, 5), weights=[0.15, 0.35, 0.35, 0.15])[0]

        # Make sure we don't attempt to sample more subgoals than items in the
        # results
        num_sub_goals = \
            num_sub_goals if len(results) > num_sub_goals else len(results)

        subgoal_attempts = 0

        # As there is no guarantee that the sampled slots for the sampled
        # subgoals exist (and we do not want empty subgoals), make three
        # attempts at sampling subgoals.
        while not goal.subgoals and subgoal_attempts < 3:
            print(f'DBG: Sampling Results {len(results)}, {num_sub_goals}')
            results = random.sample(results, num_sub_goals)
            subgoal_attempts += 1

            # For each sub-goal, sample "local" constraints and requests
            # (must be on different slots than the global)
            for sg in range(num_sub_goals):
                local_inf_slots = \
                    random.sample(self.local_slots,
                                  random.randint(1, len(self.local_slots)))

                # Create new subgoal
                subgoal = Goal()

                for lif in local_inf_slots:
                    # Check that the slot has indeed a value
                    if results[sg][lif] and \
                            results[sg][lif] not in ['None', 'Other']:
                        subgoal.constraints[lif] = \
                            DialogueActItem(lif, Operator.EQ, results[sg][lif])

                if subgoal.constraints:
                    goal.subgoals.append(subgoal)

        # TODO Check if a constraint exists in all subgoals, in which case
        # remove from all subgoals and put in global constr

        return goal
