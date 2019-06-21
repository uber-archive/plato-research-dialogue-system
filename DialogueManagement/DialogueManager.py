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

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from DialogueStateTracker.DialogueStateTracker import DummyStateTracker
from DialogueStateTracker.CamRestDST import CamRestDST

from DialogueManagement.Policy.HandcraftedPolicy import HandcraftedPolicy
from DialogueManagement.Policy.Calculated_Policy import Calculated_Policy
from DialogueManagement.Policy.ReinforcementLearning.Q_Policy import Q_Policy
from DialogueManagement.Policy.ReinforcementLearning.DistributedQ_Policy import DistributedQ_Policy
from DialogueManagement.Policy.ReinforcementLearning.MinimaxQ_Policy import MinimaxQ_Policy
from DialogueManagement.Policy.ReinforcementLearning.WoLF_PHC_Policy import WoLF_PHC_Policy
from DialogueManagement.Policy.DeepLearning.PolicyGradient_Policy import PolicyGradient_Policy
from DialogueManagement.Policy.DeepLearning.OnlineDRL import OnlineDRL
from DialogueManagement.Policy.LudwigPolicy import LudwigPolicy
from DialogueManagement.Policy.LudwigDoublePolicy import LudwigDoublePolicy
from DialogueManagement.Policy.DeepLearning.Supervised_Policy import Supervised_Policy
from DialogueManagement.Policy.DeepLearning.Reinforce_Policy import Reinforce_Policy

from Ontology.Ontology import Ontology
from Ontology.DataBase import DataBase

from copy import deepcopy

from ConversationalAgent.ConversationalModule import ConversationalModule

import random
import math


class DialogueManager(ConversationalModule):
    def __init__(self, args):
        if 'settings' not in args:
            raise AttributeError('DialogueManager: Please provide settings (config)!')
        if 'ontology' not in args:
            raise AttributeError('DialogueManager: Please provide ontology!')
        if 'database' not in args:
            raise AttributeError('DialogueManager: Please provide database!')
        if 'domain' not in args:
            raise AttributeError('DialogueManager: Please provide domain!')

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
        self.policy = None
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
            self.database = DataBase(database)
        else:
            raise ValueError('Unacceptable database type %s ' % database)

        # if self.settings:
        #     ag_id_str = 'AGENT_' + str(self.agent_id)
        #     
        #     # DM and policy settings
        #     if ag_id_str not in self.settings:
        #         raise ValueError(f'Dialogue Manager cannot instantiate without AGENT {self.agent_id} settings!')
                
        if args and args['policy']:
            if 'domain' in self.settings['DIALOGUE']:
                self.domain = self.settings['DIALOGUE']['domain']
            else:
                raise ValueError('Domain is not specified in DIALOGUE at config.')

            if 'calculate_slot_entropies' in args:
                self.CALCULATE_SLOT_ENTROPIES = bool(args['calculate_slot_entropies'])

            if args['policy']['type'] == 'handcrafted':
                self.policy = HandcraftedPolicy(self.ontology)

            elif args['policy']['type'] == 'q_learning':
                if 'learning_rate' in args['policy']:
                    alpha = float(args['policy']['learning_rate'])

                if 'discount_factor' in args['policy']:
                    gamma = float(args['policy']['discount_factor'])

                if 'exploration_rate' in args['policy']:
                    epsilon = float(args['policy']['exploration_rate'])

                if 'learning_decay_rate' in args['policy']:
                    alpha_decay = float(args['policy']['learning_decay_rate'])

                if 'exploration_decay_rate' in args['policy']:
                    epsilon_decay = float(args['policy']['exploration_decay_rate'])

                self.policy = Q_Policy(self.ontology, self.database, self.agent_id, self.agent_role, self.domain,
                                       alpha=alpha, epsilon=epsilon, gamma=gamma, alpha_decay=alpha_decay,
                                       epsilon_decay=epsilon_decay)

            elif args['policy']['type'] == 'distr_q':
                self.policy = DistributedQ_Policy(self.ontology, self.agent_id, self.agent_role)

            elif args['policy']['type'] == 'minimax_q':
                alpha = 0.25
                gamma = 0.95
                epsilon = 0.25
                alpha_decay = 0.9995
                epsilon_decay = 0.995

                if 'learning_rate' in args['policy']:
                    alpha = float(args['policy']['learning_rate'])

                if 'discount_factor' in args['policy']:
                    gamma = float(args['policy']['discount_factor'])

                if 'exploration_rate' in args['policy']:
                    epsilon = float(args['policy']['exploration_rate'])

                if 'learning_decay_rate' in args['policy']:
                    alpha_decay = float(args['policy']['learning_decay_rate'])

                if 'exploration_decay_rate' in args['policy']:
                    epsilon_decay = float(args['policy']['exploration_decay_rate'])

                self.policy = MinimaxQ_Policy(self.ontology, self.database, self.agent_id, self.agent_role,
                                              alpha=alpha, epsilon=epsilon, gamma=gamma, alpha_decay=alpha_decay,
                                              epsilon_decay=epsilon_decay)

            elif args['policy']['type'] == 'wolf_phc':
                alpha = 0.25
                gamma = 0.95
                epsilon = 0.25
                alpha_decay = 0.9995
                epsilon_decay = 0.995

                if 'learning_rate' in args['policy']:
                    alpha = float(args['policy']['learning_rate'])

                if 'discount_factor' in args['policy']:
                    gamma = float(args['policy']['discount_factor'])

                if 'exploration_rate' in args['policy']:
                    epsilon = float(args['policy']['exploration_rate'])

                if 'learning_decay_rate' in args['policy']:
                    alpha_decay = float(args['policy']['learning_decay_rate'])

                if 'exploration_decay_rate' in args['policy']:
                    epsilon_decay = float(args['policy']['exploration_decay_rate'])

                self.policy = WoLF_PHC_Policy(self.ontology, self.database, self.agent_id, self.agent_role,
                                              alpha=alpha, epsilon=epsilon, gamma=gamma, alpha_decay=alpha_decay,
                                              epsilon_decay=epsilon_decay)

            elif args['policy']['type'] == 'pol_grad':
                if 'learning_rate' in args['policy']:
                    alpha = float(args['policy']['learning_rate'])

                if 'exploration_rate' in args['policy']:
                    epsilon = float(args['policy']['exploration_rate'])

                if 'discount_factor' in args['policy']:
                    gamma = float(args['policy']['discount_factor'])

                if 'learning_decay_rate' in args['policy']:
                    alpha_decay = float(args['policy']['learning_decay_rate'])

                if 'exploration_decay_rate' in args['policy']:
                    epsilon_decay = float(args['policy']['exploration_decay_rate'])

                self.policy = PolicyGradient_Policy(self.ontology, self.database, self.agent_id, self.agent_role,
                                                    self.domain, alpha=alpha, epsilon=epsilon, gamma=gamma,
                                                    alpha_decay=alpha_decay, epsilon_decay=epsilon_decay)

            elif args['policy']['type'] == 'reinforce':
                if 'learning_rate' in args['policy']:
                    alpha = float(args['policy']['learning_rate'])

                if 'discount_factor' in args['policy']:
                    gamma = float(args['policy']['discount_factor'])

                if 'exploration_rate' in args['policy']:
                    epsilon = float(args['policy']['exploration_rate'])

                if 'learning_decay_rate' in args['policy']:
                    alpha_decay = float(args['policy']['learning_decay_rate'])

                if 'exploration_decay_rate' in args['policy']:
                    epsilon_decay = float(args['policy']['exploration_decay_rate'])

                self.policy = Reinforce_Policy(self.ontology, self.database, self.agent_id, self.agent_role,
                                               self.domain, alpha=alpha, epsilon=epsilon, gamma=gamma,
                                               alpha_decay=alpha_decay, epsilon_decay=epsilon_decay)

            elif args['policy']['type'] == 'calculated':
                self.policy = Calculated_Policy(self.ontology, self.database, self.agent_id, self.agent_role, self.domain)

            elif args['policy']['type'] == 'online_drl':
                if args['policy']['policy_path']:
                    self.policy_path = args['policy']['policy_path']
                    self.policy = OnlineDRL(self.policy_path)

                else:
                    raise ValueError('Cannot find policy_path in the config for dialogue policy.')

            elif args['policy']['type'] == 'supervised':
                self.policy = Supervised_Policy(self.ontology, self.database, self.agent_id, self.agent_role, self.domain)

            elif args['policy']['type'] == 'ludwig':
                if args['policy']['policy_path']:
                    self.policy = LudwigPolicy(args['policy']['policy_path'])
                else:
                    raise ValueError('Cannot find policy_path in the config for dialogue policy.')

            elif args['policy']['type'] == 'double':
                if args['policy']['policy_path'] and args['policy']['metadata_path']:
                    if args['policy']['slot_model_path'] and args['policy']['slot_metadata_path']:
                        self.policy = LudwigDoublePolicy(args['policy']['model_path'],
                                                         args['policy']['metadata_path'],
                                                         args['policy']['slot_model_path'],
                                                         args['policy']['slot_metadata_path'])
                    else:
                        raise ValueError(
                            'Cannot find slot_model_path or slot_metadata_path in the config for dialogue policy.')
                else:
                    raise ValueError('Cannot find model_path or metadata_path in the config for dialogue policy.')
            else:
                raise ValueError('DialogueManager: Unsupported policy type!'.format(args['policy']['type']))

            if 'train' in args['policy']:
                self.TRAIN_POLICY = bool(args['policy']['train'])

            if 'policy_path' in args['policy']:
                self.policy_path = args['policy']['policy_path']

        # DST Settings
        if 'DST' in args and args['DST']['dst']:
                if args['DST']['dst'] == 'CamRest':
                    if args['DST']['policy']['model_path'] and args['DST']['policy']['metadata_path']:
                        self.DSTracker = CamRestDST({'model_path': args['DST']['policy']['model_path']})
                    else:
                        raise ValueError('Cannot find model_path or metadata_path in the config for dialogue state tracker.')

        # Default to dummy DST
        if not self.DSTracker:
            dst_args = dict(zip(['ontology', 'database', 'domain'], [self.ontology, self.database, domain]))
            self.DSTracker = DummyStateTracker(dst_args)

        self.load('')

        # Get Table name
        cursor = self.database.SQL_connection.cursor()
        result = cursor.execute("select * from sqlite_master where type = 'table';").fetchall()
        if result and result[0] and result[0][1]:
            self.db_table_name = result[0][1]
        else:
            raise ValueError(
                'Dialogue Manager cannot specify Table Name from database {0}'.format(self.database.db_file_name))

    def initialize(self, args):
        '''
        Initialize the relevant structures and variables of the Dialogue Manager.

        :return: Nothing
        '''

        self.DSTracker.initialize()
        if 'goal' not in args:
            self.policy.initialize(
                **{'is_training': self.TRAIN_POLICY, 'policy_path': self.policy_path, 'ontology': self.ontology})
        else:
            self.policy.initialize(
                **{'is_training': self.TRAIN_POLICY, 'policy_path': self.policy_path, 'ontology': self.ontology,
                   'goal': args['goal']})

        self.dialogue_counter = 0

    def receive_input(self, inpt):
        '''
        Receive input and update the dialogue state.

        :return: Nothing
        '''

        # Update dialogue state given the new input
        self.DSTracker.update_state(inpt)

        if self.domain and self.domain in ['CamRest', 'SFH', 'SlotFilling']:
            if self.agent_role == 'system':
                # Perform a database lookup
                db_result, sys_req_slot_entropies = self.db_lookup()

                # Update the dialogue state again to include the database results
                self.DSTracker.update_state_db(db_result=db_result, sys_req_slot_entropies=sys_req_slot_entropies)

            else:
                # Update the dialogue state again to include the system actions
                self.DSTracker.update_state_db(db_result=None, sys_acts=inpt)

        return inpt

    def generate_output(self, args=None):
        '''
        Consult the current policy to respond.

        :return: List of DialogueAct representing the system's output.
        '''
        
        DState = self.DSTracker.get_state()

        sys_acts = self.policy.next_action(DState)
        # Copy the sys_acts to be able to iterate over all sys_acts while also replacing some acts
        sys_acts_copy = deepcopy(sys_acts)
        new_sys_acts = []

        # Safeguards to support policies that make decisions on intents only (i.e. do not output slots or values)
        for sys_act in sys_acts:
            if sys_act.intent == 'canthelp' and not sys_act.params:
                slots = [s for s in DState.slots_filled if DState.slots_filled[s]]
                if slots:
                    slot = random.choice(slots)

                    # Remove the empty canthelp
                    sys_acts_copy.remove(sys_act)

                    new_sys_acts.append(DialogueAct('canthelp', [DialogueActItem(slot, Operator.EQ, DState.slots_filled[slot])]))

                else:
                    print('DialogueManager Warning! No slot provided by policy for canthelp and cannot find a reasonable one!')

            if sys_act.intent == 'offer' and not sys_act.params:
                # Remove the empty offer
                sys_acts_copy.remove(sys_act)

                if DState.item_in_focus:
                    new_sys_acts.append(DialogueAct('offer', [DialogueActItem('name', Operator.EQ, DState.item_in_focus['name'])]))

                    # informed_requested_slot = False

                    # if not informed_requested_slot and DState.requested_slot:
                    #     slot = DState.requested_slot
                    #
                    #     if slot in DState.item_in_focus and DState.item_in_focus[slot]:
                    #         new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, DState.item_in_focus[slot])]))

                    # Only add these slots if no other acts were output by the DM
                    if len(sys_acts) == 1:
                        for slot in DState.slots_filled:
                            if slot in DState.item_in_focus:
                                if slot not in ['id', 'name'] and slot != DState.requested_slot:
                                    new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, DState.item_in_focus[slot])]))

                                    # if DState.requested_slot == slot:
                                    #     informed_requested_slot = True
                            else:
                                new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, 'no info')]))

            elif sys_act.intent == 'inform':
                if self.agent_role == 'system':
                    if sys_act.params and sys_act.params[0].value:
                        continue

                    if sys_act.params:
                        slot = sys_act.params[0].slot
                    else:
                        slot = DState.requested_slot

                    if not slot:
                        slot = random.choice(list(DState.slots_filled.keys()))

                    if DState.item_in_focus:
                        if slot not in DState.item_in_focus or not DState.item_in_focus[slot]:
                            new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, 'no info')]))
                        else:
                            if slot == 'name':
                                new_sys_acts.append(DialogueAct('offer', [DialogueActItem(slot, Operator.EQ, DState.item_in_focus[slot])]))
                            else:
                                new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, DState.item_in_focus[slot])]))

                    else:
                        new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, 'no info')]))

                elif self.agent_role == 'user':
                    if sys_act.params:
                        slot = sys_act.params[0].slot

                        # Do nothing if the slot is already filled
                        if sys_act.params[0].value:
                            continue

                    elif DState.last_sys_acts and DState.user_acts and DState.user_acts[0].intent == 'request':
                        slot = DState.user_acts[0].params[0].slot

                    else:
                        slot = random.choice(list(DState.user_goal.constraints.keys()))

                    # Populate the inform with a slot from the user goal
                    if DState.user_goal:
                        # Look for the slot in the user goal
                        if slot in DState.user_goal.constraints:
                            value = DState.user_goal.constraints[slot].value
                        else:
                            value = 'dontcare'

                        new_sys_acts.append(DialogueAct('inform', [DialogueActItem(slot, Operator.EQ, value)]))

                # Remove the empty inform
                sys_acts_copy.remove(sys_act)

            elif sys_act.intent == 'request':
                # If the policy did not select a slot
                if not sys_act.params:
                    found = False

                    if self.agent_role == 'system':
                        # Select unfilled slot
                        for slot in DState.slots_filled:
                            if not DState.slots_filled[slot]:
                                found = True
                                new_sys_acts.append(DialogueAct('request', [DialogueActItem(slot, Operator.EQ, '')]))
                                break

                    elif self.agent_role == 'user':
                        # Select request from goal
                        if DState.user_goal:
                            for req in DState.user_goal.requests:
                                if not DState.user_goal.requests[req].value:
                                    found = True
                                    new_sys_acts.append(DialogueAct('request', [DialogueActItem(req, Operator.EQ, '')]))
                                    break

                    if not found:
                        # All slots are filled
                        new_sys_acts.append(DialogueAct('request', [DialogueActItem(random.choice(list(DState.slots_filled.keys())[:-1]), Operator.EQ, '')]))

                    # Remove the empty request
                    sys_acts_copy.remove(sys_act)

        # Append unique new sys acts
        for sa in new_sys_acts:
            if sa not in sys_acts_copy:
                sys_acts_copy.append(sa)
        # sys_acts_copy += new_sys_acts

        self.DSTracker.update_state_sysact(sys_acts_copy)

        return sys_acts_copy

    def db_lookup(self):
        # TODO: Add check to assert if each slot in DState.slots_filled actually exists in the schema.

        DState = self.DSTracker.get_state()

        # Query the database
        cursor = self.database.SQL_connection.cursor()
        sql_command = " SELECT * FROM " + self.db_table_name + " "

        args = ''
        prev_arg = False
        prev_query_arg = False

        # Impose constraints
        for slot in DState.slots_filled:
            if DState.slots_filled[slot] and DState.slots_filled[slot] != 'dontcare':
                if prev_arg:
                    args += " AND "

                args += slot + " = \"" + DState.slots_filled[slot] + "\""
                prev_arg = True

        # Impose queries
        if prev_arg and DState.slot_queries:
            args += " AND ("

        for slot in DState.slot_queries:
            for slot_query in DState.slot_queries[slot]:
                query = slot_query[0]
                op = slot_query[1]

                if prev_query_arg:
                    args += f" {op} "

                args += slot + " LIKE \'%" + query + "%\' "
                prev_query_arg = True

        if prev_arg and DState.slot_queries:
            args += " ) "

        if args:
            sql_command += " WHERE " + args + ";"

        cursor.execute(sql_command)
        db_result = cursor.fetchall()

        if db_result:
            # Get the slot names
            slot_names = [i[0] for i in cursor.description]
            result = []
            for db_item in db_result:
                result.append(dict(zip(slot_names, db_item)))

            # Calculate entropy of requestable slot values in results - if the flag is off this will be empty
            entropies = dict.fromkeys(self.ontology.ontology['system_requestable'])

            if self.CALCULATE_SLOT_ENTROPIES:
                value_probabilities = {}

                # Count the values
                for req_slot in self.ontology.ontology['system_requestable']:
                    value_probabilities[req_slot] = {}

                    for db_item in result:
                        if db_item[req_slot] not in value_probabilities[req_slot]:
                            value_probabilities[req_slot][db_item[req_slot]] = 1
                        else:
                            value_probabilities[req_slot][db_item[req_slot]] += 1

                # Calculate probabilities
                for slot in value_probabilities:
                    for value in value_probabilities[slot]:
                        value_probabilities[slot][value] /= len(result)

                # Calculate entropies
                for slot in entropies:
                    entropies[slot] = 0

                    if slot in value_probabilities:
                        for value in value_probabilities[slot]:
                            entropies[slot] += value_probabilities[slot][value] * math.log(value_probabilities[slot][value])

                    entropies[slot] = -entropies[slot]

            return result[:self.MAX_DB_RESULTS], entropies

        # Failed to retrieve anything
        print('Warning! Database call retrieved zero results.')
        return ['empty'], {}

    def restart(self, args):
        '''
        Restart the relevant structures or variables, e.g. at the beginning of a new dialogue.

        :return: Nothing
        '''

        self.DSTracker.initialize(args)
        self.policy.restart(args)
        self.dialogue_counter += 1

    def update_goal(self, goal):
        if self.DSTracker:
            self.DSTracker.update_goal(goal)
        else:
            print('WARNING: Dialogue Manager goal update failed: No Dialogue State Tracker!')

    def get_state(self):
        return self.DSTracker.get_state()

    def at_terminal_state(self):
        return self.DSTracker.get_state().is_terminal()

    def train(self, dialogues):
        if self.TRAIN_POLICY:
            self.policy.train(dialogues)

        if self.TRAIN_DST:
            self.DSTracker.train(dialogues)

    def is_training(self):
        return self.TRAIN_DST or self.TRAIN_POLICY

    def load(self, path):
        # TODO: Handle path and loading properly
        self.DSTracker.load('')
        self.policy.load(self.policy_path)

    def save(self):
        if self.DSTracker:
            self.DSTracker.save()

        if self.policy:
            self.policy.save(self.policy_path)
