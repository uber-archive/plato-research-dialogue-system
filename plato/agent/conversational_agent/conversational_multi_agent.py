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

import os
import random
from copy import deepcopy

from gtts import gTTS

from plato.agent.conversational_agent.conversational_agent \
    import ConversationalAgent
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent
from plato.dialogue.action import DialogueAct
from plato.agent.component.dialogue_policy.\
    reinforcement_learning.reward_function import SlotFillingReward
from plato.domain import ontology, database
from plato.agent.component.user_simulator.goal \
    import GoalGenerator
from plato.agent.component.user_simulator.user_model \
    import UserModel
from plato.utilities.dialogue_episode_recorder import DialogueEpisodeRecorder

"""
The ConversationalMultiAgent is a conversational agent that can 
operate in a multi-agent environment. Practically this means 
that the agent's role and id are taken into account during the 
interaction and there are some special cases when processing 
input and when creating the output.
"""

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class ConversationalMultiAgent(ConversationalAgent):
    """
    Essentially the dialogue system. Will be able to interact with:

    - Simulated Users via:
        - dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - parser
    """

    def __init__(self, configuration, agent_id):
        """
        Initialize the internal structures of this agent.

        :param configuration: a dictionary representing the configuration file
        :param agent_id: an integer, this agent's id
        """

        super(ConversationalMultiAgent, self).__init__()

        self.agent_id = agent_id

        # Flag to alternate training between roles
        self.train_system = True

        # dialogue statistics
        self.dialogue_episode = 1
        # Note that since this is a multi-agent setting, dialogue_turn refers
        # to this agent's turns only
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 200
        self.train_interval = 50
        self.train_epochs = 3
        self.global_args = {}

        # Alternate training between the agents
        self.train_alternate_training = True
        self.train_switch_trainable_agents_every = self.train_interval

        self.configuration = configuration

        # True values here would imply some default modules
        self.USE_NLU = False
        self.USE_NLG = False
        self.USE_SPEECH = False
        self.USER_HAS_INITIATIVE = True
        self.SAVE_LOG = True
        self.SAVE_INTERVAL = 10000

        # The dialogue will terminate after MAX_TURNS (this agent will issue
        # a bye() dialogue act.
        self.MAX_TURNS = 10

        self.ontology = None
        self.database = None
        self.domain = None
        self.dialogue_manager = None
        self.user_model = None
        self.nlu = None
        self.nlg = None

        self.agent_role = None
        self.agent_goal = None
        self.goal_generator = None
        self.goals_path = None

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.user_model = UserModel()

        # The size of the experience pool is a hyperparameter.

        # Do not have an experience window larger than the current batch,
        # as past experience may not be relevant since both agents learn.
        self.recorder = DialogueEpisodeRecorder(size=20000)

        # TODO: Get reward type from the config
        self.reward_func = SlotFillingReward()

        if self.configuration:
            ag_id_str = 'AGENT_' + str(self.agent_id)
            
            # Error checks for options the config must have
            if not self.configuration['GENERAL']:
                raise ValueError('Cannot run Plato without GENERAL settings!')

            elif not self.configuration['GENERAL']['interaction_mode']:
                raise ValueError('Cannot run Plato without an interaction '
                                 'mode!')

            elif not self.configuration['DIALOGUE']:
                raise ValueError('Cannot run Plato without DIALOGUE settings!')

            elif not self.configuration[ag_id_str]:
                raise ValueError('Cannot run Plato without at least '
                                 'one agent!')

            # General settings
            if 'GENERAL' in self.configuration and \
                    self.configuration['GENERAL']:
                if 'global_arguments' in self.configuration['GENERAL']:
                    self.global_args = \
                        self.configuration['GENERAL']['global_arguments']

                if 'experience_logs' in self.configuration['GENERAL']:
                    dialogues_path = None
                    if 'path' in \
                            self.configuration['GENERAL']['experience_logs']:
                        dialogues_path = \
                            self.configuration['GENERAL'][
                                'experience_logs']['path']

                    if 'load' in \
                            self.configuration['GENERAL'][
                                'experience_logs'] and \
                            bool(
                                self.configuration[
                                    'GENERAL'
                                ]['experience_logs']['load']
                            ):
                        if dialogues_path and os.path.isfile(dialogues_path):
                            self.recorder.load(dialogues_path)
                        else:
                            raise FileNotFoundError(
                                'dialogue Log file %s not found (did you '
                                'provide one?)' % dialogues_path
                            )

                    if 'save' in \
                            self.configuration['GENERAL']['experience_logs']:
                        self.recorder.set_path(dialogues_path)
                        self.SAVE_LOG = bool(
                            self.configuration['GENERAL'][
                                'experience_logs']['save']
                        )

                        # Retrieve agent role
                        if 'role' in self.configuration[ag_id_str]:
                            self.agent_role = self.configuration[ag_id_str][
                                'role']
                        else:
                            raise ValueError(
                                'agent: No role assigned for '
                                'agent {0} in '
                                'config!'.format(self.agent_id)
                            )

            # Dialogue settings
            if 'DIALOGUE' in self.configuration and \
                    self.configuration['DIALOGUE']:
                if 'initiative' in self.configuration['DIALOGUE']:
                    self.USER_HAS_INITIATIVE = bool(
                        self.configuration['DIALOGUE']['initiative'] == 'user'
                    )

                if 'domain' in self.configuration['DIALOGUE']:
                    self.domain = self.configuration['DIALOGUE']['domain']
                elif 'domain' in self.global_args:
                    self.domain = self.global_args['domain']

                ontology_path = None
                if 'ontology_path' in self.configuration['DIALOGUE']:
                    ontology_path = \
                        self.configuration['DIALOGUE']['ontology_path']
                elif 'ontology' in self.global_args:
                    ontology_path = self.global_args['ontology']

                if ontology_path and os.path.isfile(ontology_path):
                    self.ontology = ontology.Ontology(ontology_path)
                else:
                    raise FileNotFoundError(
                        'domain file %s not found' % ontology_path
                    )

                db_path = None
                if 'db_path' in self.configuration['DIALOGUE']:
                    db_path = self.configuration['DIALOGUE']['db_path']
                elif 'database' in self.global_args:
                    db_path = self.global_args['database']

                if db_path and os.path.isfile(db_path):
                    if db_path[-3:] == '.db':
                        self.database = database.SQLDataBase(db_path)
                    else:
                        self.database = database.DataBase(db_path)
                else:
                    raise FileNotFoundError(
                        'Database file %s not found' % db_path
                    )

                if 'goals_path' in self.configuration['DIALOGUE']:
                    if os.path.isfile(
                        self.configuration['DIALOGUE']['goals_path']
                    ):
                        self.goals_path = \
                            self.configuration['DIALOGUE']['goals_path']
                    else:
                        raise FileNotFoundError(
                            'Goals file %s not '
                            'found' % self.configuration[
                                'DIALOGUE'
                            ]['goals_path']
                        )

            # Agent Settings

            # Retrieve agent role
            if 'role' in self.configuration[ag_id_str]:
                self.agent_role = self.configuration[ag_id_str][
                    'role']
            else:
                raise ValueError(
                    'agent: No role assigned for agent {0} in '
                    'config!'.format(self.agent_id)
                )

            if self.agent_role == 'user':
                if self.ontology and self.database:
                    self.goal_generator = GoalGenerator({
                        'ontology': self.ontology,
                        'database': self.database
                    })
                else:
                    raise ValueError(
                        'Conversational Multi Agent (user): Cannot generate '
                        'goal without ontology and database.'
                    )

            # Retrieve agent parameters
            if 'max_turns' in self.configuration[ag_id_str]:
                self.MAX_TURNS = self.configuration[ag_id_str][
                    'max_turns']

            if 'train_interval' in self.configuration[ag_id_str]:
                self.train_interval = \
                    self.configuration[ag_id_str]['train_interval']

            if 'train_minibatch' in self.configuration[ag_id_str]:
                self.minibatch_length = \
                    self.configuration[ag_id_str]['train_minibatch']

            if 'train_epochs' in self.configuration[ag_id_str]:
                self.train_epochs = \
                    self.configuration[ag_id_str]['train_epochs']

            if 'save_interval' in self.configuration[ag_id_str]:
                self.SAVE_INTERVAL = \
                    self.configuration[ag_id_str]['save_interval']
                
            # NLU Settings
            if 'NLU' in self.configuration[ag_id_str]:
                nlu_args = {}
                if 'package' in self.configuration[ag_id_str]['NLU'] and \
                        'class' in self.configuration[ag_id_str]['NLU']:
                    if 'arguments' in \
                            self.configuration[ag_id_str]['NLU']:
                        nlu_args = \
                            self.configuration[ag_id_str]['NLU'][
                                'arguments']

                    nlu_args.update(self.global_args)

                    self.nlu = \
                        ConversationalGenericAgent.load_module(
                            self.configuration[ag_id_str]['NLU'][
                                'package'],
                            self.configuration[ag_id_str]['NLU'][
                                'class'],
                            nlu_args
                        )

            # DM Settings
            if 'DM' in self.configuration[ag_id_str]:
                dm_args = dict(
                    zip(
                        ['settings', 'ontology', 'database',
                         'domain',
                         'agent_id',
                         'agent_role'],
                        [self.configuration,
                         self.ontology,
                         self.database,
                         self.domain,
                         self.agent_id,
                         self.agent_role
                         ]
                    )
                )

                if 'package' in self.configuration[ag_id_str]['DM'] and \
                        'class' in self.configuration[ag_id_str]['DM']:
                    if 'arguments' in \
                            self.configuration[ag_id_str]['DM']:
                        dm_args.update(
                            self.configuration[ag_id_str]['DM'][
                                'arguments']
                        )

                    dm_args.update(self.global_args)

                    self.dialogue_manager = \
                        ConversationalGenericAgent.load_module(
                            self.configuration[ag_id_str]['DM'][
                                'package'],
                            self.configuration[ag_id_str]['DM'][
                                'class'],
                            dm_args
                        )

            # NLG Settings
            if 'NLG' in self.configuration[ag_id_str]:
                nlg_args = {}
                if 'package' in self.configuration[ag_id_str]['NLG'] and \
                        'class' in self.configuration[ag_id_str]['NLG']:
                    if 'arguments' in \
                            self.configuration[ag_id_str]['NLG']:
                        nlg_args = \
                            self.configuration[ag_id_str]['NLG'][
                                'arguments']

                    nlg_args.update(self.global_args)

                    self.nlg = \
                        ConversationalGenericAgent.load_module(
                            self.configuration[ag_id_str]['NLG'][
                                'package'],
                            self.configuration[ag_id_str]['NLG'][
                                'class'],
                            nlg_args
                        )

                if self.nlg:
                    self.USE_NLG = True

        # True if at least one module is training
        self.IS_TRAINING = self.nlu and self.nlu.training or \
            self.dialogue_manager and self.dialogue_manager.training or \
            self.nlg and self.nlg.training

    def __del__(self):
        """
        Do some house-keeping and save the models.

        :return: nothing
        """

        if self.recorder and self.SAVE_LOG:
            self.recorder.save()

        if self.nlu:
            self.nlu.save()

        if self.dialogue_manager:
            self.dialogue_manager.save()

        if self.nlg:
            self.nlg.save()

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None

    def initialize(self):
        """
        Initializes the conversational agent based on settings in the
        configuration file.

        :return: Nothing
        """

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0

        if self.nlu:
            self.nlu.initialize({})

        self.dialogue_manager.initialize({})

        if self.nlg:
            self.nlg.initialize({})

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

    def start_dialogue(self, args=None):
        """
        Perform initial dialogue turn.

        :param args: optional arguments
        :return:
        """

        self.dialogue_turn = 0

        if self.agent_role == 'user':
            self.agent_goal = self.goal_generator.generate()
            self.dialogue_manager.update_goal(self.agent_goal)

            print('DEBUG > usr goal:')
            for c in self.agent_goal.constraints:
                print(f'\t\tConstr({self.agent_goal.constraints[c].slot}='
                      f'{self.agent_goal.constraints[c].value})')
            print('\t\t-------------')
            for r in self.agent_goal.requests:
                print(f'\t\tReq({self.agent_goal.requests[r].slot})')
            print('\n')

        elif args and 'goal' in args:
            # No deep copy here so that all agents see the same goal.
            self.agent_goal = args['goal']
        else:
            raise ValueError(
                'ConversationalMultiAgent - no goal provided '
                'for agent {0}!'.format(self.agent_role)
            )

        self.dialogue_manager.restart({'goal': self.agent_goal})

        response = [DialogueAct('welcomemsg', [])]
        response_utterance = ''

        # The above forces the DM's initial utterance to be the welcomemsg act.
        # The code below can be used to push an empty list of DialogueActs to
        # the DM and get a response from the model.
        # self.dialogue_manager.receive_input([])
        # response = self.dialogue_manager.respond()

        if self.agent_role == 'system':
            response = [DialogueAct('welcomemsg', [])]
            if self.USE_NLG:
                response_utterance = self.nlg.generate_output(
                    {
                        'dacts': response,
                        'system': self.agent_role == 'system',
                        'last_sys_utterance': ''
                    }
                )

                print(
                    '{0} > {1}'.format(
                        self.agent_role.upper(),
                        response_utterance
                    )
                )

                if self.USE_SPEECH:
                    tts = gTTS(text=response_utterance, lang='en')
                    tts.save('sys_output.mp3')
                    os.system('afplay sys_output.mp3')
            else:
                print(
                    '{0} > {1}'.format(
                        self.agent_role.upper(),
                        '; '.join([str(sr) for sr in response])
                    )
                )

        # TODO: Generate output depending on initiative - i.e.
        # have users also start the dialogue

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None

        return {'input_utterance': None,
                'output_raw': response_utterance,
                'output_dacts': response,
                'goal': self.agent_goal
                }

    def continue_dialogue(self, args):
        """
        Perform next dialogue turn.

        :return: this agent's output
        """

        if 'other_input_raw' not in args:
            raise ValueError(
                'ConversationalMultiAgent called without raw input!'
            )

        other_input_raw = args['other_input_raw']

        other_input_dacts = None
        if 'other_input_dacts' in args:
            other_input_dacts = args['other_input_dacts']

        goal = None
        if 'goal' in args:
            goal = args['goal']

        if goal:
            self.agent_goal = goal

        sys_utterance = ''

        other_input_nlu = deepcopy(other_input_raw)

        if self.nlu and isinstance(other_input_raw, str):
            # Process the other agent's utterance
            other_input_nlu = self.nlu.process_input(
                other_input_raw,
                self.dialogue_manager.get_state()
            )

        elif other_input_dacts:
            # If no utterance provided, use the dacts
            other_input_nlu = other_input_dacts

        self.dialogue_manager.receive_input(other_input_nlu)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue
        # manager will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())

        # Update goal's ground truth
        if self.agent_role == 'system':
            self.agent_goal.ground_truth = deepcopy(
                self.curr_state.item_in_focus
            )

        if self.dialogue_turn < self.MAX_TURNS:
            response = self.dialogue_manager.generate_output()
            self.agent_goal = self.dialogue_manager.DSTracker.DState.user_goal

        else:
            # Force dialogue stop
            response = [DialogueAct('bye', [])]

        rew, success, task_success = self.reward_func.calculate(
            self.dialogue_manager.get_state(),
            response,
            goal=self.agent_goal,
            agent_role=self.agent_role
        )

        if self.USE_NLG:
            sys_utterance = self.nlg.generate_output(
                {
                    'dacts': response,
                    'system': self.agent_role == 'system',
                    'last_sys_utterance': other_input_raw
                }
            ) + ' '

            print('{0} > {1}'.format(self.agent_role.upper(), sys_utterance))

            if self.USE_SPEECH:
                tts = gTTS(text=sys_utterance, lang='en')
                tts.save('sys_output.mp3')
                os.system('afplay sys_output.mp3')
        else:
            print(
                '{0} > {1} \n'.format(
                    self.agent_role.upper(),
                    '; '.join([str(sr) for sr in response])
                )
            )

        if self.prev_state:
            self.recorder.record(
                self.prev_state,
                self.curr_state,
                self.prev_action,
                self.prev_reward,
                self.prev_success,
                input_utterance=self.prev_usr_utterance,
                output_utterance=self.prev_sys_utterance,
                task_success=self.prev_task_success
            )

        self.dialogue_turn += 1

        self.prev_state = deepcopy(self.curr_state)
        self.prev_usr_utterance = deepcopy(other_input_raw)
        self.prev_sys_utterance = deepcopy(sys_utterance)
        self.prev_action = deepcopy(response)
        self.prev_reward = rew
        self.prev_success = success
        self.prev_task_success = task_success

        return {'input_utterance': None,
                'output_raw': sys_utterance,
                'output_dacts': response,
                'goal': self.agent_goal
                }

    def end_dialogue(self):
        """
        Perform final dialogue turn. Train and ave models if applicable.

        :return:
        """

        if self.dialogue_episode % \
                self.train_switch_trainable_agents_every == 0:
            self.train_system = not self.train_system

            # Record final state
            if self.curr_state:
                if not self.curr_state.is_terminal_state:
                    self.curr_state.is_terminal_state = True
                    self.prev_reward, self.prev_success, self.prev_task_success = \
                        self.reward_func.calculate(
                            self.curr_state,
                            [DialogueAct('bye', [])],
                            goal=self.agent_goal,
                            agent_role=self.agent_role
                        )
            else:
                print(
                    'Warning! Conversational Multi Agent attempted to end the'
                    'dialogue with no state. This dialogue will NOT be saved.')
                return

            self.recorder.record(
                self.curr_state,
                self.curr_state,
                self.prev_action,
                self.prev_reward,
                self.prev_success,
                input_utterance=self.prev_usr_utterance,
                output_utterance=self.prev_sys_utterance,
                task_success=self.prev_task_success,
                role=self.agent_role,
                force_terminate=True
            )

        self.dialogue_episode += 1

        if self.IS_TRAINING:
            if not self.train_alternate_training or \
                    (self.train_system and
                     self.agent_role == 'system' or
                     not self.train_system and
                     self.agent_role == 'user'):

                if self.dialogue_episode % self.train_interval == 0 and \
                        len(self.recorder.dialogues) >= self.minibatch_length:
                    for epoch in range(self.train_epochs):
                        print(
                            '{0}: Training epoch {1} of {2}'.format(
                                self.agent_role,
                                (epoch+1),
                                self.train_epochs
                            )
                        )

                        # Sample minibatch
                        minibatch = random.sample(
                            self.recorder.dialogues,
                            self.minibatch_length
                        )

                        if self.nlu:
                            self.nlu.train(minibatch)

                        self.dialogue_manager.train(minibatch)

                        if self.nlg:
                            self.nlg.train(minibatch)

        self.cumulative_rewards += \
            self.recorder.dialogues[-1][-1]['cumulative_reward']

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % self.SAVE_INTERVAL == 0:
            if self.nlu:
                self.nlu.save()

            if self.dialogue_manager:
                self.dialogue_manager.save()

            if self.nlg:
                self.nlg.save()

        # Count successful dialogues
        if self.recorder.dialogues[-1][-1]['success']:
            print(
                '{0} SUCCESS! (reward: {1})'.format(
                    self.agent_role,
                    sum([t['reward'] for t in self.recorder.dialogues[-1]])
                )
            )
            self.num_successful_dialogues += \
                int(self.recorder.dialogues[-1][-1]['success'])

        else:
            print(
                '{0} FAILURE. (reward: {1})'.format(
                    self.agent_role,
                    sum([t['reward'] for t in self.recorder.dialogues[-1]])
                )
            )

        if self.recorder.dialogues[-1][-1]['task_success']:
            self.num_task_success += int(
                self.recorder.dialogues[-1][-1]['task_success']
            )

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        # Hard coded response to bye to enforce dialogue_policy according to
        # which if any agent issues a 'bye' then the dialogue
        # terminates. Otherwise in multi-agent settings it is very hard to
        # learn the association and learn to terminate
        # the dialogue.
        if self.dialogue_manager.get_state().user_acts:
            for act in self.dialogue_manager.get_state().user_acts:
                if act.intent == 'bye':
                    return True

        return self.dialogue_manager.at_terminal_state()

    def get_state(self):
        """
        Get this agent's state

        :return: a DialogueState
        """
        return self.dialogue_manager.get_state()

    def get_goal(self):
        """
        Get this agent's goal

        :return: a Goal
        """
        return self.agent_goal

    def set_goal(self, goal):
        """
        Set or update this agent's goal

        :param goal: a Goal
        :return: nothing
        """

        # Note: reason for non-deep copy is that if this agent changes the goal
        #       these changes are propagated to e.g. the reward function, and
        #       the reward calculation is up to date.
        self.agent_goal = goal
