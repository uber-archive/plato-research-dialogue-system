"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

import os
import random
from copy import deepcopy

from gtts import gTTS

from ConversationalAgent.ConversationalAgent import ConversationalAgent
from Dialogue.Action import DialogueAct
from DialogueManagement import DialogueManager
from DialogueManagement.DialoguePolicy.ReinforcementLearning.RewardFunction \
    import SlotFillingReward
from NLG.CamRestNLG import CamRestNLG
from NLG.DummyNLG import DummyNLG
from NLU.CamRestNLU import CamRestNLU
from NLU.DummyNLU import DummyNLU
from Domain import Ontology, DataBase
from UserSimulator.AgendaBasedUserSimulator.Goal import GoalGenerator
from UserSimulator.UserModel import UserModel
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder

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
        - Dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - Data
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

        # Dialogue statistics
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

        # Alternate training between the agents
        self.train_alternate_training = True
        self.train_switch_trainable_agents_every = self.train_interval

        self.configuration = configuration

        # True values here would imply some default modules
        self.USE_USR_SIMULATOR = False
        self.USER_SIMULATOR_NLG = False
        self.USE_NLU = False
        self.USE_NLG = False
        self.USE_SPEECH = False
        self.USER_HAS_INITIATIVE = True
        self.SAVE_LOG = True

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
        self.recorder = DialogueEpisodeRecorder(size=self.minibatch_length)

        # TODO: Get reward type from the config
        self.reward_func = SlotFillingReward()
        # self.reward_func = SlotFillingGoalAdvancementReward()

        if self.configuration:
            agent_id_str = 'AGENT_' + str(self.agent_id)
            
            # Error checks for options the config must have
            if not self.configuration['GENERAL']:
                raise ValueError('Cannot run Plato without GENERAL settings!')

            elif not self.configuration['GENERAL']['interaction_mode']:
                raise ValueError('Cannot run Plato without an interaction '
                                 'mode!')

            elif not self.configuration['DIALOGUE']:
                raise ValueError('Cannot run Plato without DIALOGUE settings!')

            elif not self.configuration[agent_id_str]:
                raise ValueError('Cannot run Plato without at least '
                                 'one agent!')

            # Dialogue domain self.settings
            if 'DIALOGUE' in self.configuration and \
                    self.configuration['DIALOGUE']:
                if 'initiative' in self.configuration['DIALOGUE']:
                    self.USER_HAS_INITIATIVE = bool(
                        self.configuration['DIALOGUE']['initiative'] == 'user'
                    )

                if self.configuration['DIALOGUE']['domain']:
                    self.domain = self.configuration['DIALOGUE']['domain']

                if self.configuration['DIALOGUE']['ontology_path']:
                    if os.path.isfile(
                            self.configuration['DIALOGUE']['ontology_path']
                    ):
                        self.ontology = \
                            Ontology.Ontology(
                                self.configuration['DIALOGUE']['ontology_path']
                            )
                    else:
                        raise FileNotFoundError(
                            'Domain file %s not '
                            'found' % self.configuration[
                                'DIALOGUE'
                            ]['ontology_path']
                        )

                if self.configuration['DIALOGUE']['db_path']:
                    if os.path.isfile(
                            self.configuration['DIALOGUE']['db_path']
                    ):
                        if 'db_type' in self.configuration['DIALOGUE']:
                            if self.configuration['DIALOGUE']['db_type'] == \
                                    'sql':
                                self.database = DataBase.SQLDataBase(
                                    self.configuration['DIALOGUE']['db_path']
                                )
                            else:
                                self.database = DataBase.DataBase(
                                    self.configuration['DIALOGUE']['db_path']
                                )
                        else:
                            # Default to SQL
                            self.database = DataBase.SQLDataBase(
                                self.configuration['DIALOGUE']['db_path']
                            )
                    else:
                        raise FileNotFoundError(
                            'Database file %s not '
                            'found' % self.configuration['DIALOGUE']['db_path']
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

            # General settings
            if 'GENERAL' in self.configuration and \
                    self.configuration['GENERAL']:
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
                                'Dialogue Log file %s not found (did you '
                                'provide one?)' % dialogues_path
                            )

                    if 'save' in \
                            self.configuration['GENERAL']['experience_logs']:
                        self.recorder.set_path(dialogues_path)
                        self.SAVE_LOG = bool(
                            self.configuration['GENERAL'][
                                'experience_logs']['save']
                        )

                if self.configuration['GENERAL']['interaction_mode'] == \
                        'simulation':
                    self.USE_USR_SIMULATOR = True

            # NLU Settings
            if 'NLU' in self.configuration[agent_id_str] and \
                    self.configuration[agent_id_str]['NLU'] and \
                    self.configuration[agent_id_str]['NLU']['nlu']:
                nlu_args = dict(
                    zip(
                        ['ontology', 'database'],
                        [self.ontology, self.database]
                    )
                )

                if self.configuration[agent_id_str]['NLU']['nlu'] == 'dummy':
                    self.nlu = DummyNLU(nlu_args)
                    self.USE_NLU = True

                elif self.configuration[agent_id_str]['NLU']['nlu'] == \
                        'CamRest':
                    if self.configuration[agent_id_str]['NLU']['model_path']:
                        nlu_args['model_path'] = \
                            self.configuration[
                                agent_id_str
                            ]['NLU']['model_path']
                        self.nlu = CamRestNLU(nlu_args)
                        self.USE_NLU = True
                    else:
                        raise ValueError(
                            'Cannot find model_path in the config.'
                        )

            # NLG Settings
            if 'NLG' in self.configuration[agent_id_str] and \
                    self.configuration[agent_id_str]['NLG'] and \
                    self.configuration[agent_id_str]['NLG']['nlg']:
                if self.configuration[agent_id_str]['NLG']['nlg'] == 'dummy':
                    self.nlg = DummyNLG()

                elif self.configuration[agent_id_str]['NLG']['nlg'] == \
                        'CamRest':
                    if self.configuration[agent_id_str]['NLG']['model_path']:
                        self.nlg = CamRestNLG(
                            {
                                'model_path': self.configuration[
                                    agent_id_str
                                ]['NLG']['model_path']}
                        )
                    else:
                        raise ValueError(
                            'Cannot find model_path in the config.'
                        )

                if self.nlg:
                    self.USE_NLG = True

            # Retrieve agent role
            if 'role' in self.configuration[agent_id_str]:
                self.agent_role = self.configuration[agent_id_str]['role']
            else:
                raise ValueError('ConversationalMultiAgent: No role assigned '
                                 'for agent {0} in '
                                 'config!'.format(self.agent_id))

            if self.agent_role == 'user':
                if self.ontology and self.database:
                    self.goal_generator = GoalGenerator(
                        ontology=self.ontology,
                        database=self.database,
                        goals_file=self.goals_path
                    )
                else:
                    raise ValueError(
                        'Conversational Multi Agent (user): '
                        'Cannot generate goal without ontology '
                        'and database.'
                    )

        dm_args = dict(zip([
            'settings',
            'ontology',
            'database',
            'domain',
            'agent_id',
            'agent_role'
        ],
            [self.configuration, self.ontology, self.database,
             self.domain, self.agent_id, self.agent_role]
        ))

        dm_args.update(self.configuration['AGENT_' + str(agent_id)]['DM'])
        self.dialogue_manager = DialogueManager.DialogueManager(dm_args)

    def __del__(self):
        """
        Do some house-keeping and save the models.

        :return: nothing
        """

        if self.recorder and self.SAVE_LOG:
            self.recorder.save()

        if self.dialogue_manager:
            self.dialogue_manager.save()

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

    def start_dialogue(self, goal=None):
        """
        Perform initial dialogue turn.

        :param goal: an optional Goal
        :return:
        """

        self.dialogue_turn = 0

        if self.agent_role == 'user':
            self.agent_goal = self.goal_generator.generate()
            self.dialogue_manager.update_goal(self.agent_goal)

            print('DEBUG > Usr goal:')
            for c in self.agent_goal.constraints:
                print(f'\t\tConstr({self.agent_goal.constraints[c].slot}='
                      f'{self.agent_goal.constraints[c].value})')
            print('\t\t-------------')
            for r in self.agent_goal.requests:
                print(f'\t\tReq({self.agent_goal.requests[r].slot})')
            print('\n')

        elif goal:
            # No deep copy here so that all agents see the same goal.
            self.agent_goal = goal
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

        return response_utterance, response, self.agent_goal

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

        other_input_dact = None
        if 'other_input_dact' in args:
            other_input_dact = args['other_input_dact']

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

        elif other_input_dact:
            # If no utterance provided, use the dacts
            other_input_nlu = other_input_dact

        # print(
        #     '{0} Recognised Input: {1}'.format(
        #         self.agent_role.upper(),
        #         '; '.join([str(ui) for ui in other_input_nlu])
        #     )
        # )

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
            # print('{0}: terminating dialogue due to too many turns'.
            #       format(self.agent_role))
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
                input_utterance=other_input_raw,
                output_utterance=sys_utterance,
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

        return sys_utterance, response, self.agent_goal

    def end_dialogue(self):
        """
        Perform final dialogue turn. Train and ave models if applicable.

        :return:
        """

        if self.dialogue_episode % \
                self.train_switch_trainable_agents_every == 0:
            self.train_system = not self.train_system

        # Record final state
        if not self.curr_state.is_terminal_state:
            self.curr_state.is_terminal_state = True
            self.prev_reward, self.prev_success, self.prev_task_success = \
                self.reward_func.calculate(
                    self.curr_state,
                    [DialogueAct('bye', [])],
                    goal=self.agent_goal,
                    agent_role=self.agent_role
                )

        self.recorder.record(
            self.curr_state,
            self.curr_state,
            self.prev_action,
            self.prev_reward,
            self.prev_success,
            input_utterance=self.prev_usr_utterance,
            output_utterance=self.prev_sys_utterance,
            task_success=self.prev_task_success,
            force_terminate=True
        )

        if self.dialogue_manager.is_training():
            if not self.train_alternate_training or \
                    (self.train_system and
                     self.agent_role == 'system' or
                     not self.train_system and
                     self.agent_role == 'user'):

                if (self.dialogue_episode+1) % self.train_interval == 0 and \
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

        self.dialogue_episode += 1
        self.cumulative_rewards += \
            self.recorder.dialogues[-1][-1]['cumulative_reward']

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % 10000 == 0:
            self.dialogue_manager.save()

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

        # Hard coded response to bye to enforce policy according to which
        # if any agent issues a 'bye' then the dialogue
        # terminates. Otherwise in multi-agent settings it is very hard to
        # learn the association and learn to terminate
        # the dialogue.
        if self.dialogue_manager.get_state().user_acts:
            for act in self.dialogue_manager.get_state().user_acts:
                if act.intent == 'bye':
                    return True

        return self.dialogue_manager.at_terminal_state()

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

        # TODO: Deep copy?
        # Note: reason for non-deep copy is that if this agent changes the goal
        #       these changes are propagated to e.g. the reward function, and
        #       the reward calculation is up to date.
        self.agent_goal = goal
