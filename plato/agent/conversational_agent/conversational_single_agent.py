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

from plato.agent.conversational_agent.conversational_agent import \
    ConversationalAgent
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent

from plato.agent.component.user_simulator.agenda_based_user_simulator.\
    agenda_based_us import AgendaBasedUS
from plato.agent.component.user_simulator.dact_to_language_user_simulator.\
    dtl_user_simulator import DTLUserSimulator
from plato.agent.component.user_simulator.user_model import UserModel
from plato.agent.component.user_simulator.goal import GoalGenerator
from plato.agent.component.dialogue_policy.reinforcement_learning.\
    reward_function import SlotFillingReward
from plato.utilities.dialogue_episode_recorder import DialogueEpisodeRecorder
from plato.domain import ontology, database
from plato.dialogue.action import DialogueAct

from gtts import gTTS
from copy import deepcopy

import os
import random
import speech_recognition as speech_rec

"""
The ConversationalSingleAgent implements the standard architecture of a 
dialogue system, with some flexibility in the input and output (either 
can be dialogue Acts, Text, or Speech). Each component's parameters must 
be in the config, otherwise this agent will either raise an error or use 
default values.
"""

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class ConversationalSingleAgent(ConversationalAgent):
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

    def __init__(self, configuration):
        """
        Initialize the internal structures of this agent.

        :param configuration: a dictionary representing the configuration file
        """

        super(ConversationalSingleAgent, self).__init__()

        self.configuration = configuration

        # There is only one agent in this setting
        self.agent_id = 0

        # dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        # Default meta-parameter values
        self.minibatch_length = 500
        self.train_interval = 50
        self.train_epochs = 3

        # True values here would imply some default modules
        self.USE_USR_SIMULATOR = False
        self.USER_SIMULATOR_NLU = False
        self.USER_SIMULATOR_NLG = False
        self.USE_NLG = False
        self.USE_SPEECH = False
        self.USER_HAS_INITIATIVE = True
        self.SAVE_LOG = True
        self.SAVE_INTERVAL = 10000

        # The dialogue will terminate after MAX_TURNS (this agent will issue
        # a bye() dialogue act.
        self.MAX_TURNS = 15

        self.dialogue_turn = -1
        self.ontology = None
        self.database = None
        self.domain = None
        self.global_args = {}
        self.dialogue_manager = None
        self.user_model = None
        self.user_simulator = None
        self.user_simulator_args = {}
        self.nlu = None
        self.nlg = None

        self.agent_role = None
        self.agent_goal = None
        self.goal_generator = None

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.user_model = UserModel()

        self.recorder = DialogueEpisodeRecorder()

        # TODO: Handle this properly - get reward function type from config
        self.reward_func = SlotFillingReward()
        # self.reward_func = SlotFillingGoalAdvancementReward()

        if self.configuration:
            # Error checks for options the config must have
            if not self.configuration['GENERAL']:
                raise ValueError('Cannot run Plato without GENERAL settings!')

            elif not self.configuration['GENERAL']['interaction_mode']:
                raise ValueError('Cannot run Plato without an '
                                 'interaction mode!')

            elif not self.configuration['DIALOGUE']:
                raise ValueError('Cannot run Plato without DIALOGUE settings!')

            elif not self.configuration['AGENT_0']:
                raise ValueError('Cannot run Plato without at least '
                                 'one agent!')

            # dialogue domain self.settings
            if 'DIALOGUE' in self.configuration and \
                    self.configuration['DIALOGUE']:
                if 'initiative' in self.configuration['DIALOGUE']:
                    self.USER_HAS_INITIATIVE = bool(
                        self.configuration['DIALOGUE']['initiative'] == 'user'
                    )
                    self.user_simulator_args['us_has_initiative'] = \
                        self.USER_HAS_INITIATIVE

                if self.configuration['DIALOGUE']['domain']:
                    self.domain = self.configuration['DIALOGUE']['domain']

                if 'ontology_path' in self.configuration['DIALOGUE']:
                    if os.path.isfile(
                            self.configuration['DIALOGUE']['ontology_path']
                    ):
                        self.ontology = ontology.Ontology(
                            self.configuration['DIALOGUE']['ontology_path']
                        )
                    else:
                        raise FileNotFoundError(
                            'domain file %s not found' %
                            self.configuration['DIALOGUE']['ontology_path'])

                # Alternatively, look at global_arguments for ontology path
                elif 'global_arguments' in self.configuration['GENERAL'] \
                        and 'ontology' in \
                        self.configuration['GENERAL']['global_arguments']:
                    if os.path.isfile(
                            self.configuration['GENERAL'][
                                'global_arguments']['ontology']
                    ):
                        self.ontology = ontology.Ontology(
                            self.configuration['GENERAL'][
                                'global_arguments']['ontology']
                        )
                    else:
                        raise FileNotFoundError(
                            'domain file %s not found' %
                            self.configuration['GENERAL'][
                                'global_arguments']['ontology'])

                if 'db_path' in self.configuration['DIALOGUE']:
                    if os.path.isfile(
                            self.configuration['DIALOGUE']['db_path']
                    ):
                        if 'db_type' in self.configuration['DIALOGUE']:
                            if self.configuration['DIALOGUE']['db_type'] == \
                                    'sql':
                                self.database = database.SQLDataBase(
                                    self.configuration['DIALOGUE']['db_path']
                                )
                            else:
                                self.database = database.DataBase(
                                    self.configuration['DIALOGUE']['db_path']
                                )
                        else:
                            # Default to SQL
                            self.database = database.SQLDataBase(
                                self.configuration['DIALOGUE']['db_path']
                            )
                    else:
                        raise FileNotFoundError(
                            'Database file %s not found' %
                            self.configuration['DIALOGUE']['db_path']
                        )

                # Alternatively, look at global arguments for db path
                elif 'global_arguments' in self.configuration['GENERAL'] \
                        and 'database' in \
                        self.configuration['GENERAL']['global_arguments']:
                    if os.path.isfile(
                            self.configuration['GENERAL'][
                                'global_arguments']['database']
                    ):
                        self.database = database.DataBase(
                            self.configuration['GENERAL'][
                                'global_arguments']['database']
                        )
                    else:
                        raise FileNotFoundError(
                            'domain file %s not found' %
                            self.configuration['GENERAL'][
                                'global_arguments']['ontology'])

                if 'goals_path' in self.configuration['DIALOGUE']:
                    if os.path.isfile(
                            self.configuration['DIALOGUE']['goals_path']
                    ):
                        self.goals_path = \
                            self.configuration['DIALOGUE']['goals_path']
                    else:
                        raise FileNotFoundError(
                            'Goals file %s not found' %
                            self.configuration['DIALOGUE']['goals_path']
                        )

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
                            self.configuration['GENERAL']['experience_logs'] \
                            and bool(self.configuration['GENERAL'][
                                         'experience_logs']['load']):
                        if dialogues_path and os.path.isfile(dialogues_path):
                            self.recorder.load(dialogues_path)
                        else:
                            raise FileNotFoundError(
                                'dialogue Log file %s not found (did you '
                                'provide one?)' % dialogues_path)

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

                elif self.configuration['GENERAL']['interaction_mode'] == \
                        'speech':
                    self.USE_SPEECH = True
                    self.asr = speech_rec.Recognizer()

            # Agent Settings

            # Retrieve agent role
            if 'role' in self.configuration['AGENT_0']:
                self.agent_role = self.configuration['AGENT_0']['role']
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
                        'Conversational Single Agent (user): Cannot generate '
                        'goal without ontology and database.'
                    )
                
            # Retrieve agent parameters
            if 'max_turns' in self.configuration['AGENT_0']:
                self.MAX_TURNS = self.configuration['AGENT_0']['max_turns']
                
            if 'train_interval' in self.configuration['AGENT_0']:
                self.train_interval = \
                    self.configuration['AGENT_0']['train_interval']
                
            if 'train_minibatch' in self.configuration['AGENT_0']:
                self.minibatch_length = \
                    self.configuration['AGENT_0']['train_minibatch']
                
            if 'train_epochs' in self.configuration['AGENT_0']:
                self.train_epochs = \
                    self.configuration['AGENT_0']['train_epochs']

            if 'save_interval' in self.configuration['AGENT_0']:
                self.SAVE_INTERVAL = \
                    self.configuration['AGENT_0']['save_interval']
            
            # usr Simulator
            # Check for specific simulator self.settings, otherwise
            # default to agenda
            if 'USER_SIMULATOR' in self.configuration['AGENT_0']:
                # Agent 0 simulator configuration
                if 'package' in \
                    self.configuration['AGENT_0']['USER_SIMULATOR'] and \
                        'class' in \
                        self.configuration['AGENT_0']['USER_SIMULATOR']:
                    if 'arguments' in \
                            self.configuration['AGENT_0']['USER_SIMULATOR']:
                        self.user_simulator_args =\
                            self.configuration[
                                'AGENT_0']['USER_SIMULATOR']['arguments']

                    self.user_simulator_args.update(self.global_args)
                    
                    self.user_simulator = \
                        ConversationalGenericAgent.load_module(
                            self.configuration['AGENT_0']['USER_SIMULATOR'][
                                'package'],
                            self.configuration['AGENT_0']['USER_SIMULATOR'][
                                'class'],
                            self.user_simulator_args
                        )

                    if hasattr(self.user_simulator, 'nlu'):
                        self.USER_SIMULATOR_NLU = self.user_simulator.nlu
                    if hasattr(self.user_simulator, 'nlg'):
                        self.USER_SIMULATOR_NLG = self.user_simulator.nlg
                else:
                    # Fallback to agenda based simulator with default settings
                    self.user_simulator = AgendaBasedUS(
                        self.user_simulator_args
                    )

            # NLU Settings
            if 'NLU' in self.configuration['AGENT_0']:
                nlu_args = {}
                if 'package' in self.configuration['AGENT_0']['NLU'] and \
                        'class' in self.configuration['AGENT_0']['NLU']:
                    if 'arguments' in \
                            self.configuration['AGENT_0']['NLU']:
                        nlu_args = \
                            self.configuration['AGENT_0']['NLU']['arguments']

                    nlu_args.update(self.global_args)

                    self.nlu = \
                        ConversationalGenericAgent.load_module(
                            self.configuration['AGENT_0']['NLU'][
                                'package'],
                            self.configuration['AGENT_0']['NLU'][
                                'class'],
                            nlu_args
                        )
                    
            # DM Settings
            if 'DM' in self.configuration['AGENT_0']:
                dm_args = dict(
                    zip(
                        ['settings', 'ontology', 'database', 'domain',
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

                if 'package' in self.configuration['AGENT_0']['DM'] and \
                        'class' in self.configuration['AGENT_0']['DM']:
                    if 'arguments' in \
                            self.configuration['AGENT_0']['DM']:
                        dm_args.update(
                            self.configuration['AGENT_0']['DM']['arguments']
                        )

                    dm_args.update(self.global_args)

                    self.dialogue_manager = \
                        ConversationalGenericAgent.load_module(
                            self.configuration['AGENT_0']['DM'][
                                'package'],
                            self.configuration['AGENT_0']['DM'][
                                'class'],
                            dm_args
                        )

            # NLG Settings
            if 'NLG' in self.configuration['AGENT_0']:
                nlg_args = {}
                if 'package' in self.configuration['AGENT_0']['NLG'] and \
                        'class' in self.configuration['AGENT_0']['NLG']:
                    if 'arguments' in \
                            self.configuration['AGENT_0']['NLG']:
                        nlg_args = \
                            self.configuration['AGENT_0']['NLG']['arguments']

                    nlg_args.update(self.global_args)

                    self.nlg = \
                        ConversationalGenericAgent.load_module(
                            self.configuration['AGENT_0']['NLG'][
                                'package'],
                            self.configuration['AGENT_0']['NLG'][
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

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

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

        self.curr_state = None
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

        :param args: optional args
        :return:
        """

        self.dialogue_turn = 0
        sys_utterance = ''

        if self.USE_USR_SIMULATOR:
            self.user_simulator.initialize(self.user_simulator_args)

            print('DEBUG > usr goal:')
            print(self.user_simulator.goal)

        self.dialogue_manager.restart({})

        if not self.USER_HAS_INITIATIVE:
            # sys_response = self.dialogue_manager.respond()
            sys_response = [DialogueAct('welcomemsg', [])]

            if self.USE_NLG:
                sys_utterance = self.nlg.generate_output(
                    {'dacts': sys_response}
                )
                print('SYSTEM > %s ' % sys_utterance)

                if self.USE_SPEECH:
                    try:
                        tts = gTTS(sys_utterance)
                        tts.save('sys_output.mp3')
                        os.system('afplay sys_output.mp3')

                    except Exception as e:
                        print(
                            'WARNING: gTTS encountered an error: {0}. '
                            'Falling back to sys TTS.'.format(e)
                        )
                        os.system('say ' + sys_utterance)
            else:
                print(
                    'SYSTEM > %s ' % '; '.
                    join([str(sr) for sr in sys_response])
                )

            if self.USE_USR_SIMULATOR:
                usim_input = sys_response

                if self.USER_SIMULATOR_NLU and self.USE_NLG:
                    usim_input = self.user_simulator.nlu.process_input(
                        sys_utterance
                    )

                self.user_simulator.receive_input(usim_input)
                rew, success, task_success = self.reward_func.calculate(
                    self.dialogue_manager.get_state(),
                    sys_response,
                    self.user_simulator.goal
                )
            else:
                rew, success, task_success = 0, None, None

            self.recorder.record(
                deepcopy(self.dialogue_manager.get_state()),
                self.dialogue_manager.get_state(),
                sys_response,
                rew,
                success,
                task_success,
                output_utterance=sys_utterance
            )

            self.dialogue_turn += 1

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.continue_dialogue()

    def continue_dialogue(self):
        """
        Perform next dialogue turn.

        :return: nothing
        """

        usr_utterance = ''
        sys_utterance = ''

        if self.USE_USR_SIMULATOR:
            usr_input = self.user_simulator.respond()

            # TODO: THIS FIRST IF WILL BE HANDLED BY ConversationalAgentGeneric
            #  -- SHOULD NOT LIVE HERE
            if isinstance(self.user_simulator, DTLUserSimulator):
                print('USER (nlg) > %s \n' % usr_input)
                usr_input = self.nlu.process_input(
                    usr_input,
                    self.dialogue_manager.get_state()
                )

            elif self.USER_SIMULATOR_NLG:
                print('USER > %s \n' % usr_input)

                if self.nlu:
                    usr_input = self.nlu.process_input(usr_input)

                    # Otherwise it will just print the user's nlg but use the
                    # simulator's output DActs to proceed.

            else:
                print('USER (DACT) > %s \n' % '; '.join(
                    [str(ui) for ui in usr_input]))

        else:
            if self.USE_SPEECH:
                # Listen for input from the microphone
                with speech_rec.Microphone() as source:
                    print('(listening...)')
                    audio = self.asr.listen(source, phrase_time_limit=3)

                try:
                    # This uses the default key
                    usr_utterance = self.asr.recognize_google(audio)
                    print("Google ASR: " + usr_utterance)

                except speech_rec.UnknownValueError:
                    print("Google ASR did not understand you")

                except speech_rec.RequestError as e:
                    print("Google ASR request error: {0}".format(e))

            else:
                usr_utterance = input('USER > ')

            # Process the user's utterance
            if self.nlu:
                usr_input = self.nlu.process_input(
                    usr_utterance,
                    self.dialogue_manager.get_state()
                )
            else:
                raise EnvironmentError(
                    'agent: No nlu defined for '
                    'text-based interaction!'
                )

        self.dialogue_manager.receive_input(usr_input)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue manager
        # will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())

        if self.dialogue_turn < self.MAX_TURNS:
            sys_response = self.dialogue_manager.generate_output()

        else:
            # Force dialogue stop
            sys_response = [DialogueAct('bye', [])]

        if self.USE_NLG:
            sys_utterance = self.nlg.generate_output({'dacts': sys_response})
            print('SYSTEM > %s ' % sys_utterance)

            if self.USE_SPEECH:
                try:
                    tts = gTTS(text=sys_utterance, lang='en')
                    tts.save('sys_output.mp3')
                    os.system('afplay sys_output.mp3')

                except:
                    print('WARNING: gTTS encountered an error. '
                          'Falling back to sys TTS.')
                    os.system('say ' + sys_utterance)
        else:
            print('SYSTEM > %s ' % '; '.join([str(sr) for sr in sys_response]))

        if self.USE_USR_SIMULATOR:
            usim_input = sys_response

            if self.USER_SIMULATOR_NLU and self.USE_NLG:
                usim_input = \
                    self.user_simulator.nlu.process_input(sys_utterance)

            self.user_simulator.receive_input(usim_input)
            rew, success, task_success = \
                self.reward_func.calculate(
                    self.dialogue_manager.get_state(),
                    sys_response,
                    self.user_simulator.goal
                )
        else:
            rew, success, task_success = 0, None, None

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
        self.prev_action = deepcopy(sys_response)
        self.prev_usr_utterance = deepcopy(usr_utterance)
        self.prev_sys_utterance = deepcopy(sys_utterance)
        self.prev_reward = rew
        self.prev_success = success
        self.prev_task_success = task_success

    def end_dialogue(self):
        """
        Perform final dialogue turn. Train and save models if applicable.

        :return: nothing
        """

        # Record final state
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

        self.dialogue_episode += 1

        if self.IS_TRAINING:
            if self.dialogue_episode % self.train_interval == 0 and \
                    len(self.recorder.dialogues) >= self.minibatch_length:
                for epoch in range(self.train_epochs):
                    print('Training epoch {0} of {1}'.format(
                        (epoch+1),
                        self.train_epochs)
                    )

                    # Sample minibatch
                    minibatch = random.sample(
                        self.recorder.dialogues,
                        self.minibatch_length
                    )

                    if self.nlu and self.nlu.training:
                        self.nlu.train(minibatch)

                    if self.dialogue_manager.is_training():
                        self.dialogue_manager.train(minibatch)

                    if self.nlg and self.nlg.training:
                        self.nlg.train(minibatch)

        # Keep track of dialogue statistics
        self.cumulative_rewards += \
            self.recorder.dialogues[-1][-1]['cumulative_reward']
        print('CUMULATIVE REWARD: {0}'.
              format(self.recorder.dialogues[-1][-1]['cumulative_reward']))

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
            print('SUCCESS (Subjective)!')
            self.num_successful_dialogues += \
                int(self.recorder.dialogues[-1][-1]['success'])

        else:
            print('FAILURE (Subjective).')

        if self.recorder.dialogues[-1][-1]['task_success']:
            self.num_task_success += \
                int(self.recorder.dialogues[-1][-1]['task_success'])

        print('OBJECTIVE TASK SUCCESS: {0}'.
              format(self.recorder.dialogues[-1][-1]['task_success']))

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        return self.dialogue_manager.at_terminal_state() or \
            self.dialogue_turn > self.MAX_TURNS
