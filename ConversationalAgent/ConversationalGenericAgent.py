"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from ConversationalAgent.ConversationalAgent import ConversationalAgent
from ConversationalAgent.ConversationalModule import ConversationalFrame
from DialogueManagement.DialoguePolicy.ReinforcementLearning.RewardFunction \
    import SlotFillingGoalAdvancementReward
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder
from Dialogue.Action import DialogueAct

from copy import deepcopy

import os
import speech_recognition as speech_rec

"""
ConversationalGenericAgent is a Conversational Agent that is 
agnostic to its internal modules. It is the most flexible
Plato Conversational Agent as it simply needs a list of modules 
(defined as python classes in the config) and will handle the 
interaction by chaining those modules. This allows for anything 
from a single neural-network module to systems that have tens 
of modules.
"""

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class ConversationalGenericAgent(ConversationalAgent):
    """
    The ConversationalGenericAgent receives a list of modules in
    its configuration file, that are chained together serially -
    i.e. the input to the agent is passed to the first module,
    the first module's output is passed as input to the second
    module and so on. Modules are wrapped using ConversationalModules.
    The input and output passed between modules is wrapped into
    ConversationalFrames.
    """

    def __init__(self, configuration, agent_id):
        """
        Initialize the internal structures of this agent.

        :param configuration: a dictionary representing the configuration file
        :param agent_id: an integer, this agent's id
        """
        self.agent_id = agent_id

        # Dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 250
        self.TRAIN_INTERVAL = 50
        self.train_epochs = 10

        self.configuration = configuration

        self.recorder = DialogueEpisodeRecorder()

        self.SAVE_LOG = True
        self.SAVE_INTERVAL = 10000
        self.MAX_TURNS = 15
        self.INTERACTION_MODE = 'simulation'

        # This indicates which module controls the state so that we can query
        # it for dialogue termination (e.g. at end_dialogue)
        self.STATEFUL_MODULE = -1

        self.reward_func = SlotFillingGoalAdvancementReward()

        self.ConversationalModules = []
        self.prev_m_out = ConversationalFrame({})

        self.goal_generator = None
        self.agent_goal = None

        ag_id_str = 'AGENT_' + str(agent_id)

        if self.configuration:
            if 'GENERAL' not in self.configuration:
                raise ValueError('No GENERAL section in config!')
            if 'AGENT_'+str(agent_id) not in self.configuration:
                raise ValueError(f'NO AGENT_{agent_id} section in config!')

            if 'interaction_mode' in self.configuration['GENERAL']:
                self.INTERACTION_MODE = \
                    self.configuration['GENERAL']['interaction_mode']

            if 'experience_logs' in self.configuration['GENERAL']:
                dialogues_path = None
                if 'path' in self.configuration['GENERAL']['experience_logs']:
                    dialogues_path = \
                        self.configuration['GENERAL'][
                            'experience_logs']['path']

                if 'load' in self.configuration['GENERAL']['experience_logs'] \
                        and bool(self.configuration['GENERAL']
                                 ['experience_logs']['load']
                                 ):
                    if dialogues_path and os.path.isfile(dialogues_path):
                        self.recorder.load(dialogues_path)
                    else:
                        raise FileNotFoundError(
                            'Dialogue Log file %s not found (did you '
                            'provide one?)' % dialogues_path)

                if 'save' in self.configuration['GENERAL']['experience_logs']:
                    self.recorder.set_path(dialogues_path)
                    self.SAVE_LOG = bool(
                        self.configuration['GENERAL'][
                            'experience_logs']['save']
                    )

            self.NModules = 0
            if 'modules' in self.configuration[ag_id_str]:
                self.NModules = int(
                    self.configuration[ag_id_str]['modules']
                )

            if 'stateful_module' in self.configuration[ag_id_str]:
                self.STATEFUL_MODULE = int(
                    self.configuration[ag_id_str]['stateful_module']
                )

            # Note: Since we pass settings as a default argument, any
            #       module can access the global args. However, we
            #       add it here too for ease of use.
            self.global_arguments = {'settings': self.configuration}
            if 'global_arguments' in self.configuration['GENERAL']:
                self.global_arguments.update(
                    self.configuration['GENERAL']['global_arguments']
                )

            # Load the modules
            for m in range(self.NModules):
                if 'MODULE_'+str(m) not in \
                        self.configuration[ag_id_str]:
                    raise ValueError(f'No MODULE_{m} section in config!')

                if 'parallel_modules' in self.configuration[
                    ag_id_str
                ]['MODULE_' + str(m)]:

                    n_parallel_modules = self.configuration[
                        ag_id_str][
                        'MODULE_' + str(m)]['parallel_modules']

                    parallel_modules = []

                    for pm in range(n_parallel_modules):
                        if 'package' not in self.configuration[
                            ag_id_str
                        ]['MODULE_' + str(m)]['PARALLEL_MODULE_' + str(pm)]:
                            raise ValueError(
                                f'No arguments provided for parallel module '
                                f'{pm} of module {m}!')

                        package = self.configuration[
                            ag_id_str
                            ]['MODULE_' + str(m)][
                            'PARALLEL_MODULE_' + str(pm)]['package']

                        if 'class' not in self.configuration[
                            ag_id_str
                        ]['MODULE_' + str(m)]['PARALLEL_MODULE_' + str(pm)]:
                            raise ValueError(
                                f'No arguments provided for parallel module '
                                f'{pm} of module {m}!')

                        klass = self.configuration[
                            ag_id_str
                        ]['MODULE_' + str(m)][
                            'PARALLEL_MODULE_' + str(pm)]['class']

                        # Append global arguments
                        # (add configuration by default)
                        args = deepcopy(self.global_arguments)
                        if 'arguments' in \
                                self.configuration[
                                    ag_id_str
                                ]['MODULE_' + str(m)][
                                    'PARALLEL_MODULE_' + str(pm)]:
                            args.update(
                                self.configuration[
                                    ag_id_str
                                ]['MODULE_' + str(m)][
                                    'PARALLEL_MODULE_' + str(pm)]['arguments'])

                        parallel_modules.append(
                            self.load_module(package, klass, args))

                    self.ConversationalModules.append(
                        parallel_modules
                    )

                else:
                    if 'package' not in self.configuration[
                        ag_id_str
                    ]['MODULE_' + str(m)]:
                        raise ValueError(f'No arguments provided for module '
                                         f'{m}!')

                    package = self.configuration[
                        ag_id_str
                    ]['MODULE_' + str(m)]['package']

                    if 'class' not in self.configuration[
                        ag_id_str
                    ]['MODULE_' + str(m)]:
                        raise ValueError(f'No arguments provided for module '
                                         f'{m}!')

                    klass = self.configuration[
                        ag_id_str
                    ]['MODULE_' + str(m)]['class']

                    # Append global arguments (add configuration by default)
                    args = deepcopy(self.global_arguments)
                    if 'arguments' in \
                            self.configuration[
                                ag_id_str
                            ]['MODULE_' + str(m)]:
                        args.update(
                            self.configuration[
                                'AGENT_'+str(agent_id)
                            ]['MODULE_'+str(m)]['arguments'])

                    self.ConversationalModules.append(
                        self.load_module(package, klass, args)
                    )

        else:
            raise AttributeError('ConversationalGenericAgent: '
                                 'No settings (config) provided!')

        # TODO: Parse config modules I/O and raise error if
        #       any inconsistencies found

        # Initialize automatic speech recognizer, if necessary
        self.asr = None
        if self.INTERACTION_MODE == 'speech':
            self.asr = speech_rec.Recognizer()

    def __del__(self):
        """
        Do some house-keeping, save the models.

        :return: nothing
        """

        if self.recorder and self.SAVE_LOG:
            self.recorder.save()

        for m in self.ConversationalModules:
            if isinstance(m, list):
                for sm in m:
                    sm.save()
            else:
                m.save()

    # Dynamically load classes
    @staticmethod
    def load_module(package_path, class_name, args):
        """
        Dynamically load the specified class.

        :param package_path: Path to the package to load
        :param class_name: Name of the class within the package
        :param args: arguments to pass when creating the object
        :return: the instantiated class object
        """
        module = __import__(package_path, fromlist=[class_name])
        klass = getattr(module, class_name)
        return klass(args)

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
        self.agent_goal = None

        # For each module
        for m in self.ConversationalModules:
            if isinstance(m, list):
                for sm in m:
                    sm.initialize({})
            else:
                # Load and initialize
                m.initialize({})

    def start_dialogue(self, args=None):
        """
        Reset or initialize internal structures at the beginning of the
        dialogue. May issue first utterance if this agent has the initiative.

        :param args:
        :return:
        """

        self.initialize()
        self.dialogue_turn = 0

        # TODO: Get initial trigger from config
        if self.INTERACTION_MODE in ['speech', 'text', 'simulation']:
            self.prev_m_out = ConversationalFrame({'utterance': 'hello'})
        else:
            self.prev_m_out = \
                ConversationalFrame([DialogueAct('hello')])

        self.continue_dialogue()

        return self.prev_m_out.content, '', self.agent_goal

    def continue_dialogue(self, args=None):
        """
        Perform one dialogue turn

        :param args: input to this agent
        :return: output of this agent
        """

        if self.INTERACTION_MODE == 'text':
            self.prev_m_out = input('USER > ')

        elif self.INTERACTION_MODE == 'speech':
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

        for m in self.ConversationalModules:
            # If executing parallel sub-modules
            if isinstance(m, list):
                idx = 0
                prev_m_out = deepcopy(self.prev_m_out)
                self.prev_m_out.content = {}

                for sm in m:
                    # WARNING! Module compatibility cannot be guaranteed here!
                    sm.generic_receive_input(prev_m_out)
                    sm_out = sm.generic_generate_output(prev_m_out)

                    if not isinstance(sm_out, ConversationalFrame):
                        sm_out = ConversationalFrame(sm_out)

                    self.prev_m_out.content['sm'+str(idx)] = sm_out.content
                    idx += 1

            else:
                # WARNING! Module compatibility cannot be guaranteed here!
                m.generic_receive_input(self.prev_m_out)
                self.prev_m_out = m.generic_generate_output(self.prev_m_out)

                # Make sure prev_m_out is a Conversational Frame
                if not isinstance(self.prev_m_out, ConversationalFrame):
                    self.prev_m_out = ConversationalFrame(self.prev_m_out)

            # DEBUG:
            if isinstance(self.prev_m_out.content, str):
                print('DEBUG> ' + str(self.prev_m_out.content))

        self.dialogue_turn += 1

        return self.prev_m_out.content, '', self.agent_goal

    def end_dialogue(self):
        """
        Perform final dialogue turn. Save models if applicable.

        :return:
        """

        if self.dialogue_episode % self.TRAIN_INTERVAL == 0:
            for m in self.ConversationalModules:
                if isinstance(m, list):
                    for sm in m:
                        sm.train(self.recorder.dialogues)
                else:
                    m.train(self.recorder.dialogues)

        if self.dialogue_episode % self.SAVE_INTERVAL == 0:
            for m in self.ConversationalModules:
                if isinstance(m, list):
                    for sm in m:
                        sm.save()
                else:
                    m.save()

        self.dialogue_episode += 1

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        return self.ConversationalModules[
                   self.STATEFUL_MODULE
               ].at_terminal_state() or \
            self.dialogue_turn > self.MAX_TURNS

    def set_goal(self, goal):
        """
        Set or update this agent's goal.

        :param goal: a Goal
        :return: nothing
        """

        self.agent_goal = goal

    def get_goal(self):
        """
        Get this agent's goal.

        :return: a Goal
        """

        return self.agent_goal
