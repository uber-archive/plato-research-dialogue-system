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

from ConversationalAgent.ConversationalAgent import ConversationalAgent
from ConversationalAgent.ConversationalModule import ConversationalFrame
from DialogueManagement.Policy.ReinforcementLearning.RewardFunction import SlotFillingGoalAdvancementReward
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder

from UserSimulator.AgendaBasedUserSimulator.Goal import GoalGenerator

from copy import deepcopy

import os


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class ConversationalGenericAgent(ConversationalAgent):
    '''
    Essentially the dialogue system. Will be able to interact with:

    - Simulated Users via:
        - Dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - Data
    '''

    def __init__(self, settings, agent_id):
        self.agent_id = agent_id

        # Dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 250
        self.TRAIN_INTERVAL = 50
        self.train_epochs = 10

        self.settings = settings

        self.recorder = DialogueEpisodeRecorder()

        self.SAVE_LOG = True
        self.SAVE_INTERVAL = 10000
        self.MAX_TURNS = 15

        self.reward_func = SlotFillingGoalAdvancementReward()

        self.ConversationalModules = []
        self.prev_m_out = ConversationalFrame({})

        self.goal_generator = None
        self.agent_goal = None

        if self.settings:
            if 'GENERAL' not in self.settings:
                raise ValueError('No GENERAL section in config!')
            if 'AGENT_'+str(agent_id) not in self.settings:
                raise ValueError(f'NO AGENT_{agent_id} section in config!')

            if 'dialogues' in self.settings['GENERAL']:
                dialogues_path = None
                if 'path' in self.settings['GENERAL']['dialogues']:
                    dialogues_path = self.settings['GENERAL']['dialogues']['path']

                if 'load' in self.settings['GENERAL']['dialogues'] and bool(self.settings['GENERAL']['dialogues']['load']):
                    if dialogues_path and os.path.isfile(dialogues_path):
                        self.recorder.load(dialogues_path)
                    else:
                        raise FileNotFoundError('Dialogue Log file %s not found (did you provide one?)' % dialogues_path)

                if 'save' in self.settings['GENERAL']['dialogues']:
                    self.recorder.set_path(dialogues_path)
                    self.SAVE_LOG = bool(self.settings['GENERAL']['dialogues']['save'])

            self.NModules = 0
            if 'modules' in self.settings['AGENT_'+str(agent_id)]:
                self.NModules = int(self.settings['AGENT_'+str(agent_id)]['modules'])

            # Note: Since we pass settings as a default argument, any module can access the global args. However, we
            #       add it here too for ease of use.
            self.global_arguments = {'settings': self.settings}
            if 'arguments' in self.settings['GENERAL']:
                self.global_arguments.update(self.settings['GENERAL']['arguments'])

            # Load the modules
            for m in range(self.NModules):
                if 'MODULE_'+str(m) not in self.settings['AGENT_'+str(agent_id)]:
                    raise ValueError(f'No MODULE_{m} section in config!')

                if 'package' not in self.settings['AGENT_'+str(agent_id)]['MODULE_'+str(m)]:
                    raise ValueError(f'No arguments provided for module {m}!')

                package = self.settings['AGENT_'+str(agent_id)]['MODULE_'+str(m)]['package']

                if 'class' not in self.settings['AGENT_' + str(agent_id)]['MODULE_' + str(m)]:
                    raise ValueError(f'No arguments provided for module {m}!')

                klass = self.settings['AGENT_' + str(agent_id)]['MODULE_' + str(m)]['class']

                # Collect arguments (add settings / config by default)
                args = deepcopy(self.global_arguments)
                if 'arguments' in self.settings['AGENT_'+str(agent_id)]['MODULE_'+str(m)]:
                    args.update(settings['AGENT_'+str(agent_id)]['MODULE_'+str(m)]['arguments'])

                self.ConversationalModules.append(self.load_module(package, klass, args))

        else:
            raise AttributeError('ConversationalGenericAgent: No settings (config) provided!')

        # TODO: Parse config modules I/O and raise error if any inconsistencies found

    def __del__(self):
        if self.recorder and self.SAVE_LOG:
            self.recorder.save()

        for m in self.ConversationalModules:
            m.save()

    # Dynamically load classes
    def load_module(self, package_path, class_name, args):
        module = __import__(package_path, fromlist=[class_name])
        klass = getattr(module, class_name)
        return klass(args)

    def initialize(self):
        '''
        Initializes the conversational agent based on settings in the configuration file.

        :return: Nothing
        '''

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.cumulative_rewards = 0
        self.agent_goal = None

        # For each module
        for m in self.ConversationalModules:
            # Load and initialize
            m.initialize({})

    def start_dialogue(self, args=None):
        self.initialize()
        self.dialogue_turn = 0
        # TODO: Get initial trigger from config
        self.prev_m_out = ConversationalFrame({'utterance': 'Hello'})
        self.continue_dialogue()

        return self.prev_m_out.content, '', self.agent_goal

    def continue_dialogue(self, args=None):
        for m in self.ConversationalModules:
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
        '''
        Perform final dialogue turn. Save models if applicable.

        :return:
        '''

        if self.dialogue_episode % self.TRAIN_INTERVAL == 0:
            for m in self.ConversationalModules:
                m.train(self.recorder.dialogues)

        if self.dialogue_episode % self.SAVE_INTERVAL == 0:
            for m in self.ConversationalModules:
                m.save()

        self.dialogue_episode += 1

    def terminated(self):
        return self.ConversationalModules[-1].at_terminal_state() or self.dialogue_turn > self.MAX_TURNS

    def set_goal(self, goal):
        self.agent_goal = goal

    def get_goal(self):
        return self.agent_goal

