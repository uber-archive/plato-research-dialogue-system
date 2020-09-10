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

from plato.controller.controller import Controller
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent
from plato.agent.conversational_agent.generic_helpers import \
    GenericSpeechSynthesiserHelper

import PySimpleGUI as sg
import configparser
import yaml
import sys
import os.path
import time
import random
import speech_recognition as speech_rec

"""
This is the main entry point to Plato using the Graphical usr Interface.

The controller class is responsible for running each dialogue until the 
agent(s) terminate. In the multi-agent case,
the controller will pass the appropriate input to each agent.
"""


class SGUIController(Controller):
    def __init__(self):
        # These are needed to keep track of the dialogue turn by turn when the 
        # GUI is enabled
        self.GUI_dialogue_initialized = False
        self.GUI_system_turn = True
        self.GUI_SHOWN = False
        self.INTERACTION_MODE = 'simulation'

        self.sys_output = ''
        self.user_output = ''
        self.sys_output_history = ''
        self.user_output_history = ''
        self.sys_output_dacts = []
        self.user_output_dacts = []
        self.goal = None
        self.initiative = 'system'
        self.agent_0_title = ''
        self.agent_1_title = ''
        self.asr = None
        self.tts = GenericSpeechSynthesiserHelper({})

        # GUI
        self.window = None

    def run_single_agent(self, settings, human_role='user'):

        print('PATH:')
        print(os.system('pwd'))

        # Initialize automatic speech recognizer, if necessary
        if self.INTERACTION_MODE == 'speech':
            self.asr = speech_rec.Recognizer()

        ConvSysAgent = None
        ConvUserAgent = None

        # Verify that we have an AGENT section
        ag_id = 'AGENT_0'

        if ag_id in settings:
            if 'role' in settings[ag_id]:
                if settings[ag_id]['role'] == 'system':
                    ConvSysAgent = ConversationalGenericAgent(settings, 0)
                    ConvSysAgent.initialize()

                elif settings[ag_id]['role'] == 'user':
                    ConvUserAgent = ConversationalGenericAgent(settings, 0)
                    ConvUserAgent.initialize()

                else:
                    print('WARNING: Unknown agent role: {0}!'.format(
                        settings[ag_id]['role']))
            else:
                raise ValueError(
                    'Role for agent {0} not defined in config.'.format(0))

        # Lookup dictionary that maps button to function to call
        func_dict = {'Start Call': self.run_speech}

        # Layout the design of the GUI
        layout = [
            [sg.Image(r'plato/resources/PlatoRDSLogo_small.gif')],

            [sg.Text('User: ',
                     size=(27, 1),
                     justification='left',
                     font=("Helvetica", 50)),

             sg.Text('System: ',
                     size=(27, 1),
                     justification='left',
                     font=("Helvetica", 50)),
             ],

            [sg.Text('',
                     size=(60, 15),
                     justification='left',
                     font=("Helvetica", 20),
                     key='_USER_'),

             sg.Text('',
                     size=(60, 15),
                     justification='left',
                     font=("Helvetica", 20),
                     key='_SYS_')
             ],

            [sg.Button('Start Call'),
             sg.Quit(),
             sg.Text('',
                     size=(25, 1),
                     auto_size_text=True,
                     font=("Helvetica", 15),
                     key='_ASR_')]]

        # Show the Window to the user
        self.window = sg.Window('Plato Conversational Agent', layout)

        # Event loop. Read buttons, make callbacks
        while True:
            # Read the Window
            event, value = self.window.Read()

            if event in ('Quit', None):
                break

            # Lookup event in function dictionary
            try:
                func_to_call = func_dict[event]
                func_to_call(ConvSysAgent)

            except:
                pass

        self.window.Close()

    def speak(self, utterance):
        self.window.Refresh()
        self.tts.generate_output({'utterance': utterance})

    def update_text(self, key, history, utterance):
        self.window.Element(key).Update(history)
        if utterance and self.tts:
            self.speak(utterance)

    def run_speech(self, sys_agent):
        self.sys_output = ''
        self.sys_output_dacts = []
        self.goal = None
        asr_utterance = ''

        while 'bye' not in self.user_output:

            # Listen for input from the microphone
            with speech_rec.Microphone() as source:
                print('(listening...)')
                self.window.Element('_ASR_').Update('       ... Listening ...')
                self.window.Refresh()

                audio = self.asr.listen(source, phrase_time_limit=3)

                self.window.Element('_ASR_').Update('')
                self.window.Refresh()

                try:
                    # This uses the default key
                    asr_utterance = self.asr.recognize_google(audio)
                    print("Google ASR: " + asr_utterance)

                except speech_rec.UnknownValueError:
                    print("Google ASR did not understand you")

                except speech_rec.RequestError as e:
                    print("Google ASR request error: {0}".format(e))

                self.user_output = asr_utterance

                if self.user_output_history:
                    self.user_output_history += '\n\n' + asr_utterance
                else:
                    self.user_output_history = asr_utterance

                self.window.Element('_USER_').Update(self.user_output_history)
                self.window.Refresh()

            self.sys_output = \
                sys_agent.continue_dialogue({'input': self.user_output})

            self.sys_output_history += '\n' + self.sys_output['output_raw']

            self.update_text(
                '_SYS_',
                self.sys_output_history,
                self.sys_output['output_raw'])

            if 'bye' in asr_utterance:
                self.GUI_dialogue_initialized = False

                self.sys_output_history += \
                    '\n==================================\n\n'

                self.window.Element('_SYS_').Update(self.sys_output_history)
                self.window.Refresh()

                sys_agent.end_dialogue()

    def update_GUI_human(self, sys_agent, human_role='user'):
        # In here only one of sys_agent and usr_agent should be instantiated.

        asr_utterance = ''

        if self.INTERACTION_MODE == 'speech':
            if not self.GUI_dialogue_initialized:
                self.GUI_dialogue_initialized = True

            # Listen for input from the microphone
            with speech_rec.Microphone() as source:
                print('(listening...)')
                audio = self.asr.listen(source, phrase_time_limit=3)

                try:
                    # This uses the default key
                    asr_utterance = self.asr.recognize_google(audio)
                    print("Google ASR: " + asr_utterance)

                except speech_rec.UnknownValueError:
                    print("Google ASR did not understand you")

                except speech_rec.RequestError as e:
                    print("Google ASR request error: {0}".format(e))

        if human_role == 'user':
            if self.INTERACTION_MODE == 'speech':
                self.user_output = asr_utterance

                if self.user_output_history:
                    self.user_output_history += '\n' + asr_utterance
                else:
                    self.user_output_history = asr_utterance

                self.window.Element('_USER_').Update(self.user_output_history)
                self.window.Refresh()

            self.sys_output = \
                sys_agent.continue_dialogue({'input': self.user_output})

            self.sys_output_history += '\n' + self.sys_output['output_raw']

            self.update_text(
                '_SYS_',
                self.sys_output_history,
                self.sys_output['output_raw'])

            if 'bye' in self.user_output:
                self.GUI_dialogue_initialized = False

                self.sys_output_history += \
                    '\n==================================\n\n'

                self.window.Element('_SYS_').Update(self.sys_output_history)
                self.window.Refresh()

                sys_agent.end_dialogue()

                self.sys_output = ''
                self.sys_output_dacts = []
                self.goal = None

    def run_multi_agent(self, settings, num_agents):
        raise NotImplementedError

    def update_GUI(self, sys_agent, usr_agent, text_sys_agent, text_usr_agent):
        raise NotImplementedError

    def arg_parse(self, args=None):
        """
        This function will parse the configuration file that was provided as a
        system argument into a dictionary.

        :return: a dictionary containing the parsed config file.
        """

        arg_vec = args if args else sys.argv

        # Parse arguments
        if len(arg_vec) < 3:
            print('WARNING: No configuration file.')

        # Initialize random seed
        random.seed(time.time())

        cfg_filename = arg_vec[2]
        if isinstance(cfg_filename, str):
            if os.path.isfile(cfg_filename):
                # Choose config parser
                if cfg_filename[-5:] == '.yaml':
                    with open(cfg_filename, 'r') as file:
                        cfg_parser = yaml.load(file, Loader=yaml.Loader)
                elif cfg_filename[-4:] == '.cfg':
                    cfg_parser = configparser.ConfigParser()
                    cfg_parser.read(cfg_filename)
                else:
                    raise ValueError('Unknown configuration file type: %s'
                                     % cfg_filename)
            else:
                raise FileNotFoundError('Configuration file %s not found'
                                        % cfg_filename)
        else:
            raise ValueError('Unacceptable value for configuration file name: '
                             '%s ' % cfg_filename)

        tests = 1
        dialogues = 10
        interaction_mode = 'simulation'
        num_agents = 1

        if cfg_parser:
            dialogues = int(cfg_parser['DIALOGUE']['num_dialogues'])

            if 'interaction_mode' in cfg_parser['GENERAL']:
                interaction_mode = cfg_parser['GENERAL']['interaction_mode']

                if 'agents' in cfg_parser['GENERAL']:
                    num_agents = int(cfg_parser['GENERAL']['agents'])

                elif interaction_mode == 'multi_agent':
                    print('WARNING! Multi-Agent interaction mode selected but '
                          'number of agents is undefined in config.')

            if 'initiative' in cfg_parser['DIALOGUE']:
                self.initiative = cfg_parser['DIALOGUE']['initiative']

        return {'cfg_parser': cfg_parser,
                'tests': tests,
                'dialogues': dialogues,
                'interaction_mode': interaction_mode,
                'num_agents': num_agents,
                'test_mode': False}

    def run_controller(self, args):
        """
        This function will create and run a controller. It iterates over the
        desired number of tests and prints some basic results.

        :param args: the parsed configuration file
        :return: nothing
        """

        # Extract arguments
        cfg_parser = args['cfg_parser']
        tests = args['tests']
        num_dialogues = args['dialogues']
        self.INTERACTION_MODE = args['interaction_mode']
        num_agents = args['num_agents']
        statistics = []

        for test in range(tests):
            # Run simulation
            print('\n\n=======================================')
            print('# Running {0} dialogues (test {1} of {2}) #'.format(
                num_dialogues,
                (test + 1),
                tests))
            print('=======================================\n')

            try:
                if self.INTERACTION_MODE == 'multi_agent':
                    statistics = self.run_multi_agent(
                        cfg_parser, num_agents)

                elif self.INTERACTION_MODE in ['text', 'speech']:
                    self.run_single_agent(
                        cfg_parser
                    )

                else:
                    raise ValueError('Unknown interaction mode: {0}. '
                                     'Please select "multi_agent" or "text"'
                                     ' for GUIController'
                                     .format(self.INTERACTION_MODE))

            except (ValueError, FileNotFoundError, TypeError, AttributeError) \
                    as err:
                print('\nPlato error! {0}\n'.format(err))
                return -1

        print(f'Results:\n{statistics}')
        return 0


def run(config):
    # Create a basic controller
    ctrl = SGUIController()

    if config:
        if os.path.isfile(config):
            # Pass the configuration file provided
            arguments = ctrl.arg_parse(['_', '-config', config])

        else:
            # Else look for the config file in the example folder
            import plato

            # __file__ points to __init__.py, which is 11 characters but we
            # want the root path only.
            plato_path = "/".join(plato.__file__.split("/")[:-1])[:-6] + '/'
            new_config_path = \
                plato_path + 'example/config/application/' + config

            if os.path.isfile(new_config_path):
                # Parse the example configuration file
                arguments = ctrl.arg_parse(
                    ['_', '-config', new_config_path])
            else:
                raise ValueError(f'Configuration file {config} not found!')

    else:
        # Get arguments from command line (sys.argv)
        arguments = ctrl.arg_parse()

    # Normal Plato execution
    ctrl.run_controller(arguments)
