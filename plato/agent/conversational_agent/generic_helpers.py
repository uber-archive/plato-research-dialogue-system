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

from plato.agent.component.conversational_module import ConversationalModule
from gtts import gTTS

import os


class GenericTextInputHelper(ConversationalModule):
    """
    This class is a helper for listening to text input for Generic agents.
    """

    def __init__(self, args):
        super(GenericTextInputHelper, self).__init__()

    def initialize(self, args):
        pass

    def receive_input(self, args):
        pass

    def generate_output(self, args=None):

        # Listen for input
        utterance = input('USER > ')

        return utterance

    def train(self, dialogue_episodes):
        pass

    def load(self, path):
        pass

    def save(self):
        pass


class GenericSpeechSynthesiserHelper(ConversationalModule):
    """
    This class is a helper for listening to text input for Generic agents.
    """

    def __init__(self, args):
        super(GenericSpeechSynthesiserHelper, self).__init__()

    def initialize(self, args):
        pass

    def receive_input(self, args):
        pass

    def generate_output(self, args=None):

        utterance = ''
        if 'utterance' in args:
            utterance = args['utterance']

        elif 'args' in args and isinstance(args['args'], str):
            utterance = args['args']

        # Synthesise speech
        try:
            tts = gTTS(utterance)
            tts.save('sys_output.mp3')
            os.system('afplay sys_output.mp3')

        except Exception as e:
            print(
                'WARNING: gTTS encountered an error: {0}. '
                'Falling back to System TTS.'.format(e)
            )
            os.system('say ' + utterance)

        return utterance

    def train(self, dialogue_episodes):
        pass

    def load(self, path):
        pass

    def save(self):
        pass
