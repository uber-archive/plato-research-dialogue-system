"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from plato.agent.component.conversational_module import ConversationalModule
from gtts import gTTS
from google.cloud import texttospeech

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

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
            "/Users/alexandrospapangelis/Projects/GCP_key.json"

        try:
            # Instantiates a client
            self.client = texttospeech.TextToSpeechClient()

        except:
            self.client = None

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
            # Set the text input to be synthesized
            synthesis_input = texttospeech.types.SynthesisInput(
                text=utterance)

            # Build the voice request, select the language code ("en-US")
            #  and the ssml voice gender ("neutral")
            voice = texttospeech.types.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

            # Select the type of audio file you want returned
            audio_config = texttospeech.types.AudioConfig(
                audio_encoding=texttospeech.enums.AudioEncoding.MP3)

            # Perform the text-to-speech request on the text input with the
            # selected voice parameters and audio file type
            response = self.client.synthesize_speech(synthesis_input, voice,
                                                     audio_config)

            # The response's audio_content is binary.
            with open('sys_output.mp3', 'wb') as out:
                # Write the response to the output file.
                out.write(response.audio_content)

            os.system('afplay sys_output.mp3')

        except Exception as e:
            try:
                print(
                    'WARNING: GCP encountered an error: {0}. '
                    'Falling back to gTTS.'.format(e)
                )

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
