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

__author__ = "Yi-Chia Wang"

from plato.agent.component.nlg.ludwig_nlg import LudwigNLG
import re
import random
import pandas as pd

"""
camrest_nlg is an implementation of nlg for the Cambridge Restaurants domain, 
using a Ludwig model.
"""


class CamRestNLG(LudwigNLG):
    def __init__(self, args):
        """
        Load the Ludwig model and parse the arguments.

        :param args:
        """
        super(CamRestNLG, self).__init__(args)

        self.TRAIN_ONLINE = False
        if 'train_online' in args:
            self.TRAIN_ONLINE = bool(args['train_online'])

    def initialize(self, args):
        """
        Nothing to do here

        :param args:
        :return:
        """
        pass

    def generate_output(self, args=None):
        """
        Generate text output by querying the Ludwig model with the dialogue
        acts in the arguments.

        :param args: a dictionary of arguments that contain the dialogue acts
        :return: the output utterance
        """
        if not args:
            print('WARNING! camrest_nlg called without arguments!')
            return ''

        if 'args' in args:
            dacts = args['args']

        elif 'dacts' not in args:
            print('WARNING! camrest_nlg called without dacts!')
            return ''

        else:
            dacts = args['dacts']

        system = True
        if 'system' in args:
            system = bool(args['system'])

        last_sys_utterance = None
        if 'last_sys_utterance' in args:
            last_sys_utterance = args['last_sys_utterance']

        dacts_str = ''
        dacts_str_tmp = ''
        slot2value = {}

        # Convert dacts to templates
        for dact in dacts:
            dacts_str_tmp += ' ' + str(dact)
            dacts_str += 'act_' + dact.intent + ' '

            for param in dact.params:

                if param.value:
                    dacts_str += '<' + param.slot + '> '
                    slot2value['<' + param.slot + '>'] = param.value
                elif param.slot:
                    dacts_str += 'slot_' + param.slot + ' '

        dacts_str = re.sub("\s+", " ", dacts_str.strip())

        # 'act_inform <name>' is not in the dstc2
        if dacts_str == 'act_inform <name>':
            dacts_str = 'act_offer <name>'

        if not system and last_sys_utterance and \
                'welcome' not in last_sys_utterance:
            dacts_str = last_sys_utterance + ' ' + dacts_str

        # Apply nlg model
        result = \
            super(CamRestNLG, self).generate_output(
                {'dacts': dacts_str, 'system': system})

        sys_text = ' '.join([x for x in result['nlg_output_predictions'][0]])
        sys_text = sys_text.replace(' <PAD>', '')

        # Replace template slots with values
        for key, value in slot2value.items():
            # Add some variability to 'dontcare'
            if value == 'dontcare':
                value = random.choice(['dont care', 'any', 'i dont care',
                                       'i do not care'])

            sys_text = sys_text.replace(key, value)

        sys_text = sys_text.strip()

        return sys_text

    def train(self, data):
        """
        Pass to train online.

        :param data: dialogue experience
        :return: nothing
        """
        if self.TRAIN_ONLINE:
            self.train_online(data)

    def train_online(self, data):
        """
        Train the model.

        :param data: dialogue experience
        :return: nothing
        """
        self.model.train_online(pd.DataFrame(data={'transcript': [data]}))

    def save(self, path=None):
        """
        Saves the Ludwig model.

        :param path: path to save the model to
        :return:
        """
        super(CamRestNLG, self).save(path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        super(CamRestNLG, self).load(model_path)
