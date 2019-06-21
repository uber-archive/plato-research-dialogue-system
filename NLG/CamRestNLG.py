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

__author__ = "Yi-Chia Wang"

from NLG.LudwigNLG import LudwigNLG
import re
import random


class CamRestNLG(LudwigNLG):
    def __init__(self, args):
        super(CamRestNLG, self).__init__(args)

    def initialize(self, args):
        pass

    def generate_output(self, args=None):
        if not args:
            print('WARNING! CamRestNLG called without arguments!')
            return ''

        if 'args' in args:
            dacts = args['args']
        elif 'dacts' not in args:
            print('WARNING! CamRestNLG called without dacts!')
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

        # print(dacts_str_tmp.strip())

        if not system and last_sys_utterance and 'welcome' not in last_sys_utterance:
            dacts_str = last_sys_utterance + ' ' + dacts_str

        # print(f'CamRestNLG Input: {dacts_str}')

        # Apply NLG model
        result = super(CamRestNLG, self).generate_output({'dacts': dacts_str, 'system': system})

        sys_text = ' '.join([x for x in result['nlg_output_predictions'][0]])
        sys_text = sys_text.replace(' <PAD>', '')

        # Replace template slots with values
        for key, value in slot2value.items():
            # Add some variability to 'dontcare'
            if value == 'dontcare':
                value = random.choice(['dont care', 'any', 'i dont care', 'i do not care'])

            sys_text = sys_text.replace(key, value)

        sys_text = sys_text.strip()

        return sys_text

    def train(self, data):
        pass

    def save(self, path=None):
        pass

    def load(self, path):
        pass
