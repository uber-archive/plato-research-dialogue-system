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

from NLG.NLG import NLG

import random


class DummyNLG(NLG):
    def __init__(self, args):
        super(DummyNLG, self).__init__(args)

    def initialize(self, args):
        pass

    def generate_output(self, args=None):
        if not args:
            print('WARNING! DummyNLG called without arguments!')
            return ''

        if 'dacts' not in args:
            print('WARNING! DummyNLG called without dacts!')
            return ''

        dacts = args['dacts']

        system = True
        if 'system' in args:
            system = bool(args['system'])

        response = ''

        for dact in dacts:
            if dact.intent == 'request':
                if dact.params and dact.params[0].slot:
                    if system:
                        response += 'Which ' + dact.params[0].slot + ' do you prefer?'
                    else:
                        response += 'What is the ' + dact.params[0].slot + '?'
                else:
                    response += 'Which one?'

            elif dact.intent in ['inform', 'offer']:
                for dact_item in dact.params:
                    if system:
                        if dact_item.slot == 'name' and dact_item.value == 'not found':
                            response += 'Sorry, I cannot find such an item. '

                        else:
                            if not dact_item.value:
                                response += 'its ' + dact_item.slot + ' is unknown, '

                            elif dact_item.slot == 'name' and len(dact.params) > 1:
                                response += dact_item.value + '  '

                            elif dact_item.slot in ['food', 'cuisine']:
                                response += 'is serving ' + dact_item.value + ' food, '

                            elif dact_item.slot == 'endorsement':
                                response += 'is ' + dact_item.value + ', '

                            else:
                                response += 'its ' + dact_item.slot + ' is ' + dact_item.value + ', '

                    else:
                        if dact.intent == 'offer':
                            if dact_item.value:
                                response += dact_item.slot + ' is ' + dact_item.value + ', '
                            else:
                                response += dact_item.slot + ' is unknown, '
                        else:
                            r = random.random()

                            if r < 0.33:
                                response += 'I prefer ' + dact_item.value + ' ' + dact_item.slot + ', '

                            elif r < 0.66:
                                response += 'um i want ' + dact_item.value + ' ' + dact_item.slot + ', '

                            else:
                                response += dact_item.value + ' ' + dact_item.slot + ' please, '

                if response:
                    # Trim trailing comma and space
                    response = response[:-2]

            elif dact.intent == 'bye':
                response += 'Thank you, goodbye'

            elif dact.intent == 'deny':
                response += 'No'

            elif dact.intent == 'negate':
                response += 'No '

                if dact.params and dact.params[0].slot and dact.params[0].value:
                    response += dact.params[0].slot + ' is not ' + dact.params[0].value

            elif dact.intent == 'ack':
                response += 'Ok'

            elif dact.intent == 'affirm':
                response += 'Yes '

                if dact.params and dact.params[0].slot and dact.params[0].value:
                    response += dact.params[0].slot + ' is ' + dact.params[0].value

            elif dact.intent == 'thankyou':
                response += 'Thank you'

            elif dact.intent == 'reqmore':
                response += 'Can you tell me more?'

            elif dact.intent == 'repeat':
                response += 'Can you please repeat?'

            elif dact.intent == 'restart':
                response += 'Can we start over?'

            elif dact.intent == 'expl-conf':
                response += 'Alright '

                if dact.params and dact.params[0].slot and dact.params[0].value:
                    response += dact.params[0].slot + ' is ' + dact.params[0].value

            elif dact.intent == 'select':
                response += 'Which one do you prefer '

                if dact.params and dact.params[0].slot:
                    response += 'for ' + dact.params[0].slot

            elif dact.intent == 'reqalts':
                response += 'Is there anything else?'

            elif dact.intent in ['confirm', 'confirm-domain']:
                response += 'So is '

                if dact.params and dact.params[0].slot and dact.params[0].value:
                    response += dact.params[0].slot + ' ' + dact.params[0].value

            # Plato Navigates

            elif dact.intent == 'navigate':
                response += 'Navigating to ' + dact.params[0].value

            elif dact.intent == 'traffic':
                response += 'Traffic '

                if len(dact.params) > 1 and dact.params[1].value:
                    response += 'to ' + dact.params[1].value + ' '

                response += 'is ' + dact.params[0].value

            elif dact.intent == 'surge':
                response += 'The surge '

                if len(dact.params) > 1 and dact.params[1].value:
                    response += 'at ' + dact.params[1].value + ' '

                response += dact.params[0].value

            elif dact.intent == 'earnings':
                response += 'Your earnings '

                if len(dact.params) > 1 and dact.params[1].value:
                    response += 'since ' + dact.params[1].value + ' '

                    if len(dact.params) > 2 and dact.params[2].value:
                        response += 'at ' + dact.params[2].value + ' '

                response += 'are ' + dact.params[0].value

            elif dact.intent == 'queue':
                response += 'The queue '

                if len(dact.params) > 1 and dact.params[1].value:
                    response += 'to ' + dact.params[1].value + ' '

                response += 'is ' + dact.params[0].value

            elif dact.intent == 'time_on_road':
                response += 'You have been on the road for ' + dact.params[0].value

            elif dact.intent == 'canthelp':
                response += 'Sorry, I cannot help you with that.'

            elif dact.intent == 'welcomemsg':
                response += 'Hello, how may I help you?'

            elif dact.intent == 'hello':
                response = 'Hi'

            elif dact.intent == 'welcome':
                response += random.choice(['Hey Alex, how can I help you today?',
                                           'Speak, human.'])

            elif dact.intent == 'na':
                response += '(no system response)'

            else:
                response += 'DummyNLG %s' % dact

            response += ' '

        response = response.replace('addr', 'address')
        response = response.replace('pricerange', 'price range')
        response = response.replace('postcode', 'post code')
        response = response.replace('dontcare', 'any')

        return response

    def train(self, data):
        pass

    def save(self, path=None):
        pass

    def load(self, path):
        pass

