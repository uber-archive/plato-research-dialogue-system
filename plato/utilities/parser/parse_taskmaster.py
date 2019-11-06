"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from plato.utilities.parser.data_parser import DataParser

import csv
import json
import re

"""
parser is an abstract parent class for data parsers and defines the 
interface that should be used.
"""


class Parser(DataParser):

    def __init__(self, args):
        super(Parser, self).__init__()

        self.data_path = None
        self.simple_annotation = {}

    def initialize(self, **kwargs):
        """
        Initialize the internal structures of the data parser.

        :param args:
        :return:
        """

        if 'data_path' in kwargs:
            self.data_path = kwargs['data_path']

        # Map slots to simpler names
        self.simple_annotation = {
            "pizza_ordering.name.store": "restaurant",
            "pizza_ordering.name.pizza": "name",
            "pizza_ordering.size.pizza": "size",
            "pizza_ordering.type.topping": "topping",
            "pizza_ordering.type.crust": "crust",
            "pizza_ordering.preference": "preference",
            "pizza_ordering.location.store": "location",

            "coffee_ordering.location.store": "location",
            "coffee_ordering.name.drink": "name",
            "coffee_ordering.size.drink": "size",
            "coffee_ordering.num.drink": "number",
            "coffee_ordering.type.milk": "milk",
            "coffee_ordering.preference": "preference"
        }

    def parse_data(self):
        """
        Parse the data and generate Plato dialogue Experience logs.

        :return:
        """

        if not self.data_path:
            raise ValueError('Parse_Taskmaster: No data_path provided')

        with open(self.data_path, "r") as data_file, \
                open('Data/taskmaster_ordering.csv',
                     'w') as parsed_data_file, \
                open('Data/taskmaster_ordering_goals.json',
                     'w') as goals_file, \
                open('Data/seq2seq.csv', 'w') as seq2seq_file, \
                open('Data/uLabel.tsv', 'w') as uLabel_file:
                # open('Data/gpt_train.txt', 'w') as gpt_file:
            data = json.load(data_file)
            csv_writer = csv.writer(parsed_data_file, delimiter=',')
            csv_s2s_writer = csv.writer(seq2seq_file, delimiter=',')
            csv_ulabel_writer = csv.writer(uLabel_file, delimiter='\t')

            # Write headers
            csv_writer.writerow(['dialogue_id', 'domain', 'turn_no', 'speaker',
                                 'slot', 'utterance'])

            csv_s2s_writer.writerow(['context', 'output'])

            csv_ulabel_writer.writerow(['dialogID', 'utteranceIndex',
                                        'role', 'text', 'ignore'])

            goals = {}
            uLabel_dialogue_num = 0

            for dialogue in data:
                dialogue_id = dialogue['conversation_id']
                domain = dialogue['instruction_id']
                goal = {}

                if 'pizza-ordering' not in domain:
                    continue

                # To avoid bug in data where there are duplicate utterances
                prev_utterance = ''
                uLabel_turn_id = 0

                for turn in dialogue['utterances']:
                    slot = ''
                    if 'segments' in turn:
                        # TODO: Get all segments and all annotations not just the first!
                        slot = turn['segments'][0]['annotations'][0]['name']

                        for segment in turn['segments']:
                            for annotation in segment['annotations']:
                                # Ignore .accept / .reject slots as these are
                                # confirmations
                                if any(s in annotation['name']
                                       for s in ['.accept', '.reject']):
                                    continue

                                goal[self.simple_annotation[
                                    annotation['name']]
                                ] = segment['text']

                    if prev_utterance != turn['text']:
                        csv_writer.writerow([dialogue_id,
                                             domain,
                                             turn['index'],
                                             turn['speaker'],
                                             slot,
                                             turn['text'].replace(',', ' ')])

                        if uLabel_dialogue_num < 100:
                            # For uLabel, we need to split the sentences
                            sentences = \
                                re.split('[?.!]', turn['text'].
                                         replace(',', ' '))

                            for sentence in sentences:
                                sentence = sentence.rstrip()

                                if sentence and sentence != '(deleted)':
                                    csv_ulabel_writer.writerow([uLabel_dialogue_num,
                                                                uLabel_turn_id,
                                                                turn['speaker'],
                                                                sentence,
                                                                False])
                            uLabel_turn_id += 1

                        prev_utterance = turn['text']

                goals[dialogue_id] = goal
                uLabel_dialogue_num += 1

                t = 0
                prev_utterance = ''
                while t < len(dialogue['utterances'])-1:
                    assistant_text = ''
                    user_text = ''

                    while t < len(dialogue['utterances']) and \
                            dialogue['utterances'][t]['speaker'] == 'ASSISTANT':
                        assistant_text += dialogue['utterances'][t]['text'].replace(',', ' ') + ' '
                        t += 1

                    while t < len(dialogue['utterances']) and \
                            dialogue['utterances'][t]['speaker'] == 'USER':
                        user_text += dialogue['utterances'][t]['text'].replace(',', ' ') + ' '
                        t += 1

                    context = '[GOAL] ' + self.flatten_goal(goal) + \
                              ' [REST] ' + assistant_text + ' [SEP] '

                    if prev_utterance != user_text:
                        csv_s2s_writer.writerow(
                            [context,
                             user_text]
                        )

                        prev_utterance = user_text

                    # context += ' [USER] ' + user_turn['text']

                    # t += 2

                    # assistant_turn = dialogue['utterances'][t]
                    # user_turn = dialogue['utterances'][t+1]
                    #
                    # if assistant_turn['speaker'] != 'ASSISTANT' or\
                    #         user_turn['speaker'] != 'USER':
                    #     t += 1
                    #     continue
                    #
                    # gpt_file.write('[BOS] ' + assistant_turn['speaker'] + ' ' +
                    #                assistant_turn['text'] + ' [SEP] ' +
                    #                user_turn['speaker'] + ' ' +
                    #                user_turn['text'] +
                    #                ' [EOS]\n')
                    #
                    # context = '[GOAL] ' + self.flatten_goal(goal) + \
                    #           ' [REST] ' + assistant_turn['text']
                    # # context += ' [REST] ' + assistant_turn['text']
                    #
                    # csv_s2s_writer.writerow(
                    #     [context + ' [SEP] ',
                    #      user_turn['text']]
                    # )
                    #
                    # # context += ' [USER] ' + user_turn['text']
                    #
                    # t += 2

            json.dump(goals, goals_file)

        print('Taskmaster parser Reading done.')

    def flatten_goal(self, goal):
        return ' '.join([f'{k}: {v} ' for k, v in goal.items()])

    def save(self, path):
        """
        Save the experience

        :param path: path to save the experience to
        :return:
        """

        pass
