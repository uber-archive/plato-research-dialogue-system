"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from Dialogue.Action import Operator
import random

"""
The ErrorModel simulates ASR or NLU errors when the Simulated Usr emits 
actions.
"""


# Class modeling semantic and other errors
class ErrorModel:
    def __init__(self, ontology, database, slot_confuse_prob, op_confuse_prob,
                 value_confuse_prob):
        """
        Initialize the internal structures of the Error Model

        :param ontology: the domain Domain
        :param database: the domain Database
        :param slot_confuse_prob: a list of probabilities by which slots will
                                  be confused
        :param op_confuse_prob: a list of probabilities by which operators will
                                be confused
        :param value_confuse_prob: a list of probabilities by which values will
                                   be confused
        """
        self.slot_confuse_prob = slot_confuse_prob
        self.op_confuse_prob = op_confuse_prob
        self.value_confuse_prob = value_confuse_prob

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = None
        if isinstance(database, DataBase):
            self.database = database
        else:
            raise ValueError('Unacceptable database type %s ' % database)

    def semantic_noise(self, act):
        """
        Simulates semantic noise. It receives an act and introduces errors
        given the Error Model's probabilities.

        :param act: the act to be confused
        :return: the confused act
        """
        if act.intent == 'inform':
            for item in act.params:
                if item.slot in self.ontology.ontology['informable']:
                    if random.random() < self.slot_confuse_prob and item.slot:
                        item.slot = \
                            random.choice(
                                list(
                                    self.ontology
                                        .ontology['informable'].keys()))
                        item.value = \
                            random.choice(
                                self.ontology
                                    .ontology['informable'][item.slot])

                    if random.random() < self.op_confuse_prob:
                        item.op = random.choice(Operator)

                    if random.random() < self.value_confuse_prob:
                        item.value = \
                            random.choice(
                                self.ontology
                                    .ontology['informable'][item.slot])
                else:
                    # We're not raising errors here because the simulated user
                    # may be following a statistical policy
                    print('Warning! ErrorModel: Slot {0} not in informable '
                          'slots!'.format(item.slot))

        elif act.intent == 'request':
            for item in act.params:
                if random.random() < self.slot_confuse_prob:
                    if item.slot in self.ontology.ontology['requestable']:
                        item.slot = \
                            random.choice(
                                self.ontology.ontology['requestable'])
                        item.value = ''
                    else:
                        # We're not raising an error here because the simulated
                        # user may be following a statistical policy
                        print('Warning! ErrorModel: Slot {0} not in '
                              'requestable slots!'.format(item.slot))

        return act
