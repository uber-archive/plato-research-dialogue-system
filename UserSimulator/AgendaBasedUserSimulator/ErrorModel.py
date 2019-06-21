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

from Ontology.Ontology import Ontology
from Ontology.DataBase import DataBase
from Dialogue.Action import Operator
import random


# Class modeling semantic and other errors
class ErrorModel:
    def __init__(self, ontology, database, slot_confuse_prob, op_confuse_prob, value_confuse_prob):
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

    def semanticNoise(self, act):
        # TODO: With this implementation there is still a small probability by which we can select the exact same slot, op, value

        if act.intent == 'inform':
            for item in act.params:
                if item.slot in self.ontology.ontology['informable']:
                    if random.random() < self.slot_confuse_prob and item.slot:
                        item.slot = random.choice(list(self.ontology.ontology['informable'].keys()))
                        item.value = random.choice(self.ontology.ontology['informable'][item.slot])

                    if random.random() < self.op_confuse_prob:
                        item.op = random.choice(list(Operator))

                    if random.random() < self.value_confuse_prob:
                        item.value = random.choice(self.ontology.ontology['informable'][item.slot])
                else:
                    # We're not raising errors here because the simulated user may be following a statistical policy
                    print('Warning! ErrorModel: Slot {0} not in informable slots!'.format(item.slot))

        elif act.intent == 'request':
            for item in act.params:
                if random.random() < self.slot_confuse_prob:
                    if item.slot in self.ontology.ontology['requestable']:
                        item.slot = random.choice(self.ontology.ontology['requestable'])
                        item.value = ''
                    else:
                        # We're not raising errors here because the simulated user may be following a statistical policy
                        print('Warning! ErrorModel: Slot {0} not in requestable slots!'.format(item.slot))

        return act
