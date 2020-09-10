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

from enum import Enum

"""
The Action class models actions that a Conversational Agent or Simulated usr 
can take. It defines the interface that any other act (e.g. dialogue act, 
multi-modal act, etc.) should follow.
"""


class Action:
    def __init__(self):
        self.name = None
        self.funcName = None    # Function name to be called, if applicable?
        self.params = {}        # dialogue Act Items (slot - operator - value)


"""
Summary Action is a simple class to represent actions in Summary Space. 
"""


class SummaryAction(Enum):
    INFORM_X = 1
    INFORM_XY = 2
    AFFIRM = 3
    AFFIRM_X = 4
    CONFIRM = 5
    CONFIRM_X = 6
    NEGATE = 7
    NEGATE_X = 8
    REQUEST_X = 9
    NOTHING = 10


""" 
The DialogueAct models dialogue acts, each of which has an intent and a list 
of parameters.
"""


class DialogueAct(Action):
    """
    Represents a dialogue act, which as a type (e.g. inform, request, etc.)
    and a list of DialogueActItem parameters, which are triplets of
    <slot, operator, value>.
    """
    def __init__(self, intent='', params=None):
        super(DialogueAct, self).__init__()

        self.name = 'dialogue_act'
        self.intent = ''
        if isinstance(intent, str) and intent is not '':
            self.intent = intent
        else:
            raise ValueError('Unacceptable dialogue act type: %s ' % intent)

        self.params = params
        if self.params is None:
            self.params = []

    def __eq__(self, other):
        """
        Equality operator.

        :param other: the dialogue Act to compare against
        :return: True of False
        """

        # TODO: Make the check more efficient
        return self.funcName == other.funcName and \
            self.intent == other.intent and \
            self.name == other.name and \
            [s for s in self.params if s not in other.params] == []

    def __str__(self):
        """
        Pretty print dialogue Act.

        :return: string representation of the dialogue Act
        """

        if self.intent:
            return self.intent + \
                   '(' + \
                   ''.join([str(param)+', ' for param in self.params]) + ')'
        else:
            return 'None (DialogueAct)'

    def add_item(self, item):
        """
        Appends a dialogue act item to params, if it does not already exist.

        :param item: a dialogue act item to be appended to params
        :return: nothing
        """

        if item not in self.params:
            self.params.append(item)


"""
The DialogueActItem models a parameter of a DialogueAct. It is essentially a 
triplet of (slot, operator, value).
"""


class DialogueActItem:
    def __init__(self, slot, op, value):
        """
        Initialize a dialogue Act Item (slot - operator - value)

        :param slot: a string, representing the slot
        :param op: an Operator
        :param value: the value of the slot
        """
        if isinstance(slot, str):
            self.slot = slot
        else:
            raise ValueError('Unacceptable slot type: %s ' % slot)

        if op in Operator:
            self.op = op
        else:
            raise ValueError('Unacceptable operator: %s ' % op)

        # TODO: Check for all acceptable value types here?
        self.value = value

    def __eq__(self, other):
        """
        Equality operator

        :param other: the dialogue Act Item to compare against
        :return: True or False
        """

        # TODO: Will need some kind of constraint satisfaction (with tolerance)
        # to efficiently handle all operators
        return self.slot == other.slot and self.op == other.op and \
            self.value == other.value

    def __str__(self):
        """
        Pretty print dialogue Act Item.

        :return: string
        """

        opr = 'UNK'
        if self.op == Operator.EQ:
            opr = '='
        elif self.op == Operator.NE:
            opr = '!='
        elif self.op == Operator.LT:
            opr = '<'
        elif self.op == Operator.LE:
            opr = '<='
        elif self.op == Operator.GT:
            opr = '>'
        elif self.op == Operator.GE:
            opr = '>='
        elif self.op == Operator.AND:
            opr = 'AND'
        elif self.op == Operator.OR:
            opr = 'OR'
        elif self.op == Operator.NOT:
            opr = 'NOT'
        elif self.op == Operator.IN:
            opr = 'IN'

        result = self.slot

        if self.value:
            result += ' ' + opr + ' ' + self.value

        return result


"""
The Expression class models complex expressions and defines how to compute 
them.
"""


class Expression:
    
    # An Expression will allow us dialogue acts of the form:
    # inform( 50 < price < 225, food: chinese or italian, ...)
    def __init__(self):
        """
        Not implemented.
        """
        pass


"""
The Operator class defines acceptable operators.
"""


class Operator(Enum):
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    AND = 7
    OR = 8
    NOT = 9
    IN = 10

    def __str__(self):
        """
        Print the Operator

        :return: a string representation of the Operator
        """
        return f"{self.name}"


# Represents an event of the (simulated) user tapping onto something in the
# screen.
class TapAct(Action):
    def __init__(self):
        """
        Example, not implemented.
        """
        super(TapAct, self).__init__()
