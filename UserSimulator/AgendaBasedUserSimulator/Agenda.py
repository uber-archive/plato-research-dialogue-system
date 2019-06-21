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

from Dialogue.Action import DialogueAct, DialogueActItem, Operator


class Agenda:
    def __init__(self):
        self.agenda = []
        self.goal = None

    def initialize(self, goal, us_has_initiative=False):
        self.goal = goal

        self.clear()

        # Generate candidate actions
        dacts = []

        # If there are subgoals
        # Iterate from last to first because the acts will be popped in reverse order.
        for i in range(len(self.goal.subgoals)-1, -1, -1):
            sg = self.goal.subgoals[i]
            prev_sg = None

            # if i > 0:
            #     prev_sg = self.goal.subgoals[i-1]

            # Acknowledge completion of subgoal
            dacts.append(DialogueAct('ack_subgoal', []))

            # if not prev_sg:
            for constr in sg.constraints.values():
                dacts.append(DialogueAct('inform', [constr]))
            # else:
            #     for constr in sg.constraints.values():
            #         dacts.append(DialogueAct('inform', [constr]))
            #
            #     # Update any constraints the previous subgoal enforced. Put these after the new constraints so they
            #     # end up before the new constraints in the agenda (remember, the agenda is a stack).
            #     for constr in prev_sg.constraints.values():
            #         # Explicitly remove constraint (otherwise it is implicitly updated)
            #         if constr.slot not in sg.constraints:
            #             dacts.append(DialogueAct('inform', [DialogueActItem(constr.slot, Operator.EQ, 'dontcare')]))

        for req in goal.requests.values():
            dacts.append((DialogueAct('request', [req])))

        for constr in goal.constraints.values():
            dacts.append(DialogueAct('inform', [constr]))

        # Push actions into the agenda
        self.push(DialogueAct('bye', []))

        for da in dacts:
            self.push(da, force=True)

        if us_has_initiative:
            self.push(DialogueAct('hello', []))

    def push(self, act, force=False):
        '''
        Pushes a dialogue act into the agenda.

        :param act: dialogue act to be appended
        :param force: does not remove act if it already is in the agenda, potentially resulting in duplicates
        :return: Nothing
        '''

        if act is not None and isinstance(act, DialogueAct):
            # This is unnecessary only if the act is already on the top of the agenda.
            if act in self.agenda and not force:
                self.remove(act)

            self.agenda.append(act)
        else:
            # TODO: RAISE ERROR
            print("Error! Cannot add item %s in the agenda." % act)

    def pop(self):
        '''
        Pop top item from the agenda.

        :return: top item
        '''

        if self.agenda:
            return self.agenda.pop()
        else:
            # TODO: LOG WARNING INSTEAD OF PRINTING
            print('Warning! Attempted to pop an empty agenda.')
            return None

    def peek(self):
        '''
        Peek top item from the agenda.

        :return: Nothing
        '''

        if self.agenda:
            return self.agenda[-1]
        else:
            # TODO: LOG WARNING INSTEAD OF PRINTING
            print('Warning! Attempted to peek an empty agenda.')
            return None

    def remove(self, act):
        '''
        Remove a specific dialogue act from the agenda

        :param act: the dialogue act to be removed
        :return: Nothing
        '''

        if act in self.agenda:
            self.agenda.remove(act)

    def clear(self):
        '''
        Clear all items from the agenda.

        :return: Nothing
        '''

        self.agenda = []

    def consistency_check(self):
        '''
        Perform some basic checks to ensure that items in the agenda are consistent - i.e. not duplicate, not
        contradicting with current goal, etc.

        :return: Nothing
        '''

        # Remove all requests for slots that are filled in the goal
        if self.goal:
            for slot in self.goal.actual_requests:
                if self.goal.actual_requests[slot].value:
                    self.remove(DialogueAct('request', [DialogueActItem(slot, Operator.EQ, '')]))
        else:
            print('Warning! Agenda consistency check called without goal. Did you forget to initialize?')

    def size(self):
        return len(self.agenda)
