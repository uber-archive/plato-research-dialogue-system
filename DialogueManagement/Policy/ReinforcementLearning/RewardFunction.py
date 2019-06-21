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

from abc import ABC, abstractmethod
from copy import deepcopy


class Reward(ABC):

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def calculate(self, state, action):
        pass


class SlotFillingReward(Reward):
    def __init__(self):
        self.goal = None

        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20

    def initialize(self, **kwargs):
        '''

        :param kwargs: turn penalty, failure penalty, and success reward
        :return: Nothing
        '''

        if 'turn_penalty' in kwargs:
            self.turn_penalty = kwargs['turn_penalty']
        if 'failure_penalty' in kwargs:
            self.failure_penalty = kwargs['failure_penalty']
        if 'success_reward' in kwargs:
            self.success_reward = kwargs['success_reward']

    def calculate(self, state, actions, goal=None, force_terminal=False, agent_role='system'):
        '''

        :param state:
        :param action:
        :param goal:
        :return:
        '''

        reward = self.turn_penalty
        dialogue_success = None
        task_success = None

        if goal is None:
            print('Warning: SlotFillingReward() called without a goal.')
            return 0, False, False

        # if actions[0].intent == 'bye' and goal.requests != goal.actual_requests:
        #     success = False
        #     reward += self.failure_penalty

        else:
            dialogue_success = False

            if state.is_terminal() or force_terminal:
                # Check that an offer has actually been made
                if state.system_made_offer:
                    dialogue_success = True

                    # if agent_role == 'system':
                    #     # Check that the item meets all constraints that were expressed by the user
                    #     for constr in goal.actual_constraints:
                    #         # Penalise items that have unknown / no values for known constraints
                    #         if constr not in state.item_in_focus or goal.actual_requests[constr].value != state.item_in_focus[constr]:
                    #             reward = self.failure_penalty
                    #             dialogue_success = False
                    #             break

                    # else:
                    # Check that the item offered meets the user's constraints
                    for constr in goal.constraints:
                        if goal.ground_truth:
                            # Multi-agent case
                            if goal.ground_truth[constr] != goal.constraints[constr].value and \
                                    goal.constraints[constr].value != 'dontcare':
                                reward += self.failure_penalty
                                dialogue_success = False
                                break

                        elif state.item_in_focus:
                            # Single-agent case
                            if state.item_in_focus[constr] != goal.constraints[constr].value and \
                                    goal.constraints[constr].value != 'dontcare':
                                reward += self.failure_penalty
                                dialogue_success = False
                                break

                    # Check that all requests have been addressed
                    if dialogue_success:
                        not_met = 0

                        if agent_role == 'system':
                            # Check that the system has responded to all requests (actually) made by the user
                            for req in goal.actual_requests:
                                if not goal.actual_requests[req].value:
                                    not_met += 1
                                    # reward += self.failure_penalty
                                    # dialogue_success = False
                                    # break

                        elif agent_role == 'user':
                            # Check that the user has provided all the requests in the goal
                            for req in goal.requests:
                                if not goal.requests[req].value:
                                    not_met += 1
                                    # reward += self.failure_penalty
                                    # dialogue_success = False
                                    # break

                        if not_met > 0: #and goal.actual_requests:
                            # reward += self.success_reward * 0.5 * (1 + float((len(goal.actual_requests) - not_met) / len(goal.actual_requests)))
                            # reward += self.success_reward * float((len(goal.actual_requests) - not_met) / len(goal.actual_requests))

                            # if goal.actual_requests:
                            #     ratio = float((len(goal.actual_requests) - not_met) / len(goal.actual_requests))
                            # else:
                            #     ratio = 1
                            # reward += self.success_reward * ratio + self.failure_penalty * (1-ratio)

                            reward += self.failure_penalty
                            dialogue_success = False
                        else:
                            reward = self.success_reward

                    # if dialogue_success:
                    #     reward += self.success_reward

                else:
                    reward += self.failure_penalty
                    dialogue_success = False

        # Liu & Lane ASRU 2017 Definition of task success
        task_success = None
        if agent_role == 'system':
            task_success = True
            # We don't care for slots that are not in the goal constraints
            for slot in goal.constraints:
                # If the system proactively informs about a slot the user has not yet put a constraint upon,
                # the user's DState is updated accordingly and the user would not need to put that constraint.
                if goal.ground_truth:
                    if goal.ground_truth[slot] != goal.constraints[slot].value and goal.constraints[slot].value != 'dontcare':
                        task_success = False
                        break

                # Fall back to the noisier signal, that is the slots filled.
                elif state.slots_filled[slot] != goal.constraints[slot].value and goal.constraints[slot].value != 'dontcare':
                    task_success = False
                    break

            for req in goal.requests:
                if not goal.requests[req].value:
                    task_success = False
                    break

        return reward, dialogue_success, task_success


class SlotFillingGoalAdvancementReward(Reward):
    def __init__(self):
        self.prev_state = None
        self.prev_goal = None

        self.failure_penalty = -1
        self.success_reward = 1

    def initialize(self, **kwargs):
        '''

        :param kwargs: turn penalty, failure penalty, and success reward
        :return: Nothing
        '''

        if 'failure_penalty' in kwargs:
            self.failure_penalty = kwargs['failure_penalty']
        if 'success_reward' in kwargs:
            self.success_reward = kwargs['success_reward']
        if 'state' in kwargs:
            self.prev_state = deepcopy(kwargs['state'])
        else:
            self.prev_state = None
        if 'goal' in kwargs:
            self.prev_goal = deepcopy(kwargs['goal'])
        else:
            self.prev_goal = None

    def calculate(self, state, actions, goal=None, force_terminal=False, agent_role='system'):
        '''

        :param state:
        :param actions:
        :param goal:
        :return:
        '''

        if goal is None:
            print('Warning: SlotFillingGoalAdvancementReward() called without a goal.')
            return -1, False, False

        elif self.prev_state is None or self.prev_goal is None:
            reward = 1

        # Check if the goal has been advanced
        else:
            # If the new state has more slots filled than the old one
            if sum([1 if self.prev_state.slots_filled[s] else 0 for s in self.prev_state.slots_filled]) < \
                    sum([1 if state.slots_filled[s] else 0 for s in state.slots_filled]):
                reward = 1

            # Or if the new state has more requests filled than the old one
            elif sum([1 if self.prev_goal.actual_requests[r] else 0 for r in self.prev_goal.actual_requests]) < \
                    sum([1 if goal.actual_requests[r] else 0 for r in goal.actual_requests]):
                reward = 1

            # Or if the system made a request for an unfilled slot?
        
            else:
                reward = -1

        success = False
        if state.is_terminal() or force_terminal:
            # Check that an offer has actually been made
            if state.system_made_offer:
                success = True

                # Check that the item offered meets the user's constraints
                for constr in goal.constraints:
                    if goal.ground_truth:
                        # Multi-agent case
                        if goal.ground_truth[constr] != goal.constraints[constr].value:
                            success = False
                            break

                    elif state.item_in_focus:
                        # Single-agent case
                        if state.item_in_focus[constr] != goal.constraints[constr].value:
                            success = False
                            break

        self.prev_state = deepcopy(state)
        self.prev_goal = deepcopy(goal)

        # Liu & Lane ASRU 2017 Definition of task success
        task_success = None
        if agent_role == 'system':
            task_success = True
            # We don't care for slots that are not in the goal constraints
            for slot in goal.constraints:
                # If the system proactively informs about a slot the user has not yet put a constraint upon,
                # the user's DState is updated accordingly and the user would not need to put that constraint.
                if goal.ground_truth:
                    if goal.ground_truth[slot] != goal.constraints[slot].value and goal.constraints[slot].value != 'dontcare':
                        task_success = False

                # Fall back to the noisier signal, that is the slots filled.
                elif state.slots_filled[slot] != goal.constraints[slot].value and goal.constraints[slot].value != 'dontcare':
                    task_success = False

            for req in goal.requests:
                if not goal.requests[req].value:
                    task_success = False

        return reward, success, task_success


class SlotFillingMultiAgentReward(Reward):
    def __init__(self):
        self.goal = None

        self.turn_penalty = -0.25
        self.failure_penalty = -1
        self.success_reward = 25

    def initialize(self, **kwargs):
        '''

        :param kwargs: turn penalty, failure penalty, and success reward
        :return: Nothing
        '''

        if 'turn_penalty' in kwargs:
            self.turn_penalty = kwargs['turn_penalty']
        if 'failure_penalty' in kwargs:
            self.failure_penalty = kwargs['failure_penalty']
        if 'success_reward' in kwargs:
            self.success_reward = kwargs['success_reward']

    def calculate(self, state, actions, goal=None, force_terminal=False, agent_role=None):
        '''

        :param state:
        :param action:
        :param goal:
        :return:
        '''

        if agent_role is None:
            print('Warning: SlotFillingMultiAgentReward() called without an agent role.')
            return 0, False

        if goal is None:
            print('Warning: SlotFillingMultiAgentReward() called without a goal.')
            return 0, False

        reward = self.turn_penalty
        success = False

        if state.is_terminal() or force_terminal:
            # If the system terminated the dialogue and the user had things yet to request, mark as failed
            if agent_role == 'system' and actions[0].intent == 'bye' and goal.requests != goal.actual_requests:
                reward += self.failure_penalty
                success = False

            else:
                # Check that an offer has actually been made
                if state.system_made_offer:
                    success = True

                    # Check that the item offered meets the user's constraints
                    for constr in goal.constraints:
                        if goal.ground_truth:
                            if goal.ground_truth[constr] != goal.constraints[constr].value:
                                reward += self.failure_penalty
                                success = False
                                break

                        elif state.item_in_focus:
                            if state.item_in_focus[constr] != goal.constraints[constr].value:
                                reward += self.failure_penalty
                                success = False
                                break

                    # Check that all requests have been addressed
                    if success:
                        not_met = 0

                        for req in goal.actual_requests:
                            if not goal.actual_requests[req].value:
                                # not_met += 1
                                reward += self.failure_penalty
                                success = False
                                break
                        #
                        # if not_met > 0:
                        #     reward += self.success_reward * 0.5 * (1 + float((len(goal.actual_requests) - not_met) / len(goal.actual_requests)))
                        # else:
                        #     reward += self.success_reward

                    if success:
                        reward += self.success_reward

                else:
                    reward += self.failure_penalty
                    success = False

        return reward, success