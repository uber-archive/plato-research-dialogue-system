"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from abc import ABC, abstractmethod
from copy import deepcopy

"""
Reward is the parent abstract class for reward functions, primarily used for 
reinforcement learning. 
"""


class Reward(ABC):

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize internal parameters

        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def calculate(self, state, action):
        """
        Calculate the reward to be assigned for taking action from state.

        :param state: the current state
        :param action: the action taken from the current state
        :return: the calculated reward
        """
        pass


class SlotFillingReward(Reward):
    def __init__(self):
        """
        Set default values for turn penalty, success, and failure.
        """

        self.goal = None

        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20

    def initialize(self, **kwargs):
        """
        Initialize parameters for turn penalty, success, and failure.

        :param kwargs: turn penalty, failure penalty, and success reward
        :return: Nothing
        """

        if 'turn_penalty' in kwargs:
            self.turn_penalty = kwargs['turn_penalty']
        if 'failure_penalty' in kwargs:
            self.failure_penalty = kwargs['failure_penalty']
        if 'success_reward' in kwargs:
            self.success_reward = kwargs['success_reward']

    def calculate(self, state, actions, goal=None, force_terminal=False,
                  agent_role='system'):
        """
        Calculate the reward to be assigned for taking action from state.


        :param state: the current state
        :param actions: the action taken from the current state
        :param goal: the agent's goal, used to assess success
        :param force_terminal: force state to be terminal
        :param agent_role: the role of the agent
        :return: a number, representing the calculated reward
        """

        reward = self.turn_penalty

        if goal is None:
            print('Warning: SlotFillingReward() called without a goal.')
            return 0, False, False

        else:
            dialogue_success = False

            if state.is_terminal() or force_terminal:
                # Check that an offer has actually been made
                if state.system_made_offer:
                    dialogue_success = True

                    # Check that the item offered meets the user's constraints
                    for constr in goal.constraints:
                        if goal.ground_truth:
                            # Multi-agent case
                            if goal.ground_truth[constr] != \
                                    goal.constraints[constr].value and \
                                    goal.constraints[constr].value != \
                                    'dontcare':
                                reward += self.failure_penalty
                                dialogue_success = False
                                break

                        elif state.item_in_focus:
                            # Single-agent case
                            if state.item_in_focus[constr] != \
                                    goal.constraints[constr].value and \
                                    goal.constraints[constr].value != \
                                    'dontcare':
                                reward += self.failure_penalty
                                dialogue_success = False
                                break

                    # Check that all requests have been addressed
                    if dialogue_success:
                        not_met = 0

                        if agent_role == 'system':
                            # Check that the system has responded to all
                            # requests (actually) made by the user
                            for req in goal.actual_requests:
                                if not goal.actual_requests[req].value:
                                    not_met += 1

                        elif agent_role == 'user':
                            # Check that the user has provided all the
                            # requests in the goal
                            for req in goal.requests:
                                if not goal.requests[req].value:
                                    not_met += 1

                        if not_met > 0:
                            reward += self.failure_penalty
                            dialogue_success = False
                        else:
                            reward = self.success_reward

                else:
                    reward += self.failure_penalty
                    dialogue_success = False

        # Liu & Lane ASRU 2017 Definition of task success
        task_success = None
        if agent_role == 'system':
            task_success = True
            # We don't care for slots that are not in the goal constraints
            for slot in goal.constraints:
                # If the system proactively informs about a slot the user has
                # not yet put a constraint upon,
                # the user's DState is updated accordingly and the user would
                # not need to put that constraint.
                if goal.ground_truth:
                    if goal.ground_truth[slot] != \
                            goal.constraints[slot].value and \
                            goal.constraints[slot].value != 'dontcare':
                        task_success = False
                        break

                # Fall back to the noisier signal, that is the slots filled.
                elif state.slots_filled[slot] != \
                        goal.constraints[slot].value and \
                        goal.constraints[slot].value != 'dontcare':
                    task_success = False
                    break

            for req in goal.requests:
                if not goal.requests[req].value:
                    task_success = False
                    break

        return reward, dialogue_success, task_success


class SlotFillingGoalAdvancementReward(Reward):
    def __init__(self):
        """
        Initialize the internal structures.
        """
        self.prev_state = None
        self.prev_goal = None

        self.failure_penalty = -1
        self.success_reward = 1

    def initialize(self, **kwargs):
        """
        Initialize the failure penalty and success reward

        :param kwargs: dictionary containing failure penalty and success reward
        :return: Nothing
        """

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

    def calculate(self, state, actions, goal=None, force_terminal=False,
                  agent_role='system'):
        """
        Calculate the reward based on whether the action taken advanced the
        goal or not. For example, if the action resulted in filling one more
        slot the conversation advanced towards the goal.

        :param state: the current state
        :param actions: the action taken from the current state
        :param goal: the agent's goal, used to assess success
        :param force_terminal: force state to be terminal
        :param agent_role: the role of the agent
        :return: a number, representing the calculated reward
        """

        if goal is None:
            print('Warning: SlotFillingGoalAdvancementReward() called without '
                  'a goal.')
            return -1, False, False

        elif self.prev_state is None or self.prev_goal is None:
            reward = 1

        # Check if the goal has been advanced
        else:
            # If the new state has more slots filled than the old one
            if sum([1 if self.prev_state.slots_filled[s] else 0 for s in
                    self.prev_state.slots_filled]) < \
                    sum([1 if state.slots_filled[s] else 0 for s in
                         state.slots_filled]):
                reward = 1

            # Or if the new state has more requests filled than the old one
            elif sum([1 if self.prev_goal.actual_requests[r] else 0 for r in
                      self.prev_goal.actual_requests]) < \
                    sum([1 if goal.actual_requests[r] else 0 for r in
                         goal.actual_requests]):
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
                        if goal.ground_truth[constr] != \
                                goal.constraints[constr].value:
                            success = False
                            break

                    elif state.item_in_focus:
                        # Single-agent case
                        if state.item_in_focus[constr] != \
                                goal.constraints[constr].value:
                            success = False
                            break

        self.prev_state = deepcopy(state)
        self.prev_goal = deepcopy(goal)

        task_success = None
        if agent_role == 'system':
            task_success = True
            # We don't care for slots that are not in the goal constraints
            for slot in goal.constraints:
                # If the system proactively informs about a slot the user has
                # not yet put a constraint upon,
                # the user's DState is updated accordingly and the user would
                # not need to put that constraint.
                if goal.ground_truth:
                    if goal.ground_truth[slot] != \
                            goal.constraints[slot].value and \
                            goal.constraints[slot].value != 'dontcare':
                        task_success = False

                # Fall back to the noisier signal, that is the slots filled.
                elif state.slots_filled[slot] != \
                        goal.constraints[slot].value and \
                        goal.constraints[slot].value != 'dontcare':
                    task_success = False

            for req in goal.requests:
                if not goal.requests[req].value:
                    task_success = False

        return reward, success, task_success
