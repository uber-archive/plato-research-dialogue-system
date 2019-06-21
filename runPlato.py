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

from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent
from ConversationalAgent.ConversationalMultiAgent import ConversationalMultiAgent
from ConversationalAgent.ConversationalGenericAgent import ConversationalGenericAgent

import configparser
import yaml
import sys
import os.path
import time
import random


class Controller(object):
    def __init__(self):
        # These are needed to keep track of the dialogue turn by turn when the GUI is enabled
        self.GUI_dialogue_initialized = False
        self.GUI_system_turn = True

        self.sys_output = ''
        self.user_output = ''
        self.sys_output_dacts = []
        self.user_output_dacts = []
        self.goal = None

    # Run agent against user simulator
    def run(self, settings, num_dialogues):
        if 'GENERAL' in settings and 'generic' in settings['GENERAL'] and settings['GENERAL']['generic']:
            ca = ConversationalGenericAgent(settings, 0)
        else:
            ca = ConversationalSingleAgent(settings)

        ca.initialize()

        for dialogue in range(num_dialogues):
            print('\n======================================================\n\nDialogue %d (out of %d)\n' % (dialogue+1, num_dialogues))

            ca.start_dialogue()

            while not ca.terminated():
                ca.continue_dialogue()

            ca.end_dialogue()

        print('\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}\nAverage Turns: {2}'.format(
            float(ca.num_successful_dialogues / num_dialogues),
            float(ca.cumulative_rewards / num_dialogues),
            float(ca.total_dialogue_turns / num_dialogues)))

        print('OBJECTIVE TASK COMPLETION RATE: {0}'.format(float(ca.num_task_success / num_dialogues)))

    # Run multiple agents against each other
    def run_multi_agent(self, settings, num_dialogues, num_agents):
        ConvSysAgents = []
        ConvUserAgents = []
        objective_success = 0

        generic_agents = False
        if 'GENERAL' in settings and 'generic' in settings['GENERAL'] and settings['GENERAL']['generic']:
            generic_agents = bool(settings['GENERAL']['generic'])

        # Verify that we have a DM section for each agent in the config and initialize agents
        for a in range(num_agents):
            ag_id_str = 'AGENT_' + str(a)
            if 'role' in settings[ag_id_str]:
                if settings[ag_id_str]['role'] == 'system':
                    if generic_agents:
                        ConvSysAgents.append(ConversationalGenericAgent(settings, a))
                    else:
                        ConvSysAgents.append(ConversationalMultiAgent(settings, a))

                elif settings[ag_id_str]['role'] == 'user':
                    if generic_agents:
                        ConvUserAgents.append(ConversationalGenericAgent(settings, a))
                    else:
                        ConvUserAgents.append(ConversationalMultiAgent(settings, a))

                else:
                    print('WARNING: Unknown agent role: {0}!'.format(settings[ag_id_str]['role']))
            else:
                raise ValueError('Role for agent {0} not defined in config.'.format(a))

        # TODO: WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AND ONE USER AGENT
        ConvUserAgents[0].initialize()

        ConvSysAgents[0].initialize()

        for dialogue in range(num_dialogues):
            print('\n======================================================\n\nDialogue %d (out of %d)\n' % (dialogue+1, num_dialogues))

            # TODO: WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AGENT.
            _, _, goal = ConvUserAgents[0].start_dialogue()
            sys_output, sys_output_dacts, _ = ConvSysAgents[0].start_dialogue(goal)

            while not all(ca.terminated() for ca in ConvSysAgents) and not all(ca.terminated() for ca in ConvUserAgents):
                # TODO: WARNING: FOR NOW ASSUMING WE HAVE ONE USER AGENT.
                user_output, user_output_dacts, goal = ConvUserAgents[0].continue_dialogue({'other_input_raw': sys_output,
                                                                                            'other_input_dacts': sys_output_dacts})

                # Need to check for termination condition again, for the case where the system says 'bye' and then the
                # user says bye too.
                if all(ca.terminated() for ca in ConvSysAgents) or all(ca.terminated() for ca in ConvUserAgents):
                    break

                # TODO: WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AGENT.
                sys_output, sys_output_dacts, goal = ConvSysAgents[0].continue_dialogue({'other_input_raw': user_output,
                                                                                         'other_input_dacts': user_output_dacts})

                # Sync goals (user has ground truth)
                ConvSysAgents[0].set_goal(ConvUserAgents[0].get_goal())

            # Check if there is a goal. For example, if the agents are generic there may not be a tracked goal.
            if not goal:
                continue

            # Consolidate goals to track objective success (each agent tracks different things)

            # From the System we keep everything except the status of actual requests (filled or not)
            goal = ConvSysAgents[0].agent_goal
            goal.actual_requests = ConvUserAgents[0].agent_goal.actual_requests

            _, _, obj_succ = ConvSysAgents[0].reward_func.calculate(ConvSysAgents[0].curr_state, [], goal=goal,
                                                                    agent_role="system")

            objective_success += 1 if obj_succ else 0

            print(f'OBJECTIVE TASK COMPLETION: {obj_succ}')

            for ca in ConvSysAgents:
                ca.end_dialogue()

            for ca in ConvUserAgents:
                ca.end_dialogue()

        print('\n\nSYSTEM Dialogue Success Rate: {0}\nAverage Cumulative Reward: {1}\nAverage Turns: {2}'.format(
            100 * float(ConvSysAgents[0].num_successful_dialogues / num_dialogues),
            float(ConvSysAgents[0].cumulative_rewards / num_dialogues),
            float(ConvSysAgents[0].total_dialogue_turns / num_dialogues)))

        print('\n\nUSER Dialogue Success Rate: {0}\nAverage Cumulative Reward: {1}\nAverage Turns: {2}'.format(
            100 * float(ConvUserAgents[0].num_successful_dialogues / num_dialogues),
            float(ConvUserAgents[0].cumulative_rewards / num_dialogues),
            float(ConvUserAgents[0].total_dialogue_turns / num_dialogues)))

        avg_rew = 0.5 * (float(ConvSysAgents[0].cumulative_rewards / num_dialogues) + float(ConvUserAgents[0].cumulative_rewards / num_dialogues))
        print(f'\n\nAVERAGE rewards: {avg_rew}')

        print('\n\nObjective Task Success Rate: {0}'.format(100 * float(objective_success / num_dialogues)))
        # print('\n\nObjective Task Success Rate: {0}'.format(100 * float(ConvSysAgents[0].num_task_success / num_dialogues)))
        # print('\n\nObjective Task Success Rate: {0}'.format(float(ConvUserAgents[0].num_task_success / num_dialogues)))


def argParse():
    cfg_parser = None

    # Parse arguments
    if len(sys.argv) < 3:
        print('WARNING: No configuration file.')

    # Initialize random seed
    random.seed(time.time())

    cfg_filename = sys.argv[2]
    if isinstance(cfg_filename, str):
        if os.path.isfile(cfg_filename):
            # Choose config parser
            parts = cfg_filename.split('.')
            if len(parts) > 1:
                if parts[1] == 'yaml':
                    with open(cfg_filename, 'r') as file:
                        cfg_parser = yaml.load(file, Loader=yaml.CLoader)
                elif parts[1] == 'cfg':
                    cfg_parser = configparser.ConfigParser()
                    cfg_parser.read(cfg_filename)
                else:
                    raise ValueError('Unknown configuration file type: %s' % parts[1])
        else:
            raise FileNotFoundError('Configuration file %s not found' % cfg_filename)
    else:
        raise ValueError('Unacceptable value for configuration file name: %s ' % cfg_filename)

    tests = 1
    dialogues = 10
    interaction_mode = 'simulation'
    num_agents = 1

    if cfg_parser:
        dialogues = int(cfg_parser['DIALOGUE']['num_dialogues'])

        if 'interaction_mode' in cfg_parser['GENERAL']:
            interaction_mode = cfg_parser['GENERAL']['interaction_mode']

            if 'agents' in cfg_parser['GENERAL']:
                num_agents = int(cfg_parser['GENERAL']['agents'])

            elif interaction_mode == 'multi_agent':
                print('WARNING! Multi-Agent interaction mode selected but number of agents is undefined in config.')

        if 'tests' in cfg_parser['GENERAL']:
            tests = int(cfg_parser['GENERAL']['tests'])

    return {'cfg_parser': cfg_parser,
            'tests': tests,
            'dialogues': dialogues,
            'interaction_mode': interaction_mode,
            'num_agents': num_agents}


def runController(args):
    # Extract arguments
    cfg_parser = args['cfg_parser']
    tests = args['tests']
    dialogues = args['dialogues']
    interaction_mode = args['interaction_mode']
    num_agents = args['num_agents']

    # Create a Controller
    controller = Controller()

    for test in range(tests):
        # Run simulation
        print('\n\n=======================================')
        print('# Running {0} dialogues (test {1} of {2}) #'.format(dialogues, (test + 1), tests))
        print('=======================================\n')

        try:
            if interaction_mode in ['simulation', 'text', 'speech']:
                if hasattr(cfg_parser, '_sections'):
                    # CFG version
                    controller.run(cfg_parser._sections, dialogues)
                else:
                    # YAML version
                    controller.run(cfg_parser, dialogues)

            elif interaction_mode == 'multi_agent':
                if hasattr(cfg_parser, '_sections'):
                    # CFG version
                    controller.run_multi_agent(cfg_parser._sections, dialogues, num_agents)
                else:
                    # YAML version
                    controller.run_multi_agent(cfg_parser, dialogues, num_agents)

            else:
                ValueError('Unknown interaction mode: {0}'.format(interaction_mode))

        except (ValueError, FileNotFoundError, TypeError, AttributeError) as err:
            print('\nPlato error! {0}\n'.format(err))


if __name__ == '__main__':
    arguments = argParse()
    runController(arguments)



