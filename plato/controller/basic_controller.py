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

from plato.controller import controller
from plato.agent.conversational_agent.conversational_single_agent import \
    ConversationalSingleAgent
from plato.agent.conversational_agent.conversational_multi_agent import \
    ConversationalMultiAgent
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent

import configparser
import yaml
import sys
import os
import time
import random

"""
The controller class is responsible for running each dialogue until the 
agent(s) terminate. In the multi-agent case, the controller will pass the 
appropriate input to each agent.
"""


class BasicController(controller.Controller):
    def __init__(self):
        """
        Initializes some basic structs for the controller.
        """

        self.sys_output = ''
        self.user_output = ''
        self.sys_output_dacts = []
        self.user_output_dacts = []
        self.goal = None

    @staticmethod
    def run_single_agent(config, num_dialogues):
        """
        This function will create an agent and orchestrate the conversation.

        :param config: a dictionary containing settings
        :param num_dialogues: how many dialogues to run for
        :return: some statistics
        """
        if 'GENERAL' in config and 'generic' in config['GENERAL'] \
                and config['GENERAL']['generic']:
            ca = ConversationalGenericAgent(config, 0)
        else:
            ca = ConversationalSingleAgent(config)

        ca.initialize()

        for dialogue in range(num_dialogues):
            print('\n=====================================================\n\n'
                  'Dialogue %d (out of %d)\n' % (dialogue + 1, num_dialogues))

            ca.start_dialogue()

            while not ca.terminated():
                ca.continue_dialogue()

            ca.end_dialogue()

        # Collect statistics
        statistics = {'AGENT_0': {}}

        statistics['AGENT_0']['dialogue_success_percentage'] = \
            100 * float(ca.num_successful_dialogues / num_dialogues)
        statistics['AGENT_0']['avg_cumulative_rewards'] = \
            float(ca.cumulative_rewards / num_dialogues)
        statistics['AGENT_0']['avg_turns'] = \
            float(ca.total_dialogue_turns / num_dialogues)
        statistics['AGENT_0']['objective_task_completion_percentage'] = \
            100 * float(ca.num_task_success / num_dialogues)

        print('\n\ndialogue Success Rate: {0}\nAverage Cumulative Reward: {1}'
              '\nAverage Turns: {2}'.
              format(statistics['AGENT_0']['dialogue_success_percentage'],
                     statistics['AGENT_0']['avg_cumulative_rewards'],
                     statistics['AGENT_0']['avg_turns']))

        return statistics

    @staticmethod
    def run_multi_agent(config, num_dialogues, num_agents):
        """
        This function will create multiple conversational agents and
        orchestrate the conversation among them.

        Note: In Plato v. 0.1 this function will create two agents.

        :param config: a dictionary containing settings
        :param num_dialogues: how many dialogues to run for
        :param num_agents: how many agents to spawn
        :return: some statistics
        """

        conv_sys_agents = []
        conv_user_agents = []
        objective_success = 0

        generic_agents = False
        if 'GENERAL' in config and 'generic' in config['GENERAL'] \
                and config['GENERAL']['generic']:
            generic_agents = bool(config['GENERAL']['generic'])

        # Verify that we have a DM section for each agent in the config and
        # initialize agents
        for a in range(num_agents):
            ag_id_str = 'AGENT_' + str(a)
            if 'role' in config[ag_id_str]:
                if config[ag_id_str]['role'] == 'system':
                    if generic_agents:
                        conv_sys_agents.append(
                            ConversationalGenericAgent(config, a))
                    else:
                        conv_sys_agents.append(
                            ConversationalMultiAgent(config, a))

                elif config[ag_id_str]['role'] == 'user':
                    if generic_agents:
                        conv_user_agents.append(
                            ConversationalGenericAgent(config, a))
                    else:
                        conv_user_agents.append(
                            ConversationalMultiAgent(config, a))

                else:
                    print('WARNING: Unknown agent role: {0}!'
                          .format(config[ag_id_str]['role']))
            else:
                raise ValueError('Role for agent {0} not defined in config.'
                                 .format(a))

        # TODO: WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AND ONE USER AGENT
        conv_user_agents[0].initialize()

        conv_sys_agents[0].initialize()

        for dialogue in range(num_dialogues):
            print('\n=====================================================\n\n'
                  'Dialogue %d (out of %d)\n' % (dialogue + 1, num_dialogues))

            # WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AGENT.
            user_output = conv_user_agents[0].start_dialogue()
            sys_output = conv_sys_agents[0].start_dialogue(
                {'goal': user_output['goal']}
            )

            while not all(ca.terminated() for ca in conv_sys_agents) \
                    and not all(ca.terminated() for ca in conv_user_agents):
                # WARNING: FOR NOW ASSUMING WE HAVE ONE USER AGENT.
                user_output = conv_user_agents[0].continue_dialogue(
                    {
                        'other_input_raw': sys_output['output_raw'],
                        'other_input_dacts': sys_output['output_dacts'],
                        'goal': sys_output['goal']
                    })

                # Need to check for termination condition again, for the case
                # where the system says 'bye' and then the user says bye too.
                if all(ca.terminated() for ca in conv_sys_agents) \
                        or all(ca.terminated() for ca in conv_user_agents):
                    break

                # WARNING: FOR NOW ASSUMING WE HAVE ONE SYSTEM AGENT.
                sys_output = conv_sys_agents[0].continue_dialogue(
                    {
                        'other_input_raw': user_output['output_raw'],
                        'other_input_dacts': user_output['output_dacts'],
                        'goal': user_output['goal']
                    }
                )

                # Sync goals (user has ground truth)
                conv_sys_agents[0].set_goal(conv_user_agents[0].get_goal())

            # Check if there is a goal. For example, if the agents are generic
            # there may not be a tracked goal.
            if not conv_sys_agents[0].agent_goal:
                continue

            # Consolidate goals to track objective success (each agent tracks
            # different things)

            # From the sys we keep everything except the status of actual
            # requests (filled or not)
            goal = conv_sys_agents[0].agent_goal
            goal.actual_requests = \
                conv_user_agents[0].agent_goal.actual_requests

            _, _, obj_succ = conv_sys_agents[0].reward_func.calculate(
                conv_sys_agents[0].get_state(), [], goal=goal,
                agent_role="system")

            objective_success += 1 if obj_succ else 0

            print(f'OBJECTIVE TASK COMPLETION: {obj_succ}')

            for ca in conv_sys_agents:
                ca.end_dialogue()

            for ca in conv_user_agents:
                ca.end_dialogue()

        # Collect statistics
        statistics = {}

        for i in range(num_agents):
            ag_id_str = 'AGENT_' + str(i)
            statistics[ag_id_str] = {'role': config[ag_id_str]['role']}

        statistics['AGENT_0']['dialogue_success_percentage'] = \
            100 * \
            float(conv_sys_agents[0].num_successful_dialogues / num_dialogues)
        statistics['AGENT_0']['avg_cumulative_rewards'] = \
            float(conv_sys_agents[0].cumulative_rewards / num_dialogues)
        statistics['AGENT_0']['avg_turns'] = \
            float(conv_sys_agents[0].total_dialogue_turns / num_dialogues)
        statistics['AGENT_0']['objective_task_completion_percentage'] = \
            100 * float(objective_success / num_dialogues)

        statistics['AGENT_1']['dialogue_success_percentage'] = \
            100 * \
            float(conv_user_agents[0].num_successful_dialogues / num_dialogues)
        statistics['AGENT_1']['avg_cumulative_rewards'] = \
            float(conv_user_agents[0].cumulative_rewards / num_dialogues)
        statistics['AGENT_1']['avg_turns'] = \
            float(conv_user_agents[0].total_dialogue_turns / num_dialogues)
        statistics['AGENT_1']['objective_task_completion_percentage'] = \
            100 * float(objective_success / num_dialogues)

        print('\n\nSYSTEM dialogue Success Rate: {0}\n'
              'Average Cumulative Reward: {1}\n'
              'Average Turns: {2}'.
              format(100 *
                     float(conv_sys_agents[0].num_successful_dialogues
                           / num_dialogues),
                     float(conv_sys_agents[0].cumulative_rewards
                           / num_dialogues),
                     float(conv_sys_agents[0].total_dialogue_turns
                           / num_dialogues)))

        print('\n\nUSER dialogue Success Rate: {0}\n'
              'Average Cumulative Reward: {1}\n'
              'Average Turns: {2}'.
              format(100 *
                     float(conv_user_agents[0].num_successful_dialogues
                           / num_dialogues),
                     float(conv_user_agents[0].cumulative_rewards
                           / num_dialogues),
                     float(conv_user_agents[0].total_dialogue_turns
                           / num_dialogues)))

        avg_rew = 0.5 * (
                float(conv_sys_agents[0].cumulative_rewards / num_dialogues) +
                float(conv_user_agents[0].cumulative_rewards / num_dialogues))
        print(f'\n\nAVERAGE rewards: {avg_rew}')

        print('\n\nObjective Task Success Rate: {0}'.format(
            100 * float(objective_success / num_dialogues)))

        return statistics

    def arg_parse(self, args=None):
        """
        This function will parse the configuration file that was provided as a
        system argument into a dictionary.

        :return: a dictionary containing the parsed config file.
        """

        arg_vec = args if args else sys.argv

        # Parse arguments
        if len(arg_vec) < 3:
            print('WARNING: No configuration file.')

        test_mode = arg_vec[1] == '--test'

        if test_mode:
            return {'test_mode': test_mode}

        # Initialize random seed
        random.seed(time.time())

        cfg_filename = arg_vec[2]
        if isinstance(cfg_filename, str):
            if os.path.isfile(cfg_filename):
                # Choose config parser
                if cfg_filename[-5:] == '.yaml':
                    with open(cfg_filename, 'r') as file:
                        cfg_parser = yaml.load(file, Loader=yaml.Loader)
                elif cfg_filename[-4:] == '.cfg':
                    cfg_parser = configparser.ConfigParser()
                    cfg_parser.read(cfg_filename)
                else:
                    raise ValueError('Unknown configuration file type: %s'
                                     % cfg_filename)
            else:
                raise FileNotFoundError('Configuration file %s not found'
                                        % cfg_filename)
        else:
            raise ValueError('Unacceptable value for configuration file name: '
                             '%s ' % cfg_filename)

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
                    print('WARNING! Multi-Agent interaction mode selected but '
                          'number of agents is undefined in config.')

            if 'tests' in cfg_parser['GENERAL']:
                tests = int(cfg_parser['GENERAL']['tests'])

        return {'cfg_parser': cfg_parser,
                'tests': tests,
                'dialogues': dialogues,
                'interaction_mode': interaction_mode,
                'num_agents': num_agents,
                'test_mode': False}

    def run_controller(self, args):
        """
        This function will create and run a controller. It iterates over the
        desired number of tests and prints some basic results.

        :param args: the parsed configuration file
        :return: nothing
        """

        # Extract arguments
        cfg_parser = args['cfg_parser']
        tests = args['tests']
        num_dialogues = args['dialogues']
        interaction_mode = args['interaction_mode']
        num_agents = args['num_agents']

        for test in range(tests):
            # Run simulation
            print('\n\n=======================================')
            print('# Running {0} dialogues (test {1} of {2}) #'.format(
                num_dialogues,
                (test + 1),
                tests))
            print('=======================================\n')

            try:
                if interaction_mode in ['simulation', 'text', 'speech']:
                    # YAML version
                    statistics = self.run_single_agent(
                        cfg_parser, num_dialogues)

                elif interaction_mode == 'multi_agent':
                    # YAML version
                    statistics = self.run_multi_agent(
                        cfg_parser, num_dialogues, num_agents)

                else:
                    ValueError('Unknown interaction mode: {0}'.format(
                        interaction_mode))
                    return -1

            except (ValueError, FileNotFoundError, TypeError, AttributeError) \
                    as err:
                print('\nPlato error! {0}\n'.format(err))
                return -1

        print(f'Results:\n{statistics}')
        return 0


def run(config, test=None):
    # Create a basic controller
    ctrl = BasicController()

    if test:
        arguments = {'test_mode': True}

    elif config:
        if os.path.isfile(config):
            # Pass the configuration file provided
            arguments = ctrl.arg_parse(['_', '-config', config])

        else:
            # Else look for the config file in the example folder
            import plato

            # __file__ points to __init__.py but we want the root path only.
            plato_path = "/".join(plato.__file__.split("/")[:-1])[:-6] + '/'
            new_config_path = \
                plato_path + 'example/config/application/' + config

            if os.path.isfile(new_config_path):

                # Parse the example configuration file
                arguments = ctrl.arg_parse(
                    ['_', '-config', new_config_path])

                # Correct relative paths to absolute paths. This is instead of
                # having yet another option in the configuration files to
                # declare a root directory (that itself is not without
                # drawbacks), since the paths to the examples are anyway
                # hard-coded.
                if 'GENERAL' in arguments['cfg_parser'] and \
                        'global_arguments' in \
                        arguments['cfg_parser']['GENERAL']:
                    if 'ontology' in \
                            arguments['cfg_parser'][
                                'GENERAL']['global_arguments']:
                        arguments[
                            'cfg_parser'
                        ]['GENERAL']['global_arguments']['ontology'] = \
                            plato_path + \
                            arguments[
                                'cfg_parser'
                            ]['GENERAL']['global_arguments']['ontology']

                    if 'database' in \
                            arguments['cfg_parser'][
                                'GENERAL']['global_arguments']:
                        arguments[
                            'cfg_parser'
                        ]['GENERAL']['global_arguments']['database'] = \
                            plato_path + \
                            arguments[
                                'cfg_parser'
                            ]['GENERAL']['global_arguments']['database']

                if 'DIALOGUE' in arguments['cfg_parser']:
                    if 'ontology_path' in arguments['cfg_parser']['DIALOGUE']:
                        arguments['cfg_parser'][
                            'DIALOGUE']['ontology_path'] = \
                            plato_path + \
                            arguments['cfg_parser'][
                                'DIALOGUE']['ontology_path']

                    if 'db_path' in arguments['cfg_parser']['DIALOGUE']:
                        arguments['cfg_parser']['DIALOGUE']['db_path'] = \
                            plato_path + \
                            arguments['cfg_parser']['DIALOGUE']['db_path']

            else:
                raise ValueError(f'Configuration file {config} not found!')

    else:
        # Get arguments from command line (sys.argv)
        arguments = ctrl.arg_parse()

    if 'test_mode' in arguments and arguments['test_mode']:
        # Runs Plato with all configuration files in the config/tests/
        # directory and prints a FAIL message upon any exception raised.
        passed = []
        failed = []

        import plato

        # __file__ points to __init__.py, which is 11 characters, plus 6 characters for plato/ but we
        # want the root path only.
        plato_path = plato.__file__[:-17]

        for (dirpath, dirnames, filenames) in \
                os.walk(plato_path + 'test/config/'):
            if not filenames or filenames[0] == '.DS_Store':
                continue

            for config_file in filenames:
                print(f'\n\nRunning test with configuration '
                      f'{config_file}\n')

                args = ctrl.arg_parse(['_', '-config',
                                             dirpath + config_file])

                if ctrl.run_controller(args) < 0:
                    print(f'FAIL! With {config_file}')
                    failed.append(config_file)

                else:
                    print('PASS!')
                    passed.append(config_file)

        print('\nTEST RESULTS:')
        print(f'Passed {len(passed)} out of {(len(passed) + len(failed))}')

        print(f'Failed on: {failed}')

        return not failed

    else:
        # Normal Plato execution
        ctrl.run_controller(arguments)
