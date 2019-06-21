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

from ConversationalAgent.ConversationalAgent import ConversationalAgent

from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal, GoalGenerator
from UserSimulator.DActToLanguageUserSimulator.DTLUserSimulator import DTLUserSimulator
from UserSimulator.UserModel import UserModel
from DialogueManagement import DialogueManager
from DialogueManagement.Policy.ReinforcementLearning.RewardFunction import SlotFillingReward, SlotFillingGoalAdvancementReward
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder
from Ontology import Ontology, DataBase
from NLU.DummyNLU import DummyNLU
from NLU.CamRestNLU import CamRestNLU
from NLG.DummyNLG import DummyNLG
from NLG.CamRestNLG import CamRestNLG
from Dialogue.Action import DialogueAct, DialogueActItem, Operator

from copy import deepcopy
import os

import random

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class ConversationalMultiAgent(ConversationalAgent):
    '''
    Essentially the dialogue system. Will be able to interact with:

    - Simulated Users via:
        - Dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - Data
    '''

    def __init__(self, settings, agent_id):
        self.agent_id = agent_id

        # Flag to alternate training between roles
        self.train_system = True

        # Dialogue statistics
        self.dialogue_episode = 1
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 200
        self.train_interval = 50
        self.train_epochs = 3

        # Alternate training between the agents
        self.train_alternate_training = True
        self.train_switch_trainable_agents_every = self.train_interval

        # TODO Load all settings from the configuration file
        self.settings = settings

        self.USE_USR_SIMULATOR = False  # True values here would imply some default modules
        self.USER_SIMULATOR_NLG = False
        self.USE_NLU = False
        self.USE_NLG = False
        self.USE_SPEECH = False
        self.USER_HAS_INITIATIVE = True
        self.SAVE_LOG = True
        self.MAX_TURNS = 10              # This counts this agent's turns only

        self.ontology = None
        self.database = None
        self.domain = None
        self.dialogue_manager = None
        self.user_model = None
        self.nlu = None
        self.nlg = None

        self.agent_role = None
        self.agent_goal = None
        self.goal_generator = None
        self.goals_path = None

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.user_model = UserModel()

        # The size of the experience pool is a hyperparameter
        # self.recorder = DialogueEpisodeRecorder(3 * self.train_epochs * self.minibatch_length)

        # Do not have an experience window larger than the current batch, as past experience may not be relevant
        # since both agents learn.
        self.recorder = DialogueEpisodeRecorder(size=self.minibatch_length)

        # TODO: Handle this properly
        self.reward_func = SlotFillingReward()
        # self.reward_func = SlotFillingGoalAdvancementReward()

        if self.settings:
            agent_id_str = 'AGENT_' + str(self.agent_id)
            
            # Error checks for options the config must have
            if not self.settings['GENERAL']:
                raise ValueError('Cannot run Plato without GENERAL settings!')

            elif not self.settings['GENERAL']['interaction_mode']:
                raise ValueError('Cannot run Plato without an interaction mode!')

            elif not self.settings['DIALOGUE']:
                raise ValueError('Cannot run Plato without DIALOGUE settings!')

            elif not self.settings[agent_id_str]:
                raise ValueError('Cannot run Plato without at least one agent!')

            # Dialogue domain self.settings
            if 'DIALOGUE' in self.settings and self.settings['DIALOGUE']:
                if 'initiative' in self.settings['DIALOGUE']:
                    self.USER_HAS_INITIATIVE = bool(self.settings['DIALOGUE']['initiative'] == 'user')

                if self.settings['DIALOGUE']['domain']:
                    self.domain = self.settings['DIALOGUE']['domain']

                if self.settings['DIALOGUE']['ontology_path']:
                    if os.path.isfile(self.settings['DIALOGUE']['ontology_path']):
                        self.ontology = Ontology.Ontology(self.settings['DIALOGUE']['ontology_path'])
                    else:
                        raise FileNotFoundError(
                            'Ontology file %s not found' % self.settings['DIALOGUE']['ontology_path'])

                if self.settings['DIALOGUE']['db_path']:
                    if os.path.isfile(self.settings['DIALOGUE']['db_path']):
                        self.database = DataBase.DataBase(self.settings['DIALOGUE']['db_path'])
                    else:
                        raise FileNotFoundError('Database file %s not found' % self.settings['DIALOGUE']['db_path'])

                if 'goals_path' in self.settings['DIALOGUE']:
                    if os.path.isfile(self.settings['DIALOGUE']['goals_path']):
                        self.goals_path = self.settings['DIALOGUE']['goals_path']
                    else:
                        raise FileNotFoundError('Goals file %s not found' % self.settings['DIALOGUE']['goals_path'])

            # Interaction mode and User Simulator self.settings
            if 'GENERAL' in self.settings and self.settings['GENERAL']:
                if 'dialogues' in self.settings['GENERAL']:
                    dialogues_path = None
                    if 'path' in self.settings['GENERAL']['dialogues']:
                        dialogues_path = self.settings['GENERAL']['dialogues']['path']

                    if 'load' in self.settings['GENERAL']['dialogues'] and bool(self.settings['GENERAL']['dialogues']['load']):
                        if dialogues_path and os.path.isfile(dialogues_path):
                            self.recorder.load(dialogues_path)
                        else:
                            raise FileNotFoundError('Dialogue Log file %s not found (did you provide one?)' % dialogues_path)

                    if 'save' in self.settings['GENERAL']['dialogues']:
                        self.recorder.set_path(dialogues_path)
                        self.SAVE_LOG = bool(self.settings['GENERAL']['dialogues']['save'])

                if self.settings['GENERAL']['interaction_mode'] == 'simulation':
                    self.USE_USR_SIMULATOR = True

            # NLU Settings
            if 'NLU' in self.settings[agent_id_str] and self.settings[agent_id_str]['NLU'] and self.settings[agent_id_str]['NLU']['nlu']:
                nlu_args = dict(zip(['ontology', 'database'], [self.ontology, self.database]))

                if self.settings[agent_id_str]['NLU']['nlu'] == 'dummy':
                    self.nlu = DummyNLU(nlu_args)
                    self.USE_NLU = True

                elif self.settings[agent_id_str]['NLU']['nlu'] == 'CamRest':
                    if settings[agent_id_str]['NLU']['model_path']:
                        nlu_args['model_path'] = settings[agent_id_str]['NLU']['model_path']
                        self.nlu = CamRestNLU(nlu_args)
                        self.USE_NLU = True
                    else:
                        raise ValueError('Cannot find model_path in the config.')

            # NLG Settings
            if 'NLG' in self.settings[agent_id_str] and self.settings[agent_id_str]['NLG'] and self.settings[agent_id_str]['NLG']['nlg']:
                if self.settings[agent_id_str]['NLG']['nlg'] == 'dummy':
                    self.nlg = DummyNLG({})

                elif self.settings[agent_id_str]['NLG']['nlg'] == 'CamRest':
                    if settings[agent_id_str]['NLG']['model_path']:
                        self.nlg = CamRestNLG({'model_path': settings[agent_id_str]['NLG']['model_path']})
                    else:
                        raise ValueError('Cannot find model_path in the config.')

                if self.nlg:
                    self.USE_NLG = True

            # Retrieve agent role
            if 'role' in self.settings[agent_id_str]:
                self.agent_role = self.settings[agent_id_str]['role']
            else:
                raise ValueError('ConversationalMultiAgent: No role assigned for agent {0} in config!'.format(self.agent_id))

            if self.agent_role == 'user':
                if self.ontology and self.database:
                    self.goal_generator = GoalGenerator(ontology=self.ontology, database=self.database,
                                                        goals_file=self.goals_path)
                else:
                    raise ValueError('Conversational Multi Agent (user): Cannot generate goal without ontology and database.')

        dm_args = dict(zip(['settings', 'ontology', 'database', 'domain', 'agent_id', 'agent_role'],
                           [self.settings, self.ontology, self.database, self.domain, self.agent_id, self.agent_role]))
        dm_args.update(self.settings['AGENT_' + str(agent_id)]['DM'])
        self.dialogue_manager = DialogueManager.DialogueManager(dm_args)

    def __del__(self):
        if self.recorder and self.SAVE_LOG:
            self.recorder.save()

        if self.dialogue_manager:
            self.dialogue_manager.save()

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None

    def initialize(self):
        '''
        Initializes the conversational agent based on settings in the configuration file.

        :return: Nothing
        '''

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0

        if self.nlu:
            self.nlu.initialize({})

        self.dialogue_manager.initialize({})

        if self.nlg:
            self.nlg.initialize({})

        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

    def start_dialogue(self, goal=None):
        '''
        Perform initial dialogue turn.

        :return:
        '''

        self.dialogue_turn = 0

        if self.agent_role == 'user':
            self.agent_goal = self.goal_generator.generate()
            self.dialogue_manager.update_goal(self.agent_goal)

            print('DEBUG > User goal:')
            for c in self.agent_goal.constraints:
                print(f'\t\tConstr({self.agent_goal.constraints[c].slot}={self.agent_goal.constraints[c].value})')
            print('\t\t-------------')
            for r in self.agent_goal.requests:
                print(f'\t\tReq({self.agent_goal.requests[r].slot})')
            print('\n')

        elif goal:
            # No deep copy here so that all agents see the same goal.
            self.agent_goal = goal
        else:
            raise ValueError('ConversationalMultiAgent - no goal provided for agent {0}!'.format(self.agent_role))

        self.dialogue_manager.restart({'goal': self.agent_goal})

        # self.dialogue_manager.receive_input([])
        # response = self.dialogue_manager.respond()
        response = [DialogueAct('welcomemsg', [])]
        response_utterance = ''

        if self.agent_role == 'system':
            response = [DialogueAct('welcomemsg', [])]
            if self.USE_NLG:
                response_utterance = self.nlg.generate_output({'dacts': response, 'system': self.agent_role == 'system',
                                                               'last_sys_utterance': ''})

                print('{0} > {1}'.format(self.agent_role.upper(), response_utterance))

                # if self.USE_SPEECH:
                #     tts = gTTS(text=response_utterance, lang='en')
                #     tts.save('sys_output.mp3')
                #     os.system('afplay sys_output.mp3')
            else:
                print('{0} > {1}'.format(self.agent_role.upper(),  '; '.join([str(sr) for sr in response])))

        # TODO: Generate output depending on initiative - i.e. have users also start the dialogue

        # rew, success = self.reward_func.calculate(self.dialogue_manager.get_state(), response, self.agent_goal)
        # self.recorder.record(deepcopy(self.dialogue_manager.get_state()), self.dialogue_manager.get_state(), response, rew, success)
        # self.dialogue_turn += 1

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None

        # if self.USE_NLG and self.USE_NLU:
        #     return response_utterance, self.agent_goal
        # else:
        #     return response, self.agent_goal
        return response_utterance, response, self.agent_goal

    def continue_dialogue(self, args):
        '''
        Perform next dialogue turn.

        :return:
        '''

        if 'other_input_raw' not in args:
            raise ValueError('ConversationalMultiAgent called without raw input!')

        other_input_raw = args['other_input_raw']

        other_input_dact = None
        if 'other_input_dact' in args:
            other_input_dact = args['other_input_dact']

        goal = None
        if 'goal' in args:
            goal = args['goal']

        if goal:
            self.agent_goal = goal

        sys_utterance = ''

        other_input_nlu = deepcopy(other_input_raw)

        if self.nlu and isinstance(other_input_raw, str):
            # Process the other agent's utterance
            other_input_nlu = self.nlu.process_input(other_input_raw, self.dialogue_manager.get_state())

        elif other_input_dact:
            # If no utterance provided, use the dacts
            other_input_nlu = other_input_dact

        # print('{0} Recognised Input: {1}'.format(self.agent_role.upper(), '; '.join([str(ui) for ui in other_input_nlu])))

        self.dialogue_manager.receive_input(other_input_nlu)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue manager will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())

        # Update goal's ground truth
        if self.agent_role == 'system':
            self.agent_goal.ground_truth = deepcopy(self.curr_state.item_in_focus)

        if self.dialogue_turn < self.MAX_TURNS:
            response = self.dialogue_manager.generate_output()
            self.agent_goal = self.dialogue_manager.DSTracker.DState.user_goal

            # if self.agent_role == 'user':
            #     for dact in response:
            #         if dact.intent == 'request' and dact.params:
            #             for item in dact.params:
            #                 if item.slot not in self.agent_goal.actual_requests:
            #                     self.agent_goal.actual_requests[item.slot] = DialogueActItem(item.slot, Operator.EQ, '')
            #                     self.dialogue_manager.update_goal(self.agent_goal)
            #
            # elif self.agent_role == 'system':
            #     for dact in response:
            #         if dact.intent == 'offer':
            #             # Clear any old requests
            #             self.agent_goal.actual_requests = {}
            #             self.dialogue_manager.update_goal(self.agent_goal)

        else:
            # Force dialogue stop
            # print('{0}: terminating dialogue due to too many turns'.format(self.agent_role))
            response = [DialogueAct('bye', [])]

        rew, success, task_success = self.reward_func.calculate(self.dialogue_manager.get_state(), response, goal=self.agent_goal, agent_role=self.agent_role)

        if self.USE_NLG:
            sys_utterance = self.nlg.generate_output({'dacts': response, 'system': self.agent_role == 'system',
                                                      'last_sys_utterance': other_input_raw}) + ' '

            print('{0} > {1}'.format(self.agent_role.upper(), sys_utterance))

            # if self.USE_SPEECH:
            #     tts = gTTS(text=sys_utterance, lang='en')
            #     tts.save('sys_output.mp3')
            #     os.system('afplay sys_output.mp3')
        else:
            print('{0} > {1} \n'.format(self.agent_role.upper(), '; '.join([str(sr) for sr in response])))

        if self.prev_state:
            self.recorder.record(self.prev_state, self.curr_state, self.prev_action, self.prev_reward, self.prev_success,
                                 input_utterance=other_input_raw,
                                 output_utterance=sys_utterance,
                                 task_success=self.prev_task_success)

        self.dialogue_turn += 1

        self.prev_state = deepcopy(self.curr_state)
        self.prev_usr_utterance = deepcopy(other_input_raw)
        self.prev_sys_utterance = deepcopy(sys_utterance)
        self.prev_action = deepcopy(response)
        self.prev_reward = rew
        self.prev_success = success
        self.prev_task_success = task_success

        return sys_utterance, response, self.agent_goal

    def end_dialogue(self):
        '''
        Perform final dialogue turn. Save models if applicable.

        :return:
        '''

        if self.dialogue_episode % self.train_switch_trainable_agents_every == 0:
            self.train_system = not self.train_system

        # Record final state
        if not self.curr_state.is_terminal_state:
            self.curr_state.is_terminal_state = True
            self.prev_reward, self.prev_success, self.prev_task_success = \
                self.reward_func.calculate(self.curr_state, [DialogueAct('bye', [])], goal=self.agent_goal, agent_role=self.agent_role)

        self.recorder.record(self.curr_state, self.curr_state, self.prev_action, self.prev_reward, self.prev_success,
                             input_utterance=self.prev_usr_utterance,
                             output_utterance=self.prev_sys_utterance,
                             task_success=self.prev_task_success,
                             force_terminate=True)

        if self.dialogue_manager.is_training():
            if not self.train_alternate_training or \
                    (self.train_system and self.agent_role == 'system' or not self.train_system and self.agent_role == 'user'):

                if (self.dialogue_episode+1) % self.train_interval == 0 and len(self.recorder.dialogues) >= self.minibatch_length:
                    for epoch in range(self.train_epochs):
                        print('{0}: Training epoch {1} of {2}'.format(self.agent_role, (epoch+1), self.train_epochs))

                        # Sample minibatch
                        minibatch = random.sample(self.recorder.dialogues, self.minibatch_length)
                        self.dialogue_manager.train(minibatch)

        self.dialogue_episode += 1
        self.cumulative_rewards += self.recorder.dialogues[-1][-1]['cumulative_reward']

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % 10000 == 0:
            self.dialogue_manager.save()

        # Count successful dialogues
        if self.recorder.dialogues[-1][-1]['success']:
            print('{0} SUCCESS! (reward: {1})'.format(self.agent_role, sum([t['reward'] for t in self.recorder.dialogues[-1]])))
            self.num_successful_dialogues += int(self.recorder.dialogues[-1][-1]['success'])

        else:
            print('{0} FAILURE. (reward: {1})'.format(self.agent_role, sum([t['reward'] for t in self.recorder.dialogues[-1]])))

        if self.recorder.dialogues[-1][-1]['task_success']:
            self.num_task_success += int(self.recorder.dialogues[-1][-1]['task_success'])

    def terminated(self):
        # Hard coded response to bye to enforce policy according to which if any agent issues a 'bye' then the dialogue
        # terminates. Otherwise in multi-agent settings it is very hard to learn the association and learn to terminate
        # the dialogue.
        if self.dialogue_manager.get_state().user_acts:
            for act in self.dialogue_manager.get_state().user_acts:
                if act.intent == 'bye':
                    return True

        return self.dialogue_manager.at_terminal_state()

    def get_goal(self):
        return self.agent_goal

    def set_goal(self, goal):
        # TODO: Deep copy?
        self.agent_goal = goal

