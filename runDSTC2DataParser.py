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


from Data import Parse_DSTC2

from DialogueManagement.Policy.DeepLearning.Supervised_Policy import Supervised_Policy
from DialogueManagement.Policy.DeepLearning.PolicyGradient_Policy import PolicyGradient_Policy

from Ontology.Ontology import Ontology
from Ontology.DataBase import DataBase

from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder

import sys

if __name__ == '__main__':
    parser = Parse_DSTC2.Parser()

    # Default values
    data_path = '/Users/alexandrospapangelis/Projects/Data/DSTC2/dstc2_traindev/data/'
    ontology_path = 'Ontology/Ontologies/CamRestaurants-rules_DSTC2.json'
    database_path = 'Ontology/Ontologies/CamRestaurants-dbase.db'

    if len(sys.argv) > 2:
        if sys.argv[1] == '-data':
            data_path = sys.argv[2]
    
    ontology = Ontology(ontology_path)
    database = DataBase(database_path)

    args = {'path': data_path,
            'ontology': ontology_path,
            'database': database_path}

    parser.initialize(**args)

    print('Parsing {0}'.format(args['path']))

    parser.parse_data()

    print('Data parsing complete.')

    # Save data

    parser.save('Logs')

    # Load data
    recorder_sys = DialogueEpisodeRecorder(path='Logs/DSTC2_system')
    recorder_usr = DialogueEpisodeRecorder(path='Logs/DSTC2_user')

    # Train Supervised Models using the recorded data
    system_policy_supervised = Supervised_Policy(ontology, database, agent_role='system', agent_id=0,
                                                 domain='CamRest')
    user_policy_supervised = Supervised_Policy(ontology, database, agent_role='user', agent_id=1,
                                               domain='CamRest')

    lr = 0.02
    epochs = 100

    system_policy_supervised.initialize(**{'is_training': True,
                                           'policy_path': 'Models/CamRestPolicy/Sys/sys_supervised_data',
                                           'learning_rate': lr})
    user_policy_supervised.initialize(**{'is_training': True,
                                         'policy_path': 'Models/CamRestPolicy/Usr/usr_supervised_data',
                                         'learning_rate': lr})

    # Train Policy Gradient Models using the recorded data
    # system_policy_gradient = PolicyGradient_Policy(self.ontology, self.database, agent_id=0, agent_role='system',
    #                                                domain='CamRest')
    # user_policy_gradient = PolicyGradient_Policy(self.ontology, self.database, agent_id=1, agent_role='user',
    #                                              domain='CamRest')
    #
    # system_policy_gradient.initialize(**{'is_training': True,
    #                                      'policy_path': 'Models/Policies/Sys/PG/policy_gradient_sys',
    #                                      'policy_alpha': 0.00001,
    #                                      'value_alpha': 0.001})
    # user_policy_gradient.initialize(**{'is_training': True,
    #                                    'policy_path': 'Models/Policies/User_0/PG/policy_gradient_usr',
    #                                    'policy_alpha': 0.00001,
    #                                    'value_alpha': 0.001})

    minibatch_length = round(len(recorder_sys.dialogues) / epochs) if epochs > 10 * len(recorder_sys.dialogues) else 100

    for epoch in range(1, epochs):
        print(f'\nTraining epoch {epoch}\n')

        user_policy_supervised.train(recorder_usr.dialogues)
        system_policy_supervised.train(recorder_sys.dialogues)

        # minibatch_sys = random.sample(recorder_sys.dialogues, minibatch_length)
        # system_policy_supervised.train(minibatch_sys)
        #
        # minibatch_usr = random.sample(recorder_usr.dialogues, minibatch_length)
        # user_policy_supervised.train(minibatch_usr)

        # system_policy_gradient.train(recorder_sys.dialogues)
        # user_policy_gradient.train(recorder_usr.dialogues)
        #
        # minibatch_sys = random.sample(recorder_sys.dialogues, minibatch_length)
        # minibatch_usr = random.sample(recorder_usr.dialogues, minibatch_length)
        # system_policy_gradient.train(minibatch_sys)
        # user_policy_gradient.train(minibatch_usr)

    # system_policy_gradient.save()
    # user_policy_gradient.save()

    system_policy_supervised.save()
    user_policy_supervised.save()
