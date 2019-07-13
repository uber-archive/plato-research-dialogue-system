"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"


import sys

from Data import Parse_DSTC2
from DialogueManagement.DialoguePolicy.DeepLearning.SupervisedPolicy import \
    SupervisedPolicy
from Domain.DataBase import SQLDataBase
from Domain.Ontology import Ontology
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder

"""
This script runs the DSTC2 Data Parser and trains a Supervised DialoguePolicy 
for the user and the system respectively.
"""


if __name__ == '__main__':
    """
    This script will create a DSCT2-specific data parser, and run it. It will
    then load the parsed experience and train two supervised dialogue policies,
    one for the system and one for the user.
    """

    if len(sys.argv) < 3:
        raise AttributeError('Please provide a path to the DSTC2 data. For'
                             'example: .../DSTC2/dstc2_traindev/data/')

    if sys.argv[1] == '-data_path':
        data_path = sys.argv[2]

    else:
        raise TypeError(f'Incorrect option: {sys.argv[1]}')

    parser = Parse_DSTC2.Parser()

    # Default values
    ontology_path = 'Domain/Domains/CamRestaurants-rules.json'
    database_path = 'Domain/Domains/CamRestaurants-dbase.db'

    if len(sys.argv) > 2:
        if sys.argv[1] == '-data':
            data_path = sys.argv[2]
    
    ontology = Ontology(ontology_path)
    database = SQLDataBase(database_path)

    args = {'path': data_path,
            'ontology': ontology,
            'database': database}

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
    system_policy_supervised = SupervisedPolicy(ontology,
                                                database,
                                                agent_role='system',
                                                agent_id=0,
                                                domain='CamRest')

    user_policy_supervised = SupervisedPolicy(ontology,
                                              database,
                                              agent_role='user',
                                              agent_id=1,
                                              domain='CamRest')

    # Set learning rate and number of epochs
    learning_rate = 0.02
    epochs = 100

    system_policy_supervised.initialize(**{
        'is_training': True,
        'policy_path': 'Models/CamRestPolicy/Sys/sys_supervised_data',
        'learning_rate': learning_rate})

    user_policy_supervised.initialize(**{
        'is_training': True,
        'policy_path': 'Models/CamRestPolicy/Usr/usr_supervised_data',
        'learning_rate': learning_rate})

    for epoch in range(1, epochs):
        print(f'\nTraining epoch {epoch}\n')

        user_policy_supervised.train(recorder_usr.dialogues)
        system_policy_supervised.train(recorder_sys.dialogues)

    system_policy_supervised.save()
    user_policy_supervised.save()
