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

from plato.agent.component.dialogue_state_tracker.dialogue_state_tracker \
    import DialogueStateTracker
from ludwig.api import LudwigModel

from os import path

import pandas as pd

"""
LudwigDST provides an interface to Ludwig models for dialogue State Tracking.
"""


class LudwigDST(DialogueStateTracker):
    def __init__(self, args):
        """
        Load the Ludwig DST model.

        :param args: a dictionary containing the path to the model.
        """

        super(LudwigDST, self).__init__()

        model_path = None
        if 'model_path' in args:
            model_path = args['model_path']

        self.model = None

        self.load(model_path)

    def __del__(self):
        """
        Close the model.

        :return: nothing
        """

        if self.model:
            self.model.close()

    def initialize(self, args):
        """
        Nothing to do here

        :return:
        """
        pass

    def update_state(self, inpt):
        """
        Retrieve updated state by querying the Ludwig model.

        :param inpt: the current input (usually the nlu output)
        :return:
        """

        if not self.model:
            print('ERROR! Ludwig DST model not initialized!')
            return pd.DataFrame({'empty': [0]})

        # Warning: Make sure the same tokenizer that was used to train the
        # model is used during prediction
        return self.model.predict(pd.DataFrame(data=inpt))

    def update_state_db(self, db_result):
        """
        Nothing to do here.

        :param db_result:
        :return: nothing
        """

        pass

    def train(self, dialogue_episodes):
        """
        Not implemented.

        We can use Ludwig's API to train the model given the experience.

        :param dialogue_episodes: dialogue experience
        :return:
        """

        pass

    def save(self, model_path=None):
        """
        Saves the Ludwig model.

        :return:
        """

        self.model.save(model_path)

    def load(self, model_path):
        """
        Load the Ludwig model from the path provided

        :param model_path: the path to load the model from
        :return: nothing
        """

        if isinstance(model_path, str):
            if path.isdir(model_path):
                print('Loading Ludwig DST model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig DST model directory {0} not '
                                        'found'.format(model_path))
        else:
            raise ValueError('Ludwig DST: Unacceptable value for model file '
                             'name: {0}'.format(model_path))
