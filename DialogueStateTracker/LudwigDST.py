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

from DialogueStateTracker.DialogueStateTracker import DialogueStateTracker
from ludwig.api import LudwigModel

from os import path

import pandas as pd


class LudwigDST(DialogueStateTracker):
    def __init__(self, args):
        super(LudwigDST, self).__init__(args)

        model_path = None
        if 'model_path' in args:
            model_path = args['model_path']

        self.model = None

        if isinstance(model_path, str):
            if path.isdir(model_path):
                print('Loading Ludwig DST model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig DST model directory {0} not found'.format(model_path))
        else:
            raise ValueError('Ludwig DST: Unacceptable value for model file name: {0}'.format(model_path))

    def initialize(self):
        pass

    def update_state(self, inpt):
        if not self.model:
            print('ERROR! Ludwig DST model not initialized!')
            return pd.DataFrame({'empty': [0]})

        # Warning: Make sure the same tokenizer that was used to train the model is used during prediction
        return self.model.predict(pd.DataFrame(data=inpt))

    def update_state_db(self, db_result):
        pass

    def train(self, dialogue_episodes):
        pass

    def save(self):
        pass

    def load(self, path):
        pass
