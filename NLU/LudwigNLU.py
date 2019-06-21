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

Connects Ludwig to NLU via PyML

'''


__author__ = "Alexandros Papangelis"


from ludwig.api import LudwigModel
from NLU.NLU import NLU
import os.path
import pandas as pd


class LudwigNLU(NLU):
    def __init__(self, args):
        super(LudwigNLU, self).__init__(args)

        model_path = None
        if 'model_path' in args:
            model_path = args['model_path']

        self.model = None

        if isinstance(model_path, str):
            if os.path.isdir(model_path):
                print('Loading Ludwig NLU model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig NLU: Model directory {0} not found'.format(model_path))
        else:
            raise ValueError('Ludwig NLU: Unacceptable value for model file name: {0}'.format(model_path))

    def initialize(self, args):
        pass

    def process_input(self, text, dialogue_state=None):
        if not self.model:
            print('ERROR! Ludwig NLU model not initialized!')
            return pd.DataFrame({'empty': [0]})

        # Warning: Make sure the same tokenizer that was used to train the model is used during prediction
        return self.model.predict(pd.DataFrame(data={'transcription': [text]}), logging_level='logging.INFO')

    def train(self, data):
        pass

    def train_online(self, data):
        pass

    def save(self, path=None):
        pass

    def load(self, path):
        pass

