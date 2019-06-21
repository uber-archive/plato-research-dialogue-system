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
from NLG.NLG import NLG
import os.path
import pandas as pd


class LudwigNLG(NLG):
    def __init__(self, args):
        super(LudwigNLG, self).__init__(args)

        model_path = None
        if 'model_path' in args:
            model_path = args['model_path']

        self.model = None

        if isinstance(model_path, str):
            if os.path.isdir(model_path):
                print('Loading Ludwig NLG model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig NLG model directory {0} not found'.format(model_path,))
        else:
            raise ValueError('Ludwig NLG: Unacceptable value for NLG model file name: {0} '.format(model_path,))

    def initialize(self, args):
        pass

    def generate_output(self, args=None):
        if not args:
            print('WARNING! LudwigNLG called without arguments!')
            return ''

        if 'dacts' not in args:
            print('WARNING! LudwigNLG called without dacts!')
            return ''

        dacts = args['dacts']

        if not self.model:
            print('ERROR! Ludwig NLG model not initialized!')
            return pd.DataFrame({'empty': [0]})

        return self.model.predict(pd.DataFrame(data={'nlg_input': [dacts]}))

    def train(self, data):
        pass

    def train_online(self, data):
        self.train_online(data)

    def save(self, path=None):
        pass

    def load(self, path):
        pass

