"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"


from ludwig.api import LudwigModel
from NLG.NLG import NLG
import os.path
import pandas as pd

"""
LudwigNLG is an NLG class that defines an interface to Ludwig models.
"""


class LudwigNLG(NLG):
    def __init__(self, args):
        """
        Load the Ludwig NLG model.

        :param args: a dictionary containing the path to the model.
        """
        super(LudwigNLG, self).__init__()

        model_path = None
        if 'model_path' in args:
            model_path = args['model_path']

        self.model = None

        self.load(model_path)

    def __del__(self):
        """
        Close the Ludwig model.

        :return:
        """

        if self.model:
            self.model.close()

    def initialize(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def generate_output(self, args=None):
        """
        Generate output, used by the Generic Agent.

        :param args:
        :return:
        """
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
        """
        Not implemented.

        We can use Ludwig's API to train the model given the experience.

        :param data: dialogue experience
        :return:
        """
        pass

    def train_online(self, data):
        """
        Not implemented.

        We can use Ludwig's API to train the model online (i.e. for a single
        dialogue).

        :param data: dialogue experience
        :return:
        """
        pass

    def save(self, path=None):
        """
        Save the Ludwig model.

        :param path: the path to save to
        :return:
        """
        if not path:
            print('WARNING: Ludwig NLG model not saved (no path provided).')
        else:
            self.model.save(path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        if isinstance(model_path, str):
            if os.path.isdir(model_path):
                print('Loading Ludwig NLG model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig NLG: Model directory {0} not '
                                        'found'.format(model_path))
        else:
            raise ValueError('Ludwig NLG: Unacceptable value for model file '
                             'name: {0}'.format(model_path))
