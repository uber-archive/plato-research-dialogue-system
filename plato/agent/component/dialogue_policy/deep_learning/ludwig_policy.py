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


from ludwig.api import LudwigModel
from .. import dialogue_policy
import os.path

"""
LudwigPolicy is a dialogue_policy class that defines an interface to Ludwig 
models.

This is an example class that you can extend to interface with your Ludwig
dialogue_policy model.
"""


class LudwigPolicy(dialogue_policy):
    def __init__(self, args):
        """
        Load the Ludwig Policy model.

        :param args: a dictionary containing the path to the model.
        """
        super(LudwigPolicy, self).__init__()

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

    def restart(self, args):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return:
        """

        pass

    def next_action(self, state):
        """
        Consult the internal model and produce the agent's response, given
        the current state

        :param state: the current dialogue state
        :return:
        """
        pass

    def train(self, data):
        """
        Not implemented.

        We can use Ludwig's API to train the model given the experience.

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
            print('WARNING: Ludwig Policy model not saved (no path provided).')
        else:
            # If the directory does not exist, create it
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            self.model.save(path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        if isinstance(model_path, str):
            if os.path.isdir(model_path):
                print('Loading Ludwig Policy model...')
                self.model = LudwigModel.load(model_path)
                print('done!')

            else:
                raise FileNotFoundError('Ludwig Policy: Model directory {0} '
                                        'not found'.format(model_path))
        else:
            raise ValueError('Ludwig Policy: Unacceptable value for model '
                             'file name: {0}'.format(model_path))
