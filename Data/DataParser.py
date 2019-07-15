"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from abc import ABC, abstractmethod

"""
DataParser is an abstract parent class for data parsers and defines the 
interface that should be used.
"""


class DataParser(ABC):

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the internal structures of the data parser.

        :param kwargs:
        :return:
        """

        pass

    @abstractmethod
    def parse_data(self):
        """
        Parse the data and generate Plato Dialogue Experience Logs.

        :return:
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the experience

        :param path: path to save the experience to
        :return:
        """

        pass
