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

from abc import ABC, abstractmethod

"""
parser is an abstract parent class for data parsers and defines the 
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
        Parse the data and generate Plato dialogue Experience logs.

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
