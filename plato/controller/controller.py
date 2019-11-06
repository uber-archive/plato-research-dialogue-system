"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from abc import abstractmethod


class Controller:
    def __int__(self):
        pass

    @abstractmethod
    def arg_parse(self, args=None):
        pass

    @abstractmethod
    def run_controller(self, args):
        pass
