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

import os.path
import sqlite3


class DataBase:
    def __init__(self, filename):
        self.db_file_name = None
        self.SQL_connection = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.db_file_name = filename
                self.SQL_connection = sqlite3.connect(self.db_file_name)

            else:
                raise FileNotFoundError('Database file %s not found' % filename)
        else:
            raise ValueError('Unacceptable value for database file name: %s ' % filename)

