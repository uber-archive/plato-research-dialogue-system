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

from abc import abstractmethod

import os.path
import sqlite3

"""
DataBase is the abstract parent class for all DataBase classes and defines the 
interface that should be followed.

SQLDataBase is an implementation of a DataBase class that can interface with 
SQL databases.

JSONDataBase is an implementation of a DataBase class that can interface with 
JSON databases (i.e. databases represented
as .json files).
"""


class DataBase:
    def __init__(self, filename):
        """
        Initialize the internal structures of the parser Base
        :param filename: path to the database file
        """

        self.db_file_name = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.db_file_name = filename

            else:
                raise FileNotFoundError('Database file %s not found'
                                        % filename)
        else:
            raise ValueError('Unacceptable value for database file name: %s '
                             % filename)

    @abstractmethod
    def db_lookup(self, dialogue_state):
        """
        Perform a database query.

        :param dialogue_state: the current dialogue state
        :return: result of the query
        """
        pass

    @abstractmethod
    def get_table_name(self):
        """
        Return the database table's name
        :return: the table's name
        """

        pass


class SQLDataBase(DataBase):
    def __init__(self, filename):
        """
        Initialize the internal structures of the SQL parser Base
        :param filename: path to load the database from
        """

        super(SQLDataBase, self).__init__(filename)

        self.SQL_connection = None
        self.db_table_name = None

        if isinstance(filename, str):
            if os.path.isfile(filename):
                self.SQL_connection = sqlite3.connect(self.db_file_name)

                # Get Table name
                cursor = self.SQL_connection.cursor()
                result = \
                    cursor.execute(
                        "select * from sqlite_master "
                        "where type = 'table';").fetchall()
                if result and result[0] and result[0][1]:
                    self.db_table_name = result[0][1]
                else:
                    raise ValueError(
                        'dialogue Manager cannot specify Table Name from '
                        'database {0}'.format(
                            self.db_file_name))

            else:
                raise FileNotFoundError('Database file %s not found'
                                        % filename)
        else:
            raise ValueError('Unacceptable value for database file name: %s '
                             % filename)

    def db_lookup(self, DState, MAX_DB_RESULTS=None):
        """
        Perform an SQL query

        :param DState: the current dialogue state
        :param MAX_DB_RESULTS: upper limit for results to be returned
        :return: the results of the SQL query
        """
        # Query the database
        cursor = self.SQL_connection.cursor()
        sql_command = " SELECT * FROM " + self.db_table_name + " "

        args = ''
        prev_arg = False
        prev_query_arg = False

        # Impose constraints
        for slot in DState.slots_filled:
            if DState.slots_filled[slot] and DState.slots_filled[slot] != \
                    'dontcare':
                if prev_arg:
                    args += " AND "

                args += slot + " = \"" + DState.slots_filled[slot] + "\""
                prev_arg = True

        # Impose queries
        if prev_arg and DState.slot_queries:
            args += " AND ("

        for slot in DState.slot_queries:
            for slot_query in DState.slot_queries[slot]:
                query = slot_query[0]
                op = slot_query[1]

                if prev_query_arg:
                    args += f" {op} "

                args += slot + " LIKE \'%" + query + "%\' "
                prev_query_arg = True

        if prev_arg and DState.slot_queries:
            args += " ) "

        if args:
            sql_command += " WHERE " + args + ";"

        cursor.execute(sql_command)
        db_result = cursor.fetchall()

        result = []

        if db_result:
            # Get the slot names
            slot_names = [i[0] for i in cursor.description]
            for db_item in db_result:
                result.append(dict(zip(slot_names, db_item)))

        if MAX_DB_RESULTS:
            return result[:MAX_DB_RESULTS]
        else:
            return result

    def get_table_name(self):
        """
        Get the SQL database's table name

        :return: the table name
        """

        cursor = self.SQL_connection.cursor()
        result = cursor.execute("select * from sqlite_master "
                                "where type = 'table';").fetchall()

        if result and result[0] and result[0][1]:
            db_table_name = result[0][1]
        else:
            raise ValueError(
                'dialogue State Tracker cannot specify Table Name from '
                'database {0}'.format(self.db_file_name))

        return db_table_name


class JSONDataBase(DataBase):
    def __init__(self, filename):
        """
        Initializes the internal structures of the json parser Base

        :param filename: path to the json database
        """
        super(JSONDataBase, self).__init__(filename)

    def db_lookup(self, dialogue_state):
        """
        Placeholder to query the json database

        :param dialogue_state: the current dialogue state
        :return: the result of the query
        """
        return []

    def get_table_name(self):
        """
        Placeholder to get the json database's table name
        :return: the table name
        """

        return ''
