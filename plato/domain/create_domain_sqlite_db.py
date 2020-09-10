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

import sqlite3
from sqlite3 import Error
import csv
import json
import yaml
import os
import string

"""
This script creates an domain .json file and an SQL .db file, given a .csv or
 .tsv data file.

Please note that this script assumes that the first column of the data file is 
the primary key of the .db file.
"""


class CreateSQLiteDB:

    def create_sql_connection(self, db_file):
        """
        Creates a connection to an SQL database

        :param db_file: database file path
        :return: the connection object or None
        """
        try:
            connection = sqlite3.connect(db_file)
            return connection

        except Error as err:
            print(err)

        return None

    def create_sql_table(self, sql_conn, create_table_sql):
        """
        This function creates a table given an SQL connection

        :param sql_conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            sql_cursor = sql_conn.cursor()
            sql_cursor.execute(create_table_sql)

        except Error as e:
            print(e)

    def create_ontology(self, sql_conn, tab_name, ontol_name, inf_slots,
                        req_slots, sys_req_slots):
        """
        This function will create the .json ontology file.

        :param sql_conn: an sql connection
        :param tab_name: the table name
        :param ontol_name: the ontology name (to be created)
        :param inf_slots: a list of informable slots
        :param req_slots: a list of requestable slots
        :param sys_req_slots: a list of system requestable slots
        :return: nothing
        """

        # Create the ontology
        ontology = {'type': tab_name,
                    'informable': {slot: [] for slot in inf_slots},
                    'requestable': req_slots,
                    'system_requestable': sys_req_slots}

        cursor = sql_conn.cursor()

        for slot in ontology['informable']:
            sql_command = 'SELECT DISTINCT ' + slot + ' FROM ' + tab_name + ';'

            cursor.execute(sql_command)
            db_result = cursor.fetchall()

            if db_result:
                ontology['informable'][slot] = [t[0] for t in db_result]
            else:
                print(f'Warning! CreateSQLiteDB query for distinct {slot} '
                      f'values did not return results.')

        with open(ontol_name, 'w') as ontology_file:
            json.dump(ontology, ontology_file, separators=(',', ':'), indent=4)

    def check_float(self, number):
        """
        Checks to see if number is float or not

        :param number: the number to check
        :return: True or False
        """
        try:
            float(number)
            return True
        except ValueError:
            return False

    def arg_parse(self, cfg_filename):
        """
        This function parses the configuration file into a dictionary.

        :return: a dictionary with the settings
        """
        # Parse arguments
        if not cfg_filename:
            print('WARNING CreateSQLiteDB: No source CSV file provided.')

        if isinstance(cfg_filename, str):
            if os.path.isfile(cfg_filename):
                with open(cfg_filename, 'r') as file:
                    cfg_parser = yaml.load(file, Loader=yaml.Loader)
            else:
                raise FileNotFoundError('Configuration file %s not found'
                                        % cfg_filename)
        else:
            raise ValueError('Unacceptable value for configuration file name: '
                             '%s ' % cfg_filename)

        return cfg_parser


def run(config):
    """
    This script will create an SQL database and the corresponding ontology,
    given a .csv or .tsv file containing the data.
    
    It will produce a .db file (the database) and a .json file (the ontology),
    reflecting the settings in the configuration file.
    
    Warning: The first column of the .csv / .tsv will be treated as the primary
             key.
    
    """

    db_creator = CreateSQLiteDB()

    plato_path = ''

    if not os.path.isfile(config):
        # Look for the config file in the example
        import plato

        # __file__ points to __init__.py, which is 11 characters but we
        # want the root path only.
        plato_path = "/".join(plato.__file__.split("/")[:-1])[:-6] + '/'
        config = plato_path + 'example/config/domain/' + config

    args = db_creator.arg_parse(config)

    if not args:
        raise ValueError('Terminating')

    csv_filename = plato_path + args['GENERAL']['csv_file_name']
    table_name = args['GENERAL']['db_table_name']
    db_name = plato_path + args['GENERAL']['db_file_path']
    ontology_name = plato_path + args['GENERAL']['ontology_file_path']

    informable_slots = []
    requestable_slots = []
    system_requestable_slots = []

    if 'ONTOLOGY' in args:
        if 'informable_slots' in args['ONTOLOGY']:
            informable_slots = args['ONTOLOGY']['informable_slots']

        if 'requestable_slots' in args['ONTOLOGY']:
            requestable_slots = args['ONTOLOGY']['requestable_slots']

        if 'system_requestable_slots' in args['ONTOLOGY']:
            system_requestable_slots = \
                args['ONTOLOGY']['system_requestable_slots']

    column_names = []

    MAX_DB_ENTRIES = -1

    delim = '\t' if csv_filename.split('.')[1] == 'tsv' else ','

    # Read csv entries and create items
    with open(csv_filename) as csv_input:
        csv_reader = csv.reader(csv_input, delimiter=delim)

        for entry in csv_reader:
            column_names = entry

            if not informable_slots:
                # Skip the primary key (first column by default)
                informable_slots = column_names[1:]

            if not requestable_slots:
                requestable_slots = column_names[1:]

            if not system_requestable_slots:
                system_requestable_slots = column_names[1:]

            break

    # Warning! This treats all entries as strings by default
    sqlcmd_create_table = 'CREATE TABLE IF NOT EXISTS ' + \
                          table_name + '(' + column_names[0] + \
                          ' text PRIMARY KEY,' + \
                          ' text,'.join(
                              [column_names[c]
                               for c in range(1, len(column_names))]) \
                          + ');'

    # Create a database connection
    conn = db_creator.create_sql_connection(db_name)
    with conn:
        if conn is not None:
            # Create the table
            db_creator.create_sql_table(conn, sqlcmd_create_table)

        else:
            print("Error! cannot create the database connection.")

        # Read csv entries and create items
        with open(csv_filename) as csv_input:
            csv_reader = csv.reader(csv_input, delimiter=delim)
            first_entry = True

            punctuation = string.punctuation.replace('$', '')
            punctuation = punctuation.replace('-', '')
            punctuation = punctuation.replace('_', '')
            punctuation = punctuation.replace('.', '')
            punctuation = punctuation.replace('&', '')
            punctuation_remover = str.maketrans('', '', punctuation)

            print('Populating database ')
            entries_count = 1

            for entry in csv_reader:
                # Discard first entry (column names)
                if first_entry:
                    first_entry = False

                else:
                    # Create item
                    sql_cmd = \
                        "INSERT INTO " + table_name + '(' + \
                        ','.join([c for c in column_names]) + ')' + \
                        " VALUES(" + ','.join(
                            ['?' for c in column_names]) + \
                        ')'

                    # Round to one decimal digit
                    entry = [str(round(float(e), 1)) if
                             db_creator.check_float(e) else e for e in entry]

                    # Remove non-ascii characters
                    entry = \
                        [str(''.join(i for i in e if
                                     ord(i) < 128)).replace('\"', '')
                         for e in entry]
                    entry = [e.replace('\'', '') for e in entry]

                    # Remove punctuation
                    entry = [e.rstrip().lower().translate(punctuation_remover)
                             for e in entry]

                    # Replace empty values with None
                    # Put an actual string here so the slot entropy
                    # calculation can take
                    # the absence of values into account
                    entry = [e if e else 'None' for e in entry]

                    cur = conn.cursor()
                    cur.execute(sql_cmd, tuple(entry))

                    entries_count += 1
                    if entries_count % 10000 == 0:
                        print(f'(added {entries_count} entries)')

                    if 0 < MAX_DB_ENTRIES <= entries_count:
                        break

    print(f'{table_name} database created with {entries_count} items!\n'
          f'Creating ontology...')

    db_creator.create_ontology(conn, table_name, ontology_name,
                               informable_slots, requestable_slots,
                               system_requestable_slots)

    print(f'{table_name} ontology created!')
