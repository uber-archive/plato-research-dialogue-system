from setuptools import setup
import os
from plato import __version__

def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


# Parse requirements
reqs = list(map(lambda l: l.strip(), open('requirements.txt').readlines()))

# Create directories that will be used by Plato
os.system('mkdir applications')
os.system('mkdir data')
os.system('mkdir logs')
os.system('mkdir models')
os.system('cd docs')
os.system('make html')
os.system('cd ..')

setup(
    name='plato',
    version=__version__,

    package_data={'plato': ['example/config/application/*.yaml',
                            'example/config/domain/*.yaml',
                            'example/config/ludwig/*.yaml',
                            'example/config/parser/*.yaml',
                            'example/data/*.csv',
                            'example/domains/*.db',
                            'example/domains/*.json',
                            'example/test/*.yaml']},

    packages=['plato',
              'plato.controller',
              'plato.agent',
              'plato.agent.conversational_agent',
              'plato.agent.component',
              'plato.agent.component.dialogue_manager',
              'plato.agent.component.dialogue_policy',
              'plato.agent.component.dialogue_policy.deep_learning',
              'plato.agent.component.dialogue_policy.reinforcement_learning',
              'plato.agent.component.dialogue_state_tracker',
              'plato.agent.component.joint_model',
              'plato.agent.component.nlg',
              'plato.agent.component.nlu',
              'plato.agent.component.user_simulator',
              'plato.agent.component.user_simulator.agenda_based_user_simulator',
              'plato.agent.component.user_simulator.dact_to_language_user_simulator',
              'plato.domain',
              'plato.dialogue',
              'plato.utilities',
              'plato.utilities.parser',
              'applications',
              'applications.cambridge_restaurants'
              ],

    python_requires='>=3.6',

    install_requires=reqs,

    url='https://uber-research.github.io/plato-research-dialogue-system/',
    license=readfile('LICENSE'),
    author='Alexandros Papangelis',
    author_email='al3x.papangelis@gmail.com',

    py_modules=['run_plato_rds'],

    entry_points={
        'console_scripts': [
            'plato = plato.run_plato_rds:entry_point'
        ]
    },

    description='Plato Research dialogue System is a platform for building, '
                'training, and deploying conversational AI agents that allows '
                'us to conduct state of the art research in conversational '
                'AI, quickly create prototypes and demonstration systems, as'
                ' well as facilitate conversational data collection'
)
