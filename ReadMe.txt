Plato
=====

This is a beta release!

About Plato:

Plato can be used to create, train, and evaluate conv-AI agents in various environments and domains.
It was used to conduct the research (submitted to SIGDial 2019) regarding two agents that learn by talking to each
other via self-generated language.

Publication citation:

{To appear here upon acceptance in the conference.}



Quick Start Guide
=================

To run the Plato Research Dialogue System, you need to run the following command with the appropriate config file:

python runPlato.py -config config/CONFIG_FILE_NAME.yaml


Some examples are listed below.


To run a simulation using the agenda based simulator:

python runPlato.py -config Examples/config/simulate_agenda.yaml



To run a text based interaction using the agenda based simulator:

python runPlato.py -config Examples/config/simulate_text.yaml



To run a simulation using the Dialogue Act to Language user simulator:

python runPlato.py -config Examples/config/simulate_dtl.yaml



To run multiple agents

(train)

python runPlato.py -config Examples/config/CamRest_MA_train.yaml

(test)

python runPlato.py -config Examples/config/CamRest_MA_train.yaml


To run in generic module mode

(single agent)

python runPlato.py -config Examples/config/simulate_agenda_generic.yaml

(multiple agents)

python runPlato.py -config Examples/config/MA_test_generic.yaml

