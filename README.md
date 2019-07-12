![PlatoRDS-Logo](Resources/PlatoRDSLogo.png)

# Plato Research Dialogue System

This is a v 0.1 release.

The Plato Research Dialogue System is a flexible framework that can be used to 
create, train, and evaluate conversational AI agents in various environments. 
It supports interactions through speech, text, or dialogue acts and each 
conversational agent can interact with data, human users, or other 
conversational agents (in a multi-agent setting). Every component of every 
agent can be trained independently online or offline and Plato provides an 
easy way of wrapping around virtually any existing model, as long as Plato's 
interface is adhered to. 

Publication citations:

Alexandros Papangelis, Yi-Chia Wang, Piero Molino, and Gokhan Tur, 
“Collaborative Multi-Agent Dialogue Model Training Via Reinforcement Learning”, 
SIGDIAL 2019

Alexandros Papangelis, Mahdi Namazifar, Chandra Khatri, "Plato Research 
Dialogue System: A Flexible Conversational AI Research Platform", ArXiv 
(to appear)


# Introduction

Plato has been designed to be as modular and flexible as possible; it supports 
traditional as well as custom spoken dialogue system / conversational AI 
architectures and, importantly, allows multi-party interactions, where multiple
agents - potentially with a different role for each - can interact with each 
other, train concurrently, etc. 

The figures below show an example Plato conversational agent architecture when 
interacting with human users or with simulated users. Each individual component 
can be trained online or offline using your favourite framework ([Ludwig](https://uber.github.io/ludwig),
[TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), 
your own implementations). Ludwig is one of the easiest frameworks to get 
started with as it does not require writing code and is fully compatible with 
Plato.

![PlatoRDS-Humans](Resources/PlatoRDS-Humans.png)
Figure 1: This a Plato agent that follows the standard conversational agent 
architecture and interacts with a human user over text or speech. Any component
can be trained online or offline and can be replaced by custom or pre-trained
models. Grayed components are not core Plato components.

![PlatoRDS-Simulator](Resources/PlatoRDS-Simulator.png)
Figure 2: This is a Plato agent that follows the standard conversational agent
architecture and interacts with a Simulated User via structured information 
(dialogue acts) or text. Interacting with simulated users can generate
simulated conversational data that can be used to pre-train statistical 
models for the various components. These can then be used to create a 
prototype conversational agent that can interact with human users to collect
more natural data that can be subsequently used to train better statistical 
models. Grayed components are not Plato core components.


In addition to Single-Agent interactions, Plato supports agents communicating 
with other agents while adhering to some dialogue principles. These principles 
define *what each agent can understand* (an ontology of entities, or meanings; 
for example: price, location, users, preferences, cuisine types, etc.) and 
*what it can do* (ask for more information, provide some information, call an 
API, etc.). The Agents can communicate over speech, text, or structured 
information (Dialogue Acts) and each Agent has its own configuration. The 
figure below shows this architecture, outlining the communication between two 
Agents and the various components.

![PlatoRDS-MultiAgent](Resources/PlatoRDS-MultiAgent.png)
Figure 3: This is a Plato agent interacting with another Plato agent, over 
structured information (dialogue acts), text, and in the future, speech. This 
architecture allows concurrent training of multiple agents, each with 
potentially different roles and objectives and can facilitate research in 
fields such as multi-party interactions and multi-agent learning. Grayed 
components are not core Plato components.


Last, Plato supports custom architecture and jointly-trained components 
(e.g. end to end) via the Generic-Agent architecture shown below.

![PlatoRDS-GenericAgent](Resources/PlatoRDS-GenericAgent.png)
Figure 4: This is the most abstract version of a Plato agent (referred to as 
generic agent). This mode moves away from the standard conversational agent 
architecture and supports any kind of architecture (e.g., with joint 
components, text-to-text or speech-to-speech components, or any other set-up) 
and allows loading existing or pre-trained models into Plato.


This agent allows you to define your own architecture and/or plug your own 
components into Plato by simply providing a python class name and path to your 
module, as well as its initialization arguments. All you need to do is list the
modules in the order they should be executed in and Plato will take care of the 
rest (wrapping I/O, chaining the modules, and handling the dialogues).

Plato also provides support for Bayesian Optimisation of Conversational AI 
architectures or individual module parameters, through BOCS (Bayesian 
Optimisation of Combinatorial Structures). 


Contents:
* Quick Start Guide
    * Understanding the configuration files
    * Running Plato Agents
    * Running multiple Plato Agents
    * Running generic Plato Agents
* Training from data
    * Plato internal experience
    * Training with Plato
    * Training with Ludwig
* Create a new domain
* Create a new component
    * Inheriting from the abstract classes
    * Creating a custom component
* Bayesian Optimisation in Plato
* Conclusion


# Quick Start Guide

## Installation

1. Clone this repository:
    ````
    git clone git@github.com:uber-research/plato-research-dialogue-system.git
    ````
2. Install the requirements:

    For MacOS:
    ````
    brew install portaudio
    pip install -r requirements.txt
    ````
    
    For Ubuntu/Debian:
    ````
    sudo apt-get install python3-pyaudio
    pip install -r requirements.txt
    ````
    
    For Windows:
    ````
    pip install -r requirements.txt
    ````
    
3. Run Plato! 

    See below for a quick introduction to the configuration files and how to 
    run your first Plato agent.


#### Common Issues During Installation:
"CommonIssues.md" file contains common issues and their resolution that a user 
might encounter while installation.


## Running Plato Agents

To run the Plato Research Dialogue System, you need to run the following 
command with the appropriate configuration file 
(see Examples/config/simulate_agenda.yaml for an example configuration file):

````
python runPlatoRDS.py -config PATH_TO_CONFIG_FILE.yaml
````

Some examples are listed below.


To run a simulation using the agenda based simulator:

````
python runPlatoRDS.py -config Examples/config/simulate_agenda.yaml
````


To run a text based interaction using the agenda based simulator:

````
python runPlatoRDS.py -config Examples/config/simulate_text.yaml
````



To run a simulation using the Dialogue Act to Language user simulator:

````
python runPlatoRDS.py -config Examples/config/simulate_dtl.yaml
````


## Running multiple Plato Agents

One of Plato's main features is that it allows two agents to interact with each 
other. Each agent can have a different role (e.g. "system" and "user"), 
different objectives, and receive different reward signals. If the agents
are cooperating, some of these can be shared (e.g. what constitutes a
successful dialogue). In the future Plato will support more than two agents 
interacting.

To run multiple Plato agents:

- Training phase
    ````
    python runPlatoRDS.py -config Examples/config/CamRest_MA_train.yaml
    ````

- Testing phase
    ````
    python runPlatoRDS.py -config Examples/config/CamRest_MA_test.yaml
    ````


## Running generic Plato Agents

Most of the discussion and examples in this guide revolve around the 
traditional conversational agent architecture. Plato, however, does not need to 
adhere to that pipeline; it supports any range of custom modules, from 
splitting the NLU into many components to having just a single text to text
model. This is achieved by the Generic Agents. See 
Examples/config/simulate_agenda_generic.yaml for an example configuration file.

Generic Agents allow you to load your custom modules as python class objects. 
For each module listed in the config, Plato will instantiate the class using
the given path and arguments. Then during each dialogue turn, the Generic
Agent will sequentially call each module (in the order provided in the 
config) and will pass the output of the current module to the next. 

To run Plato in generic module mode:

- Single agent

    ````
    python runPlatoRDS.py -config Examples/config/simulate_agenda_generic.yaml
    ````

- Multiple agents
    ````
    python runPlatoRDS.py -config Examples/config/MultiAgent_test_generic.yaml
    ````


# Training from data

Plato has been designed to support training of internal components in an
online (i.e. during the interaction) or offline (i.e. from data) manner, using
your favourite framework (ludwig, tensorflow, pytorch, numpy, etc.). Virtually 
any model can be loaded into Plato as long as the I/O is respected, for example
if your model is an NLU it simply needs to inherit from Plato's NLU abstract
class, implement the necessary functions and pack / unpack the data into and
out of your model.


## Plato internal experience 

To facilitate learning online, debugging, and evaluation, Plato keeps track of
its internal "experience", which contains information about previous dialogue
state, action taken, current dialogue state, utterance received and utterance
produced, reward received, and a few other structs including a custom field 
that you can use to track anything that doesn't fall under the above. 

The idea behind this is that while training, each Conversational Agent, at the 
end of a dialogue or at specified intervals will call the train() function 
of all of its components, passing the dialogue experience as training data. 
Each component then picks the parts it needs for training.

If you want to use learning algorithms that are implemented inside Plato, any 
external data, such as DSTC2, should be parsed into this Plato experience, so
that they may be loaded and used by the corresponding component under training.

Alternatively, you may parse the data and train your models outside of Plato
and simply load the trained model when you want to use it for a Plato agent.

## Training with Plato

Training online (i.e. only using the current experience) is as easy as flipping
the 'Train' flags to 'True' in the configuration, for each component you wish
to train. 

To train from data, you simply need to load the experience you parsed from your
dataset. As an example of offline training in Plato, we will use the DSTC2
dataset that you can download from here:

````
http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz
````

Once the download is complete, you need to unzip the file.

The runDSTC2DataParser.py script will parse the DSTC2 data for you, and save it
as Plato experience. It will then load that experience and train a Supervised
Policy:

````
python runDSTC2DataParser.py -data_path <PATH_TO_DSTC2_DATA>/dstc2_traindev/data/
````

You can test using the following configuration:

````
python runPlatoRDS.py -config Examples/config/simulate_agenda_supervised.yaml
````

Note that you may load your experience into Plato and then keep training your
model with Reinforcement Learning or other learning methods.


## Training with Ludwig
Ludwig (https://uber.github.io/ludwig/) is an open source deep 
learning framework that allows you to train models without writing any code. 
You only need to parse your data into .csv files, create a ludwig config 
(in YAML), that describes the architecture you want, which features to use from 
the .csv and other parameters and then simply run a command in a terminal.

Ludwig also provides an API, that Plato is compatible with. This allows Plato
to integrate with Ludwig models, i.e. load / save the models, train and query
them. 

In the previous section, the runDSTC2DataParser.py actually generated some 
.csv files as well that can be used to train NLU and NLG. If all went well, 
you can find them here: Data/data/. Now, you need to write a config that
looks like this:

````

input_features:
    -
        name: nlg_input
        type: sequence
        encoder: rnn
        cell_type: lstm
        
output_features:
    -
        name: nlg_output
        type: sequence
        decoder: generator
        cell_type: lstm

training:
    epochs: 20
    learning_rate: 0.001
    dropout: 0.2
````

and train your model:

````
ludwig experiment --model_definition_file Examples/config/ludwig_nlg_train.yaml 
--data_csv Data/data/DSTC2_NLG_sys.csv --output_directory Models/CamRestNLG/Sys/
````

The next step is to load the model in Plato. Go to the simulate_agenda_nlg.yaml
configuration file and update the path if necessary:

````
...

NLG:
    nlg: CamRest
    model_path: <PATH_TO_YOUR_LUDWIG_MODEL>/model

...
````


and test that the model works:

````
python runPlatoRDS.py -config Examples/config/simulate_agenda_nlg.yaml
````

Remember that Ludwig will create a new *experiment_run_i* directory each time 
it is called, so please make sure you keep the correct path in Plato's config 
up to date.

Note that Ludwig also offers a method to train your model online, so in 
practice you need to write very little code to build, train, and evaluate a
new deep learning component in Plato. 



# Create a new domain

In order to build a conversational agent for task-oriented applications (such
as slot-filling), you need a *database* of items and an *ontology* describing your
domain. Plato provides a script for automating this process.

Let's say for example that you want to build a conversational agent for a 
flower shop, and you have the following items in a .csv:

````
id,type,color,price,occasion
1,rose,red,1,any
2,rose,white,2,anniversary
3,rose,yellow,2,celebration
4,lilly,white,5,any
5,orchid,pink,30,any
````

You can simply call createSQLiteDB.py to automatically generate a .db SQL file 
and a .json Ontology file. If you want to specify informable, requestable, and
system-requestable slots, you may do so in the configuration file:

````
GENERAL:
  csv_file_name: Data/data/flowershop.csv
  db_table_name: estore
  db_file_path: Ontology/Ontologies/flowershop-dbase.db
  ontology_file_path: Ontology/Ontologies/flowershop-rules.json

ONTOLOGY:
  informable_slots: [type, price, occasion]

  requestable_slots: [price, color]

  System_requestable_slots: [type, price, occasion]

````

and run the script:

````
python createSQLiteDB.py -config config/create_flowershop_DB.yaml
````

If all went well, you should have a flowershop.json and a flowershop.db into 
the Data/data/ folder.

You can now simply run Plato's dummy components as a sanity check:

````
python runPlatoRDS.py -config Examples/config/flowershop_text.yaml
````


# Create a new component

There are two ways to create a new module and this largely depends on its 
function. If your module implements a new way of doing NLU or Dialogue Policy,
then you should write a class that inherits from the corresponding abstract
class.

If, however, your module does not fit one of the Single Agent basic components, 
for example doesNamed Entity Recognition or predicts Dialogue Acts from text, 
then you need to write a class that inherits from the ConversationalModule
directly. This can then only be used by the Generic Agents. 

## Inheriting from the abstract classes

Create a new class inheriting from the corresponding Plato abstract class. You
should have a unique name for that class (e.g. 'myNLG') that will be used to
distinguish it from other options when parsing the config. At this version of
Plato, you will need to manually add some conditions where the configuration
files are being parsed (e.g. in the Conversational Agent, Dialogue Manager, 
etc.). 


## Creating a custom component

Create a new class inheriting from Conversational Module, and add your code 
there. You can then load your module via a Generic Agent by providing the 
appropriate path, class name, and arguments in the config.

````
...
MODULE_i:
    package: myPackage.myModule
    Class: myModule
    arguments:
      model_path: Models/myModule/parameters/
      ...
...

````

**Be careful!** *You are responsible for guaranteeing that the I/O of this 
module can be processed and consumed appropriately by modules before and after, 
as provided in your generic configuration file.*

Plato also supports parallel execution of modules. To enable that you need to
have the following structure in your config:

````
...
MODULE_i:
    parallel_modules: 5
    
    PARALLEL_MODULE_0:
        package: myPackage.myModule
        Class: myModule
        arguments:
          model_path: Models/myModule/parameters/
          ...
          
    PARALLEL_MODULE_1:
        package: myPackage.myModule
        Class: myModule
        arguments:
          model_path: Models/myModule/parameters/
          ...
          
    ...
...

````

**Be careful!** Outputs from the modules executed in parallel will be packed
into a list. The next module (e.g. ````MODULE_i+1````) will need to be able
to handle this kind of input. **The provided Plato modules are not designed to 
handle this, you will need to write a custom module to process input from
multiple sources.**

# Bayesian Optimisation in Plato

Coming soon!

# Acknowledgements

Special thanks to Mahdi Namazifar, Chandra Khatri, Piero Molino, Yi-Chia Wang, 
Michael Pearce and Gokhan Tur for their contributions and support.

# Conclusion

This is the very first release of Plato. Please understand that many features
are still being implemented and some use cases may not be supported yet.

Enjoy!
