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

To support speech it is necessary to install [PyAudio](https://pypi.org/project/PyAudio/), 
which has a number of dependencies that might not exist on a developer's 
machine. If the steps above are unsuccessful, this [post](https://stackoverflow.com/questions/5921947/pyaudio-installation-error-command-gcc-failed-with-exit-status-1) 
on a PyAudio installation error includes instructions on how to get these 
dependencies and install PyAudio. 

#### Common Issues During Installation:
"CommonIssues.md" file contains common issues and their resolution that a user 
might encounter while installation.


## Running Plato Agents

Plato uses a Controller class to orchestrate the conversation between the
agents. The Controller will instantiate the agents, initialize them for each
dialogue, pass input and output appropriately, and keep track of statistics.


<!---! ![PlatoRDS-Controller](Resources/PlatoRDS-Controller.png) --->

To run a Plato conversational agent, the user must run the following command 
with the appropriate configuration file (see Examples/simulate_agenda.yaml for 
an example configuration file which contains a number of settings on the 
environment and the agent(s) to be created as well as their components):


````
python runPlatoRDS.py -config PATH_TO_CONFIG_FILE.yaml
````

Some examples are listed below.


To run a simulation using the agenda based user simulator in the Cambridge 
Restaurants domain:

````
python runPlatoRDS.py -config Examples/config/simulate_agenda.yaml
````


To run a text based interaction using the agenda based simulator in the 
Cambridge Restaurants domain:

````
python runPlatoRDS.py -config Examples/config/simulate_text.yaml
````



To run a speech based interaction using the agenda based simulator in the 
Cambridge Restaurants domain:


````
python runPlatoRDS.py -config Examples/config/simulate_speech.yaml
````


## Running multiple Plato Agents

One of Plato's main features allows two agents to interact with each other. 
Each agent can have a different role (for instance, system and user), different 
objectives, and receive different reward signals. If the agents are 
cooperating, some of these can be shared (e.g., what constitutes a successful 
dialogue). (In the future, we plan to build support for Plato to enable 
interaction between more than two agents at a time.) 

For example, to run multiple Plato agents on the benchmark Cambridge 
Restaurants domain, we run the following commands to train the agents’ 
dialogue policies and test them:


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
adhere to that pipeline; its generic agents support any range of custom 
modules, from splitting natural language understanding into many components to 
having multiple components running in parallel to having just a single 
text-to-text model. 

Generic agents allow users to load their custom modules as Python class 
objects. For each module listed in the configuration file, Plato will 
instantiate the class using the given path and arguments. Then, during each 
dialogue turn, the generic agent will sequentially call each module (in the 
order provided in its configuration file) and will pass the output of the 
current module to the next module in the list. The generic agent will return 
the last module’s output. 

the following are two examples of running a single Plato agent or multiple 
Plato agents in the generic module mode:

- Single generic agent, used to implement custom architectures or to use 
existing, pre-trained statistical models:

    ````
    python runPlatoRDS.py -config Examples/config/simulate_agenda_generic.yaml
    ````

- Multiple generic agents, same as above but for multiple agents (assuming you 
have trained dialogue policies using Examples/config/CamRest_MA_train.yaml):

    ````
    python runPlatoRDS.py -config Examples/config/MultiAgent_test_generic.yaml
    ````


# Training from data

Plato supports the training of agents’ internal components in an online 
(during the interaction) or offline (from data) manner, using any deep learning 
framework. Virtually any model can be loaded into Plato as long as Plato’s 
interface Input/Output is respected; for example, if a model is a custom NLU 
it simply needs to inherit from Plato's NLU abstract class, implement the 
necessary functions and pack/unpack the data into and out of the custom model.



## Plato internal experience 

To facilitate online learning, debugging, and evaluation, Plato keeps track of 
its internal experience in a structure called the Dialogue Episode Recorder, 
which contains information about previous dialogue states, actions taken, 
current dialogue states, utterances received and utterances produced, rewards 
received, and a few other structs including a custom field that can be used to 
track anything else that cannot be contained by the aforementioned categories

At the end of a dialogue or at specified intervals, each conversational agent 
will call the train() function of each of its internal components, passing the 
dialogue experience as training data. Each component then picks the parts it 
needs for training.

To use learning algorithms that are implemented inside Plato, any external 
data, such as DSTC2 data, should be parsed into this Plato experience so that 
they may be loaded and used by the corresponding component under training.

Alternatively, users may parse the data and train their models outside of Plato 
and simply load the trained model when they want to use it for a Plato agent.


## Training with Plato

Training online is as easy as flipping the 'Train' flags to 'True' in the 
configuration for each component users wish to train. 

To train from data, users simply need to load the experience they parsed from 
their dataset. As an example of offline training in Plato, we will use the 
DSTC2 dataset, which can be obtained from the 2nd Dialogue State Tracking 
Challenge website:

````
http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz
```` 

Once the download is complete, you need to unzip the file.

The ````runDSTC2DataParser.py```` script will parse the DSTC2 data for you, and save it
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

While each component has its own training parameters (e.g. learning rate), the 
Conversational Agent defines the meta-parameters of training such as:
- length of the experience
- length of the minibatch 
- training interval (train after that many dialogues)
- how many epochs to train for at each training interval


## Training with Plato and Ludwig
[Ludwig](https://uber.github.io/ludwig/) is an open source deep 
learning framework that allows you to train models without writing any code. 
You only need to parse your data into .csv files, create a ludwig config 
(in YAML), that describes the architecture you want, which features to use from 
the .csv and other parameters and then simply run a command in a terminal.

Ludwig also provides an API, that Plato is compatible with. This allows Plato
to integrate with Ludwig models, i.e. load / save the models, train and query
them. 

In the previous section, the ````runDSTC2DataParser.py```` actually generated 
some .csv files as well that can be used to train NLU and NLG. If all went well, 
you can find them here: ````Data/data/````. Now, you need to write a 
configuration file that looks like this:

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

The next step is to load the model in Plato. Go to the 
````simulate_agenda_nlg.yaml```` configuration file and update the path if 
necessary:

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
as slot-filling), you need a *database* of items and an *ontology* describing 
your domain. Plato provides a script for automating this process.

Let's say for example that you want to build a conversational agent for a 
flower shop, and you have the following items in a .csv:

````
id,type,color,price,occasion
1,rose,red,1,any
2,rose,white,2,anniversary
3,rose,yellow,2,celebration
4,lilly,white,5,any
5,orchid,pink,30,any
6,dahlia,blue,15,any
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
python createSQLiteDB.py -config Examples/config/create_flowershop_DB.yaml
````

If all went well, you should have a ````flowershop.json```` and a 
````flowershop.db```` into the ````Data/data/```` folder.

You can now simply run Plato's dummy components as a sanity check and talk to
your flower shop agent:

````
python runPlatoRDS.py -config Examples/config/flowershop_text.yaml
````


# Create a new component

There are two ways to create a new module depending on its function. If a 
module implements a new way of performing NLU or dialogue policy, then the user 
should write a class that inherits from the corresponding abstract class.

If, however, a module does not fit one of the single agent basic components, 
for example, it performs Named Entity Recognition or predicts dialogue acts 
from text, then the user must write a class that inherits from the 
ConversationalModule directly, which can then only be used by the generic 
agents. 


## Inheriting from the abstract classes

Users need to create a new class inheriting from the corresponding Plato 
abstract class and implement the interface defined by the abstract class and 
any other functionality they wish. This class should have a unique name (e.g. 
'myNLG') that will be used to distinguish it from other options when parsing 
the configuration file. At this version of Plato, users will need to manually 
add some conditions where the configuration files are being parsed (e.g. in 
the Conversational Agent, Dialogue Manager, etc.) unless the generic agent is 
used. 

## Creating a custom component

To construct a new module, the user must add their code to a new class 
inheriting from the conversational module. They can then load the module via a
generic agent by providing the appropriate package path, class name, and 
arguments in the configuration.


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

**Be careful!** You are responsible for guaranteeing that the I/O of this 
module can be processed and consumed appropriately by modules before and after, 
as provided in your generic configuration file.

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

Tutorial coming soon!

# Acknowledgements

Special thanks to Yi-Chia Wang, Mahdi Namazifar, Chandra Khatri, Piero Molino, 
Michael Pearce, and Gokhan Tur for their contributions and support.

# Conclusion

This is the very first release of Plato. Please understand that many features
are still being implemented and some use cases may not be supported yet.

Enjoy!
