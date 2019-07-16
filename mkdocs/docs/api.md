<span style="float:right;">[[source]](https://github.com/uber/ludwig/blob/master/runPlatoRDS.py#L37)</span>
# Controller class

```python
runPlatoRDS.Controller(
)
```


---
# Controller methods

## run_multi_agent


```python
run_multi_agent(
  config,
  num_dialogues,
  num_agents
)
```



This function will create multiple conversational agents and
orchestrate the conversation among them.

Note: In Plato v. 0.1 this function will create two agents.

:param config: a dictionary containing settings
:param num_dialogues: how many dialogues to run for
:param num_agents: how many agents to spawn
:return: some statistics

---
## run_single_agent


```python
run_single_agent(
  config,
  num_dialogues
)
```



This function will create an agent and orchestrate the conversation.

:param config: a dictionary containing settings
:param num_dialogues: how many dialogues to run for
:return: some statistics

----

<span style="float:right;">[[source]](https://github.com/uber/ludwig/blob/master/ConversationalAgent/ConversationalGenericAgent.py#L35)</span>
# ConversationalGenericAgent class

```python
ConversationalAgent.ConversationalGenericAgent.ConversationalGenericAgent(
  configuration,
  agent_id
)
```


The ConversationalGenericAgent receives a list of modules in
its configuration file, that are chained together serially -
i.e. the input to the agent is passed to the first module,
the first module's output is passed as input to the second
module and so on. Modules are wrapped using ConversationalModules.
The input and output passed between modules is wrapped into
ConversationalFrames.


---
# ConversationalGenericAgent methods

## continue_dialogue


```python
continue_dialogue(
  args=None
)
```



Perform one dialogue turn

:param args: input to this agent
:return: output of this agent

---
## end_dialogue


```python
end_dialogue(
)
```



Perform final dialogue turn. Save models if applicable.

:return:

---
## get_goal


```python
get_goal(
)
```



Get this agent's goal.

:return: a Goal

---
## initialize


```python
initialize(
)
```



Initializes the conversational agent based on settings in the
configuration file.

:return: Nothing

---
## load_module


```python
load_module(
  package_path,
  class_name,
  args
)
```



Dynamically load the specified class.

:param package_path: Path to the package to load
:param class_name: Name of the class within the package
:param args: arguments to pass when creating the object
:return: the instantiated class object

---
## set_goal


```python
set_goal(
  goal
)
```



Set or update this agent's goal.

:param goal: a Goal
:return: nothing

---
## start_dialogue


```python
start_dialogue(
  args=None
)
```



Reset or initialize internal structures at the beginning of the
dialogue. May issue first utterance if this agent has the initiative.

:param args:
:return:

---
## terminated


```python
terminated(
)
```



Check if this agent is at a terminal state.

:return: True or False
