# SNaiK
Machine learning project made by third year computer science bachelors Magnus Hektoen Steensland, Tommy René Sætre, Norbert Arkadiusz Görke and Christian Ryddheim Dahlin.

<br >

## Project
The goal of the project was to create an enviroment for a snake game, and implement three types of agents to controll the snake:

* Random agent: Actions are chosen at random
* DQN agent: Actions are chosen through Deep Q-learning (DQN)
* Q-learning agent: Actions are chosen through Q-learing

Further, the group were to gather data and create graphs of the different models' perfomance, and compare the results.

<br >

## Prerequisites
You need Python 3.10 and the packages specified in `requirements.txt` installed. You can download the packages however you like, but this installation-guide will do so through the Python package-manager `pip`.  

<br >

## Installation
Clone the project, and in the root folder of the project (~/.../SNaiK) run the following command:

```
pip install -r requirements.txt
```

<br >

## Run SNaiK
Run SNaiK via a command line. For a detalied guide, you can run `--help` for every option, command or argument added after typing `python main.py`.

<br >

### New run
Minimal options:
```
python main.py {OPTIONS} new {OPTIONS} {AGENT} {OPTIONS} {AGENT TYPE}
```      
<br >

Agent: random
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| N/A  | N/A | N/A |

<br >

Agent: dqn
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| convolutional  | N/A | dqn using a convolutional network |
| linear  | N/A | dqn using a linear network |

<br >

Agent: qlearning
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| full  | N/A | qlearning with maximum state space |
| angular  | -ns `int` | qlearning where the state space is divided into `int` sectors |
| quad  | -qs `int`|  qlearning where the state space is divided into `int x int` quads |

<br >

All options:
```
python main.py new -a -g -sz -e -s -r {AGENT} {AGENT TYPE} {AGENT TYPE ARUMENTS} 
```
|  OPTION | DESCRIPTION  |
|---|---|
| -a `float`|  sets the learning rate of the agent to `float`. Must be in `[0,1]`. Default is `0.1`|
| -g  `float`|  sets the gamma value of the agent to `float`. Must be in `[0,1]`. Default is `0.9`|
|  -sz `int`|  sets the grid size to `int x int`. Default is `4` |
|  -e `int`| sets the amount of episodes the agent should do before terminating. Default is `inf` |
|  -s `int` |  sets the seed to `int`. A seed makes the apple spawn in a pattern. Default is `None`|
|  -r | enables live rendering of the agent playing snake |

<br >

### Load run
Allows you to continue a previously saved SNaiK run.

<br >

Minimal options:
```
python main.py load <filename>.qbf
```

<br >

All options:
```
python main.py load -r -e <filename>.qbf
```
For description of options, see table above.

<br ><br >

## LIVE INTERACTIVE CLI
You can interact with a running instance of SNaiK by typing in the command line.

<br >

### exit
Terminates program.

<br >

### info
Prints info about the running instance in the terminal.

<br >

### save 
Allows you to save the running instance in different ways. 

* #### save &#8594; model
  * Saves an image of the running instance that can be loaded and resumed at a later time by using `python main load <filename>.qbf`

* #### save &#8594; stats
  * Saves the statistics of the running instance in a JSON-file.
  
* #### save &#8594; graph
  * Creates a graph of the running instance, showcasing score over episodes. There are two options:
    * graph &#8594; best: graphs the best scores over epsiodes
    * graph &#8594; avg: graphs the average scores over epsiodes

