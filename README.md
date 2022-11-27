# SNaiK
Machine learning project made by third year computer science bachelors Magnus Hektoen Steensland, Tommy René Sætre, Norbert Arkadiusz Gõrke and Christian Ryddheim Dahlin.

## Project
The goal of the project was to create an enviroment for a snake game, and implement three types of agents to controll the snake:

* Random agent: Actions are chosen at random
* DQN agent: Actions are chosen through Deep Q-learning (DQN)
* Q-learning agent: Actions are chosen through Q-learing

Further, the group were to gather data and create graphs of the different models' perfomance, and compare the results.

## Prerequisites
You need Python 3.10 and the packages specified in `requirements.txt` installed. You can download the packages however you like, but this installation-guide will do so through the Python package-manager `pip`.  

## Installation
Clone the project, and in the root folder of the project (~/.../SNaiK) run the following command:

```
pip install -r requirements.txt
```

## Run SNaiK
Run SNaiK via a command line. For a detalied guide, you can run `--help` for every option, command or argument added after typing `python main.py`.

### New run
Minimal options:
```
python main.py new {AGENT} {AGENT TYPE} {AGENT TYPE ARUMENTS}  
```      

Agent: random
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| N/A  | N/A | N/A |


Agent: dqn
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| convolutional  | N/A | dqn using a convolutional network |
| linear  | N/A | dqn using a linear network |

Agent: qlearning
| AGENT TYPE  | AGENT TYPE ARGUMENTS  | DESCRIPTON |
|---|---|---|
| full  | N/A | qlearning with maximum state space |
| angular  | -ns `num` | qlearning where the state space is divided into `num` sectors |
| quad  | -qs `num`|  qlearning where the state space is divided into `num x num` quads |



All options:
```
python main.py new -a -g -sz -e -s -r {AGENT} {AGENT TYPE} {AGENT TYPE ARUMENTS} 
```
|  PARAMTER | DESCRIPTION  |
|---|---|
| -a `num`|  sets the learning rate of the agent to `num`. Must be in `[0,1]`. Default is `0.1`|
| -g  `num`|  sets the gamma value of the agent to `num`. Must be in `[0,1]`. Default is `0.9`|
|  -sz `num`|  sets the grid size to `num x num`. Default is `4` |
|  -e `num`| sets the amount of episodes the agent should do before terminating. Default is `inf` |
|  -s `num` |  sets the seed to `num`. A seed makes the apple spawn in a pattern. Default is `None`|
|  -r | enables rendering of the snake |

### Load run
Allows you to continue a previously saved SNaiK run.
Minimal options:
```
python main.py load <filename>.qbf
```

All options:
```
python main.py load -r -e <filename>.qbf
```
For description of options, see table above.
