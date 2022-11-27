# SNaiK 🤖🐍
Machine learning project for the 5th semester [applied machine learning course (IDATT2502)](https://www.ntnu.no/studier/emner/IDATT2502) created by:
 * [Magnus Hektoen Steensland](https://github.com/magsteen)
 * [Tommy René Sætre](https://github.com/tomrsae)
 * [Norbert Arkadiusz Görke](https://github.com/norgor)
 * [Christian Ryddheim Dahlin](https://github.com/chrisryda)


## Goal 🥅
The goal of the project was to create an enviroment for a snake game, and implement following types of machine learning agents to control the snake:

* Random agent: Actions are chosen at random.
* DQN agent: Actions are chosen by using Deep Q-Network (DQN)
* Q-Learning agent: Actions are chosen through Q-Learning

Further, the group was to gather data, create graphs of the different models' performance, and compare the results.

## Prerequisites 📦
You need the following to run SNaiK:
 * Python 3.10 or greater
 * Packages from the `requirements.txt` file.

### Install packages using pip
Run the following command in the root directory of the repository.
```
pip install -r requirements.txt
```

## Running SNaiK 🏃
Run SNaiK via a command line. For a detalied guide, you can run `--help` for every option, command or argument added after typing `python main.py`.

### New run
```
python main.py new [options] <agent> <agent argument> <agent options>  
```

#### Options
|  Option | Description  |
|---|---|
| -a `num`|  Sets the learning rate of the agent to `num`. Must be in `[0,1]`. Default is `0.1`.|
| -g  `num`|  Sets the gamma value of the agent to `num`. Must be in `[0,1]`. Default is `0.9`.|
|  -sz `num`|  Sets the grid size to `num x num`. Default is `4`. |
|  -e `num`| Sets the number of episodes the agent should do before stopping. Default is `inf`. |
|  -s `num` |  Sets the seed to `num`. A seed makes the apple spawn in a deterministic pattern every run. Default is `None`. |
|  -r | Enables rendering of the snake. |

Available agents:
 * [random](#random-agent-random)
 * [dqn](#dqn-agent-dqn)
 * [qlearning](#q-learning-agent-qlearning)

#### Random Agent (`random`)
| Agent Argument  | Agent Option  | Description |
|---|---|---|
| N/A  | N/A | N/A |

#### DQN Agent (`dqn`)
| Agent Argument  | Agent Option  | Description |
|---|---|---|
| convolutional  | N/A | dqn using a convolutional network |
| linear  | N/A | dqn using a linear network |

#### Q-Learning Agent (`qlearning`)
| Agent Argument  | Agent Option  | Description |
|---|---|---|
| full  | N/A | qlearning with maximum state space |
| angular  | -ns `num` | qlearning where the state space is divided into `num` sectors |
| quad  | -qs `num`|  qlearning where the state space is divided into `num x num` quads |

### Load run
Allows you to load and continue a previously saved SNaiK run.
```
python main.py load [options] <filename>.qbf
```
#### Options
|  Option | Description  |
|---|---|
|  -e `num`| Sets the number of episodes the agent should do before stopping. Default is `inf`. |
|  -r | Enables rendering of the snake. |

## Interactive CLI
You can interact with a running instance of SNaiK by typing in the command line.

### Commands
 * `exit`: Exits the application gracefully.
 * `info`: Shows information about the current program run.
 * `save`: Save an aspect of the program specified by the subcommand.
   * `model`: Saves the currently learned state of the agent. Used for [loading the agent](#load-run).
   * `stats`: Saves the statistics specified by the subcommand to file.
     * `best`: Moving maximum of all the episodes.
     * `avg`: Simple moving average of all the episodes.
   * `graph`: Saves an image of the graphed data specified by the subcommand.
     * `best`: Moving maximum of all the episodes.
     * `avg` : Simple moving average of all the episodes.
