# SNaiK
Machine learning project made by third year computer science bachelors Magnus Hektoen Steensland, Tommy René Sætre, Norbert Arkadiusz Gõrke and Christian Ryddheim Dahlin.

## Project
The goal of the project was to create an enviroment for a snake game, and implement three types of agents to controll the snake:

* Random agent: Actions are chosen at random
* Q-learning agent: Actions are chosen through Q-learing
* DQN agent: Actions are chosen through Deep Q-learning (DQN)

Further, the group were to gather data and create graphs of the different models' perfomance, and compare the results.

## Prerequisites
You need Python 3.10 and the packages specified in `requirements.txt` installed. You can download the packages however you like, but this installation-guide will do so through the Python package-manager `pip`.  

## Installation
Clone the project, and in the root folder of the project (~/.../SNaiK) run the following command:

```
pip install -r requirements.txt
```

## Run SNaiK
Run SNaiK is via a command line. For a detalied guide, you can run `--help` for every option, command or argument added after typing `python main.py` 

### New run
Minimal options:
```
python main.py new {AGENT} {AGENT TYPE} {AGENT TYPE ARUMENTS}
```


All options:
```
python main.py new -a -g -sz -e -r -s {AGENT} {AGENT TYPE} {AGENT TYPE ARUMENTS} 
```

