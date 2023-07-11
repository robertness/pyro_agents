# pyro agents
Agent models and decision theory implemented with Pyro. 

# Findings & Write-up:

"A Causal Approach to Modeling Rational Agents", here: https://github.com/robertness/pyro_agents/blob/master/A%20causal%20approach%20to%20modeling%20rational%20agents.pdf

## Directory Structure

```
.
├── causal_agents               # Directory for Pyro implementation of causal agents, based on this semester's research
│   ├── causal_agents           
│   │   ├── agents.py           # script with causal agent class
│   │   ├── environments.py     # script with environments class
│   │   ├── utils.py            # script with helper functions 
│   │   └── __init__.py         
│   └── tests                   # causal agent tests folder
│       ├── test_app.py         # script with unittests for causal agent
│       └── __init__.py         
│
└──tutorial_agents             # Directory for Pyro implementation of tutorial agents (non-causal), using agentmodels.org structure
    ├── tutorial_agents         
    │   ├── agents.py           # script for tutorial agents (includes environment)
    │   ├── Agent_Flowchart.pdf # flowchart showing structure of agent.py script
    │   └── __init__.py         
    └── tests                   # tutorial agent tests folder
        ├── test_app.py         # script with unittests for tutorial agent
        └── __init__.py        
```

## Installing

Python 3.6 or later is required. If using MacOSX, we suggest using [pyenv](https://github.com/pyenv/pyenv) instead of the native OSX python. I assume you are using a virtual environment called .env.

```
git clone git@github.com:robertness/pyro_agents.git
cd pyro_agents
python3 -m venv .env
. .env/bin/activate
pip install --upgrade pip setuptools wheel
pip install . -r requirements.txt
```

## Running the unit tests

Go to desired folder:
```
cd causal_agents
```

and run
```
python -m unittest
```
