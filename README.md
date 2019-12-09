# pyro agents
Agent models and decision theory implemented with Pyro. 

## Directory Structure

```
.
©À©¤©¤ causal_agents               # Directory for Pyro implementation of causal agents, based on this semester's research
©¦   ©À©¤©¤ causal_agents           
©¦   ©¦   ©À©¤©¤ agents.py           # script for causal agent class
©¦   ©¦   ©À©¤©¤ environments.py     # script for environments class
©¦   ©¦   ©À©¤©¤ utils.py            # script for environments class
©¦   ©¦   ©¸©¤©¤ __init__.py         
©¦   ©¸©¤©¤ tests                   # causal agent tests folder
©¦       ©À©¤©¤ test_app.py         # script with unittests for causal agent
©¦       ©¸©¤©¤ __init__.py         
©¦
©¸©¤©¤tutorial_agents             # Directory for Pyro implementation of tutorial agents (non-causal), using agentmodels.org structure
    ©À©¤©¤ tutorial_agents         
    ©¦   ©À©¤©¤ agents.py           # script for tutorial agents (includes environment)
    ©¦   ©À©¤©¤ Agent_Flowchart.pdf # flowchart showing structure of agent.py script
    ©¦   ©¸©¤©¤ __init__.py         
    ©¸©¤©¤ tests                   # tutorial agent tests folder
        ©À©¤©¤ test_app.py         # script with unittests for tutorial agent
        ©¸©¤©¤ __init__.py        
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