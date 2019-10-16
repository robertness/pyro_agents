# pyro agents
Agent models and decision theory implemented with Pyro. 

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

Run

```
python -m unittest
```