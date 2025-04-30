1) Setup
Create .venv (python3.12 -m venv .venv in the project root)
Activate .venv (source .venv/bin/activate)
In case you want to stop the .venv, run deactivate

2) Get project onboarding instructions
Execute python main.py (wait until debugger is activated)
Open your browser: http://192.168.0.19:5001/
Write prompts and you will get answers

3) Evaluate agent
Execute python model_eval.py
Click on a link that is generated
Open your browser
Analyze the agent performance

TODO
4) Add more question-answers to the eval
5) Debug the eval (json-schema required)
6) Test the reactivity and domain-only responses