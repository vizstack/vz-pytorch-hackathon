mkdir venv
virtualenv -p python3.7 venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m ensurepip --upgrade
npm install -g @vizstack/vz-logger
