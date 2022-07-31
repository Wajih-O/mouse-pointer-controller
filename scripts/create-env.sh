ENV=".venv"
python -m venv $ENV
. $ENV/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt
