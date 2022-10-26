Create virtual environment:

```
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

Run flask app:

```
export FLASK_APP=main
export FLASK_ENV=development
flask --app main --debug run
```
