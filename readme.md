Servidor realizado en Flask que procesa peticiones del frontend para crear modelos, realizar predicciones y hacer un análisis de los datos

# Run app

Se necesita tener Python3 en la computadora para realizar

1. Create virtual environment:

Windows
```
python -m venv ./venv
source venv/Scripts/activate
pip install -r requirements.txt

```
Linux

```
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run flask app:

```
export FLASK_APP=main
export FLASK_ENV=development
flask --app main --debug run
```
