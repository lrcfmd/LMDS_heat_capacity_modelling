from flask import Flask

app = Flask(__name__, static_url_path='/static', static_folder="static")
app.config['SECRET_KEY'] = "SOME_SECRET_KEY"

from app import routes
