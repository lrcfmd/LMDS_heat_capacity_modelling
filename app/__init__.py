from flask import Flask
from flask_restful import Resource, Api
app = Flask(__name__, static_url_path='/static', static_folder="static")
app.config['SECRET_KEY'] = "SOME_SECRET_KEY"
api = Api(app)

from app import routes

