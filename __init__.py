from flask import Flask 
def create_app():
    app = Flask(__name__)
    
    from.app import main
    app.register_Blueprint('main')

    return app