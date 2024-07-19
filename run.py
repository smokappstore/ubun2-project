from flask import Blueprint
def create_app():
    app = Blueprint(__name__)
    return Blueprint