"""Shared app server."""
from dash import Dash
from flask import Flask
server = Flask(__name__)
app = Dash(
    __name__,
    server=server
)

app.css.config.serve_locally = False
app.scripts.config.serve_locally = False
app.config.suppress_callback_exceptions = True
