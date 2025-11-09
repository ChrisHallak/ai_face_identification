from flask import Flask, render_template
from routes import api_blueprint

app = Flask(__name__)

app.register_blueprint(api_blueprint, url_prefix='/')

if __name__ == '__main__':
    app.run(debug=True)