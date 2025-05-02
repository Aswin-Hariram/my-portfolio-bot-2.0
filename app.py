from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Hello, World!'

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 for local dev
    app.run(host="0.0.0.0", port=port)
