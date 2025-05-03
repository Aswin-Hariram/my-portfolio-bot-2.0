from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()
import logging

# Import the handlers
from contact_handler import handle_contact
from chat_handler import handle_chat, initialize_chat, reset_chat_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))  # For session management
CORS(app, supports_credentials=True)  # Enable credentials for session cookies

# Initialize global variables
try:
    pdf_path = os.environ.get("PDF_PATH", 'document.pdf')
    chain = initialize_chat(pdf_path)
    logger.info("Application initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize application: {str(e)}")
    # We'll continue and handle this in the routes

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Portfolio AI API is running',
        'version': '2.0'
    })

@app.route('/chat', methods=['POST'])
def chat():
    return handle_chat(chain)

@app.route('/reset', methods=['POST'])
def reset_chat():
    return reset_chat_history()

@app.route('/contact', methods=['POST'])
def contact():
    return handle_contact()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default to 8000 for local dev
    app.run(host="0.0.0.0", port=port)