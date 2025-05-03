import os
import logging
from flask import Flask, request, jsonify
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twilio credentials from environment
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
NOTIFICATION_NUMBER = os.getenv('NOTIFICATION_NUMBER')

# Initialize Twilio client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized.")
else:
    logger.warning("Twilio credentials missing. SMS will not be sent.")

def send_sms_notification(name, email, phone, message):
    """Send SMS notification."""
    if not twilio_client:
        logger.warning("Twilio client not initialized.")
        return False

    if not TWILIO_PHONE_NUMBER or not NOTIFICATION_NUMBER:
        logger.warning("Twilio phone numbers not configured.")
        return False
        
    # Check if the numbers are the same
    if TWILIO_PHONE_NUMBER == NOTIFICATION_NUMBER:
        logger.error("Twilio error: 'To' and 'From' numbers cannot be the same.")
        return False

    try:
        short_message = message[:50] + '...' if len(message) > 50 else message
        sms_body = (
            f"🔔 New Contact Form Submission\n"
            f"👤 Name: {name}\n"
            f"📧 Email: {email}\n"
            f"📱 Phone: {phone}\n"
            f"💬 Message: {short_message}"
        )

        sms = twilio_client.messages.create(
            body=sms_body,
            from_=TWILIO_PHONE_NUMBER,
            to=NOTIFICATION_NUMBER
        )

        logger.info(f"SMS sent. SID: {sms.sid}, Status: {sms.status}")
        if sms.error_message:
            logger.error(f"SMS Error: {sms.error_message}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS: {str(e)}")
        return False

@app.route('/contact', methods=['POST'])
def handle_contact():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

        required = ['name', 'email', 'message']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'status': 'error', 'message': f'Missing fields: {", ".join(missing)}'}), 400

        # Get phone number (optional field)
        phone = data.get('phone', 'Not provided')
        
        logger.info(f"New contact: {data['name']} ({data['email']}, {phone})")

        sms_sent = send_sms_notification(data['name'], data['email'], phone, data['message'])

        return jsonify({
            'status': 'success',
            'message': 'Contact submitted successfully',
            'sms_notification': 'sent' if sms_sent else 'not sent'
        })
    except Exception as e:
        logger.error(f"Error in contact endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
