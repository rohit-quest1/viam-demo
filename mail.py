import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
from datetime import datetime



async def send_alert_email(record):
    """Send an alert email using Amazon SES when a mismatch is detected."""
    
    load_dotenv()
    
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")  # e.g., 'us-east-1'

    SENDER = "ram@quest1.io"  
    RECIPIENT = "ram@quest1.io"  

    SUBJECT = "Vehicle Number Plate Mismatch Alert"
    BODY_TEXT = f"""
    Alert: Vehicle Number Plate Mismatch Detected!
    
    Vehicle Number: {record['vehicle_number']}
    Expected Make: {record['expected_params']['make']}
    Expected Colour: {record['expected_params']['colour']}
    
    Actual Data: {record['actual_params']}
    
    Timestamp: {record['timestamp']}
    """
    
    BODY_HTML = f"""
    <html>
    <head></head>
    <body>
      <h2>🚨 Vehicle Number Plate Mismatch Detected! 🚨</h2>
      <p><strong>Vehicle Number:</strong> {record['vehicle_number']}</p>
      <p><strong>Expected Make:</strong> {record['expected_params']['make']}</p>
      <p><strong>Expected Colour:</strong> {record['expected_params']['colour']}</p>
      <p><strong>Actual Data:</strong> {record['actual_params']}</p>
      <p><strong>Timestamp:</strong> {record['timestamp']}</p>
      <p><strong>Image:</strong> <br> <img src='{record['image_url']}' alt='{record['image_url']}' width='500px'></p>
    </body>
    </html>
    """
    
    try:
        ses_client = boto3.client('ses', region_name=AWS_REGION, 
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY)  
        
        response = ses_client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,  
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': 'UTF-8',
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': 'UTF-8',
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
        
    except ClientError as e:
        print(f"Error sending email: {e}")
    except Exception as e: 
        print(f"An unexpected error occurred: {e}")
    else:
        print(f"Email sent successfully! Message ID: {response['MessageId']}")



