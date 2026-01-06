import os
import vertexai
from google.oauth2 import service_account

# Vertex AI Configuration
PROJECT_ID = "marine-set-447307-k6"
LOCATION = "us-central1"
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "marine-set-447307-k6-d2077b260b04.json")

def initialize_vertexai():
    """Initialize Vertex AI with service account credentials"""
    try:
        # Load the credentials from the JSON file
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        
        # Initialize Vertex AI
        vertexai.init(
            project=PROJECT_ID,
            location=LOCATION,
            credentials=credentials
        )
        print(f"✅ Vertex AI initialized successfully for project: {PROJECT_ID}")
        return True
    except FileNotFoundError:
        print(f"❌ Error: Credentials file not found at {CREDENTIALS_PATH}")
        print("Please ensure the service account JSON file is in the project root directory.")
        return False
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {str(e)}")
        return False
