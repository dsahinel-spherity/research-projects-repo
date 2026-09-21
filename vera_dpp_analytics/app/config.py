import os
from dotenv import load_dotenv

load_dotenv()

UMAMI_BASE_URL = os.getenv("UMAMI_BASE_URL")
UMAMI_WEBSITE_ID = os.getenv("UMAMI_WEBSITE_ID")
UMAMI_USERNAME = os.getenv("UMAMI_USERNAME")
UMAMI_PASSWORD = os.getenv("UMAMI_PASSWORD")
TOKEN = os.getenv("TOKEN")