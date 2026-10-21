import os
from dotenv import load_dotenv

load_dotenv()

# Public viewer origin, separate from the Umami analytics API.
DPP_VIEWER_BASE_URL = os.getenv('DPP_VIEWER_BASE_URL', 'https://dpp.vera.spherity.dev')

UMAMI_BASE_URL = os.getenv("UMAMI_BASE_URL")
UMAMI_WEBSITE_ID = os.getenv("UMAMI_WEBSITE_ID")
UMAMI_USERNAME = os.getenv("UMAMI_USERNAME")
UMAMI_PASSWORD = os.getenv("UMAMI_PASSWORD")
TOKEN = os.getenv("TOKEN")
