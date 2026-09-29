import os
from dotenv import load_dotenv

load_dotenv()

DB = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
BREVO_KEY = os.getenv("BRAVO_API_KEY")
