import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = "VARELab-GPT4o"
AZURE_API_VERSION = "2024-08-01-preview"
TEMPERATURE = 0.5

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))  # seconds
RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", "5"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Output configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "metadata_gen_output")
EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation")

# Debug print
print("DEBUG - Environment Variables:")
print(f"AZURE_API_KEY: {'[SET]' if AZURE_API_KEY else '[NOT SET]'}")
print(f"AZURE_ENDPOINT: {AZURE_ENDPOINT}")
print(f"AZURE_DEPLOYMENT: {AZURE_DEPLOYMENT}")
print(f"AZURE_API_VERSION: {AZURE_API_VERSION}")
print(f"TEMPERATURE: {TEMPERATURE}")