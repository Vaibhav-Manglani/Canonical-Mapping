import os

# Google Gemini API Configuration
# GOOGLE_API_KEY = "your-google-api-key"

# LangChain API Configuration
LANGCHAIN_API_KEY = "your-langchain-api-key"

# Set environment variables


def setup_environment():
    """Setup environment variables for API keys"""
    # if GOOGLE_API_KEY != "your-google-api-key":
    #     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    if LANGCHAIN_API_KEY != "your-langchain-api-key":
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
