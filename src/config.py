import os

# Google Gemini API Configuration
# GOOGLE_API_KEY = "AIzaSyDCTicTU1DkVOlq9Niyl_i76MXI7MkTwto"

# LangChain API Configuration
LANGCHAIN_API_KEY = "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"

# Set environment variables


def setup_environment():
    """Setup environment variables for API keys"""
    # if GOOGLE_API_KEY != "AIzaSyDCTicTU1DkVOlq9Niyl_i76MXI7MkTwto":
    #     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    if LANGCHAIN_API_KEY != "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":
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
