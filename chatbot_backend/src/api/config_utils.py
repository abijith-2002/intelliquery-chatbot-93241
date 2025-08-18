import os

# PUBLIC_INTERFACE
def get_gemini_api_key() -> str:
    """
    PUBLIC_INTERFACE
    Retrieve the Google Gemini API key from environment variables.

    Checks multiple common variable names in priority order to ensure compatibility
    with different deployment environments and build systems:

    1. GEMINI_API_KEY
    2. REACT_APP_GEMINI_API_KEY
    3. GOOGLE_API_KEY
    4. GOOGLE_GEMINI_API_KEY

    Returns:
        str: The API key string, or an empty string if none are set.
    """
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("REACT_APP_GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
        or ""
    )
