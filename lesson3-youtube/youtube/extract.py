import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from IPython.display import Markdown, display

def analyze_youtube_video(youtube_url: str, prompt: str) -> str:
    """
    Analyze a YouTube video using Google's Generative AI.
    
    Args:
        youtube_url (str): The URL of the YouTube video to analyze
        prompt (str): The prompt to guide the analysis
        
    Returns:
        str: The analysis response from the model
    """
    # Load environment variables
    load_dotenv()

    # Get API key from environment variable
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    # create client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Analyze the video
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=types.Content(
            parts=[
                types.Part(text=prompt),
                types.Part(
                    file_data=types.FileData(file_uri=youtube_url)
                )
            ]
        )
    )
    
    return response.text

if __name__ == "__main__":
    # Example usage
    youtube_url = "https://www.youtube.com/watch?v=LMhe2egLsrQ"
    
    # Example prompts
    summary_prompt = """Analyze the following YouTube video content. Provide a concise summary covering:

    1.  **Main Thesis/Claim:** What is the central point the creator is making?
    2.  **Key Topics:** List the main subjects discussed, referencing specific examples or technologies mentioned (e.g., AI models, programming languages, projects).
    3.  **Call to Action:** Identify any explicit requests made to the viewer.
    4.  **Summary:** Provide a concise summary of the video content.

    Use the provided title, chapter timestamps/descriptions, and description text for your analysis."""

    code_prompt = """Analyze the following YouTube video content. Please extract:

    1.  Code in that is shown in Colab/Jupyter notebook not in documentation

    return the code in the format ```code```"""

    # Get summary
    summary = analyze_youtube_video(youtube_url, summary_prompt)
    print("Summary Analysis:")
    print(summary)
    print("\n" + "="*50 + "\n")

    # Get code
    code = analyze_youtube_video(youtube_url, code_prompt)
    print("Code Analysis:")
    print(code)