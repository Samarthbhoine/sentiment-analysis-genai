import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_with_genai(text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"""
        Analyze the sentiment of this review and explain briefly:
        Review: {text}
        
        Give:
        - Sentiment (positive/negative)
        - Short explanation
        """
    )
    return response.text