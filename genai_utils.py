import google.generativeai as genai

genai.configure(api_key="AIzaSyA7PwdUsu4Ouqg059XKLziON0AHpJguLtE")

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