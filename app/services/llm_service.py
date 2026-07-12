import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-flash-latest")


def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an AI Research Assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I could not find that information in the uploaded document."

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text