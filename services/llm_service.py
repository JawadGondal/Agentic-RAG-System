from openai import OpenAI
from core.config import OPENAI_API_KEY, OPENAI_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def llm_answer(prompt: str) -> str:
    # simple single-turn chat completion using OpenAI's Chat API
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
        )
    return resp.choices[0].message.content