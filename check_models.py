from google import genai
from config import config

client = genai.Client(api_key=config.GOOGLE_API_KEY)

print("Models that support generateContent:\n")
for m in client.models.list():
    if hasattr(m, 'supported_actions') and 'generateContent' in (m.supported_actions or []):
        print(f"  {m.name}")