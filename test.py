import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()   # <-- ADD THIS LINE

print("KEY:", os.environ.get("GROQ_API_KEY"))

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
)

response = agent.run("Say hello in one word.")
print(response.content)