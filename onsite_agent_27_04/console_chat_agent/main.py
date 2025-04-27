from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
)
from dotenv import load_dotenv, find_dotenv
import os

_: bool = load_dotenv(find_dotenv())

set_tracing_disabled(True)

gemini_key = os.getenv("GOOGLE_API_KEY")

gemini_provider = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key,
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_provider
)

greeting_agent = Agent(
    name="Greeting Agent",
    model=gemini_model,
    instructions="""
You are a helpful assistant that greets the user.
""",
)


def main():
    history = []
    while True:
        user_input = input("Enter a message : ")
        history.append({"role": "user", "content": user_input})
        response = Runner.run_sync(starting_agent=greeting_agent, input=history)
        history.append({"role": "assistant", "content": response.final_output})
        print(response.final_output)


main()
