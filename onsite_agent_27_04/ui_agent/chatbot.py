import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv, find_dotenv

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


@cl.on_chat_start
async def chat_start():
    cl.user_session.set("history", [])
    await cl.Message("Hi there. How can I assist you today").send()


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    msg =cl.Message(content = "")
    await msg.send()

    response =Runner.run_streamed(starting_agent=greeting_agent, input=history)
    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta) 

    history.append({"role": "assistant", "content": response.final_output})
    cl.user_session.set("history", history)
    await msg.update()

    # Send a response back to the user
    # await cl.Message(
    #     content=f"Received: {response.final_output}",
    # ).send()
