# Load env variables and create client
import json

from anthropic import Anthropic
from anthropic.types import ToolParam,Message

# Tools and Schemas

from datetime import datetime, timedelta


client = Anthropic()
model = "claude-haiku-4-5-20251001"



# Helper functions
def add_user_message(messages, message):
    user_message = {"role": "user", "content": message.content if isinstance(message, Message) else message}
    messages.append(user_message)


def add_assistant_message(messages, message):
    assistant_message = {"role": "assistant", "content": message.content if isinstance(message, Message) else message}
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=None, tools=None,
    thinking=False,
    thinking_budget=1024):
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
    }

    if thinking:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }

    if system:
        params["system"] = system

    if tools:
        params["tools"] = tools

    message = client.messages.create(**params)
    # content[0] is only a text block when the model returns text; a tool_use
    # (or empty) response would raise on `.text`, so pick the first text block.
    return message


def text_from_message(message):
    return "\n".join(
        [block.text for block in message.content if block.type == "text"]
    )


if __name__ == "__main__":
    
    messages = []
    add_user_message(messages, "write a one pararaph guide to recursion in python")
    chat_response = chat(messages, thinking=True, thinking_budget=1024)


    print(chat_response)



    