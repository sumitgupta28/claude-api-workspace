from platform import system

from anthropic import Anthropic

client = Anthropic()
model = "claude-sonnet-4-6"
messages = []

system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""



def addUserMessage(messages, content):
    userMessage = {"role": "user", "content": content}
    messages.append(userMessage)
    return messages

def addAssistantMessage(messages, content):
    assistantMessage = {"role": "assistant", "content": content}
    messages.append(assistantMessage)
    return messages

def chat(messages,system=None,temperature=1.0):

    parameters = {
        "model": model,
        "messages": messages,
        "max_tokens": 16000,
        "temperature": temperature,
        "stream": True
    }

    if system is not None:
        parameters["system"] = system

    message = client.messages.create(**parameters)
    return message.content[0].text


exit_message = "exit"
input_message = input("Enter your message: ")


while input_message.lower() != exit_message:
    addUserMessage(messages, input_message)
    response = chat(messages)
    print("Assistant:", response)
    addAssistantMessage(messages, response)
    input_message = input("Enter your message (or type 'exit' to quit): ")


