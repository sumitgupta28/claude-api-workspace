from anthropic import Anthropic

client = Anthropic()
model = "claude-sonnet-4-6"
messages = []


def addUserMessage(messages, content):
    userMessage = {"role": "user", "content": content}
    messages.append(userMessage)
    return messages

def addAssistantMessage(messages, content):
    assistantMessage = {"role": "assistant", "content": content}
    messages.append(assistantMessage)
    return messages

def chat(messages):
    message = client.messages.create(
        model=model,
        messages=messages,
        max_tokens=16000,
    )
    return message.content[0].text

exit_message = "exit"

input_message = input("Enter your message: ")


while input_message.lower() != exit_message:
    addUserMessage(messages, input_message)
    response = chat(messages)
    print("Assistant:", response)
    addAssistantMessage(messages, response)
    input_message = input("Enter your message (or type 'exit' to quit): ")


