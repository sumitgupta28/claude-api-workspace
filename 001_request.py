from anthropic import Anthropic

client = Anthropic()
model = "claude-sonnet-4-6"
messages = []
initial_message = "what is quantum computing? answer in 1 sentences."


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

addUserMessage(messages, initial_message)
response = chat(messages)
print(response)
addAssistantMessage(messages, response)
addUserMessage(messages, "can you explain it in 1 paragraph?")
response = chat(messages)
print(response)



