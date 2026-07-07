from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-4-5-20251001"
messages = []


def addUserMessage(messages, content):
    userMessage = {"role": "user", "content": content}
    messages.append(userMessage)
    return messages

def addAssistantMessage(messages, content):
    assistantMessage = {"role": "assistant", "content": content}
    messages.append(assistantMessage)
    return messages

addUserMessage(messages, "Generate 3 different AWS CLI Commands.")
addAssistantMessage(messages, "here are all the three comamnds in a single block without any comments \n ```bash")
with client.messages.stream(
    model=model,
    messages=messages,
    max_tokens=16000,
    temperature=1.0,stop_sequences=["```"]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
        #pass
        
print("\n--- Response completed ---")
