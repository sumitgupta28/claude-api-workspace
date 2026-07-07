from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from the environment automatically (set in ~/.zshrc).
client = Anthropic()

# System prompt: sets Claude's behavior, tone, and role for the whole conversation.
# TODO: replace this placeholder with your actual instructions.
SYSTEM_PROMPT = "You are a helpful assistant."

# Stop sequences: custom strings that make Claude stop generating as soon as it
# produces one. When triggered, response.stop_reason == "stop_sequence" and
# response.stop_sequence names the match. Leave empty ([]) to disable.
# TODO: replace these placeholders with your actual stop sequences.
STOP_SEQUENCES = ["5"]


def createUserMessages(messages, text):
    """
    Append a user message to the conversation.

    Args:
        messages (list): The conversation to append to.
        text (str): The text content of the user message.
    """
    messages.append({"role": "user", "content": text})
    print("user: " + messages[-1]["content"])


def createAssistantMessages(messages, text):
    """
    Append an assistant message to the conversation.

    Args:
        messages (list): The conversation to append to.
        text (str): The text content of the assistant message.
    """
    messages.append({"role": "assistant", "content": text})
    print("assistant: " + messages[-1]["content"])

# Initialize once so the conversation history persists across turns.
messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break

    createUserMessages(messages, user_input)
    createAssistantMessages(messages, "```json")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            stop_sequences=STOP_SEQUENCES,
            messages=messages,
        )
    except Exception as e:
        print(f"Error: {e}")
        # Drop the unanswered user turn so history stays valid.
        messages.pop()
        continue

    reply = next(b.text for b in response.content if b.type == "text")
    createAssistantMessages(messages, reply)
    print("Claude: " + reply)
