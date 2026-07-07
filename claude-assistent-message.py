from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from the environment automatically (set in ~/.zshrc).
client = Anthropic()

# System prompt: sets Claude's behavior, tone, and role for the whole conversation.
# TODO: replace this placeholder with your actual instructions.
SYSTEM_PROMPT = "You are a helpful assistant."


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

    The text is not printed here — with streaming it has already been
    displayed token-by-token as it arrived.

    Args:
        messages (list): The conversation to append to.
        text (str): The text content of the assistant message.
    """
    messages.append({"role": "assistant", "content": text})

# Initialize once so the conversation history persists across turns.
messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break

    createUserMessages(messages, user_input)
    createAssistantMessages(messages, "```json")

    try:
        print("Claude: ", end="", flush=True)
        reply_parts = []
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                reply_parts.append(text)
        print()  # newline after the streamed reply
    except Exception as e:
        print(f"\nError: {e}")
        # Drop the unanswered user turn so history stays valid.
        messages.pop()
        continue

    createAssistantMessages(messages, "".join(reply_parts))
