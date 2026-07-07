# Client Setup
import voyageai
# Chunk by section
import re

client = voyageai.Client()




def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)

# Embedding Generation
def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)

    return result.embeddings[0]

with open("./report.md", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)

print("chunks: ", len(chunks))
print("chunks: ", chunks[0])

embeddings = generate_embedding(chunks[0])
print("embeddings: ", embeddings)