import os
import chromadb
import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize Voyage AI Client
# Make sure VOYAGE_API_KEY is available in your environment variables
vo = voyageai.Client()
EMBED_MODEL = "voyage-3-large"  # You can also use "voyage-3-large" depending on your needs

# 2. Connect to Chroma Cloud DB
# Chroma Cloud uses the HttpClient with your hosted authentication tokens
client = chromadb.HttpClient(
    host="api.trychroma.com",  # Standard Chroma Cloud production host
    tenant=os.environ.get("CHROMA_TENANT", "default_tenant"),
    database=os.environ.get("CHROMA_DATABASE", "default_database"),
    headers={"Authorization": f"Bearer {os.environ.get('CHROMA_CLOUD_API_KEY')}"}
)

# 3. Create or Fetch your Remote Collection
collection_name = "local_file_knowledge_base"
collection = client.get_or_create_collection(name=collection_name)

# 4. Read and Chunk the Local File
file_path = "./report.md"  # Replace with your actual local file path

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Using Recursive text splitting to preserve context around paragraphs and sentences
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Targets ~1000 characters per chunk
    chunk_overlap=100      # 100 character overlap prevents split sentences from losing context
)
chunks = text_splitter.split_text(raw_text)
print(f"File successfully split into {len(chunks)} chunks.")

# 5. Generate Embeddings & Push to Chroma Cloud
# Prepare batches for API insertion limits
documents = []
embeddings = []
metadatas = []
ids = []

for i, chunk_text in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}...")
    # Generate the vector embedding using Voyage AI
    # Specifying input_type="document" optimizes the embedding vectors for retrieval
    response = vo.embed([chunk_text], model=EMBED_MODEL, input_type="document")
    vector = response.embeddings[0]
    
    # Bundle items for insertion
    documents.append(chunk_text)
    embeddings.append(vector)
    metadatas.append({"source": file_path, "chunk_index": i})
    ids.append(f"id_chunk_{i}")

# Add all components directly into your cloud-hosted collection
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"Successfully stored {len(chunks)} vectors in Chroma Cloud collection: '{collection_name}'!")
