import os
import chromadb
import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "voyage-3-large"
BATCH_SIZE = 25


def main():
    # 1. Initialize Voyage AI Client
    voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set. Export it before running ingest.py.")

    print("Initializing Voyage AI client...")
    vo = voyageai.Client()

    # 2. Connect to Chroma Cloud DB
    chroma_api_key = os.environ.get("CHROMA_CLOUD_API_KEY")
    chroma_tenant = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")
    chroma_host = os.environ.get("CHROMA_HOST", "api.trychroma.com")
    print(f"Connecting to Chroma Cloud {chroma_host} with tenant '{chroma_tenant}' and database '{database}'...")

    if not chroma_api_key:
        raise RuntimeError("CHROMA_CLOUD_API_KEY is not set. Export it before running ingest.py.")

    print("Connecting to Chroma Cloud...")
    client = chromadb.HttpClient(
        host=chroma_host,
        tenant=chroma_tenant,
        database=database,
        headers={"Authorization": f"Bearer {chroma_api_key}"},
    )
    print("Connected  to Chroma Cloud...")
    # 3. Create or Fetch your Remote Collection
    collection_name = "local_file_knowledge_base"
    print(f"Using Chroma collection: '{collection_name}'")
    collection = client.get_or_create_collection(name=collection_name)

    # 4. Read and Chunk the Local File
    file_path = "./report.md"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Read {len(raw_text)} characters from '{file_path}'.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"File successfully split into {len(chunks)} chunks.")

    if not chunks:
        print("No chunks to ingest. Exiting.")
        return

    print(f"Starting embedding and upload for {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    # 5. Generate Embeddings & Push to Chroma Cloud in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_index in range(total_batches):
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))
        batch_chunks = chunks[start:end]

        print(f"Processing batch {batch_index + 1}/{total_batches} ({len(batch_chunks)} chunks)...")

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for offset, chunk_text in enumerate(batch_chunks):
            global_index = start + offset
            print(f"  Embedding chunk {global_index + 1}/{len(chunks)}...")
            response = vo.embed([chunk_text], model=EMBED_MODEL, input_type="document")
            vector = response.embeddings[0]

            documents.append(chunk_text)
            embeddings.append(vector)
            metadatas.append({"source": file_path, "chunk_index": global_index})
            ids.append(f"id_chunk_{global_index}")

        print(f"Writing batch {batch_index + 1}/{total_batches} to Chroma...")
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write batch {batch_index + 1} to Chroma: {exc}") from exc

        print(f"Completed batch {batch_index + 1}/{total_batches}.")

    print(f"Successfully stored {len(chunks)} vectors in Chroma Cloud collection: '{collection_name}'!")


if __name__ == "__main__":
    main()
