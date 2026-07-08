import os
import json
import ollama
from dotenv import load_dotenv
from pypdf import PdfReader
from pymongo import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()


# 1. Connect to Local MongoDB
# DB Name: vectorless_rag_db | Collection: document_structures

uri = os.getenv("MONGODB_URI")

mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["vectorless_rag_db"]
documents_collection = db["document_structures"]

def build_and_cache_document_tree(pdf_path: str, model_name: str) -> dict:
    """
    Checks MongoDB for an existing parsed document tree by filename.
    If missing, parses the PDF page-by-page using Ollama and stores it.
    """
    filename = os.path.basename(pdf_path)
    
    # Check MongoDB Cache First
    cached_doc = documents_collection.find_one({"filename": filename})
    if cached_doc:
        print(f"[*] Found cached structural map in MongoDB for: {filename}")
        return cached_doc

    print(f"[*] Cache miss. Processing local PDF via Ollama: {filename}")
    reader = PdfReader(pdf_path)
    
    document_tree = {
        "filename": filename,
        "sections": []
    }
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        text_content = raw_text.strip()
        
        if not text_content:
            continue
            
        # Call local Ollama to get a 1-sentence page summary
        summary_prompt = f"""
        Analyze the following text from Page {page_num} of a document.
        Provide a 1-sentence, high-level summary of the main topic or concepts covered on this page.
        
        Text:
        {text_content[:1500]}
        
        Summary:
        """
        
        res = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        page_summary = res['message']['content'].strip()
        print(f"    -> Indexed Page {page_num}: {page_summary[:60]}...")
        
        # Structure payload for native MongoDB insertion
        document_tree["sections"].append({
            "id": f"page_{page_num}",
            "title": f"Page {page_num}",
            "summary": page_summary,
            "content": text_content
        })
    
    # Save the parsed structure to MongoDB for future queries
    documents_collection.insert_one(document_tree)
    print(f"[*] Successfully indexed and stored structural map in MongoDB.")
    return document_tree


def mongo_vectorless_rag(pdf_path: str, query: str, model_name: str = "llama3.2") -> str:
    # 2. Retrieve document metadata tree from MongoDB or build it
    document_tree = build_and_cache_document_tree(pdf_path, model_name)
    
    # 3. Create lightweight routing structural map (Omits heavy full-text content field to keep context clear)
    structural_map = [
        {"id": sec["id"], "title": sec["title"], "summary": sec["summary"]}
        for sec in document_tree["sections"]
    ]
    
    # 4. Routing Phase: Ask local model to find the correct structural page ID
    router_prompt = f"""
    Analyze the document page map and identify the exact 'id' of the page needed to answer the user query.
    
    Document Map:
    {json.dumps(structural_map, indent=2)}
    
    User Query: {query}
    
    Output ONLY the precise target page id string (e.g., page_1) and absolutely nothing else.
    """
    
    router_res = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": router_prompt}],
        options={"temperature": 0.0}  # Force deterministic routing choice
    )
    
    target_id = router_res['message']['content'].strip()
    print(f"\n[*] [Router Log] LLM selected target structural node: {target_id}")
    
    # 5. Native Database Content Extraction 
    # Finds the specific nested page section inside the matching MongoDB structural document
    matched_section = next((s for s in document_tree["sections"] if s["id"] in target_id), None)
    
    if not matched_section:
        return f"Error: Local router returned a page node boundary not found in the DB: '{target_id}'."
        
    # 6. Generation Phase: Target text synthesis
    generation_prompt = f"""
    Answer the user query comprehensively using exclusively the source document text below.
    Source: {matched_section['title']}
    
    Context Data:
    {matched_section['content']}
    
    Query: {query}
    """
    
    final_res = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": generation_prompt}]
    )
    
    return final_res['message']['content']


# --- Execution Example ---
if __name__ == "__main__":
    # Point to your local file path
    MY_PDF = "462760_Sumit_Kumar_Gupta_Profile_v1.pdf" 
    QUESTION = "What are the termination notice requirements?"
    
    # Execute Vectorless RAG powered by MongoDB Persistence
    answer = mongo_vectorless_rag(pdf_path=MY_PDF, query=QUESTION, model_name="llama3.2")
    
    print("\n" + "="*40 + "\nFINAL ANSWER:\n" + "="*40)
    print(answer)
