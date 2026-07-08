import json
import ollama
from pypdf import PdfReader

def build_pdf_document_tree(pdf_path: str, model_name: str) -> dict:
    """
    Parses the local PDF page by page and uses Ollama to generate 
    a structural index map (Table of Contents) dynamically.
    """
    print(f"[*] Processing PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    document_tree = {
        "doc_title": pdf_path,
        "sections": []
    }
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        
        # Clean text slightly for processing
        text_content = raw_text.strip()
        if not text_content:
            continue
            
        # Get a quick 1-sentence summary of the page from the local model
        summary_prompt = f"""
        Analyze the following text from Page {page_num} of a document.
        Provide a 1-sentence, high-level summary of the main topic or concepts covered on this page.
        
        Text:
        {text_content[:1500]} # Send a snippet to keep execution fast
        
        Summary:
        """
        
        res = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        page_summary = res['message']['content'].strip()
        print(f"    -> Indexed Page {page_num}: {page_summary[:1000]}...")
        
        # Append to our local non-vector structure map
        document_tree["sections"].append({
            "id": f"page_{page_num}",
            "title": f"Page {page_num}",
            "summary": page_summary,
            "content": text_content # Stores the full literal text of the page
        })
        
    return document_tree


def local_pdf_vectorless_rag(pdf_path: str, query: str, model_name: str = "llama3.2") -> str:
    # 1. Dynamically build the structural table-of-contents from the PDF
    document_tree = build_pdf_document_tree(pdf_path, model_name)
    
    # 2. Extract only structural summaries to fit easily into the LLM context window
    structural_map = [
        {"id": sec["id"], "title": sec["title"], "summary": sec["summary"]}
        for sec in document_tree["sections"]
    ]
    
    print("structural_map:", json.dumps(structural_map, indent=2))

    # 3. Routing Phase: Ask local model to act as a human structural locator
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
        options={"temperature": 0.0} # Lower temperature for stable routing
    )
    
    target_id = router_res['message']['content'].strip()
    print(f"\n[*] [Router Log] LLM navigated to target location: {target_id}")
    
    # 4. Straight Dictionary Retrieval
    matched_section = next((s for s in document_tree["sections"] if s["id"] in target_id), None)
    
    if not matched_section:
        return f"Error: Local router returned a non-existent page boundary context: '{target_id}'."
        
    # 5. Generation Phase: Read the exact page data into memory and synthesize the final answer
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
    # Replace with the name of your local PDF file placed in the same directory
    MY_PDF = "462760_Sumit_Kumar_Gupta_Profile.pdf" 
    QUESTION = "what's sumit's education and work experience?"
    
    # Run pipeline entirely on your local machine
    answer = local_pdf_vectorless_rag(pdf_path=MY_PDF, query=QUESTION, model_name="llama3.2")
    
    print("\n" + "="*40 + "\nFINAL ANSWER:\n" + "="*40)
    print(answer)
