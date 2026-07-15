class Prompts:
    # --- P1 & P2: Direct Answering (BM25 / Dense) ---
    DIRECT_QA_SYSTEM = """You are a helpful assistant. You are given a list of retrieved documents.
Answer the user's question based strictly on the provided documents.
If the answer is not present, reply with "Information not found." """

    DIRECT_QA_USER = """Documents:
{context}

Question:
{question}
"""

    # --- P3: Extraction & Aggregation ---
    EXTRACTION_SYSTEM = """You are a data extraction assistant. You are given a document and a query.
Extract the fields required by the JSON schema. Be precise.
Extract exact text evidence where required.
If a field cannot be answered from the document, set is_missing to true. """

    EXTRACTION_USER = """Document:
{document}

Query:
{query}
"""

def build_direct_prompt(question: str, context: str) -> str:
    return Prompts.DIRECT_QA_SYSTEM + "\n\n" + Prompts.DIRECT_QA_USER.format(context=context, question=question)

def build_extract_prompt(document: str, query: str) -> str:
    return Prompts.EXTRACTION_SYSTEM + "\n\n" + Prompts.EXTRACTION_USER.format(document=document, query=query)

