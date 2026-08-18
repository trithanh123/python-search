import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

semantic = "PC máy tính gaming chơi game"
try:
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=semantic,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    vector = result['embedding']
    
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    COLLECTION = os.getenv("QDRANT_COLLECTION", "san_pham")
    
    qdrant_filter = Filter(must=[
        FieldCondition(key="gia", range=Range(gte=35000000)),
        FieldCondition(key="gia", range=Range(lte=45000000))
    ])
    
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=10,
        with_payload=True,
    )
    
    for h in hits:
        print(f"ID: {h.id}, Score: {h.score}, Name: {h.payload.get('tensp')}")
    if not hits:
        print("No hits found!")
        
except Exception as e:
    print("Error:", e)
