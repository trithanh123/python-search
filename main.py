
import os
import re
from typing import Optional
from dotenv import load_dotenv

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, Range, MatchValue,
    PointStruct,
)

load_dotenv()
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "san_pham")

print("Cau hinh Gemini API...")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type=task_type
    )
    return result['embedding']

def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document"
    )
    return result['embedding']

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Service san sang!")

app = FastAPI(title="ToiYeuPC Search Service", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    branch_id: Optional[int] = None
    top_k: int = 5

class UpsertRequest(BaseModel):
    id: int                              
    masp: Optional[str] = ""
    tensp: str
    gia: int
    motasanpham: Optional[str] = ""
    specifications: Optional[dict] = {}
    ten_danhmuc: Optional[str] = ""



def parse_query(query: str) -> dict:
    filters  = {}
    semantic = query

    # Đơn vị giá: "triệu", "trieu", hoặc "tr" (standalone - không phải trong "triệu/trieu")
    UNIT = r'(?:tri[eệ]u|tr(?![a-zA-ZàáâãèéêìíòóôõùúýăđơưÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]))'

    # Bước 1: parse khoảng "từ A đến B triệu" -> gia_gte và gia_lte
    pattern_between = r'(?:từ|tu)?\s*(\d+)\s*(?:đến|den|-|tới|toi)\s*(\d+)\s*' + UNIT
    m_between = re.search(pattern_between, query, re.I)
    if m_between:
        filters["gia_gte"] = int(m_between.group(1)) * 1_000_000
        filters["gia_lte"] = int(m_between.group(2)) * 1_000_000
        semantic = re.sub(pattern_between, "", semantic, flags=re.I).strip()

    # Bước 2: parse "dưới/tối đa/không quá X triệu" → gia_lte
    pattern_lte = r'(?:gia\s*)?(?:duoi|toi da|khong qua|dưới|tối đa|không quá)\s*(\d+)\s*' + UNIT
    m = re.search(pattern_lte, query, re.I)
    if m and "gia_lte" not in filters:
        filters["gia_lte"] = int(m.group(1)) * 1_000_000
        semantic = re.sub(pattern_lte, "", semantic, flags=re.I).strip()

    # Bước 2: parse "trên/từ/tối thiểu X triệu" → gia_gte
    pattern_gte = r'(?:gia\s*)?(?:tren|tu|toi thieu|trên|từ|tối thiểu)\s*(\d+)\s*' + UNIT
    m = re.search(pattern_gte, query, re.I)
    if m:
        filters["gia_gte"] = int(m.group(1)) * 1_000_000
        semantic = re.sub(pattern_gte, "", semantic, flags=re.I).strip()

    # Bước 3: parse số đơn "khoảng X triệu" → ±5tr
    # Chỉ áp dụng nếu CHƯA có filter giá từ bước 1 và 2 (tránh override)
    if "gia_lte" not in filters and "gia_gte" not in filters:
        pattern_range = r'(?:gia\s*|khoang\s*|giá\s*|khoảng\s*)(\d+)\s*(?:' + UNIT + r'|củ)'
        m = re.search(pattern_range, query, re.I)
        if m:
            center = int(m.group(1)) * 1_000_000
            filters["gia_gte"] = max(0, center - 5_000_000)
            filters["gia_lte"] = center + 5_000_000
            semantic = re.sub(pattern_range, "", semantic, flags=re.I).strip()

    brands = ["ASUS", "MSI", "Dell", "HP", "Lenovo", "Acer", "Apple", "Gigabyte", "Samsung", "LG"]
    for brand in brands:
        if re.search(brand, query, re.I):
            filters["brand"] = brand
            break

    # Thay thế từ đồng nghĩa để AI hiểu tốt hơn
    synonyms = {
        r'\bmáy tính để bàn\b': 'PC desktop máy tính để bàn',
        r'\bmáy tính\b': 'PC máy tính',
        r'\bchơi game\b': 'gaming chơi game',
        r'\bvăn phòng\b': 'office văn phòng',
        r'\bđồ họa\b': 'render đồ họa'
    }
    for pattern, replacement in synonyms.items():
        semantic = re.sub(pattern, replacement, semantic, flags=re.I)

    semantic = re.sub(r'\s+', ' ', semantic).strip()
    # Nếu sau khi strip, semantic quá ngắn (<= 2 ký tự) thì fallback về query gốc
    if not semantic or len(semantic) <= 2:
        semantic = query
    return {"semantic": semantic, "filters": filters}


def build_qdrant_filter(filters: dict, branch_id: Optional[int]) -> Optional[Filter]:
    conditions = []

    if "gia_lte" in filters:
        conditions.append(FieldCondition(key="gia", range=Range(lte=filters["gia_lte"])))
    if "gia_gte" in filters:
        conditions.append(FieldCondition(key="gia", range=Range(gte=filters["gia_gte"])))
    if "brand" in filters:
        conditions.append(FieldCondition(key="brand", match=MatchValue(value=filters["brand"])))

    if not conditions:
        return None
    return Filter(must=conditions)


def make_product_text(data: dict) -> str:
    """Ghép các thông tin sản phẩm thành 1 đoạn văn để encode."""
    specs = data.get("specifications") or {}
    parts = [
        f"Tên sản phẩm: {data.get('tensp', '')}",
        f"Danh mục: {data.get('ten_danhmuc', '')}",
        f"Thương hiệu: {specs.get('brand', '')}"       if specs.get('brand')       else "",
        f"CPU: {specs.get('CPU') or specs.get('cpu', '')}"       if (specs.get('CPU') or specs.get('cpu')) else "",
        f"RAM: {specs.get('RAM') or specs.get('ram', '')}"       if (specs.get('RAM') or specs.get('ram')) else "",
        f"Ổ cứng: {specs.get('storage', '')}"          if specs.get('storage')     else "",
        f"Card đồ họa: {specs.get('gpu', '')}"         if specs.get('gpu')         else "",
        f"Bo mạch chủ: {specs.get('mainboard', '')}"   if specs.get('mainboard')   else "",
        f"Bộ nguồn: {specs.get('psu', '')}"            if specs.get('psu')         else "",
        f"Vỏ case: {specs.get('case', '')}"            if specs.get('case')        else "",
        f"Màn hình: {specs.get('screen_size', '')}"    if specs.get('screen_size') else "",
        f"Mục đích: {specs.get('use_case', '')}"       if specs.get('use_case')    else "",
        f"Mô tả: {data.get('motasanpham', '')}",
    ]
    return ". ".join(p for p in parts if p).strip()



@app.get("/health")
def health():
   
    return {"status": "ok", "service": "ToiYeuPC Search"}


@app.post("/search")
def search(req: SearchRequest):
    """Tìm kiếm sản phẩm bằng AI semantic search."""
    parsed       = parse_query(req.query)
    semantic     = parsed["semantic"]
    filters      = parsed["filters"]
    query_vector = get_embedding(semantic, task_type="retrieval_query")
    qdrant_filter = build_qdrant_filter(filters, req.branch_id)

    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=req.top_k,
        with_payload=False,
    )
    results = [{"id": hit.id, "score": round(hit.score, 4)} for hit in hits]

    return {
        "query"   : req.query,
        "semantic": semantic,
        "filters" : filters,
        "results" : results,
    }


@app.post("/upsert")
def upsert(req: UpsertRequest):
    data   = req.dict()
    text   = make_product_text(data)
    vector = get_embedding(text, task_type="retrieval_document")

    payload = {
        "masp"          : req.masp,
        "tensp"         : req.tensp,
        "gia"           : req.gia,
        "ten_danhmuc"   : req.ten_danhmuc,
        "brand"         : (req.specifications or {}).get("brand", ""),
        "specifications": req.specifications or {},
    }

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id      = req.id,
                vector  = vector,
                payload = payload,
            )
        ],
    )

    return {
        "status"   : "success",
        "message"  : f"Đã upsert embedding cho sản phẩm id={req.id}",
        "text_used": text,
    }


class UpsertBatchRequest(BaseModel):
    products: list[UpsertRequest]

@app.post("/upsert-batch")
def upsert_batch(req: UpsertBatchRequest):
    if not req.products:
        return {"status": "success", "message": "No products to upsert"}
        
    texts = [make_product_text(p.dict()) for p in req.products]
    vectors = get_embedding_batch(texts)
    
    points = []
    for p, vector in zip(req.products, vectors):
        payload = {
            "masp"          : p.masp,
            "tensp"         : p.tensp,
            "gia"           : p.gia,
            "ten_danhmuc"   : p.ten_danhmuc,
            "brand"         : (p.specifications or {}).get("brand", ""),
            "specifications": p.specifications or {},
        }
        points.append(PointStruct(id=p.id, vector=vector, payload=payload))
        
    qdrant.upsert(
        collection_name=COLLECTION,
        points=points,
    )
    
    return {
        "status" : "success",
        "message": f"Đã upsert embedding cho {len(req.products)} sản phẩm",
    }

@app.delete("/delete/{product_id}")
def delete_product(product_id: int):
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=[product_id],
    )

    return {
        "status" : "success",
        "message": f"Đã xóa embedding của sản phẩm id={product_id}",
    }

class AiBuildRequest(BaseModel):
    query: str

from ai_builder import find_best_pc_build

# Patterns to detect specific component model mentions in a query
# CPU patterns: i3/i5/i7/i9 + model, Ryzen X XXXX
_CPU_PATTERN = re.compile(
    r'\b(i[3579]-?\s*\d{4,5}[A-Z]*|ryzen\s*[359]\s*\d{4,5}[A-Z]*|core\s*i[3579]-?\s*\d{4,5}[A-Z]*)\b',
    re.I
)
# VGA patterns: RTX/GTX/RX + model
_VGA_PATTERN = re.compile(
    r'\b(rtx\s*\d{3,4}(?:\s*ti)?(?:\s*super)?|gtx\s*\d{3,4}(?:\s*ti)?|rx\s*\d{3,4}(?:\s*xt)?)\b',
    re.I
)

def _search_by_keyword(keyword: str, comp_type: str, limit: int = 5) -> list:
    """Search Qdrant for products matching keyword in name, filtered by component type."""
    try:
        results = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="specifications.loai", match=MatchValue(value=comp_type))
            ]),
            limit=200,
            with_payload=True,
            with_vectors=False
        )
        keyword_lower = keyword.lower().replace(' ', '')
        matched = []
        for point in results[0]:
            name = point.payload.get('tensp', '').lower().replace(' ', '').replace('-', '')
            kw = keyword_lower.replace('-', '')
            if kw in name:
                matched.append({'id_sanpham': point.id, **point.payload})
        return matched[:limit]
    except Exception:
        return []

@app.post("/ai-build-pc")
def ai_build_pc(req: AiBuildRequest):
    query = req.query
    parsed = parse_query(query)
    
    # Extract EXACT budget from query to avoid +5M tolerance from search parser
    m = re.search(r'(\d+)\s*(?:tri[eệ]u|tr)', query, re.I)
    if m:
        budget = int(m.group(1)) * 1_000_000
    else:
        budget = 30_000_000
            
    semantic = parsed["semantic"]
    vector = get_embedding(semantic, task_type="retrieval_query")

    # Detect explicitly mentioned CPU/VGA models
    pinned_cpu_kw = _CPU_PATTERN.search(query)
    pinned_vga_kw = _VGA_PATTERN.search(query)
    
    components = ["CPU", "Mainboard", "VGA", "RAM", "SSD", "PSU", "Case"]
    components_by_type = {}
    
    for comp_type in components:
        conditions = [FieldCondition(key="specifications.loai", match=MatchValue(value=comp_type))]
        if "brand" in parsed["filters"]:
            conditions.append(FieldCondition(key="brand", match=MatchValue(value=parsed["filters"]["brand"])))
            
        qdrant_filter = Filter(must=conditions)
        
        try:
            res = qdrant.search(
                collection_name=COLLECTION,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=15
            )
            # Include the numeric DB id (Qdrant point id) in every payload so Laravel
            # can look up the fresh price from MySQL instead of relying on stale Qdrant data.
            semantic_results = [{'id_sanpham': h.id, **h.payload} for h in res]
        except Exception:
            semantic_results = []

        # If user mentioned a specific CPU/VGA model, pin it to the TOP of results
        if comp_type == "CPU" and pinned_cpu_kw:
            kw = pinned_cpu_kw.group(1)
            pinned_results = _search_by_keyword(kw, "CPU")
            if pinned_results:
                existing_masps = {p.get('masp') for p in pinned_results}
                semantic_results = pinned_results + [p for p in semantic_results if p.get('masp') not in existing_masps]

        elif comp_type == "VGA" and pinned_vga_kw:
            kw = pinned_vga_kw.group(1)
            pinned_results = _search_by_keyword(kw, "VGA")
            if pinned_results:
                existing_masps = {p.get('masp') for p in pinned_results}
                semantic_results = pinned_results + [p for p in semantic_results if p.get('masp') not in existing_masps]

        components_by_type[comp_type] = semantic_results

    # Build the pinned dict to tell ai_builder which components are explicitly requested
    pinned_for_builder = {}
    if pinned_cpu_kw and components_by_type.get('CPU'):
        pinned_for_builder['CPU'] = 1   # first CPU in list is pinned
    if pinned_vga_kw and components_by_type.get('VGA'):
        pinned_for_builder['VGA'] = 1   # first VGA in list is pinned

    best_build = find_best_pc_build(budget, components_by_type, pinned=pinned_for_builder)
    
    if not best_build:
        raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình phù hợp với ngân sách.")
        
    return {
        "budget_requested": budget,
        "total_price": sum(c.get('gia', 0) for c in best_build),
        "build": best_build
    }
