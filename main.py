
import os
import re
from typing import Optional
from dotenv import load_dotenv

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, Range, MatchValue,
    PointStruct,
)

load_dotenv()
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "san_pham")

print(" Đang load model vietnamese-sbert...")
model  = SentenceTransformer("keepitreal/vietnamese-sbert")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print(" Service sẵn sàng!")

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

    m = re.search(r'(?:gia\s*)?(?:duoi|toi da|khong qua|dưới|tối đa|không quá)\s*(\d+)\s*(?:triệu|tr)', query, re.I)
    if m:
        filters["gia_lte"] = int(m.group(1)) * 1_000_000
        semantic = re.sub(m.re, "", semantic).strip()

    m = re.search(r'(?:gia\s*)?(?:tren|tu|toi thieu|trên|từ|tối thiểu)\s*(\d+)\s*(?:triệu|tr)', query, re.I)
    if m:
        filters["gia_gte"] = int(m.group(1)) * 1_000_000
        semantic = re.sub(m.re, "", semantic).strip()

    m = re.search(r'(?:gia\s*|khoang\s*|giá\s*|khoảng\s*)?(\d+)\s*(?:triệu|tr|củ)', query, re.I)
    if m:
        center = int(m.group(1)) * 1_000_000
        filters["gia_gte"] = center - 5_000_000
        filters["gia_lte"] = center + 5_000_000
        semantic = re.sub(m.re, "", semantic).strip()

    brands = ["ASUS", "MSI", "Dell", "HP", "Lenovo", "Acer", "Apple", "Gigabyte", "Samsung", "LG"]
    for brand in brands:
        if re.search(brand, query, re.I):
            filters["brand"] = brand
            break

    semantic = re.sub(r'\s+', ' ', semantic).strip()
    if not semantic:
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
    query_vector = model.encode(semantic).tolist()
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
    vector = model.encode(text).tolist()

    payload = {
        "masp"        : req.masp,
        "tensp"       : req.tensp,
        "gia"         : req.gia,
        "ten_danhmuc" : req.ten_danhmuc,
        "brand"       : (req.specifications or {}).get("brand", ""),
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
