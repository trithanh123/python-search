
import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
load_dotenv()
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
SUPABASE_DB_URL   = os.getenv("SUPABASE_DB_URL")
COLLECTION        = os.getenv("QDRANT_COLLECTION", "san_pham")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", 768))
print(" Đang load model vietnamese-sbert...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")
print(" Model đã sẵn sàng!")

def create_product_text(product: dict) -> str:
    specs = product.get("specifications") or {}
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except Exception:
            specs = {}
    cpu       = specs.get("cpu", "")
    ram       = specs.get("ram", "")
    storage   = specs.get("storage", "")       
    gpu       = specs.get("gpu", "")            
    mainboard = specs.get("mainboard", "")     
    psu       = specs.get("psu", "")            
    case      = specs.get("case", "")           
    use_case  = specs.get("use_case", "")      
    brand     = specs.get("brand", "")          
    parts = [
        f"Tên sản phẩm: {product.get('tensp', '')}",
        f"Danh mục: {product.get('ten_danhmuc', '')}",
        f"Thương hiệu: {brand}"         if brand     else "",
        f"CPU: {cpu}"                   if cpu       else "",
        f"RAM: {ram}"                   if ram       else "",
        f"Ổ cứng SSD/HDD: {storage}"   if storage   else "",
        f"Card đồ họa: {gpu}"           if gpu       else "",
        f"Bo mạch chủ: {mainboard}"     if mainboard else "",
        f"Bộ nguồn: {psu}"              if psu       else "",
        f"Vỏ case: {case}"              if case      else "",
        f"Mục đích sử dụng: {use_case}" if use_case  else "",
        f"Mô tả: {product.get('motasanpham', '')}",
    ]
    return ". ".join(p for p in parts if p).strip()


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        print(f" Tạo collection '{COLLECTION}' trong Qdrant...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,   
            ),
        )
        print(f" Collection '{COLLECTION}' đã được tạo!")
    else:
        print(f" Collection '{COLLECTION}' đã tồn tại, dùng luôn.")

def fetch_products(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT
                sp.id_sanpham,
                sp.masp,
                sp.tensp,
                sp.gia,
                sp.motasanpham,
                sp.specifications,
                dm.ten_danhmuc
            FROM san_pham sp
            LEFT JOIN danh_muc dm ON sp.ma_danhmuc = dm.id_danhmuc
            ORDER BY sp.id_sanpham
        """)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def index_all():
    print(" Kết nối database...")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    print(" Kết nối database thành công!")
    print(" Kết nối Qdrant Cloud...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    print(" Kết nối Qdrant thành công!")
    ensure_collection(qdrant)
    print(" Đang đọc sản phẩm từ database...")
    products = fetch_products(conn)
    conn.close()
    print(f" Tìm thấy {len(products)} sản phẩm.")

    if not products:
        print("  Không có sản phẩm nào để index!")
        return
    print("  Đang tạo văn bản mô tả sản phẩm...")
    texts = [create_product_text(p) for p in products]
    print(f" Đang tạo vector bằng vietnamese-sbert (batch {len(texts)} sản phẩm)...")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)
    print(" Đã tạo xong tất cả vector!")
    print("⬆ Đang đẩy dữ liệu vào Qdrant Cloud...")
    points = []
    for product, vector in zip(products, vectors):
        specs = product.get("specifications") or {}
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except Exception:
                specs = {}

        points.append(
            PointStruct(
                id=int(product["id_sanpham"]),
                vector=vector.tolist(),
                payload={
            
                    "id_sanpham"   : int(product["id_sanpham"]),
                    "masp"         : product["masp"],
                    "tensp"        : product["tensp"],
                    "gia"          : int(product["gia"] or 0),          
                    "ten_danhmuc"  : product.get("ten_danhmuc", ""),
                    "brand"        : specs.get("brand", ""),       
                    "ram"          : specs.get("ram", ""),         
                    "gpu"          : specs.get("gpu", ""),         
                }
            )
        )
    BATCH = 100
    for i in range(0, len(points), BATCH):
        chunk = points[i:i + BATCH]
        qdrant.upsert(collection_name=COLLECTION, points=chunk)
        print(f"   Đã upload {min(i + BATCH, len(points))}/{len(points)} sản phẩm")

    print(f"\n Hoàn tất! {len(points)} sản phẩm đã được index vào Qdrant!")
    print(f"   Collection: {COLLECTION}")
    print(f"   Qdrant URL: {QDRANT_URL}")


if __name__ == "__main__":
    index_all()
