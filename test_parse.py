# encoding: utf-8
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('d:/LuanVan/python-search')
from main import parse_query

q = "máy tính chơi game dưới 30 triệu"
res = parse_query(q)
print("Result for:", q)
print(res)
