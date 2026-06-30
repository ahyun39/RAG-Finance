"""
embeddings.npy + chunks_meta.json -> FAISS 인덱스 구축 및 유사도 검색

- 정규화된 임베딩(코사인 유사도) 기준이므로 내적(IndexFlatIP) 사용
- 쿼리도 같은 모델로 임베딩 후 검색
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sroberta-multitask"

EMBEDDING_FILE = "index/embeddings.npy"
META_FILE = "data/chunks_meta.json"
INDEX_FILE = "index/faiss_index.bin"


def build_index(embedding_file=EMBEDDING_FILE, index_file=INDEX_FILE):
    """임베딩으로부터 FAISS 인덱스를 생성하고 파일로 저장"""
    embeddings = np.load(embedding_file).astype("float32")
    dim = embeddings.shape[1]

    # 내적(IP) 기반 인덱스 - 정규화된 벡터이므로 코사인 유사도와 동일
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)  # 정규화
    index.add(embeddings)

    faiss.write_index(index, index_file)
    print(f"인덱스 생성 완료: {index.ntotal}개 벡터, 차원={dim}")
    print(f"저장 완료: {index_file}")

    return index


def load_index(index_file=INDEX_FILE):
    return faiss.read_index(index_file)


def load_meta(meta_file=META_FILE):
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query, model, index, meta, top_k=5):
    """쿼리 텍스트로 유사한 chunk를 top_k개 검색"""
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = meta[idx]
        results.append({
            "chunk_id": item["chunk_id"],
            "chunk_title": item["chunk_title"],
            "chunk_content": item["chunk_content"],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    # 1. 인덱스 구축 (최초 1회 또는 임베딩 갱신 시)
    build_index()

    # 2. 검색 테스트
    print("\n모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)
    index = load_index()
    meta = load_meta()

    test_queries = [
        "예금자 보호 한도가 얼마인가요?",
        "보험 청약을 취소하고 싶어요",
        "대출 계약을 해지하는 방법",
    ]

    for q in test_queries:
        print(f"\n[검색어] {q}")
        results = search(q, model, index, meta, top_k=3)
        for r in results:
            print(f"  - ({r['score']:.4f}) {r['chunk_title']}")
            print(f"    {r['chunk_content'][:80]}...")
