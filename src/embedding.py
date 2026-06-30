"""
chunks.json -> 벡터 임베딩 생성

- 모델: jhgan/ko-sroberta-multitask (한국어 특화 sentence-transformers)
- chunk_title + chunk_content를 합쳐 임베딩 (제목의 의미 정보도 함께 반영)
- 결과: embeddings.npy (벡터 배열) + chunks_with_embedding_meta.json (chunk 메타정보, 임베딩과 순서 동일)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sroberta-multitask"

INPUT_JSON = "data/chunks.json"
EMBEDDING_OUTPUT = "index/embeddings.npy"
META_OUTPUT = "data/chunks_meta.json"


def build_embedding_text(chunk):
    """제목 + 본문을 합쳐 임베딩 입력 텍스트 생성"""
    title = chunk.get("chunk_title", "")
    content = chunk.get("chunk_content", "")
    return f"{title}\n{content}"


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"총 chunk 수: {len(chunks)}")
    print(f"모델 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_embedding_text(c) for c in chunks]

    print("임베딩 생성 중...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 코사인 유사도 검색을 위해 정규화
    )

    print(f"임베딩 shape: {embeddings.shape}")

    # 임베딩 벡터 저장
    np.save(EMBEDDING_OUTPUT, embeddings)

    # 메타정보 저장 (임베딩과 같은 순서로 chunk_id, chunk_title, chunk_content 보존)
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "chunk_title": c["chunk_title"],
            "chunk_content": c["chunk_content"],
            "chunk_length": c.get("chunk_length", len(c["chunk_content"])),
        }
        for c in chunks
    ]
    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {EMBEDDING_OUTPUT}, {META_OUTPUT}")


if __name__ == "__main__":
    main()
