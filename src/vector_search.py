"""
embeddings.npy + chunks_meta.json -> FAISS 인덱스 구축 및 유사도 검색

- 정규화된 임베딩(코사인 유사도) 기준이므로 내적(IndexFlatIP) 사용
- 쿼리도 같은 모델로 임베딩 후 검색
"""

import json
import re

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 설정값은 src/config.py 한 곳에서만 정의한다 (중복 상수가 실제로 어긋난 이력이 있다)
from config import (MODEL_NAME, EMBEDDING_FILE, META_FILE, INDEX_FILE, TOP_K,
                    RRF_K, LEXICAL_WEIGHT, LEXICAL_WEIGHT_STATUTE,
                    ANSWER_THRESHOLD, CONTEXT_FLOOR)


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


def search(query, model, index, meta, top_k=TOP_K):
    """쿼리 텍스트로 유사한 chunk를 top_k개 검색"""
    # FAISS가 돌려준 행 번호로 meta를 조회하므로 둘의 개수·순서가 어긋나면 안 된다.
    # meta가 더 길면 예외 없이 '엉뚱한 청크'가 조용히 반환되므로 여기서 막는다.
    assert index.ntotal == len(meta), (
        f"인덱스({index.ntotal})와 메타({len(meta)})의 개수가 다릅니다. "
        "embedding.py -> vector_search.py를 다시 실행해 함께 재생성하세요."
    )

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


# ===== 어휘 검색 (BM25) =====
# 밀집 벡터는 `제39조의2` 같은 저빈도 정확 토큰의 신호를 풀링 과정에서 잃는다.
# BM25는 그 반대라서 둘을 합치면 서로의 약점을 덮는다.

_WORD = re.compile(r"[가-힣A-Za-z0-9]+")
# "제39조의2제1항" 처럼 조문 번호는 붙여 쓰므로 원자 단위로 쪼개 둔다.
_STATUTE = re.compile(r"제\d+조(?:의\d+)?|제\d+항|제\d+호")
# 형태소 분석기 없이 흔한 조사만 떼어 낸다. 긴 것부터 검사해야 "에서"가 "에"로 잘리지 않는다.
_JOSA = tuple(sorted(
    ("으로서", "으로써", "에서는", "에게서", "이라는", "까지", "부터", "에서", "에게",
     "으로", "라는", "이나", "에는", "와의", "과의", "은", "는", "이", "가", "을",
     "를", "의", "에", "로", "와", "과", "도", "만"),
    key=len, reverse=True))


def tokenize(text):
    """BM25용 토크나이저.

    한국어는 공백 분리만으로는 매칭이 거의 안 된다. 실제로 걸렸던 문제:
    - 쿼리 "「예금자보호법」 제39조의2"  vs  본문 "말합니다[「예금자보호법」 제39조의2제1항"
      → 괄호가 붙어 토큰이 달라지고, 조문 번호도 뒤에 항이 붙어 안 맞는다.
    그래서 (1) 구두점 제거 (2) 조문 원자 추출 (3) 얕은 조사 분리를 한다.
    원본 토큰도 함께 남기므로 분리에 실패해도 정보가 사라지지는 않는다.
    """
    tokens = []
    for word in _WORD.findall(text):
        tokens.append(word)
        atoms = _STATUTE.findall(word)
        if atoms:
            tokens.extend(atoms)          # "제39조의2제1항" -> "제39조의2", "제1항"
            continue
        for josa in _JOSA:
            # 조사를 뗀 어간이 최소 2글자는 남아야 한다 ("가는" -> "가" 같은 과잉 분리 방지)
            if len(word) - len(josa) >= 2 and word.endswith(josa):
                tokens.append(word[:-len(josa)])
                break
    return tokens


def build_bm25(meta):
    """청크 제목 + 본문으로 BM25 인덱스 구성 (임베딩과 같은 텍스트를 쓴다)"""
    return BM25Okapi([tokenize(m["chunk_title"] + " " + m["chunk_content"]) for m in meta])


def _rank_map(order):
    """[문서번호...] (좋은 순) -> {문서번호: 순위(1부터)}"""
    return {int(idx): rank for rank, idx in enumerate(order, start=1)}


def search_hybrid(query, model, index, meta, bm25, top_k=TOP_K, rrf_k=RRF_K):
    """밀집 검색 + BM25를 RRF(Reciprocal Rank Fusion)로 결합.

    RRF를 쓰는 이유: 두 점수의 스케일을 맞출 필요 없이 순위만 쓴다. 가중합이었다면 결합 점수가
    코사인 유사도와 다른 척도가 되어 답변 게이트(ANSWER_THRESHOLD / CONTEXT_FLOOR)를 다시 잡아야 하는데, 그 값은
    이 앱의 거부(abstention) 안전장치라 함부로 흔들 수 없다.
    → **순위만 하이브리드로 바꾸고 score 필드는 코사인 유사도 그대로 둔다.**

    어휘 가중치를 질의 종류에 따라 달리 주는 이유 (골든셋 38문항 측정 결과):
    - 두 검색을 동등 가중(w=1.0)하면 전체 Recall@5가 0.889 -> 0.812로 **떨어졌다.**
      밀집 검색이 이미 강해서, 신호 없는 질의에까지 BM25를 같은 비중으로 섞으면 순위가 흐려진다.
    - 반대로 조문 번호 질의는 BM25가 압도적이다. st-01(「예금자보호법」 제39조의2)에서
      BM25는 정답을 1위로 찾는데 밀집 검색은 16위였다. 여기서는 어휘 가중치를 낮추면 못 건진다.
    그래서 질의에 조문 번호가 있으면 어휘 가중치를 올린다. 결과: Recall@5 0.889 -> 0.961,
    statute_ref 0.875 -> 1.000, 악화된 문항 0개, 거부 정확성 0.700 유지.
    """
    assert index.ntotal == len(meta), (
        f"인덱스({index.ntotal})와 메타({len(meta)})의 개수가 다릅니다. "
        "embedding.py -> vector_search.py를 다시 실행해 함께 재생성하세요."
    )

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # 코퍼스 전체를 대상으로 밀집 순위와 코사인 점수를 함께 얻는다 (현재 101청크라 비용이 없다)
    scores, indices = index.search(query_vec, len(meta))
    cosine = {int(i): float(sc) for sc, i in zip(scores[0], indices[0]) if i != -1}
    dense_rank = _rank_map([i for i in indices[0] if i != -1])

    bm25_scores = bm25.get_scores(tokenize(query))
    lexical_rank = _rank_map(np.argsort(bm25_scores)[::-1])

    # 조문 번호가 들어 있는 질의는 정확 토큰 매칭이 핵심이라 어휘 검색에 더 무게를 준다
    weight = LEXICAL_WEIGHT_STATUTE if _STATUTE.search(query) else LEXICAL_WEIGHT

    missing = len(meta) + 1          # 한쪽 랭킹에 없으면 최하위로 취급
    fused = sorted(
        range(len(meta)),
        key=lambda i: -(1.0 / (rrf_k + dense_rank.get(i, missing))
                        + weight / (rrf_k + lexical_rank.get(i, missing))),
    )

    results = []
    for idx in fused[:top_k]:
        item = meta[idx]
        results.append({
            "chunk_id": item["chunk_id"],
            "chunk_title": item["chunk_title"],
            "chunk_content": item["chunk_content"],
            # 임계값 게이트가 쓰는 값 — 하이브리드로 바뀌어도 코사인 유사도를 유지한다
            "score": cosine.get(idx, 0.0),
            "dense_rank": dense_rank.get(idx),
            "lexical_rank": lexical_rank.get(idx),
        })
    return results


def gate(results, answer_threshold=ANSWER_THRESHOLD, context_floor=CONTEXT_FLOOR):
    """LLM에 넘길 근거 청크를 고른다. 통과가 0건이면 답변을 생성하지 않는다는 뜻이다.

    두 판단을 분리하는 이유:
    코사인 점수만으로는 "답이 있는 질문"과 "답이 없는 질문"을 가를 수 없다.
    골든셋 실측에서 정답 청크 42개 중 29개가 무정답 질문 top1 점수 범위 안에 들어온다.
    단일 임계값을 낮추면 무관한 근거가 새고, 높이면 관련 문서가 있는데도 답변이 안 나온다.

    그래서 질문 단위로 먼저 답변 여부를 정하고(최고점 >= answer_threshold),
    답하기로 한 뒤에는 더 낮은 하한으로 근거를 모은다. 답할 수 있다고 이미 판단한 질문이라
    2~3위 청크가 조금 낮아도 넣는 편이 낫다 — 정답이 상위에 있는데 절대 점수만 낮아
    통째로 버려지던 경우(st-01 등)가 이 분리로 해결된다.
    """
    if not results:
        return []
    if max(r["score"] for r in results) < answer_threshold:
        return []
    return [r for r in results if r["score"] >= context_floor]


def _selfcheck():
    """토크나이저가 실제로 문제였던 케이스를 잡는지 확인한다."""
    # 조문 번호: 뒤에 항이 붙어도 조 단위로 매칭돼야 한다
    assert "제39조의2" in tokenize("제39조의2제1항")
    assert "제1항" in tokenize("제39조의2제1항")
    assert "제39조의2" in tokenize("제39조의2에")
    # 법령명: 괄호·대괄호가 붙어도 같은 토큰이 나와야 한다
    assert "예금자보호법" in tokenize("말합니다[「예금자보호법」")
    assert "예금자보호법" in tokenize("「예금자보호법」")
    # 조사 분리: 어간이 2글자 미만이면 자르지 않는다
    assert "금리인하" in tokenize("금리인하를")
    assert tokenize("가는") == ["가는"]
    # 조문 번호 유무로 어휘 가중치가 갈리는 분기 (하이브리드 설계의 핵심 조건)
    assert _STATUTE.search("「예금자보호법」 제39조의2는?")
    assert _STATUTE.search("제39조의2제1항에 근거한 신청")
    assert not _STATUTE.search("계약을 취소하고 싶어요")

    # 2단 게이트 — 최고점이 문턱 미만이면 전부 버리고, 넘으면 하한까지 모은다
    rs = [{"score": 0.60}, {"score": 0.47}, {"score": 0.40}]
    assert len(gate(rs, 0.52, 0.44)) == 2, "답변 문턱을 넘으면 하한 이상 청크를 모두 포함해야 한다"
    assert gate([{"score": 0.50}, {"score": 0.49}], 0.52, 0.44) == [], "최고점이 문턱 미만이면 거부"
    assert gate([], 0.52, 0.44) == []
    print("selfcheck: 토크나이저·조문 분기·게이트 확인 완료")


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
