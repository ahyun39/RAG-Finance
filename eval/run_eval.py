"""
골든셋 기반 검색 품질 평가

- Recall@5 / MRR: 정답이 있는 질문(legal_term, colloquial) 대상
- 전달률: 정답 청크 중 임계값까지 통과해 실제로 LLM 컨텍스트에 전달되는 비율
  (Recall@5는 검색기 성능, 전달률은 앱 최종 동작 기준 — 검색돼도 임계값 미만이면 LLM이 못 봄)
- 거부 정확성(abstention): 무정답 질문(no_answer)에서 임계값 통과 청크가 0개인 비율
  (통과 청크가 있으면 app.py가 무관한 근거로 LLM 답변을 생성하게 됨)

실행: python eval/run_eval.py  (레포 루트에서)
필요 파일: eval/golden_set.json, index/faiss_index.bin, data/chunks_meta.json
"""

import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from sentence_transformers import SentenceTransformer
from vector_search import MODEL_NAME, load_index, load_meta, search

GOLDEN_SET_FILE = "eval/golden_set.json"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.55  # app.py의 값과 반드시 일치해야 함


def evaluate(golden, model, index, meta):
    answerable, no_answer = [], []

    for g in golden:
        results = search(g["question"], model, index, meta, top_k=TOP_K)
        retrieved = [r["chunk_id"] for r in results]
        passed = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
        top = results[0] if results else None

        if g["type"] == "no_answer":
            no_answer.append({
                "id": g["id"],
                "abstained": not passed,
                "top": top,
                "passed": passed,
            })
        else:
            relevant = set(g["answer_chunk_ids"])
            recall = len(relevant & set(retrieved)) / len(relevant)
            rr = 0.0
            for rank, cid in enumerate(retrieved, start=1):
                if cid in relevant:
                    rr = 1.0 / rank
                    break
            delivered = {r["chunk_id"] for r in passed}
            deliv = len(relevant & delivered) / len(relevant)
            answerable.append({
                "id": g["id"],
                "type": g["type"],
                "recall": recall,
                "rr": rr,
                "deliv": deliv,
                "missing": sorted(relevant - set(retrieved)),
                "filtered": sorted((relevant & set(retrieved)) - delivered),
                "top": top,
            })

    return answerable, no_answer


def report(answerable, no_answer):
    print(f"\n{'='*70}")
    print(f"검색 품질 평가 — top_k={TOP_K}, threshold={SIMILARITY_THRESHOLD}")
    print(f"{'='*70}")

    print(f"\n[정답 있는 질문 {len(answerable)}개] Recall@{TOP_K} / MRR / 전달률(임계값 통과)")
    print(f"{'-'*70}")
    for r in answerable:
        mark = "YES" if r["deliv"] == 1.0 else ("PARTIAL" if r["deliv"] > 0 else "NO")
        miss = f"  누락={r['missing']}" if r["missing"] else ""
        filt = f"  임계값에 걸림={r['filtered']}" if r["filtered"] else ""
        print(f"{mark} {r['id']}  R@{TOP_K}={r['recall']:.2f}  RR={r['rr']:.2f}  전달={r['deliv']:.2f}"
              f"  top1=({r['top']['score']:.3f}) {r['top']['chunk_title'][:30]}{miss}{filt}")

    by_type = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)

    print(f"{'-'*70}")
    for t, rows in by_type.items():
        n = len(rows)
        print(f"{t:12s} (n={n}):  Recall@{TOP_K}={sum(r['recall'] for r in rows)/n:.3f}"
              f"  MRR={sum(r['rr'] for r in rows)/n:.3f}"
              f"  전달률={sum(r['deliv'] for r in rows)/n:.3f}")
    n = len(answerable)
    print(f"{'전체':12s} (n={n}):  Recall@{TOP_K}={sum(r['recall'] for r in answerable)/n:.3f}"
          f"  MRR={sum(r['rr'] for r in answerable)/n:.3f}"
          f"  전달률={sum(r['deliv'] for r in answerable)/n:.3f}")

    print(f"\n[무정답 질문 {len(no_answer)}개] 거부 정확성 — 임계값 통과 청크 0개면 성공")
    print(f"{'-'*70}")
    for r in no_answer:
        if r["abstained"]:
            print(f"{r['id']}  거부 성공  top1=({r['top']['score']:.3f}) {r['top']['chunk_title'][:30]}")
        else:
            leaked = ", ".join(f"({p['score']:.3f}) {p['chunk_title'][:25]}" for p in r["passed"])
            print(f"{r['id']}  임계값 통과 {len(r['passed'])}개 → LLM이 무관한 근거로 답변 생성: {leaked}")
    n = len(no_answer)
    ok = sum(r["abstained"] for r in no_answer)
    print(f"{'-'*70}")
    print(f"거부 정확성: {ok}/{n} = {ok/n:.3f}")


if __name__ == "__main__":
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        golden = json.load(f)

    print("모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)
    index = load_index()
    meta = load_meta()

    answerable, no_answer = evaluate(golden, model, index, meta)
    report(answerable, no_answer)
