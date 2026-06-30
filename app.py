"""
금융 법령정보 검색 - Streamlit RAG 앱

실행: streamlit run app.py

필요 파일:
- index/faiss_index.bin
- data/chunks_meta.json

환경변수:
- UPSTAGE_API_KEY (Upstage API 키)
"""

import os
import re
import json
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

MODEL_NAME = "jhgan/ko-sroberta-multitask"
UPSTAGE_MODEL = "solar-pro3"
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
INDEX_FILE = "index/faiss_index.bin"
META_FILE = "data/chunks_meta.json"
SIMILARITY_THRESHOLD = 0.5

st.set_page_config(
    page_title="금융 법령 정보 검색",
    page_icon="🗂️",
    layout="wide",
)

# ===== 리소스 로딩 (캐시) =====
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_FILE)

@st.cache_data
def load_meta():
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_llm_client():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=UPSTAGE_BASE_URL)


def search(query, model, index, meta, top_k=5):
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


def build_context(results):
    """검색 결과를 LLM 프롬프트용 컨텍스트 문자열로 변환"""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[참고 {i}] {r['chunk_title']}\n{r['chunk_content']}")
    return "\n\n".join(blocks)


def generate_answer(client, query, results):
    """검색된 chunk를 컨텍스트로 Upstage(Solar)에게 답변 생성 요청"""
    context = build_context(results)

    system_prompt = (
        "당신은 한국 금융 관련 법령 정보를 안내하는 어시스턴트입니다."
        "1. 반드시 아래 제공된 [참고 자료]의 내용만을 근거로 답변하세요."
        "2. 참고 자료에 없는 내용은 추측하지 말고, 제공된 자료에서 확인할 수 없다고 안내하세요."
        "3. 링크를 출력할 때는 반드시 [텍스트](URL) 형식만 사용하세요."
        "4. 링크 바로 뒤에 조사(은,는,이,가,을,를,에,에서,의 등)를 붙이지 마세요."
        "5. 링크 뒤에는 반드시 공백을 넣고 문장을 이어가세요."\
            "예시: [금융용어사전](https://fins.kdic.or.kr) 에서 확인할 수 있습니다."
            "잘못된 예시: [금융용어사전](https://fins.kdic.or.kr)에서 확인할 수 있습니다."
        "6. 답변 출력 형식:"
            "(1) 첫 번째 단락: 3~4문장 분량의 답변 내용"
            "(2) (매우 중요) 첫 번째 단락이 끝난 후 반드시 빈 줄 1개를 출력"
            "(3) 마지막 단락: 참고 자료 번호만 표시"
            "(4) 참고 자료 번호는 반드시 [참고 N], [참고 M] 형식만 사용. 반드시 컨텍스트에 제시된 [참고 N] 번호를 그대로 사용"
            "(5) 참고 자료 번호 앞뒤에 추가 설명 문구를 작성하지 말 것"

            "출력 예시:"
            "대출 계약 철회권은 일정 기간 내 소비자가 계약을 취소할 수 있도록 보장하는 제도입니다. 해당 권리는 관련 법령에 따라 적용 요건이 정해져 있습니다. 구체적인 행사 방법은 금융회사 안내 및 관련 규정을 통해 확인할 수 있습니다."

            "\n\n[참고 1], [참고 3]"

            "잘못된 예시:"
            "대출 계약 철회권은 일정 기간 내 소비자가 계약을 취소할 수 있도록 보장하는 제도입니다. 해당 권리는 관련 법령에 따라 적용 요건이 정해져 있습니다. 구체적인 행사 방법은 금융회사 안내 및 관련 규정을 통해 확인할 수 있습니다. [참고 1], [참고 3]"

    )

    user_message = f"[참고 자료]\n{context}\n\n[질문]\n{query}"

    response = client.chat.completions.create(
        model=UPSTAGE_MODEL,
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


# ===== 스타일 =====
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a2e4a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.4rem;
    }
    .result-score {
        font-size: 0.8rem;
        color: #2563eb;
        font-weight: 500;
        float: right;
    }
    .result-content {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.6;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ===== 헤더 =====
st.markdown('<div class="main-title">🗂️ 금융 법령 정보 검색</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">생활법령정보(easylaw.go.kr) 금융 관련 콘텐츠를 의미 기반으로 검색합니다.</div>',
    unsafe_allow_html=True,
)

# ===== 리소스 로드 =====
with st.spinner("검색 엔진을 준비하는 중..."):
    model = load_model()
    index = load_index()
    meta = load_meta()
    llm_client = load_llm_client()

if llm_client is None:
    st.warning(
        "UPSTAGE_API_KEY 환경변수가 설정되어 있지 않아 AI 답변 생성 기능을 사용할 수 없습니다. "
        "검색 결과만 표시됩니다."
    )

# ===== 검색 UI =====
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "검색어를 입력하세요",
        placeholder="예: 예금자 보호 한도, 보험 청약 철회, 대출 계약 해지 방법",
        label_visibility="collapsed",
    )
with col2:
    top_k = st.selectbox("표시 개수", [3, 5, 10], index=1, label_visibility="collapsed")

search_clicked = st.button("검색", type="primary", use_container_width=False)

st.divider()

# ===== 검색 실행 및 결과 표시 =====
if search_clicked and query:
    with st.spinner("검색 중..."):
        results = search(query, model, index, meta, top_k=top_k)

    relevant_results = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]  # ← 여기
    if not relevant_results:                                                         # ← 여기
        st.info("관련성이 높은 자료를 찾지 못했습니다.")

    if not results:
        st.info("검색 결과가 없습니다. 다른 검색어로 시도해보세요.")
    else:
        # --- AI 답변 생성 ---
        if llm_client is not None:
            with st.spinner("답변을 생성하는 중..."):
                try:
                    answer = generate_answer(llm_client, query, results)

                    answer = re.sub(
                        r'(\(https?://[^)]+\))(에서|에|의|은|는|이|가|을|를)',
                        r'\1 \2',
                        answer
                    )
                except Exception as e:
                    answer = None
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")

            if answer:
                st.markdown("### 💬 답변")
                st.markdown(answer)
                st.caption("⚠️ 이 답변은 참고용 정보이며 법률 자문을 대체할 수 없습니다.")
                st.divider()

        # --- 검색된 원본 자료 (출처) ---
        st.markdown(f"### 📄 참고한 원본 자료 ({len(results)}건)")

        for i, r in enumerate(results, start=1):
            content = r["chunk_content"]
            is_table = content.strip().startswith("|")

            with st.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="result-score">유사도 {r['score']:.3f}</span>
                        <div class="result-title">[참고 {i}] {r['chunk_title']}</div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_table:
                    st.markdown(content)  # 마크다운 표로 렌더링
                else:
                    st.markdown(f'<div class="result-content">{content}</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("검색어를 입력하면 관련된 금융 법령 정보를 바탕으로 AI가 답변을 생성합니다.")

