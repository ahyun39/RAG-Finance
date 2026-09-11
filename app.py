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
import sys
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 설정값과 검색 로직은 src/ 한 곳에서만 정의한다.
sys.path.insert(0, "src")
from config import MODEL_NAME, TOP_K
from vector_search import (build_bm25, gate, load_index as _load_index,
                           load_meta as _load_meta, search_hybrid)

UPSTAGE_MODEL = "solar-pro3"
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

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
    return _load_index()

@st.cache_data
def load_meta():
    return _load_meta()

@st.cache_resource
def load_bm25(_meta):
    # BM25는 순수 파이썬 인덱스라 101청크 기준 즉시 만들어진다. 앱 시작 시 1회만.
    return build_bm25(_meta)

@st.cache_resource
def load_llm_client():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=UPSTAGE_BASE_URL)


def build_context(results, selected):
    """게이트를 통과한 청크만 넣되, 번호는 **화면 카드와 같은 기준**으로 붙인다.

    통과분만 1..N으로 다시 세면 답변의 [참고 N]과 화면의 [참고 N]이 어긋난다.
    예전에는 결과가 코사인 내림차순이고 게이트가 앞부분만 남겨 자동으로 맞았지만,
    하이브리드 도입으로 결과 순서가 RRF 기준이 되면서 그 보장이 사라졌다
    (골든셋 56문항 중 34건에서 코사인이 순서와 어긋난다).
    그래서 아래 화면 렌더링과 똑같이 `enumerate(results)`로 번호를 매긴다.
    통과분이 연속이 아니면 번호도 건너뛰는데, 프롬프트가 "컨텍스트에 제시된
    번호를 그대로 사용"하도록 지시하므로 그대로 두는 것이 맞다.
    """
    picked = {r["chunk_id"] for r in selected}
    return "\n\n".join(
        f"[참고 {i}] {r['chunk_title']}\n{r['chunk_content']}"
        for i, r in enumerate(results, start=1)
        if r["chunk_id"] in picked
    )


def generate_answer(client, query, results, selected):
    """게이트를 통과한 chunk를 컨텍스트로 Upstage(Solar)에게 답변 생성 요청"""
    context = build_context(results, selected)

    # 링크 뒤 조사 분리는 프롬프트가 아닌 정규식 후처리(호출부)로 보장
    system_prompt = """당신은 한국 금융 관련 법령 정보를 안내하는 어시스턴트입니다.

규칙:
1. 반드시 아래 제공된 [참고 자료]의 내용만을 근거로 답변하세요.
2. 참고 자료에 없는 내용은 추측하지 말고, 제공된 자료에서 확인할 수 없다고 안내하세요.
3. 금액·기간·비율 등 수치는 참고 자료에 명시된 그대로만 인용하고, 직접 계산하거나 변환하지 마세요.
4. 링크를 출력할 때는 반드시 [텍스트](URL) 형식만 사용하세요.
5. 첫 번째 단락 마지막에 이 답변이 법률 자문이 아닌 정보 제공 목적임을 한 문장으로 안내하세요.

답변 출력 형식:
(1) 첫 번째 단락: 3~4문장 분량의 답변 내용
(2) (매우 중요) 첫 번째 단락이 끝난 후 반드시 빈 줄 1개를 출력
(3) 마지막 단락: 참고 자료 번호만 표시
(4) 참고 자료 번호는 반드시 [참고 N], [참고 M] 형식만 사용하고, 컨텍스트에 제시된 [참고 N] 번호를 그대로 사용
(5) 참고 자료 번호 앞뒤에 추가 설명 문구를 작성하지 말 것

출력 예시:
대출 계약 철회권은 일정 기간 내 소비자가 계약을 취소할 수 있도록 보장하는 제도입니다. 해당 권리는 관련 법령에 따라 적용 요건이 정해져 있습니다. 구체적인 행사 방법은 금융회사 안내 및 관련 규정을 통해 확인할 수 있습니다. 이 답변은 법률 자문이 아닌 정보 제공 목적입니다.

[참고 1], [참고 3]

잘못된 예시 (빈 줄 없이 참고 번호를 이어 붙임):
대출 계약 철회권은 일정 기간 내 소비자가 계약을 취소할 수 있도록 보장하는 제도입니다. [참고 1], [참고 3]
"""

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
    bm25 = load_bm25(meta)
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
    # 기본값은 평가와 같은 TOP_K — 평가된 적 없는 설정이 기본으로 나가지 않게 한다
    _options = [3, 5, 10]
    top_k = st.selectbox("표시 개수", _options, index=_options.index(TOP_K),
                         label_visibility="collapsed")

search_clicked = st.button("검색", type="primary", use_container_width=False)

st.divider()

# ===== 검색 실행 및 결과 표시 =====
if search_clicked and query:
    with st.spinner("검색 중..."):
        results = search_hybrid(query, model, index, meta, bm25, top_k=top_k)

    # 2단 게이트로 LLM에 넘길 근거를 고른다 (UI에는 전체 결과 표시).
    # 통과가 0건이면 답변을 생성하지 않는다 — src/vector_search.gate 참고
    relevant_results = gate(results)

    if not results:
        st.info("검색 결과가 없습니다. 다른 검색어로 시도해보세요.")
    else:
        # --- AI 답변 생성 (관련성 높은 청크가 있을 때만 호출) ---
        if not relevant_results:
            st.info("관련성이 높은 자료를 찾지 못해 AI 답변 생성을 건너뜁니다. 검색 결과만 표시합니다.")
        elif llm_client is not None:
            with st.spinner("답변을 생성하는 중..."):
                try:
                    answer = generate_answer(llm_client, query, results, relevant_results)

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

