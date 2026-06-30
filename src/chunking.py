import json
import re

# ===== 설정값 =====
# EDA 결과(text_length 분포)를 참고하여 설정
# - 한국어 법령 설명문 기준, 임베딩 모델(보통 max 512 token ≈ 800~1000자) 고려
# - 너무 짧으면 컨텍스트 부족, 너무 길면 검색 정밀도 저하
TARGET_CHUNK_SIZE = 500   # 청크 목표 길이 (문자 수)
MAX_CHUNK_SIZE = 700      # 청크 최대 길이 (문자 수) - 이 길이를 넘으면 분할
MIN_CHUNK_SIZE = 100      # 너무 짧은 chunk는 이전 chunk와 병합


def split_into_sentences(text):
    """문장 단위로 분리 (한국어 종결 어미 기준 간단 분리)"""
    # 줄바꿈 기준 우선 분리 후, 너무 긴 줄은 문장 단위로 추가 분리
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        # 마침표/물음표/느낌표 뒤에서 분리 (단, 숫자 뒤 마침표는 보존 - 간단 처리)
        parts = re.split(r'(?<=[.!?])\s+', line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def chunk_section(section_title, content, target_size=TARGET_CHUNK_SIZE,
                   max_size=MAX_CHUNK_SIZE, min_size=MIN_CHUNK_SIZE):
    """
    하나의 section(content)을 여러 chunk로 분할.
    각 chunk는 {"chunk_title": ..., "chunk_content": ...} 형태.
    """
    content = content.strip()

    # 짧으면 분할 없이 단일 chunk
    if len(content) <= max_size:
        return [{"chunk_title": section_title, "chunk_content": content}]

    sentences = split_into_sentences(content)

    chunks = []
    current = ""

    for sent in sentences:
        # 현재 chunk에 문장을 추가했을 때 target_size를 초과하고,
        # 이미 min_size 이상 채워져 있다면 chunk 완료
        if current and len(current) + len(sent) + 1 > target_size:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

        # 단일 문장이 max_size를 초과하는 극단적인 경우 -> 강제 분할
        while len(current) > max_size:
            chunks.append(current[:max_size].strip())
            current = current[max_size:].strip()

    if current:
        chunks.append(current.strip())

    # 마지막 chunk가 너무 짧으면 이전 chunk와 병합
    if len(chunks) > 1 and len(chunks[-1]) < min_size:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    # chunk_title에 분할 정보 추가
    total = len(chunks)
    if total == 1:
        return [{"chunk_title": section_title, "chunk_content": chunks[0]}]

    result = []
    for i, c in enumerate(chunks, start=1):
        title = f"{section_title} {i}/{total}"
        result.append({"chunk_title": title, "chunk_content": c})

    return result


def table_to_markdown(rows):
    """표(rows: list of list)를 마크다운 표 형식으로 변환"""
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        # 행의 컬럼 수가 header와 다를 경우 빈 값으로 채움
        r = r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)]
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)


def chunk_all_sections(input_json="text_sections.json", output_json="chunks.json"):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    chunk_id = 0

    for item in data:
        section_title = item["section_title"]
        content = item.get("content", "")
        tables = item.get("tables", [])

        # 텍스트 chunk (길이 기준 분할)
        if content.strip():
            section_chunks = chunk_section(section_title, content)
            for chunk in section_chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_title": chunk["chunk_title"],
                    "chunk_content": chunk["chunk_content"],
                    "chunk_length": len(chunk["chunk_content"]),
                })
                chunk_id += 1

        # 테이블 chunk (길이 무관, 표 1개 = chunk 1개, 마크다운 변환)
        n_tables = len(tables)
        for t_idx, table in enumerate(tables, start=1):
            if n_tables == 1:
                table_title = f"{section_title} (표)"
            else:
                table_title = f"{section_title} (표 {t_idx}/{n_tables})"

            table_md = table_to_markdown(table)

            all_chunks.append({
                "chunk_id": chunk_id,
                "chunk_title": table_title,
                "chunk_content": table_md,
                "chunk_length": len(table_md),
            })
            chunk_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"총 chunk 수: {len(all_chunks)}")
    print(f"저장 완료: {output_json}")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_sections("data/text_sections.json", "data/chunks.json")

    # 간단한 통계 출력
    lengths = [c["chunk_length"] for c in chunks]
    print(f"평균 길이: {sum(lengths)/len(lengths):.1f}")
    print(f"최소/최대 길이: {min(lengths)} / {max(lengths)}")
