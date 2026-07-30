import json
import re

# ===== 설정값 =====
# EDA 결과(text_length 분포)를 참고하여 설정
# - 너무 짧으면 컨텍스트 부족, 너무 길면 검색 정밀도 저하
#
# 임베딩 모델(nlpai-lab/KURE-v1)의 max_seq_length는 8192 토큰이므로 아래 설정은 상한 안에 넉넉히 들어온다.
# (이전 ko-sroberta-multitask는 128 토큰이라 106청크 중 49개가 벡터화 단계에서 잘렸다)
TARGET_CHUNK_SIZE = 500   # 청크 목표 길이 (문자 수)
MAX_CHUNK_SIZE = 700      # 청크 최대 길이 (문자 수) - 이 길이를 넘으면 분할
MIN_CHUNK_SIZE = 100      # 한 섹션을 쪼갠 결과의 마지막 조각이 이보다 짧으면 직전 조각에 병합
                          # (섹션 자체가 짧은 경우는 병합하지 않는다 — 섹션을 넘어 합치면
                          #  heading 단위 분리 효과가 사라지므로. 그래서 최소 청크는 38자다)

# 표는 쪼개지 않는다. 표가 청크 경계로 잘리면 헤더와 행의 대응이 끊겨 사실상 못 쓰게 되고,
# 8192 토큰 모델에서는 쪼갤 이유도 없다(현재 최대 표 1853자). 이 값은 검색 품질 조절 손잡이가
# 아니라 임베딩 상한을 넘는 비정상적으로 큰 표에 대한 안전장치다.
MAX_TABLE_CHUNK_SIZE = 4000


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


def _cell(value):
    """마크다운 표의 셀 값 정리.

    - 개행: 공백으로 접는다. 셀 안에 개행이 남으면 표 문법이 깨지는 것은 물론,
      행 단위로 동작하는 split_table_markdown이 그 조각을 별도 행으로 오인한다.
    - `|`: 이스케이프한다. 그대로 두면 컬럼이 밀린다.
    """
    return re.sub(r"[\r\n]+", " ", str(value)).replace("|", r"\|").strip()


def table_to_markdown(rows):
    """표(rows: list of list)를 마크다운 표 형식으로 변환"""
    if not rows:
        return ""

    header = [_cell(c) for c in rows[0]]
    body = rows[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        r = [_cell(c) for c in r]
        # 행의 컬럼 수가 header와 다를 경우 빈 값으로 채움
        r = r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)]
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)


def split_table_markdown(table_md, max_size=MAX_TABLE_CHUNK_SIZE, min_size=MIN_CHUNK_SIZE):
    """
    마크다운 표를 행 경계에서 분할.

    각 조각은 헤더 행 + 구분선을 반복해서 갖는다.
    - 조각 단독으로 검색돼도 컬럼의 의미를 알 수 있고
    - 같은 헤더 = 같은 표라는 식별 신호가 된다
    셀 중간을 자르면 마크다운 문법이 깨지고 app.py의 표 렌더링도 무너지므로 행 단위로만 자른다.
    """
    lines = table_md.split("\n")

    # 기본값은 MAX_TABLE_CHUNK_SIZE — 표는 웬만하면 쪼개지 않는다는 모듈 정책과 일치시킨다.
    # (기본값을 MAX_CHUNK_SIZE로 두면 인자 없이 부른 곳에서 700자 표 분할이 되살아난다)

    # 헤더(1) + 구분선(1) + 본문(1) 미만이면 나눌 것이 없음
    if len(table_md) <= max_size or len(lines) <= 3:
        return [table_md]

    head = lines[:2]
    head_len = sum(len(l) + 1 for l in head)

    parts, cur, cur_len = [], [], head_len
    for row in lines[2:]:
        # ponytail: 단일 행이 max_size를 넘으면 그 행만 담은 조각이 되어 max_size를 초과한다.
        #           셀 중간 절단보다 초과가 낫다는 판단. 현재 데이터 최장 행 569자 < 700이라 미발생.
        #           행 자체가 상한을 넘기 시작하면 그때 셀 단위 요약/분할을 검토.
        if cur and cur_len + len(row) + 1 > max_size:
            parts.append(cur)
            cur, cur_len = [], head_len
        cur.append(row)
        cur_len += len(row) + 1

    if cur:
        parts.append(cur)

    # 마지막 조각이 너무 짧으면 직전 조각에 병합 (max_size 소폭 초과 허용)
    # pop()을 먼저 꺼내 둔다 — parts[-2] += parts.pop() 은 pop이 리스트를 줄인 뒤
    # -2가 한 칸 밀린 위치에 대입되어 앞 조각을 덮어쓴다
    if len(parts) > 1 and sum(len(r) + 1 for r in parts[-1]) < min_size:
        tail = parts.pop()
        parts[-1].extend(tail)

    return ["\n".join(head + p) for p in parts]


def chunk_all_sections(input_json="text_sections.json", output_json="chunks.json"):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    chunk_id = 0

    for sec_idx, item in enumerate(data):
        section_title = item["section_title"]
        if item.get("heading"):
            section_title = f"{section_title} : {item['heading']}" 
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

        # 테이블 chunk (표 1개 = chunk 1개. MAX_TABLE_CHUNK_SIZE를 넘을 때만 행 경계로 분할)
        n_tables = len(tables)
        for t_idx, table in enumerate(tables, start=1):
            if n_tables == 1:
                table_title = f"{section_title} (표)"
            else:
                table_title = f"{section_title} (표 {t_idx}/{n_tables})"

            parts = split_table_markdown(table_to_markdown(table), max_size=MAX_TABLE_CHUNK_SIZE)
            total = len(parts)

            for p_idx, part in enumerate(parts, start=1):
                all_chunks.append({
                    "chunk_id": chunk_id,
                    # 분할된 경우에만 파트 번호 표기 (텍스트 청크의 "i/total" 관례와 동일)
                    "chunk_title": table_title if total == 1 else f"{table_title} {p_idx}/{total}",
                    "chunk_content": part,
                    "chunk_length": len(part),
                    # 제목 문자열을 파싱하지 않고도 같은 표 조각을 묶을 수 있게 하는 식별자.
                    # section_title은 유일하지 않으므로(예: "분쟁조정 절차 : 조정 신청"이 2번 등장)
                    # 섹션 순번을 넣어 서로 다른 표가 같은 그룹으로 묶이지 않게 한다.
                    "table_group": f"{section_title}#{sec_idx}-{t_idx}",
                    "part": p_idx,
                    "part_total": total,
                })
                chunk_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"총 chunk 수: {len(all_chunks)}")
    print(f"저장 완료: {output_json}")

    return all_chunks


def _selfcheck():
    """표 분할이 무손실인지 확인 — 내용 소실 방지가 이 분할의 존재 이유다."""
    header = ["구분", "내용"]

    # 아래 표본들은 MAX_CHUNK_SIZE(700) 기준으로 크기를 맞춰 두었으므로 상한을 명시해 부른다.
    # (기본값은 MAX_TABLE_CHUNK_SIZE라 이 표본들은 분할되지 않는다)
    def check(body, allow_over=False, expect=None):
        md = table_to_markdown([header] + body)
        parts = split_table_markdown(md, max_size=MAX_CHUNK_SIZE)
        assert len(parts) > 1, "700자를 훌쩍 넘는 표가 분할되지 않음"
        # 조각 수를 못 박아 둔다 — 안 그러면 분기를 안 타도 테스트가 통과해 버린다
        assert expect is None or len(parts) == expect, f"조각 수 {len(parts)}, 기대 {expect}"

        head = md.split("\n")[:2]
        restored = []
        for p in parts:
            lines = p.split("\n")
            assert lines[:2] == head, "조각에 헤더 행 + 구분선이 반복되지 않음"
            if not allow_over:
                assert len(p) <= MAX_CHUNK_SIZE, f"조각이 상한 초과: {len(p)}자"
            restored += lines[2:]
        assert restored == md.split("\n")[2:], "행이 소실되거나 중복됨"

    check([[f"항목{i}", "가" * 120] for i in range(20)])
    # 마지막 조각이 min_size 미만이라 직전 조각에 병합되는 경로
    # (실데이터 '보호대상 금융회사' 표가 이 경로에서 행을 잃었던 이력이 있다)
    # 긴 행 2개가 각자 조각을 차지하고, 마지막 짧은 행(10자)이 min_size 미만이라 앞으로 병합된다.
    # 분할 결과 3조각 → 병합 후 2조각.
    check([["가", "나" * 652], ["다", "라" * 662], ["마", "바"]], allow_over=True, expect=2)

    # 상한 이하 표는 그대로 두어야 한다
    small = table_to_markdown([header, ["가", "나"]])
    assert split_table_markdown(small, max_size=MAX_CHUNK_SIZE) == [small], "짧은 표가 불필요하게 분할됨"

    # 셀 안의 | 와 개행이 표 구조를 깨뜨리지 않아야 한다 (재크롤링 시 유입 가능)
    dirty = table_to_markdown([["구분", "내용"], ["가|나", "첫 줄\n둘째 줄"]])
    rows = dirty.split("\n")
    assert len(rows) == 3, "셀 개행이 행을 쪼갬"
    # 이스케이프되지 않은 | 만 컬럼 구분자다. 2컬럼이면 구분자는 3개여야 한다.
    assert len(re.findall(r"(?<!\\)\|", rows[2])) == 3, "셀의 | 가 이스케이프되지 않아 컬럼이 밀림"

    print("selfcheck: 표 분할 무손실 확인 완료")


if __name__ == "__main__":
    _selfcheck()
    chunks = chunk_all_sections("data/text_sections.json", "data/chunks.json")

    # 간단한 통계 출력
    lengths = [c["chunk_length"] for c in chunks]
    print(f"평균 길이: {sum(lengths)/len(lengths):.1f}")
    print(f"최소/최대 길이: {min(lengths)} / {max(lengths)}")
