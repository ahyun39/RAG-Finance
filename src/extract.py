import json
import os

TEXT_TYPES = ("heading", "text", "note")

def extract_sections(law_data):
    records = []
    for doc in law_data:
        for section in doc["sections"]:
            cur = None
            for b in section["body"]:
                # heading을 만나면 새 블록 시작 (heading 없이 시작하는 body는 빈 heading으로)
                if b["type"] == "heading" or cur is None:
                    cur = {
                        "section_title": section["title"],
                        "heading": b["text"] if b["type"] == "heading" else "",
                        "content": [],
                        "tables": [],
                    }
                    records.append(cur)
                if b["type"] in ("text", "note"):
                    cur["content"].append(b["text"])
                elif b["type"] == "table":
                    cur["tables"].append(b["rows"])

    records = [r for r in records if r["content"] or r["tables"]]
    for r in records:
        r["content"] = "\n".join(r["content"])
    return records


if __name__ == "__main__":
    with open("data/law_data.json", "r", encoding="utf-8") as f:
        law_data = json.load(f)

    records = extract_sections(law_data)

    os.makedirs("data", exist_ok=True)
    with open("data/text_sections.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: data/text_sections.json (총 {len(records)}개 섹션)")
