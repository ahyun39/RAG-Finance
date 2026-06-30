import json
import os

TEXT_TYPES = ("heading", "text", "note")


def extract_sections(law_data):
    records = []
    for doc in law_data:
        for section in doc["sections"]:
            texts = [b["text"] for b in section["body"] if b["type"] in TEXT_TYPES]
            tables = [b["rows"] for b in section["body"] if b["type"] == "table"]
            records.append({
                "section_title": section["title"],
                "content": "\n".join(texts),
                "tables": tables,
            })
    return records


if __name__ == "__main__":
    with open("data/law_data.json", "r", encoding="utf-8") as f:
        law_data = json.load(f)

    records = extract_sections(law_data)

    os.makedirs("data", exist_ok=True)
    with open("data/text_sections.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: data/text_sections.json (총 {len(records)}개 섹션)")
