import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def get_ovdiv_content(csm_seq, ccf_no, cci_no, cnp_cls_no):
    url = "https://www.easylaw.go.kr/CSP/CnpClsMain.laf"
    params = {
        "csmSeq": csm_seq,
        "ccfNo": ccf_no,
        "cciNo": cci_no,
        "cnpClsNo": cnp_cls_no,
    }
    res = requests.get(url, params=params, headers=HEADERS)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "html.parser")
    box = soup.find("div", class_="ovDivbox")
    if box is None:
        return None

    if not box.get_text(strip=True):
        return None

    sections = []
    current = None

    for div in box.find_all("div", recursive=True):
        cls = div.get("class", [])

        if "plv1a" in cls:
            title = div.get_text(strip=True)
            title = re.sub(r"(인쇄체크|주소복사|즐겨찾기에추가)", "", title).strip()
            if current:
                sections.append(current)
            current = {"title": title, "body": []}

        elif "plv2a" in cls:
            text = div.get_text(strip=True)
            if current is not None:
                current["body"].append({"type": "heading", "text": text})

        elif "plv3a" in cls:
            text = div.get_text(strip=True)
            if current is not None:
                current["body"].append({"type": "text", "text": text})

        elif "tplv2d" in cls:
            text = div.get_text(strip=True)
            if current is not None:
                current["body"].append({"type": "note", "text": text})

        elif any(re.match(r"tplv\d+$", c) for c in cls):
            table = div.find("table")
            if table and current is not None:
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
                    rows.append(cells)
                current["body"].append({"type": "table", "rows": rows})

    if current:
        sections.append(current)

    return sections if sections else None

def crawl_csm_seq(csm_seq, max_ccf=5, max_cci=5, max_cnp=5, delay=0.5):
    results = []

    for ccf_no in range(1, max_ccf + 1):
        for cci_no in range(1, max_cci + 1):
            for cnp_cls_no in range(1, max_cnp + 1):
                sections = get_ovdiv_content(csm_seq, ccf_no, cci_no, cnp_cls_no)
                time.sleep(delay)

                if sections is None:
                    continue

                results.append({
                    "csmSeq": csm_seq,
                    "ccfNo": ccf_no,
                    "cciNo": cci_no,
                    "cnpClsNo": cnp_cls_no,
                    "sections": sections
                })
                print(f"수집됨: csmSeq={csm_seq}, ccfNo={ccf_no}, cciNo={cci_no}, cnpClsNo={cnp_cls_no}")

    return results

def save_to_json(data, filepath="data/law_data.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {filepath} (총 {len(data)}건)")

if __name__ == "__main__":
    csm_seq = 1771
    data = crawl_csm_seq(csm_seq, max_ccf=5, max_cci=5, max_cnp=5)
    save_to_json(data, "data/law_data.json")