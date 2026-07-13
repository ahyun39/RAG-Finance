"""
생활법령정보(easylaw.go.kr) 100문100답 크롤러

골든셋(eval/golden_set.json) 구축의 원천 데이터로 쓸 질문·답변을 수집합니다.

사용법:
    1. URLS 리스트에 100문100답 페이지 URL을 수동으로 추가
    2. python eval/crawl_qna.py
출력: eval/qna_raw.json — [{url, category, question, answer}, ...]
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# 100문100답 페이지 URL을 여기에 수동으로 추가
URLS = [
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=1&cciNo=1&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=1&cciNo=2&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=1&cciNo=2&cnpClsNo=2&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=1&cciNo=2&cnpClsNo=4&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=2&cciNo=1&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=2&cciNo=2&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=2&cciNo=2&cnpClsNo=2&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=2&cciNo=3&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=3&cciNo=1&cnpClsNo=1&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=3&cciNo=1&cnpClsNo=2&menuType=onhunqna&search_put=",
    "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1771&ccfNo=4&cciNo=1&cnpClsNo=1&menuType=onhunqna&search_put="
]

OUTPUT_FILE = "eval/qna_raw.json"


def fetch(url, retries=2):
    """네트워크 오류 시 2회 재시도 — 조용한 누락 방지"""
    for attempt in range(retries + 1):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            res.encoding = "utf-8"
            return res.text
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"  재시도 {attempt + 1}/{retries}: {e}")
            time.sleep(2)


def parse_qna(html, url):
    soup = BeautifulSoup(html, "html.parser")
    box = soup.find("div", id="onhunqnaDiv")
    if box is None:
        return None

    # 제목 영역: <li class="title"> 직계 span들이 분류 경로 (p 안의 조회수 span은 제외)
    category = ""
    title_li = box.select_one("ul.question li.title")
    if title_li:
        spans = [s.get_text(strip=True) for s in title_li.find_all("span", recursive=False)]
        category = spans[2].strip()

    dt = box.select_one("li.qa dl.q dt")
    dd = box.select_one("li.qa dl.q dd")
    if dt is None or dd is None:
        return None

    question = dt.get_text(strip=True)
    # 답변은 plv* div + 표가 섞여 있음 — 텍스트만 개행 구분으로 추출
    answer = dd.get_text("\n", strip=True)
    answer = re.sub(r"\n{2,}", "\n", answer)

    return {
        "url": url,
        "category": category,
        "question": question,
        "answer": answer,
    }


if __name__ == "__main__":
    records = []
    for i, url in enumerate(URLS, start=1):
        print(f"[{i}/{len(URLS)}] {url}")
        item = parse_qna(fetch(url), url)
        if item is None:
            print("질문/답변을 찾지 못함")
            continue
        records.append(item)
        time.sleep(1)

    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUTPUT_FILE} (총 {len(records)}건)")
