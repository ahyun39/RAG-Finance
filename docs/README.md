# docs

금융 법령 RAG 프로젝트를 설명하는 웹 문서입니다. GitHub Pages에서 바로 열어 볼 수 있습니다.

| 문서 | 내용 |
|---|---|
| [RAG 시스템 구축](https://ahyun39.github.io/RAG-Finance/docs/finance_rag_system.html) | 시스템 전체 구조와 설계 근거 |
| [기술 아키텍처](https://ahyun39.github.io/RAG-Finance/docs/finance_rag_tech_architecture.html) | 사용한 기술을 다이어그램으로 연결하고, 기술별 이론·예시를 정리 |
| [품질 검증](https://ahyun39.github.io/RAG-Finance/docs/rag_quality_docs.html) | 정량 평가와 실제 답변 점검으로 무엇을 확인했는지 |

## [RAG 시스템 구축](https://ahyun39.github.io/RAG-Finance/docs/finance_rag_system.html)

데이터 수집부터 검색, 답변 게이트, 생성, 평가까지 시스템 전체를 설명합니다. 단계마다 어떤 문제가 있었고, 어떤 측정을 근거로 설계를 골랐는지 적었습니다.

- 데이터 파이프라인: 크롤링, heading 단위 추출, 표를 보존하는 청킹, KURE-v1 임베딩
- 검색: 밀집 검색(FAISS)과 BM25를 조건부 가중 RRF로 결합
- 답변 게이트: 답변 여부(0.52)와 근거 선택(0.44)을 나눈 2단 게이트
- 평가: 골든셋 56문항 기반 지표, 측정 후 기각한 설계, 실제 질의응답 결과
- 한계와 다음 단계

## [기술 아키텍처](https://ahyun39.github.io/RAG-Finance/docs/finance_rag_tech_architecture.html)

프로젝트에 쓰인 기술을 오프라인 → 온라인 → 생성 → 평가 순서의 아키텍처 다이어그램 한 장으로 연결합니다.

- 다이어그램의 ⓘ 버튼이나 상자 안 글자를 누르면 해당 기술의 이론, 예시, 프로젝트 적용, 출처가 열립니다.
- 기술 사전: 23개 기술을 단계별 6개 그룹으로 정리했습니다.
- 이론은 논문과 공식 문서를 근거로 썼고, 예시 수치는 프로젝트 코드와 실제 라이브러리로 계산했습니다.
- 레이어별 상세, 파일 입출력 계약, 평가 구조, 기술 스택 목록도 함께 담았습니다.

## [품질 검증](https://ahyun39.github.io/RAG-Finance/docs/rag_quality_docs.html)

구현한 시스템을 두 가지 방법으로 검증한 결과입니다.

- 정량: 골든셋 56문항으로 검색 방식(BM25 · 밀집 · 하이브리드)과 게이트 방식을 같은 조건에서 비교
- 정성: 실제 앱 답변 10건을 근거 원문과 공식 답변에 대조
- 실패 분석: 순위, 생성 누락, 데이터 결함, 어휘 함정처럼 검색 지표에 드러나지 않는 실패
- 한계와 다음 단계

> 수치는 파라미터를 고른 골든셋으로 측정한 개발 벤치마크이며, 일반화 성능을 뜻하지 않습니다.

## 그 밖의 문서

- [evaluation.md](evaluation.md) — 지표 정의, 실험 기록, 임계값 스윕
- [changelog.md](changelog.md) — 개선 이력
- [build_process.md](build_process.md) — 구축 과정과 기술 스택
