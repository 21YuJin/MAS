# Real-LLM v2 — Analysis Plan (frozen premise)

**작성일**: 2026-07-21
**상태**: LOCKED — 이 문서에 적힌 정의는 P2 이후 모든 코드/데이터 수집 작업의 전제다. 바꿔야
할 필요가 생기면 코드를 먼저 고치지 말고 이 문서를 새 커밋으로 먼저 갱신한다.

이 문서는 real_llm_v1(정상 50 + 공격 50, `lgnn_experiment.py`)에 대한 코드 리뷰에서 발견된
편향 문제 — 특히 공격이 "GCN이 유리하고 Z-score가 불리하도록" 의도적으로 설계됐다는 주석
(`lgnn_experiment.py:14-18`, `:197`) — 을 계기로 작성됐다. v1은 **프로토콜이 정상 동작하는지
확인한 pilot experiment**였고, v2는 이 문서에 적힌 중립적 전제 위에서 논문 본 실험 데이터셋을
새로 구축한다. v1 코드/캐시/결과는 그대로 보존하며 v2는 별도 파일로만 구현한다(P0).

---

## 1. 연구 목표

LightGAE가 무엇이고 무엇이 아닌지를 먼저 명확히 한다.

- **의미 기반(semantic) 탐지기의 완전한 대체재가 아니다.** 응답 내용을 읽고 판단하는 방식과
  같은 수준의 탐지력을 주장하지 않는다.
- **메타데이터 기반 1차 선별기(first-pass triage filter)다.** 세션/에이전트 단위로 "이건 더
  들여다볼 가치가 있다"는 우선순위를 싸고 빠르게 매기는 역할이며, 최종 판정 도구가 아니다.
- **목표는 "실행 행동(execution behavior)에 변화가 발생한" injection 세션을 탐지하는 것**이다
  — 즉 injection이 삽입됐다는 사실 자체가 아니라, 그로 인해 관측 가능한 실행 지표(토큰 수,
  타이밍 등)가 정상 분포에서 벗어났는지를 본다. 이 정의는 왜 "출력에 변화가 없는" 공격(예:
  length-preserving)에서 탐지력이 떨어질 수 있는지를 사전에 설명해준다 — 실패가 아니라 목표
  정의상 예상된 경계다.
- **스코프는 고정 4-agent, tool 없는(tool-free) Ollama 배포 환경에서의 초기 검증(initial
  validation)이다.** 다른 topology, 다른 모델, tool-사용 에이전트, 다른 LLM 제공자로의 일반화는
  이 연구의 주장 범위에 포함하지 않는다.

## 2. 정상과 공격의 정의

**정상과 공격은 반드시 동일한 user task를 사용한다.** 서로 다른 질문을 비교하면 결과 차이가
공격 때문인지 task 난이도/길이 차이 때문인지 구분할 수 없다 (matched-pair design).

```
Normal:
  동일 user request
  + clean external content

Attack:
  동일 user request
  + injected external content
```

차이는 **external content 내부에 injection이 있는지 여부만**이어야 한다. user request(agent가
받는 "무엇을 하라"는 지시문)는 정상/공격 조건에서 byte-identical해야 한다. 실제 채널 구현(P2:
instruction/content 분리, indirect injection 구조)은 이 문서가 아니라 코드 단계에서 이어서
진행한다 — 여기서는 이 정의를 판단 기준으로 먼저 고정해둔다.

## 3. 공격 라벨 분리

지금까지 `attack_success_observed` 하나(키워드 매칭)로 뭉뚱그려져 있던 개념을 4개의 독립된
필드로 나눈다. 서로 다른 걸 측정하면서 하나의 값으로 합치지 않는다.

| 필드 | 의미 |
|---|---|
| `injection_present` | 공격 입력이 실제로 삽입됐는가 (세션 생성 시점에 결정되는 사실 — 정상/공격 풀 소속 여부) |
| `indicator_observed` | 미리 정한 공격 지표(indicator pattern)가 응답에 나타났는가 |
| `goal_success` | 공격자의 목적(task override, workflow corruption 등)이 실제로 달성됐는가 |
| `propagation_observed` | 영향이 후속 agent까지 실제로 전달됐는가 |

**Ground truth label은 당분간 `ground_truth_label = int(injection_present)`로 유지한다** —
이전과 동일하게 응답 내용이 아니라 "어느 풀에서 수집됐는가"로만 결정한다. 다만 결과 보고 시
`goal_success`/`propagation_observed` 여부에 따라 별도로 breakdown(성공한 공격만 recall이 어떤지,
실패한 공격은 어떤지)을 함께 제시한다 — 하나의 헤드라인 숫자 뒤에 숨기지 않는다.

`indicator_observed`와 `goal_success`는 서로 다른 함수로 각각 계산해야 한다. v1의
`detect_injection_pattern()` 하나가 "attack success"와 "attack indicator occurrence"를
동시에 대표하던 문제(코드 리뷰에서 발견)를 여기서 명시적으로 금지한다.

## 4. 이 문서가 대체하는 이전 설계 원칙

| 이전 (v1) | 문제 | v2 원칙 |
|---|---|---|
| "Z-score는 개별 피처만 보므로 불리" (주석으로 명시) | 특정 baseline에 불리하도록 공격을 설계 — 실험 편향 | 공격은 공격자의 보안 목표를 기준으로 설계하고, 특정 탐지 방법의 유불리는 설계 기준에서 배제 |
| injection을 Agent_0 프롬프트에 직접 append | direct injection에 가까움, 논문 제목("indirect prompt injection")과 불일치 | user request는 고정, external content만 정상/공격으로 분기 |
| `detect_injection_pattern()` 하나가 성공 여부와 지표 발생을 동시에 대표 | 서로 다른 개념을 하나의 값으로 뭉뚱그림 | injection_present / indicator_observed / goal_success / propagation_observed 4개 독립 필드 |
| 공격 7종 전부 동일 진입점(Agent_0)·동일 전파경로(전체 4-agent) | 공격 다양성 부족, 일반화 주장 근거 약함 | goal/propagation/output-effect 축을 다양화 (아래 우선순위 9순위에서 진행) |

## 5. 다음 단계와의 관계

이 문서는 1순위(연구 전제 고정) 산출물이다. 이후 진행 순서는 다음과 같다 — 이 문서를 벗어나는
결정은 이 문서를 먼저 갱신한 뒤에 코드에 반영한다.

- 2순위: 기존 공격 설계 편향 제거 (위 §4 표의 "이전 원칙" 코드/주석 삭제·재작성)
- 3순위: indirect prompt injection 구조 구현 (§2의 external content 채널 실제 구현)
- 4순위: raw 데이터 저장 스키마 확정
- 5순위 이후: feature pool → screening → formal collection (세부는 대화 기록의 우선순위 목록 참고)
