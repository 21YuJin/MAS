# MAS Security — Lightweight GNN-based Anomaly Detection

**멀티에이전트 AI 시스템 환경 구축 및 Quick Identification 기술 개발**
1차년도 연구과제 예비 실험 | 목표: WISA 2026 포스터 발표

---

## 실험 경로 (Headline vs. Legacy)

**최종 논문 결과는 Real-LLM 단일 경로로 통일한다.**

| 구분 | 경로 | 상태 |
|------|------|------|
| **Headline (공식 최종 결과, v1)** | `experiments/real_llm/lgnn_experiment.py` | ✅ 유지보수 대상. 결과는 `output/real_llm/results_summary.json` |
| **v2 개발 중** | `experiments/real_llm/analysis_plan.md` 이하 | 🚧 진행 중 — 아래 [v2 진행상황](#v2-진행상황) 참고 |
| Legacy / synthetic (참고용) | `experiments/synthetic_legacy/` | ⚠️ 능동 유지보수 안 함, 최종 수치로 인용 금지 |
| 교차 환경 비교 (supplementary) | `experiments/synthetic_legacy/cross_env_comparison.py` | headline과 legacy를 나란히 보여줄 뿐, 하나의 벤치마크로 합치지 않음 |

---

## 문제 상황

AI 에이전트 여러 개가 서로 대화하며 일을 처리하는 시스템(MAS)이 있다.

```
사용자 → Agent_0 → Agent_1 → Agent_2 → Agent_3 → 결과 반환
```

공격자는 외부 콘텐츠(검색 결과, 문서 등) 안에 악성 명령을 숨겨 넣는다. Agent_0(진입점)가
오염된 지시를 내리면, 그 출력을 받은 하류 agent 전체가 도미노처럼 침해되는 것이 간접 프롬프트
인젝션(Indirect Prompt Injection) 공격이다.

**기존 방어의 한계:** LLM 응답 내용을 일일이 검사하면 느리고, 모든 에이전트 내부에 접근해야
하므로 실제 배포 환경에서 적용이 어렵다.

**명명 규칙:** 특정 workflow를 전제하지 않도록 에이전트는 generic ID(`Agent_0`~`Agent_3`)로
표기한다. 실제 코드에서 사용한 예시 prompt 역할(`AGENT_ROLES`: Agent_0=orchestration,
Agent_1=research, Agent_2=analysis, Agent_3=writing)은 별도 문서화만 하고, 모델 입력·그래프
구조·결과 표에는 전혀 반영되지 않는다.

---

## 핵심 아이디어

응답 **내용을 보지 않고**, 통신 메타데이터만으로 공격을 탐지한다.

```
정상 Agent_0:  토큰 504개  |  ctx_delta 1.0
침해된 Agent_0: 토큰 780개  |  ctx_delta 1.0   (하류로 갈수록 편차 커짐)
```

### 왜 GNN인가

에이전트들은 서로 연결돼 있어 **관계 구조**가 중요하다. Agent_0만 따로 보면 "살짝 이상한가?"
수준이지만, Agent_0→Agent_1→Agent_2→Agent_3 연결을 함께 보면 "Agent_0가 오염됐고 전체
파이프라인으로 전파 중"을 잡을 수 있다. 이 관계 구조를 학습하는 것이 GNN의 역할이다.

단, 16단계 ablation에서 실제로는 **이 데이터셋 조건에서 그래프 구조 자체의 명시적 이점이
측정되지 않았다**(아래 [그래프 구조 기여도](#그래프-구조-기여도-검증) 참고) — 신호가 강해서
Z-score조차 포화되기 때문이며, 더 어려운 공격 조건에서 재검증이 필요하다.

### 시스템 모델

```
G = (A, E, M)
A: 에이전트 집합  = {Agent_0, Agent_1, Agent_2, Agent_3}
E: 통신 엣지      = {(0→1), (1→2), (2→3), (0→2)}
M: 메타데이터(모델 입력) = {token_count, ctx_delta}
```

그래프 구조(`nodes`/`edges`/`primary_predecessor`)의 유일한 정의처는
`experiments/real_llm/config/topology_4agent_v1.json`이며, `load_topology()`가 매 실행마다
unknown node/중복 edge/self-loop/disconnected node/predecessor-edge 정합성을 assert로
검증한다.

**Agent_2의 predecessor 결정:** Agent_2는 들어오는 edge가 두 개다(Agent_1→Agent_2,
Agent_0→Agent_2). ctx_delta는 predecessor 하나를 전제로 하므로 topology config의
`primary_predecessor`로 명시적으로 고정했다(Agent_2 → Agent_1 기준). Agent_0→Agent_2 edge는
GCN 인접행렬에는 반영되지만 ctx_delta 계산에는 쓰이지 않는다.

```
ctx_delta_i = token_count_i / max(token_count_{primary_predecessor(i)}, 1)
ctx_delta_entry = 1.0   # predecessor가 없는 진입 노드(Agent_0)
```

### Feature 구성: Core-2

```python
CORE_FEATURES       = ["token_count", "ctx_delta"]   # 모델 입력
DIAGNOSTIC_FEATURES = ["latency", "sentence_count", "joint_deviation_flag"]  # 기록만, 학습/평가 미사용
```

**근거:** latency는 real-LLM(decode-bound Ollama) 환경에서 token_count와 상관계수
r=0.95~0.99로 사실상 중복이라(role/조건 고정 후에도 유지, pooling 아티팩트 아님) 제거해도
F1 손실이 거의 없다. sentence_count/joint_deviation_flag도 추가 이득이 없었다. 상세 ablation:
`experiments/real_llm/feature_ablation.py`, `feature_correlation_breakdown.py`.

---

## 구현: LightGAE (Lightweight Graph Autoencoder)

**학습 방식:** 정상 통신 패턴만 보고 학습하는 One-Class(novelty) detector — 재구성이 잘 되면
정상, 오차가 크면 이상(공격).

```
Input  X ∈ R^{B × |V| × 2}
  ├─ GCNLayer 1:  2  → 16
  ├─ GCNLayer 2:  16 →  8
  ├─ DecoderLayer 1:   8 → 16
  └─ DecoderLayer 2:  16 →  2

총 파라미터: 362개
추론 속도:   0.12 ms/session (CPU)
```

**Ground-truth label:** `ground_truth_label = int(injection_enabled)` — 세션 수집 시 injection
template을 삽입했는지 여부로만 결정한다. 응답 텍스트의 키워드 관측 여부는 label에 절대
반영하지 않고, `indicator_observed`라는 별도 진단 필드로만 기록한다(모델 평가에 미사용).

**Normal-only novelty detection, 3-way split:**

| Split | Normal | Attack | 용도 |
|-------|:---:|:---:|------|
| Normal Train | 30 (60%) | 0 | `model.fit`/`scaler.fit` — 비지도, attack 데이터 미사용 |
| Normal Validation | 10 (20%) | 0 | threshold(95th percentile) 산정 전용 |
| Normal Test | 10 (20%) | — | 최종 metric에만 사용 |
| Attack Test | — | 50 | test 전용 |

Split은 **원본 task_id 단위 group split**이다(`group_split_3way()`) — 같은 task를 반복
실행한 세션들은 항상 같은 split에 통째로 들어가, 반복 실행이 train/test에 걸쳐 나뉘어
결과가 부풀려지는 걸 방지한다.

**Threshold 정책:** `threshold = percentile(normal_validation_scores, 95)`,
`prediction = int(session_score > threshold)`. Percentile은 `THRESHOLD_PERCENTILE` 상수
하나로 조정 가능. 매 seed·method별 threshold 값/validation score 분포/test
score·prediction·ground-truth 전부가 `results_summary.json`에 저장돼 재실행 없이 감사 가능.

---

## 헤드라인 결과 (real_llm_v1, N=50 정상 + 50 공격, 5-seed)

`experiments/real_llm/lgnn_experiment.py` 단일 스크립트로 재현 가능. `Ollama llama3.2`, 4-agent
파이프라인, injection은 Agent_0 진입 → 하류 cascade.

| Method | AUC | F1 | Precision | Recall | FPR | AUPRC | Recall@5%FPR |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Z-score (baseline) | 1.0000 | 0.9902±0.0088 | - | - | - | - | - |
| MLPAE (no graph) | 0.9991±0.0018 | 0.9921±0.0097 | - | - | - | - | - |
| **LightGAE (제안)** | **1.0000** | **0.9941±0.0048** | **0.9882** | **1.0000** | **0.0586** | **1.0000** | **1.0000** |

paired t-test(F1, N=5 seeds): LightGAE vs MLPAE p=0.75, LightGAE vs Z-score p=0.18 — **어느
쪽도 통계적으로 유의미하지 않다.** 세 방법 모두 AUC가 1.0에 가깝게 포화되는 것은 Agent_3
token ratio가 3.97배까지 벌어질 만큼 이 데이터셋의 공격 효과크기가 커서(easy separation)
발생하는 ceiling effect이며, 그래프 구조의 우위를 가리는 원인이기도 하다(아래 참고).

### 그래프 구조 기여도 검증

기존 캐시(신규 Ollama 호출 없음)로 split/scaler/threshold/feature/파라미터 예산/optimizer/
epoch/seed를 헤드라인과 동일하게 고정하고 6개 방법만 바꿔 비교했다
(`experiments/real_llm/baseline_ablation.py`).

| Method | AUC | F1 |
|---|:---:|:---:|
| Z-score | 1.0000 | 0.9902±0.0088 |
| Node-wise MLP-AE (노드 간 정보 혼합 없음) | 1.0000 | 0.9864±0.0144 |
| Flattened MLP-AE (4 agent concat, topology 무지) | 1.0000 | 0.9941±0.0079 |
| LightGAE No-edge | 1.0000 | 0.9821±0.0220 |
| LightGAE Random-edge | 1.0000 | 0.9883±0.0141 |
| **LightGAE Correct-edge (헤드라인)** | **1.0000** | **0.9941±0.0079** |

6개 방법 모두 AUC=1.0000, F1도 paired t-test로 Correct-edge가 Random-edge(p=0.53)·
No-edge(p=0.37)·Flattened MLP-AE(p=1.00, 완전 동일)와 유의미하게 다르지 않다. **현재
데이터셋/공격 설계에서는 그래프 구조 자체의 명시적 이점이 측정되지 않는다** — 성능 저하가
아니라, 신호가 매우 강해서(Z-score조차 AUC=1.0) 구조적 이점이 가려지는 것으로 해석한다.
논문에서 그래프 구조의 기여를 주장하려면 신호가 약한 조건(length-preserving 공격 등)에서
재검증이 필요하다.

### Node-level Localization

공격 세션의 agent별 평균 이상 점수(seed=123 대표):

| Agent | Mean Score | Max Score |
|-------|:---:|:---:|
| Agent_0 (진입점) | 7.15 | 162.14 |
| Agent_1 | 10.31 | 324.79 |
| **Agent_2** | **29.48** | 358.16 |
| Agent_3 | 18.25 | 55.71 |

`configs/attacks/chain.json` × `session_metadata_attack.json` × `results_summary.json`의
`test_node_scores`를 조인해 propagation ground-truth 기반 localization을 측정했다
(`experiments/real_llm/localization_analysis.py`).

| 지표 (5-seed 평균) | 값 |
|---|:---:|
| Entry-node Top-1 accuracy | 0.02 |
| Entry-node MRR | 0.326 |
| Compromised-node mean rank | 3.336 / 4 |
| Affected-node Hit@1 | 1.00 (자명함 — 아래 참고) |

**핵심 발견:** 진입 노드(Agent_0)가 top-1으로 뽑히는 비율은 겨우 2%이고, 4개 agent 중 평균
이상 점수가 가장 낮다. 즉 LightGAE의 node score는 **공격이 실제로 들어온 지점이 아니라
cascade 효과가 가장 크게 누적되는 지점(Agent_2)을 가리킨다** — 침해 원인(attribution)이
아니라 사고 조사 우선순위(triage)를 제공하는 신호라는 뜻이다. (Affected-node Hit@1/score
ratio는 현재 7개 attack_type이 전부 `chain.json` 하나에서 나와 진입점·전파경로가 동일하므로
자명하다 — 의미 있는 측정은 진입점이 다양한 공격이 필요하다.)

---

## v2 진행상황

교수님 피드백(공격 설계 편향, 라벨 개념 혼재, task/attack 다양성 부족)에 따라 `main`
브랜치에서 진행 중인 재설계 작업. 상세 원칙은 `experiments/real_llm/analysis_plan.md` 참고,
v1 pilot 상태는 `v1` 브랜치에 그대로 보존.

**완료:**
- 연구 전제·공격 라벨 4분리(`injection_present`/`indicator_observed`/`goal_success`/
  `propagation_observed`) 문서화 및 구현
- 공격 설계 편향 문구 제거, `detect_indicator_pattern()`/`goal_success()` 분리
- Agent_0 프롬프트를 고정 instruction + external content 채널로 분리(indirect injection 구조)
- Raw Ollama telemetry 스키마 확장(`prompt_eval_count`/`eval_duration`/`hardware_backend`/
  `gpu_name`/`ollama_version` 등, `experiments/real_llm/lgnn_experiment.py`/`collect_normal.py`)
- 후보 feature pool 구현(`feature_pool_v2.py`) — token/timing/agent-normalized z-score/
  session-level/orchestration 5개 카테고리
- **GPU 환경 검증 완료** — RTX 5060, Ollama 0.32.1, 100% GPU 로딩 확인. 세션당 약 26.5초
  (CPU 대비 6~7배 개선), 300세션 정식 수집 예상 약 2.2시간
- 공격 시나리오를 5개 목표(`task_override`/`workflow_corruption`/`misinformation`/
  `unauthorized_disclosure`/`downstream_propagation`) × overt/contextual 2변형(총 10개
  템플릿)으로 재설계 중(`configs/attacks/v2/`) — 공격 성공 여부는 `goal_success`/
  `propagation` 기준으로만 판단하고 detector 점수는 튜닝에 사용하지 않음

**진행 중 / 대기:**
- 재설계 공격 템플릿의 round-2 검증(disjoint task set 기준) — screening 진입 준비조건 확인 중
- Feature screening(20-task), feature ablation set 확정, task/attack 50-set 본작업,
  300세션 정식 수집은 위 검증 통과 후 진행 예정

---

## 현재 한계

| 한계 | 상태 |
|------|------|
| **그래프 구조 우위 미측정 (real-LLM)** | ceiling effect(AUC 1.0 포화) 때문 — [그래프 구조 기여도 검증](#그래프-구조-기여도-검증) 참고, v2 length-preserving 공격으로 재검증 예정 |
| **LightGAE vs MLPAE/Z-score F1 차이 통계적 미검증** | paired t-test 모두 비유의(p=0.75, 0.18) — "어느 쪽이 높은가"는 노이즈 안에 있음 |
| **단일 모델** | llama3.2만 검증. 다른 LLM 일반화는 향후 과제 |
| **AUC 포화 (1.0)** | Agent_3 token ratio 3.97x로 효과크기가 매우 커서 발생. v1 공격셋(verbosity-inflation, 단일 진입점)의 한계이기도 함 — v2 공격 다양화로 재검증 중 |
| **공격 다양성 부족 (v1)** | 7개 attack_type 전부 동일 진입점(Agent_0)·동일 전파경로. v2에서 5개 목표로 재설계 중 |
| **observed_propagation_path 미보유 (v1)** | v1 캐시는 feature만 저장, 원문 텍스트 없음 — v2 raw telemetry부터 원문 저장으로 해결됨 |

---

## 프로젝트 구조

```
MAS/
├── data/
│   ├── tasks/                               # v1: 5 categories x 10 tasks = 50 (아직 session generator 미연동)
│   │   ├── summarization.json / qa.json / comparison.json / planning.json / technical_reasoning.json
│   │   └── v2/
│   │       ├── mini_validation.json         # round-1 attack development task (5개, 카테고리당 1개)
│   │       └── validation_round2.json       # round-2 attack validation task (round-1과 격리된 5개)
│   └── splits/
│       └── normal_task_split_v1.json        # task_id 기준 고정 split (train 30 / val 10 / test 10)
├── configs/attacks/
│   ├── direct.json / slow.json / chain.json / length_preserving.json   # v1 (chain만 실제 수집에 사용됨)
│   └── v2/                                  # 5개 목표 x overt/contextual = 10 템플릿, hop별 propagation criterion 포함
│       ├── task_override.json / workflow_corruption.json / misinformation.json
│       └── unauthorized_disclosure.json / downstream_propagation.json
├── experiments/
│   ├── real_llm/                          # ★ Headline + v2 개발
│   │   ├── config/topology_4agent_v1.json # 그래프 구조의 유일한 정의처
│   │   ├── analysis_plan.md               # v2 연구 전제·공격 라벨 정의 (LOCKED)
│   │   ├── lgnn_experiment.py             # ★★★ LightGAE + 실제 LLM — headline 진입점 (v2 스키마로 계속 진화 중)
│   │   ├── collect_normal.py              # v2 정상 세션 수집기 (lgnn_experiment.py와 스키마 동기화됨)
│   │   ├── feature_pool_v2.py             # v2 후보 feature pool (token/timing/agent-norm/session/orchestration)
│   │   ├── smoke_test.py                  # Phase 1 스키마·하드웨어 검증 (5 task x 2 condition)
│   │   ├── mini_validation.py             # 공격 템플릿 round-1/2 검증 러너 (--tasks/--attacks/--label)
│   │   ├── task_loader.py / generate_task_split.py
│   │   ├── attack_loader.py / generate_attack_pairs.py
│   │   ├── feature_ablation.py / feature_correlation_breakdown.py
│   │   ├── baseline_ablation.py           # strong baseline + graph ablation
│   │   ├── localization_analysis.py       # propagation ground-truth + node-level localization
│   │   ├── experiment.py                  # [superseded]
│   │   └── patch_*.py                     # (완료된 1회성 마이그레이션, 캐시에 이미 반영됨)
│   └── synthetic_legacy/                  # ⚠️ Legacy — 참고용, 최종 결과 아님
├── notebooks/
│   └── colab_data_collection.ipynb        # GPU 미확보 시 대체 경로 (현재는 로컬 GPU로 충분, 참고용 유지)
├── output/
│   ├── real_llm/                          # ★ Headline + v2 결과물
│   │   ├── results_summary.json           # 헤드라인 AUC/F1/threshold/per_seed 감사 기록
│   │   ├── baseline_ablation_results.json / localization_metrics.json / propagation_ground_truth.json
│   │   ├── cache_{normal,attack}.json / session_metadata_{normal,attack}.json / dataset_summary.csv
│   │   ├── smoke_test_raw_telemetry.json  # GPU 검증용 (schema-validation only)
│   │   ├── attack_development_round1_records.json / attack_validation_round2_records.json  # attack 개발용 데이터 (formal 아님)
│   │   └── lgnn_fig*.png
│   └── synthetic_legacy/                  # ⚠️ Legacy 결과물
└── .claude/
```

---

## 실행 방법

```bash
# 환경 설정
pip install numpy scikit-learn matplotlib torch requests networkx scipy

# ★ Headline 실험 (Ollama 필요, GPU 사용 시 약 5분 / CPU만 있으면 캐시로 즉시 완료)
.\.venv\Scripts\python.exe -u experiments/real_llm/lgnn_experiment.py

# v2 스키마 검증 (5 task x 2 condition = 10 세션)
.\.venv\Scripts\python.exe -u experiments/real_llm/smoke_test.py

# v2 공격 템플릿 검증
.\.venv\Scripts\python.exe -u experiments/real_llm/mini_validation.py --tasks data/tasks/v2/validation_round2.json --attacks "configs/attacks/v2/*.json" --label attack_validation_round2
```

실제 LLM 실험은 crash recovery를 지원한다 — 중단 후 재실행 시 수집된 세션은
`output/real_llm/cache_*.json`에서 자동 복원된다.

```bash
# (선택, legacy) synthetic 시뮬레이션 — 참고용, 최종 결과에는 쓰지 않음
python experiments/synthetic_legacy/lgnn/mas_lgnn.py
python experiments/synthetic_legacy/lgnn/mas_lgnn_5agent.py

# (선택, supplementary) headline vs legacy 교차 환경 비교
python experiments/synthetic_legacy/cross_env_comparison.py
```

---

## 환경

| 항목 | 값 |
|--------|------|
| Python | 3.11.x |
| PyTorch | 2.3.1+cpu |
| NumPy | 1.26.4 |
| scikit-learn | 1.6.1 |
| Ollama | 0.32.1 |
| GPU | NVIDIA RTX 5060 (100% GPU 로딩 확인, `hardware_backend` 필드로 세션마다 기록) |

> LLM 추론·모델 학습 모두 이 환경 기준 수치다. `hardware_backend`가 다르면(예: CPU-only)
> latency 계열 수치는 재현되지 않을 수 있으나, token_count 기반 Core-2 결과는 하드웨어와
> 무관하게 재현되어야 한다.
