# MAS Security — Lightweight GNN-based Anomaly Detection

**멀티에이전트 AI 시스템 환경 구축 및 Quick Identification 기술 개발**  
1차년도 연구과제 예비 실험

---

## 연구 개요

멀티에이전트 AI 시스템(MAS)에서 발생하는 **간접 프롬프트 인젝션(Indirect Prompt Injection)** 공격을 에이전트 간 통신 메타데이터와 경량 그래프 신경망(Lightweight GNN)으로 탐지하는 연구다.

### 핵심 아이디어

에이전트 응답 내용에 접근하지 않고, **통신 메타데이터만으로** 이상 탐지 가능:

```
메타데이터 피처 M = {δ: latency,  τ: token_count,  f: api_freq,
                     Δc: ctx_delta,  s: call_seq}
```

시스템 모델 `G = (A, E, M)`:
- **A** — 에이전트 집합 `{Orchestrator(v0), Researcher(v1), Writer(v2)}`
- **E** — 통신 엣지 `{(v0,v1), (v1,v2), (v0,v2)}`
- **M** — 각 엣지/노드의 메타데이터 피처 벡터

---

## 1차년도 연구목표 매핑

| 연구 목표 | 구현 상태 | 위치 |
|-----------|-----------|------|
| MAS 환경 정형화 및 구축 | ✅ `G = (A, E, M)` 수식 정의 | `experiments/lgnn/mas_lgnn.py §1` |
| 에이전트 간 통신 메타데이터 수집 | ✅ 5종 피처, 4가지 공격 유형 | `experiments/lgnn/mas_lgnn.py §2` |
| 정상/이상 통신 패턴 경계 정의 | ✅ Adaptive Threshold | `experiments/simulation/` |
| **시공간 그래프 모델링** | ✅ 슬라이딩 윈도우 + GCN | `experiments/lgnn/mas_lgnn.py §3` |
| **Lightweight GNN 아키텍처** | ✅ LightGAE (461 파라미터) | `experiments/lgnn/mas_lgnn.py §3` |
| 노드·엣지 수준 경보 생성 | ✅ Per-agent anomaly score | `experiments/lgnn/mas_lgnn.py §5` |
| Quick Identification | ✅ 0.0005 ms/sample | `experiments/lgnn/mas_lgnn.py §5` |
| 실제 LLM 환경 검증 | ✅ Ollama llama3.2 실험 | `experiments/real_llm/` |

---

## 프로젝트 구조

```
MAS/
├── experiments/
│   ├── simulation/
│   │   └── mas_experiment.py     # 시뮬레이션 실험 (4 Baseline + Adaptive Threshold)
│   ├── real_llm/
│   │   └── experiment.py         # 실제 LLM 실험 (Ollama llama3.2 + 간접 인젝션)
│   └── lgnn/
│       └── mas_lgnn.py           # ★ LightGAE 핵심 실험
└── output/
    ├── simulation/               # 시뮬레이션 실험 Figure (6종)
    ├── real_llm/                 # 실제 LLM 실험 결과 (8종)
    └── lgnn/                     # LightGAE Figure (6종)
```

---

## 실험 설명

### 1. Simulation — `experiments/simulation/mas_experiment.py`

4종 Baseline 비교 + Adaptive Threshold 메커니즘 검증.

```bash
python3 experiments/simulation/mas_experiment.py
```

| Method | TPR | FPR | F1 | AUC |
|--------|-----|-----|----|-----|
| Threshold (B1) | — | — | — | — |
| Isolation Forest (B2) | — | — | — | — |
| Z-score (B3) | — | — | — | — |
| GNN fixed θ (B4) | — | — | — | — |
| **GNN + Adaptive θ** | — | — | — | — |

> 실행 시 수치 채워짐. 격리(Isolation) 메커니즘으로 전파율 감소 검증.

---

### 2. Real LLM — `experiments/real_llm/experiment.py`

Ollama(llama3.2)로 실제 에이전트 파이프라인 구성, 3가지 간접 인젝션 공격 테스트.

```bash
# Ollama 먼저 실행 필요
ollama serve
python3 experiments/real_llm/experiment.py
```

**공격 유형:**
- 문서 메타데이터 위장 인젝션
- 역할 탈취(Role Hijack)
- 권한 상승 시도(Authority Escalation)

---

### 3. LightGAE ★ — `experiments/lgnn/mas_lgnn.py`

핵심 실험. 진짜 GCN 기반 Graph Autoencoder로 one-class 이상 탐지.

```bash
python3 experiments/lgnn/mas_lgnn.py
```

#### 모델 구조

```
Input  X ∈ R^{B × N × F}   (batch, 3 agents, 5 features)
  │
  ├─ GCNLayer 1:  H' = σ(Â H W₁)   5  → 16
  ├─ GCNLayer 2:  H  = Â H' W₂    16  →  8   (node embeddings)
  │
  ├─ Readout: z = mean(H, dim=N)        (graph embedding)
  │
  ├─ DecoderLayer 1:  8  → 16
  └─ DecoderLayer 2: 16  →  5   (reconstruction)

총 파라미터: 461개  |  추론: 0.0005 ms/sample
```

#### 학습 방식

정상 세션만으로 학습(One-Class). 이상 세션은 재구성 오차(Reconstruction Error)가 높게 나타남.

```
Loss = MSE(X_reconstructed, X_normal)
Anomaly Score = mean per-node reconstruction error
```

#### 공격 유형 (4가지)

| 유형 | 설명 | Researcher p | Writer p |
|------|------|:---:|:---:|
| Type-I Direct | 직접 역할 탈취 | 1.0 | 0.0 |
| Type-II Harvest | 권한 수집 + 전파 | 0.8 | 0.35 |
| Type-III Slow | 점진적 오염 | 0.4 | 0.15 |
| Type-IV Flood | 전체 동시 오염 | 0.65 | 0.65 |

#### 탐지 성능

| Method | TPR | FPR | F1 | AUC | 추론 속도 |
|--------|-----|-----|----|-----|-----------|
| Threshold (B1) | 0.0159 | 0.0288 | 0.0312 | 0.5247 | ~0ms |
| IsoForest (B2) | 0.9984 | 0.0994 | 0.9967 | 0.9988 | 0.071ms |
| Z-score (B3) | 1.0000 | 0.0288 | 0.9993 | 1.0000 | 0.0005ms |
| Sliding GNN (B4) | 0.9998 | 0.0000 | 0.9999 | 1.0000 | 0.0005ms |
| **LightGAE** | **0.9973** | **0.0417** | **0.9976** | **0.9996** | **0.0005ms** |

#### 노드 수준 이상 탐지 (에이전트 식별)

| 공격 유형 | Orchestrator | Researcher | Writer |
|-----------|:---:|:---:|:---:|
| Type-I Direct | 3.42 | **51.39** | 3.37 |
| Type-II Harvest | 3.86 | **27.76** | 3.15 |
| Type-III Slow | 1.63 | **7.57** | 1.01 |
| Type-IV Flood | 4.87 | 14.74 | **15.94** |

> Researcher 침해 시 Researcher 점수 급증, 전체 오염(Type-IV) 시 Writer도 함께 감지.

---

## 생성 Figure 목록

### LightGAE (`output/lgnn/`)
| 파일 | 내용 |
|------|------|
| `lgnn_fig1_mas_graph.png` | MAS 그래프 구조 + 공격 전파 패턴 |
| `lgnn_fig2_feature_dist.png` | 공격 유형별 메타데이터 분포 |
| `lgnn_fig3_embedding_pca.png` | 학습된 그래프 임베딩 PCA 시각화 |
| `lgnn_fig4_roc.png` | ROC 곡선 — 4 Baseline vs LightGAE |
| `lgnn_fig5_performance.png` | TPR/FPR/F1/AUC 성능 비교 Bar |
| `lgnn_fig6_node_timing.png` | 노드별 이상 점수 히트맵 + 추론 속도 |

### Simulation (`output/simulation/`)
Figure 1~6: 피처 분포, ROC, 성능, Adaptive Threshold, 격리 효과, 전파 분석

### Real LLM (`output/real_llm/`)
실제 llama3.2 환경 피처 분포 및 이상 점수 결과

---

## 환경 설정

```bash
pip3 install numpy pandas scikit-learn matplotlib torch networkx
```

| 패키지 | 버전 |
|--------|------|
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.9.4 |

---

## 논문 방향 (WISA 2026 포스터 목표)

**제목안:** *Lightweight Graph Autoencoder for Quick Identification of Anomalous Interactions in Multi-Agent AI Systems*

**핵심 기여:**
1. MAS 환경의 공식 그래프 모델 `G = (A, E, M)` 정의
2. 461 파라미터 LightGAE — 진짜 GCN 기반 one-class 탐지
3. 노드 수준 이상 탐지 → 침해 에이전트 직접 식별
4. Quick Identification: 0.0005 ms/sample, IsoForest 대비 142배 빠름

**다음 단계:**
- Safety Filter 신호를 탐지 증폭기로 활용하는 실험 설계
- 실제 LLM(Ollama)에 LightGAE 탑재 검증
- 세션 수 200+ 확장 + Mann-Whitney U test (p-value)
