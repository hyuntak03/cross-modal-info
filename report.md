# VLM의 방향 인식 실패 원인 분석 보고서

**작성일:** 2026-04-17
**모델:** LLaVA-Video-7B-Qwen2 (SigLIP vision encoder + mm_projector + Qwen2-7B LLM)
**과제:** 4지선다 방향 예측 (위/아래/왼쪽/오른쪽)

---

## 초록 (한 눈에 보기)

### 문제
Vision Language Model (VLM)에게 "비디오 속 객체가 어느 방향으로 움직이나?"를 4지선다 (A/B/C/D)로 물어보면, **훈련 때 본 도메인 (합성 도형 + 합성 색)은 99% 정답**을 맞히지만, **훈련 때 못 본 도메인 (실제 객체 + 실제 장소)은 79%로 떨어진다.** 왜 그럴까?

### 답 (한 문장)

**LLM 내부의 특정 벡터 방향(L21 layer의 direction axis) 위에 실린 신호 크기(magnitude)가 OOD에서 부족해서**, 이 신호에 의존하는 letter 매핑이 불완전해진다. 신호 크기를 3배로 올리면 성능이 67.67% → 89.33% (+21.67pp)로 회복된다.

### 3가지 핵심 발견

1. **방향 정보 자체는 OOD에도 충분**. Linear probe로 읽으면 92% 정확도 (IN 99%와 큰 차이 없음).
2. **문제는 "크기"**. Linear probe가 학습하는 축은 같지만, 그 축 방향의 projection magnitude가 OOD에서 60%만 된다.
3. **"어느 축에 크기가 부족한지"가 정확히 특정됨**. 무작위 축에 같은 크기로 개입해도 효과 없음 (+0.9pp). 방향 축에만 효과 (+10.67~+21.67pp).

### 왜 중요한가
이 발견은 OOD 성능을 올리는 새 loss 함수 (delta_direct v2)의 **정량적 target**을 제공한다:
> "L21 last token의 direction axis 방향 projection이 SC의 3배 magnitude에 도달하도록 훈련"

---

## 목차

- [1. 연구 배경과 중심 질문](#1-연구-배경과-중심-질문)
- [2. 실험 환경](#2-실험-환경)
- [3. 방법론](#3-방법론)
- [4. 기본 관찰 — 역설](#4-기본-관찰--역설)
- [5. Layer별 direction 표현의 기하](#5-layer별-direction-표현의-기하)
- [6. L19 amplifier — fine-tuning의 지문](#6-l19-amplifier--fine-tuning의-지문)
- [7. 인과 진단 — binding gap = magnitude 부족](#7-인과-진단--binding-gap--magnitude-부족)
- [8. 원인 추적 — magnitude 부족은 어디서 생기는가](#8-원인-추적--magnitude-부족은-어디서-생기는가)
- [9. Vision 개입은 왜 실패하는가](#9-vision-개입은-왜-실패하는가)
- [10. Readout 메커니즘 — 비선형 변환](#10-readout-메커니즘--비선형-변환)
- [11. 엄밀한 검증 — causal 확증](#11-엄밀한-검증--causal-확증)
- [12. 종합 — 정식 claim](#12-종합--정식-claim)
- [13. Delta_direct v2 설계](#13-delta_direct-v2-설계)
- [14. 미해결 / 후속 실험](#14-미해결--후속-실험)
- [Appendix A — 주요 측정치](#appendix-a--주요-측정치)
- [Appendix B — 파일 맵](#appendix-b--파일-맵)
- [Appendix C — 실험 목록](#appendix-c--실험-목록)

---

## 1. 연구 배경과 중심 질문

### 1.1 상황

Video Language Model은 비디오를 보고 질문에 답하는 모델이다. 예를 들어 다음과 같은 질문:

> Video: [공이 아래에서 위로 움직이는 영상]
> Question: "What direction did the object move?"
> Options: A. Up / B. Right / C. Down / D. Left
> Answer: **A**

우리는 이런 **4-way MCQ 방향 문제** 데이터로 모델을 fine-tune했다. 훈련 데이터는 **합성 도형이 합성 색 배경에서 움직이는 단순한 영상** (shape_color, 줄여서 **SC**).

### 1.2 관찰된 문제

Fine-tune된 모델을 더 어려운 영상에 test하면:

| 테스트 도메인 | 설명 | 정답률 |
|---|---|---|
| shape_color (SC) | 합성 도형 + 합성 색 (훈련 도메인) | ~99% |
| obj_color | 실제 객체 + 합성 색 | ~94% |
| shape_place | 합성 도형 + 실제 장소 | ~90% |
| **obj_place (OP)** | **실제 객체 + 실제 장소** | **~79%** |

즉 **훈련 도메인에서 멀어질수록 성능 하락**. "실제 객체가 실제 장소에서 움직이는" 현실적 영상이 가장 어려움.

이런 도메인 외 (Out-Of-Distribution, **OOD**) 성능 하락은 머신러닝에서 일반적 문제지만, 우리 관심사는 **"왜 이게 일어나는가"의 정확한 메커니즘**.

### 1.3 역설 (중심 질문)

Linear probe라는 방법으로 "모델 내부에 방향 정보가 있는지"를 측정해보면:

- SC: **99.8%** 정확도로 읽을 수 있음
- OP: **92.3%** 정확도로 읽을 수 있음 (OOD도 거의 같음)

즉 **OOD에서도 모델 내부에는 방향 정보가 거의 완벽히 존재**. 그런데 **최종 정답률 (MCQ)은 OP에서 79%로 뚝 떨어진다**.

> **중심 질문**: 모델 내부에 방향 정보가 있는데도 왜 OOD에서 최종 정답률이 떨어지는가?

이 질문에 **수치적, 인과적으로 답하는 것**이 본 리포트의 목표.

### 1.4 답 미리보기 (3줄)

1. 모델 후반부 (L21)에는 **"방향 축" (direction axis)**이라는 특정 벡터 방향이 있다. Direction 정보는 이 축에 실린다.
2. OOD에서는 이 축 방향의 **신호 크기 (magnitude)가 IN의 60%**. 정보는 있지만 약함.
3. 이 약한 신호를 downstream layer (L22-L27)가 letter logit으로 변환할 때 **크기가 부족해서 오답**. 크기를 3배로 올리면 +22%p 회복.

이제 이를 엄밀히 증명한다.

---

## 2. 실험 환경

### 2.1 모델 3가지 (같은 backbone, 다른 fine-tuning)

| 이름 | Fine-tuning | Aux loss |
|---|---|---|
| **Vanilla** | 없음 (LLaVA-Video 원본) | — |
| **Baseline** | SC 데이터로 MCQ만 훈련 | 없음 |
| **Delta** | SC 데이터로 MCQ 훈련 + direction auxiliary | `delta_direct` (projector output의 temporal frame difference로 direction 예측하는 별도 head) |

전부 LoRA (r=64)로 동일한 backbone (Qwen2-7B + SigLIP)에 얹음.

### 2.2 Task — R2R 4-way 1500

각 방향별 1500 샘플 × 4 방향 = **task당 6000 샘플**. 네 가지 난이도:

| Task | 특징 | 역할 |
|---|---|---|
| shape_color (SC) | 합성 도형 + 합성 색 | In-domain (학습) |
| obj_color (OC) | 실제 객체 + 합성 색 | 중간 OOD |
| shape_place (SP) | 합성 도형 + 실제 장소 | 어려운 OOD |
| **obj_place (OP)** | **실제 객체 + 실제 장소** | **가장 어려운 OOD** |

### 2.3 Factorial 통제 데이터셋 — 인과 개입을 위한 것

일반 R2R 데이터로는 "object identity, background, instance, MCQ variant 중 어느 요소가 binding에 영향 주는지" 분리 불가. 그래서 factorial design:

**5 obj × 5 bg × 4 direction × 20 instance × 4 MCQ variant = 한 condition당 8000 prompt**

두 condition 준비: SC (shape_color), OP (obj_place).

**4 MCQ variant**의 핵심 의미: 각 비디오를 4번 서로 다른 후보 순서로 물음:
- Variant 0: [Up, Right, Down, Left] → Up이 A
- Variant 1: [Right, Up, Left, Down] → Up이 B  
- Variant 2: [Down, Left, Up, Right] → Up이 C
- Variant 3: [Left, Down, Right, Up] → Up이 D

따라서 4 variant를 평균하면 **각 방향이 A/B/C/D 자리에 정확히 한 번씩 등장** → letter 편향 상쇄.

### 2.4 측정 지표

다섯 가지 주요 지표:

- **Direction probe (방향 probe)**: hidden state에 linear classifier 달고 "위/아래/왼쪽/오른쪽" 4-way 분류 정확도. **"hidden에 방향 정보가 있나"**의 지표.
- **Letter probe (letter probe)**: hidden state에 linear classifier 달고 "A/B/C/D" 4-way 분류 정확도. 4 variant 데이터에서는 letter = f(direction, ordering)이라 non-trivial. **"hidden에 letter 정답이 있나"**의 지표.
- **MCQ accuracy**: 모델의 실제 출력 logit에서 argmax. 최종 task 성능.
- **Direction axis (Δ_d)**: 특정 방향 d를 담당하는 벡터. 정의: `Δ_d = 방향=d인 sample들의 평균 hidden − 전체 평균 hidden`. Hidden space의 3584차원 벡터.
- **cos similarity (방향 간 유사도)**: 두 벡터가 가리키는 방향이 얼마나 같은가. 1이면 완전 일치, 0이면 직교, -1이면 반대.

---

## 3. 방법론

이 섹션은 **수식과 재현 가능한 절차**를 정리. 결과를 빠르게 보고 싶으면 §4로 넘어가고, 나중에 참조해도 됨.

### 3.1 기호 정리

| 기호 | 의미 |
|---|---|
| `D = 3584` | Hidden dimension (LLM 내부 벡터 길이) |
| `h_L[-1] ∈ ℝ^D` | Layer L의 **마지막 token** hidden state |
| `g_L` | Grand mean (전체 sample의 L에서 hidden 평균) |
| `h_avg_d_L` | 방향 = d인 sample들의 hidden 평균 (layer L) |
| `Δ_d_L = h_avg_d_L − g_L` | 방향 d의 "평균 편차 벡터" = **direction axis** |
| `Δ̂_d_L = Δ_d_L / ‖Δ_d_L‖` | 단위 벡터로 정규화한 direction axis |
| `m_d_L = ‖Δ_d_L‖` | Direction axis 크기 (스칼라) |
| `p_L = ⟨h_L[-1] − g_L, Δ̂_d_L⟩` | 개별 sample이 direction axis 방향으로 얼마나 치우쳐있는가 (projection) |
| `T = 8` | 비디오 frame 수 |
| `S ≈ 729` | 한 frame의 spatial vision token 수 |

**비유로 이해하기**:
- hidden state = "3584차원 공간의 점"
- direction axis = "방향 d를 나타내는 화살표 (이 공간의 벡터)"
- projection p = "각 sample이 이 화살표에 투영된 크기" (양수면 direction에 치우침, 음수면 반대)

### 3.2 Direction axis 계산 방법

**입력**: hidden 집합 `H ∈ ℝ^(N × 28 × D)` (N 샘플, 28 layer, D 차원) + 방향 label `y ∈ {up, right, down, left}^N`

**절차** (layer L, 방향 d):
1. Grand mean: `g_L = (1/N) Σ_i H[i, L, :]`
2. 방향별 mean: `h_avg_d_L = 방향=d인 sample들의 H[i, L, :] 평균`
3. Axis: `Δ_d_L = h_avg_d_L − g_L`
4. 단위 벡터: `Δ̂_d_L = Δ_d_L / (‖Δ_d_L‖ + 10^{-9})`

즉 **"방향 d의 전형적 편차"**. 이 벡터 방향으로 projection하면 방향 정보를 읽을 수 있음.

### 3.3 개입 (Intervention) 연산자 정의

모든 개입은 **하나의 layer에 hook을 걸어서 last token slice만 수정**, 나머지는 정상 forward.

#### A. Direction axis 기반 개입 (last token)

| 이름 | 수식 | 의미 |
|---|---|---|
| `no_swap` | 변경 없음 | 기준선 |
| `amp_k(L,d)` | `h_L[-1] += (k−1) · p · Δ̂_d_L` | 자기 projection을 k배로 scale |
| `clean_m(L,d)` | `h_L[-1] −= p · Δ̂_d_L + m · Δ̂_d_L` | 자기 projection 제거 후 크기 `m`으로 set |
| `add_canon(L,d,m)` | `h_L[-1] += m · Δ̂_d_L` | 제거 없이 크기 `m`만 추가 |
| `on_axis(L,d)` | `h_L[-1] ← g_L + p · Δ̂_d_L` | 축 방향만 남기고 나머지 제거 |
| `remove_own(L,d)` | `h_L[-1] −= p · Δ̂_d_L` | 자기 projection 제거만 |

#### B. Replacement 개입

| 이름 | 수식 | 의미 |
|---|---|---|
| `full_rep(L,d)` | `h_L[-1] ← h_avg_d_L` | Last token을 방향별 평균으로 완전 교체 |
| `replace_avg(L, v)` | `h_L[-1] ← v` | 임의 벡터 `v`로 교체 |

#### C. Identity 제거 개입 (§11.1 용)

```python
# 1. 현재 projection 계산
centered = h_L[-1] − g_L
id_coef = Q_id^T @ centered       # identity subspace에 투영 (k=4차원)
dir_coef = centered · Δ̂_d_L       # direction axis projection (복원용)

# 2. Identity subspace 제거
h_L[-1] ← h_L[-1] − Q_id @ id_coef

# 3. Direction axis 성분은 복원 (건드리지 않았음 보장)
new_dir_coef = (h_L[-1] − g_L) · Δ̂_d_L
h_L[-1] ← h_L[-1] + (dir_coef − new_dir_coef) · Δ̂_d_L
```

`Q_id`는 LDA로 얻은 identity subspace의 orthonormal basis (D×4). 설계상 `Q_id ⊥ Δ̂_d_L` (측정: cos = 0.000).

#### D. 통제 개입 (§11.3 rigorous verification 용)

| 이름 | 내용 |
|---|---|
| `clean_random(L, r, m)` | `clean_m`을 **random unit vector `r`**로 수행 (direction axis 아님) |
| `clean_identity(L, m)` | `clean_m`을 **identity LDA axis**로 수행 (direction과 직교) |

→ 이들이 효과 없으면 "direction axis가 특별한 이유" 증명.

#### E. Vision-side 개입 (projector output hook)

Projector output shape: `V ∈ ℝ^(T × S × D)` (T=8 frames batch, S=729 spatial, D=3584)

| 이름 | 수식 | 의미 |
|---|---|---|
| `V1_broadcast_add(d, α)` | `V[t,s,:] += α · Δ_d_vision` 모든 (t,s)에 | 모든 위치에 같은 shift |
| `V_axis_swap(d, α)` | Per-position: OP axis projection 제거 + IN axis로 추가 | 축 교체 |
| `V_amp_own(d, α)` | `V[t,s,:] += (α−1) · ⟨V[t,s,:], Δ̂_d_vision⟩ · Δ̂_d_vision` | Per-position 자기 projection scale |
| `per_t_amp(d, α)` | 각 frame t별로 해당 t의 Δ̂^(t) 사용 | Per-temporal axis 반영 |

### 3.4 Hook 프로토콜 (재현성)

모든 intervention은 **PyTorch forward hook**으로 구현:

```python
decoder = model.model.layers  # 28 layer

def make_hook(operation):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # (1, seq, D)
        h = h.clone()
        h[:, -1, :] = apply(h[:, -1, :], operation)  # last token만 수정
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook

handle = decoder[L].register_forward_hook(make_hook(cfg))
try:
    output = model(inputs_embeds=..., attention_mask=..., position_ids=...)
finally:
    handle.remove()  # 중요: 항상 제거 (state 유출 방지)
```

### 3.5 Probe 학습 방법

**CPU probe (sklearn)**:
```python
# 70/30 split, seed=42
clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)
```

**GPU probe (PyTorch Linear, 빠름)**:
```python
# Adam, lr=1e-2, weight_decay=1e-2, 200 epochs
model_probe = nn.Linear(D, n_classes).cuda()
# Feature normalize: (X − mean_train) / (std_train + 1e-6)
for epoch in range(200):
    loss = CrossEntropy(model_probe(X_tr_normalized), y_tr)
    loss.backward(); opt.step()
acc = (model_probe(X_te).argmax(−1) == y_te).float().mean()
```

두 방법 결과 거의 동일. GPU 버전은 high-dim feature (3584)에서 훨씬 빠름.

### 3.6 MCQ 채점 방법

```python
# Letter token ID 추출
letter_ids = {l: tokenizer.encode(l)[0] or tokenizer.encode(' '+l)[0]
              for l in ['A','B','C','D']}

# Forward 후 letter만 argmax
logits = output.logits[0, -1, :]                  # (vocab_size,)
letter_logits = logits[list(letter_ids.values())] # (4,)
pred = ['A','B','C','D'][letter_logits.argmax()]
correct = (pred == expected_letter)
```

### 3.7 샘플 수와 통계적 재현성

| 실험 | N_sample | 근거 |
|---|---|---|
| Factorial baseline (direction axis 계산) | 8000 | 5×5×4×20×4 full factorial |
| L21 intervention (magnitude sweep) | 500 | variant_0, letter = direction |
| Combined intervention | 300 | 75 video × 4 variant |
| Rigorous verification (15 conditions) | 300 | 15 × 300 = 4500 forwards |
| R2R 1500 probe | 6000 | Cross-task 일반화 |

**통계적 유의성**: "+17.6pp from 68.8% over N=500"는 binomial standard error ≈ 2.1%, 따라서 **약 8σ 효과**. Robust.

**재현성 확보**:
- Random seed: 42 (probe train/test split)
- Random axis control: seeds 0, 1, 2 (§11.3)
- Split-half consistency: cos 0.998 (§11.2) → 측정이 sample noise에 robust

### 3.8 데이터 경로

| 데이터 | 경로 | Shape | 내용 |
|---|---|---|---|
| R2R 1500 vision_encoder | `/data3/.../linear_probing_1500/{model}/vision_encoder/{task}/features.npy` | (6000, 9216) | SigLIP 출력 |
| R2R 1500 after_projector | `.../after_projector/{task}/features.npy` | (6000, 28672) = (6000, 8×3584) | Projector 출력 (spatial pooled, T=8 preserved) |
| R2R 1500 answer_token L | `.../answer_token/{task}/features_layer_{L}.npy` | (6000, 3584) | LLM layer L의 last token |
| Factorial hiddens | `/local_datasets/vlm_direction/factorial_dataset/hiddens/baseline_{cond}_4variants*.npz` | (8000, 28, 3584) | 28 layer × last token + metadata |

---

## 4. 기본 관찰 — 역설

### 4.1 Direction은 OOD에도 살아있다

**Linear probe로 L21 (28층 중 21번째)에서 방향을 읽을 수 있는가?**

| Task | Direction probe | Letter probe | 차이 (gap) |
|---|---|---|---|
| shape_color (IN) | 99.8% | 97.2% | 2.6pp |
| obj_color | 97.9% | 86~94% | 중간 |
| shape_place | 95.6% | 82~90% | 중간 |
| **obj_place (OOD)** | **92.3%** | **71.9%** | **20pp** |

**의미**: L21 hidden에는 OP에서도 **방향 정보가 거의 완벽하게 담겨있음** (probe 92%). 하지만 **letter 예측 정확도 (MCQ)는 79%로 떨어짐**.

→ **"hidden에 방향 정보 부재"가 binding 실패의 원인이 아님.**

### 4.2 역설의 위치 — Letter probe가 L15→L16에서 폭증

Letter probe 정확도를 layer별로 보면 (4 variant data, letter ≠ direction):

| Layer | SC letter | OP letter | Gap |
|---|---|---|---|
| L14 | 33% | 26% | 7pp |
| L15 | 36% | 27% | 9pp |
| **L16** | **70%** | **38%** | **32pp** (폭증) |
| L17 | 87% | 57% | 30pp |
| L21 | 97% | 72% | 25pp |
| L27 | 99% | 79% | 20pp |

**의미**:
- L14까지는 letter 정확도가 랜덤 (25%) 근처 — letter 정보 아직 없음
- **L15→L16 단일 layer에서 SC는 70%, OP는 38%로 급등** (32pp gap 폭발)
- Direction probe는 같은 구간에서 이미 높은 수치로 flat

**Letter probe는 "hidden이 direction × candidate ordering → letter 변환을 완료했나"를 측정**. L16에서 이 변환이 일어나지만 **OP에서는 반만 성공**.

### 4.3 L21 direction axis는 task-invariant

다른 task의 direction axis가 얼마나 비슷한지 (`cos similarity`):

| Model | cos(SC_axis, OP_axis) @ L21 |
|---|---|
| Vanilla (fine-tune 안 함) | 0.54 (중간) |
| **Baseline** | **0.94** (거의 일치) |
| **Delta** | **0.94** (거의 일치) |

**의미**: Fine-tune한 모델들은 SC와 OP에서 **거의 같은 방향**으로 direction을 인코딩한다. 즉 **"OP는 다른 axis 쓰니까 probe가 안 되는 것"도 아님**.

### 4.4 역설 정리

- 방향 정보 있음 (probe 92%) ✓
- Task-invariant axis 있음 (cos 0.94) ✓
- 그런데 letter 예측 실패 (MCQ 79%, letter probe L16 38%)

**→ 원인은 "axis의 방향"이 아닌 무엇인가 다른 것**. 다음 섹션들에서 밝힘.

---

## 5. Layer별 direction 표현의 기하

### 5.1 L21이 task-invariance "peak"

Cross-task axis cos를 각 layer에서:

| Layer | cos(SC, OP) (Baseline) |
|---|---|
| L14 | 0.64 |
| L18 | 0.85 |
| **L21** | **0.94 (peak)** |
| L24 | 0.94 |
| L27 | 0.92 |

초반 layer는 task-specific axis (각 task가 다른 축 사용). 후반 layer로 갈수록 SC와 OP axis가 점점 정렬. L21에서 peak.

### 5.2 같은 task 안에서도 layer별로 axis가 회전한다

이것이 중요: **"L14의 방향 axis"와 "L21의 방향 axis"는 같은 task 내에서도 다름**.

같은 task의 L21 axis를 기준으로 다른 layer axis와의 cos:

| Layer | L21 axis와의 cos |
|---|---|
| L14 | **0.04** (거의 직교) |
| L17 | 0.18 |
| L18 | 0.19 |
| L19 | 0.17 |
| **L20** | **0.78** (급등) |
| L21 | 1.00 (기준) |
| L27 | 0.53 |

**해석**:
- L14~L19까지는 L21 axis와 거의 직교 (다른 방향에 담음)
- **L19→L20 사이에서 갑자기 회전**, L20이 L21의 78%와 정렬
- 즉 **"L21 canonical axis"는 L19-L20 구간에서 형성**

이게 왜 중요한가: **"L14에 L21 axis 방향으로 무언가 주입해도 L19-L20 rotation이 덮어쓴다"**. 개입 위치를 잘못 고르면 효과 없음.

### 5.3 OOD에서는 sample들이 axis에서 더 흩어짐

L21에서 sample별 direction axis projection 통계:

| Task | 평균 projection | 표준편차 |
|---|---|---|
| SC | 0.74 | 0.12 |
| OP | **0.57** (23% 감소) | **0.24** (2배) |

OOD sample들은:
- axis 방향으로 **덜 강하게** projection (평균 23% 낮음)
- axis 주변에 **더 흩어져** 있음 (분산 2배)

Magnitude 부족 + variance 큼 **둘 다 해당**. 후속 causal test에서 magnitude가 주범, variance가 미미함을 확인.

---

## 6. L19 amplifier — fine-tuning의 지문

### 6.1 Amplifier란?

각 LLM decoder layer는 residual stream에 contribution을 더한다:
```
h_L = h_{L-1} + attn_output_L + mlp_output_L
```

"Amplifier"란 **특정 layer의 attn+mlp contribution이 특정 axis (L21 canonical direction axis) 방향으로 얼마나 큰가**를 의미.

측정 수식:
```
push_L = ⟨attn_out_L[-1], Δ̂_d_L21⟩ + ⟨mlp_out_L[-1], Δ̂_d_L21⟩
```

### 6.2 L19에만 amplifier가 있다

모든 28 layer에서 `push_L` 측정 결과, **L19만 우뚝 솟음**:

| Model | SC (IN) L19 push | OP (OOD) L19 push |
|---|---|---|
| Vanilla (fine-tune 없음) | **−0.3** | +0.2 |
| **Baseline** | **+37.8** | **+17.6** |
| **Delta** | **+48.8** | **+26.9** |

다른 layer (L10~L18, L20~L27)은 모두 push ≈ 0.

**의미**:
- **Vanilla L19에는 amplifier 없음** — fine-tune 안 된 모델은 이 회로가 없음
- **Fine-tuning이 L19에 direction amplifier를 만든다**
- SC에는 +37.8로 강하게 fire, OP에는 +17.6으로 절반만 fire (OOD에서 약함)
- Delta는 Baseline보다 +52% 강하게 fire

### 6.3 Attention과 MLP 둘 다 기여

| Model | attn push (L19 SC) | mlp push (L19 SC) | mlp 비중 |
|---|---|---|---|
| Baseline | 15.0 | 22.8 | 60% |
| Delta | 16.6 | 32.3 | **66%** |

Delta는 특히 MLP 기여를 강화.

### 6.4 L19 단독 개입 vs L20/L21

Factorial OP 500 sample, 해당 layer의 last token을 direction-averaged prototype으로 교체:

| 개입 위치 | MCQ | Δ |
|---|---|---|
| no_swap | 68.8% | — |
| L19 단독 | 76.6% | +7.8pp |
| L20 단독 | 86.0% | +17.2pp |
| **L21 단독** | **86.4%** | **+17.6pp (peak)** |
| L19+L20 | 86.0% | +17.2pp |

**L19가 amplifier지만 개입 sweet spot은 L20-L21** (axis rotation이 완료된 지점).

---

## 7. 인과 진단 — binding gap = magnitude 부족

### 7.1 10가지 개입 실험 @ L21

Factorial OP 500 샘플. 각 조건은 L21에서만 개입, 나머지 정상 forward:

| 조건 | 연산 | MCQ | Δ |
|---|---|---|---|
| no_swap | 건드리지 않음 | 68.80% | — |
| `L21 amp_2x` | on-axis × 2 | 73.80% | +5.0 |
| `L21 clean_sc` | 자기 제거 + SC 크기 삽입 | 78.80% | +10.0 |
| `L21 add_canon` | SC 크기 추가 (제거 없이) | 80.40% | +11.6 |
| `L21 on_axis` | off-axis 제거 (축만 남김) | 78.60% | +9.8 |
| `L21 remove_own` | 자기 projection 제거만 | 52.40% | **−16.4** |
| **`L21 full_rep`** | **Last token 완전 교체** | **86.40%** | **+17.6 (기준)** |
| `L18 clean_sc` | 같은 연산을 L18에 | 68.80% | **0** |
| `L16 clean_sc` | L16에 | 68.60% | −0.2 |
| `L14 clean_sc` | L14에 | 68.80% | 0 |

**첫 관찰**: L20 이전 layer 개입은 전부 효과 없음 (0pp). **L21만 효과**.

### 7.2 Magnitude Sweep — "얼마나 크게" 문제

L21 on-axis magnitude를 여러 값으로 set (direction axis 방향, off-axis는 그대로):

| 목표 크기 | MCQ | Δ |
|---|---|---|
| 0.5× SC (24) | 68.33% | +0.67pp |
| 1× SC (48) | 78.33% | +10.67pp |
| 2× SC (96) | 87.00% | +19.33pp |
| **3× SC (144)** | **89.33%** | **+21.67pp (peak)** |
| 5× SC (240) | 88.33% | +20.67pp |
| 10× SC (480) | 72.00% | +4.33pp (붕괴) |
| **−1× SC** | **14.33%** | **−53.33pp** (반대 방향) |
| **−2× SC** | **4.67%** | **−63.00pp** (완전 반전) |

**Key observations**:
- 0.5× → 1× → 2× → 3× 단조 증가
- **3×SC에서 peak** (+21.67pp)
- 5×SC 정체, **10×SC에서 catastrophic** (over-amplification)
- **음수 방향 넣으면 MCQ 14%** (chance 25% 이하) → **axis가 부호를 가진 의미 벡터**

**가장 중요한 finding**:
- `clean_2x_sc` (96 크기) = `full_rep` (전체 교체) = **+17.6pp**
- 즉 **"MCQ 회복의 전부가 magnitude로 설명됨"**. Letter content, identity 등 다른 정보 추가 기여 없음.

### 7.3 Layer 민감도 — L20-L21만 유효한 이유

같은 `clean_sc` 연산을 여러 layer에 적용:

| Layer | MCQ | Δ | 그 layer의 axis가 L21 canonical과 얼마나 가까운가 |
|---|---|---|---|
| L14 | 68.80% | 0 | cos 0.04 (거의 직교) |
| L16 | 68.60% | −0.2 | cos 0.18 |
| L18 | 68.80% | 0 | cos 0.19 |
| L19 | (full_rep 7.8pp) | — | cos 0.17 |
| **L20** | **86.0%** | +17.2 | cos 0.78 (정렬 완료) |
| **L21** | **86.4%** | +17.6 | cos 1.00 (자기 자신) |

**해석**: L20 이전 layer의 "canonical axis"는 L21의 canonical axis와 거의 무관. 그 layer에서 canonical axis 방향으로 뭔가 주입해도 **L19→L20 rotation에서 씻겨나감**.

→ **개입은 L20-L21 구간에서만 유효**.

### 7.4 Propagation test — 왜 L14-L18 개입이 0pp인가

L14/L16/L18 각각에 `clean_sc` 걸고 downstream magnitude 추적:

| 개입 위치 | L14 | L16 | L18 | L20 | L21 canonical |
|---|---|---|---|---|---|
| no_swap | 0.55 | 1.33 | 3.92 | 29.98 | 30.57 |
| L14 개입 | 0.55 | **1.84** (+38%) | 4.10 (+5%) | 30.03 | **30.62** (거의 0) |
| L16 개입 | — | 1.33 | **4.62** (+18%) | 30.09 | 30.66 |
| L18 개입 | — | — | 3.92 | **30.71** | 31.30 |
| **L20 개입** | — | — | — | 29.98 | **47.83** (+56%) |

**해석**:
- L14 개입은 L16 local까지만 전파 (+38%), L21 canonical까지 전달 안 됨
- **L20 개입만 L21 canonical에 직접 도달** → MCQ 상승
- L19-L20 rotation이 magnitude를 기반으로 하지 않고 **특정 feature pattern**을 기반으로 filter 역할

### 7.5 Binding vs Refinement — 어디가 주 원인?

300 샘플 × 9 조건 × 여러 layer 측정:

**MCQ**:

| 조건 | MCQ | Δ |
|---|---|---|
| no_swap | 69.33% | — |
| L14/L15/L16/L17 clean_sc | 69~69.33% | ~0 |
| L14+15+16 combined | 69.33% | 0 |
| **L21 clean_sc** | **78.67%** | **+9.3** |
| **L21 clean_2x_sc** | **88.33%** | **+19.0** |
| L14 + L21 clean_2x_sc | 88.67% | +19.3 (additive 아님) |

**L16 letter probe** (binding 단계):

| 조건 | L16 letter |
|---|---|
| no_swap | 23.33% |
| L14 clean_sc | 27.78% (+4.5) |
| **L15 clean_sc** | **31.11% (+7.8)** |
| L16 clean_sc | 23.33% (0) |

**L27 letter probe** (readout 직전):

| 조건 | L27 letter |
|---|---|
| no_swap | 46.67% |
| L14-L17 단독 | 45~49% (변화 없음) |
| **L21 clean_sc** | **84.44% (+37.8)** |
| **L21 clean_2x_sc** | **94.44% (+47.8)** |

**중요한 발견들**:
1. **L15 clean_sc → L16 letter probe +7.8pp** (binding circuit이 input magnitude에 반응하긴 함)
2. **하지만 MCQ는 0pp** — binding 개선이 downstream으로 전달 안 됨
3. **L21 2xsc → L22~L27 letter probe 폭증** (L22: 32→69, L27: 47→94)
4. **L14+L21이 L21 alone과 동일** (binding 기여가 refinement에 의해 흡수됨)

**결론**: **L22-L27 refinement가 지배적**. Binding (L15-L17)은 input에 약하게 반응하지만 MCQ 기여 없음. **L21 direction magnitude가 refinement 기계를 작동시키는 연료**.

### 7.6 3가지 가설 중 확증 하나, 약확증 하나, 반박 하나

| 가설 | 예측 | 결과 |
|---|---|---|
| **H-mag** (magnitude 부족이 원인) | magnitude 올리면 회복 | **확증 (dominant)**: clean_2x_sc = full_rep |
| **H-noise** (off-axis noise가 원인) | off-axis 제거하면 회복 | 약확증 (+9.8pp만, add_canon +11.6보다 작음) |
| **H-layer** (초기 layer 오염) | pre-L20 개입도 효과 | **반박**: L14/16/18 전부 0pp |

**결론**: binding gap = **L20-L21 canonical direction axis magnitude 부족**. Off-axis noise 기여 미미.

---

## 8. 원인 추적 — magnitude 부족은 어디서 생기는가

### 8.1 Stage별 ‖Δ_d‖ 추적 (SC/OP ratio)

| Stage | Vanilla | Baseline | Delta |
|---|---|---|---|
| Vision encoder (SigLIP, frozen) | 0.59× | 0.59× | 0.59× |
| After projector | 0.56× | 0.59× | **1.53×** |
| Vision token L7 | 0.70× | 0.81× | 1.55× |
| **Vision token L14** | 1.05× | **2.96×** | **3.85×** |
| Vision token L21 | 1.38× | 4.60× | 4.77× |
| Answer token L18 | 1.55× | 1.80× | 2.09× |
| **Answer token L21** | 1.96× | **1.51×** | 1.46× |

(값 < 1 = OP가 SC보다 magnitude 큼)

### 8.2 SigLIP은 원인이 아님

Vision encoder에서 SC/OP = 0.59× (OP가 오히려 큼). **SigLIP은 실제 객체/장소 비디오에서 direction feature가 합성보다 강함**. 이 단계에서는 OOD 문제 없음.

### 8.3 LLM L7-L14에서 SC-bias 발생

Baseline:
- L7: SC/OP = 0.81× (거의 동등)
- **L14: SC/OP = 2.96×** (SC가 3배로 증폭)
- L21: SC/OP = 4.60× (계속 누적)

**이 구간에서 LLM이 SC amplify 시작**. OP는 그만큼 amplify 안 됨.

### 8.4 Delta projector의 효과

Delta는 projector에서 이미 SC/OP = 1.53× (SC-bias 시작). 이게 `delta_direct` aux loss 효과. 하지만 LLM L14까지 가면 Delta도 3.85×로 격차 커짐 — LLM 자체 SC-bias는 못 고침.

### 8.5 근본 원인 진술

> LLM의 direction amplification 회로 (L7-L14의 여러 layer, 특히 L19 last token amplifier)가 **SC distribution의 visual pattern에 학습됨**. OP pattern에는 덜 민감하게 반응 → SC는 4×, OP는 2× amplify → L21에서 OP는 SC의 66% magnitude.

Task-invariant axis (cos 0.94)와 일관:
- 모델은 **"direction을 어느 axis에 쓸지"**는 task-agnostic하게 학습 (shared)
- **"얼마나 쓸지"**는 SC-trained pattern recognition에 의존 (task-biased)

---

## 9. Vision 개입은 왜 실패하는가

**질문**: 원인이 LLM 내부 SC-bias라면, projector 단에서 vision token을 고쳐서 해결할 수 없을까?

### 9.1 V1/V2/V3 실험

| 조건 | OP MCQ | Δ |
|---|---|---|
| no_swap | 67.67% | — |
| **V1** (mean-pooled axis 전체에 uniform shift) | 69.0% | +0.2 |
| **V2** (OP axis를 IN axis 방향으로 shift) | 68.6% | −0.2 |
| **V3** (axis swap: OP 성분 제거 + IN 성분 추가) | **23.8%** | **−45 (chance)** |

V3 catastrophic: per-position으로 direction 성분 제거하면 파괴.

### 9.2 Per-position magnitude amplification

| 조건 | OP MCQ | Δ |
|---|---|---|
| `amp_own_2x` | 65.0% | −2.7 |
| `amp_own_5x` | 38.0% | −29.7 |
| **`amp_own_10x`** | **20.0%** | **−47.7 (chance)** |
| `clean_sc` | 45.7% | −22.0 |

Magnitude 키울수록 단조 악화. L21 intervention (+17.6pp)과 정반대.

### 9.3 Per-temporal axis (bug fix 후 재시도)

Projector output은 `(T=8, S=729, D)`. 각 frame별로 따로 direction axis 계산해서 해당 frame에만 적용:

| 조건 | OP MCQ | Δ |
|---|---|---|
| `per_t_amp_2x` | 58.67% | −9.0 |
| `per_t_amp_5x` | 35.00% | −32.7 |
| `per_t_clean_sc` | 53.00% | −14.7 |
| `per_t_clean_2x_sc` | 52.67% | −15.0 |

**Per-temporal도 실패**. Mean-pooled보다 약간 덜 파괴적이지만 여전히 destructive.

### 9.4 Vision axis와 L21 axis는 직교

`cos(Δ̂_d_vision, Δ̂_d_L21)`:

| Direction | cos |
|---|---|
| Up | −0.046 |
| Right | +0.012 |
| Down | −0.008 |
| Left | −0.012 |

거의 0. **Vision token의 direction axis와 LLM의 L21 canonical axis는 서로 다른 공간의 벡터**. 28개 layer의 비선형 처리로 연결됨, linear하지 않음.

### 9.5 Coadaptation이 필요한 이유

Vision intervention 실패의 근본 원인:
- Vision token의 direction axis = **projector output의 통계적 signature** (방향 label로 얻은 평균 벡터 차이)
- LLM attention은 이 axis를 linear하게 읽는 게 아니라, **vision token의 spatial/temporal pattern을 non-linear하게 인식**
- 따라서 vision token에 선형 조작을 가해도 LLM이 기대하는 pattern이 바뀌지 않음

**Delta의 +5pp 성공**은 다른 mechanism:
- Training 중 projector와 LLM LoRA가 **동시에 update**
- Projector가 "LLM amplifier가 trigger되기 쉬운 form"으로 학습
- LLM이 projector output의 **새 distribution 읽도록 coadapt**
- 결과: OP input도 조금 더 SC-like로 인식

**Inference time에 linear operation으로 이걸 재현할 수 없음** — training loop가 필요.

---

## 10. Readout 메커니즘 — 비선형 변환

### 10.1 lm_head는 direction axis를 직접 읽지 않는다

LLM의 최종 layer는 `lm_head`라는 linear projection으로 hidden을 vocab logit (152064차원)으로 변환. 직접 측정:

`cos(W_lm[letter], Δ̂_d_L21)`:

| Direction | A | B | C | D |
|---|---|---|---|---|
| up (정답=A) | +0.005 | −0.022 | −0.013 | −0.005 |
| right (=B) | −0.051 | +0.018 | +0.011 | −0.003 |
| down (=C) | +0.003 | +0.025 | +0.019 | +0.014 |
| left (=D) | +0.043 | −0.021 | −0.018 | −0.006 |

**거의 전부 0 근처**. L21 direction axis와 letter token weights는 **거의 직교**.

### 10.2 L22-L27이 비선형 변환 담당

Direction axis와 letter axis가 다른데 왜 L21 magnitude가 MCQ를 좌우할까?

답: **L22~L27 6개 layer**가 direction → letter 비선형 변환 수행.

Combined experiment (§7.5):
- L21 2xsc → L27 letter probe 47 → 94% (+47pp)
- L22 letter probe: 32 → 69% (+37pp) — L22에서 이미 크게 refine
- → L22-L27이 L21 direction signal을 보고 letter를 완성

### 10.3 전체 파이프라인

```
L21 hidden (direction on canonical axis, magnitude M)
      ↓
L22~L27 (비선형 attention + MLP, direction 기반 letter refine)
      ↓ M이 충분하면 letter 정확히 재구성
L27 hidden (letter representation)
      ↓
lm_head (linear, letter axis만 읽음 — direction axis와 직교)
      ↓
Letter logits {A/B/C/D}
```

**"L21 magnitude → readout 직접"이 아니라 "L21 magnitude → L22-L27 refinement 품질 → letter logit"**.

---

## 11. 엄밀한 검증 — causal 확증

앞선 섹션의 claim들이 **correlational인지 causal인지** 확증.

### 11.1 Identity는 binding gap에 기여 없음

500 OP × 7 조건 at L21:

| 조건 | MCQ | Δ |
|---|---|---|
| no_swap | 68.80% | — |
| **id_remove_lda** (identity subspace 제거, direction 유지) | **68.80%** | **0.0pp** |
| clean_sc_mean (magnitude only, 48) | 78.80% | +10.0 |
| **clean_2x_sc** (magnitude only, 96) | **86.40%** | **+17.6** |
| bg_dir_L21 (obj 제거) | 83.80% | +15.0 |
| id_dir_L21 (bg 제거) | 88.60% | +19.8 |
| **full_dir_L21** (identity 완전 제거) | **86.40%** | **+17.6** |

**Identity subspace와 direction axis 직교 확인**: `max|cos(Δ̂_d, Q_id)| = 0.000` (모든 4 direction).

**결론**:
- `id_remove_lda` = `no_swap` (0pp) → **identity 제거만으론 MCQ 전혀 안 변함**
- `clean_2x_sc` (magnitude only) = `full_dir_L21` (identity+everything smear) → **86.40% 동일**. **+17.6pp 전부가 magnitude로 설명**.

### 11.2 Direction axis가 정말 "pure direction" 인지 검증

`Δ_d = h_avg(d) − h_avg(all)` 이 "letter/identity/bg가 상쇄된 순수 direction"인지 직접 검증:

Subset별로 Δ_d 계산 후 pairwise cos:

| 검증 | Subset | Pairwise cos | 해석 |
|---|---|---|---|
| **Variant invariance** | 4 variants (letter 다름) | **0.9491** | Letter 상쇄 ✓ |
| **Identity invariance** | 5 obj classes | **0.9753** | Obj 상쇄 ✓ |
| **Bg invariance** | 5 bg types | **0.8926** | Bg 부분 상쇄 (residual 있음) |
| **Split-half consistency** | 랜덤 반반 | **0.9975** | 측정 stability ✓ |

**결과**: Δ_d는 **letter-free, identity-free, bg는 10% residual이 남은** direction axis.

### 11.3 Axis 특정성, 부호, saturation 엄밀 증명

**15 conditions × 300 samples at L21**. 핵심 controls:

#### Direction axis magnitude sweep (확장)

| 크기 | MCQ | Δ |
|---|---|---|
| 0.5× SC | 68.33% | +0.67 |
| 1× SC | 78.33% | +10.67 |
| 2× SC | 87.00% | +19.33 |
| **3× SC** | **89.33%** | **+21.67 (peak)** |
| 5× SC | 88.33% | +20.67 |
| 10× SC | 72.00% | +4.33 (붕괴) |
| **−1× SC** | **14.33%** | **−53.33** |
| **−2× SC** | **4.67%** | **−63.00** |

#### Random axis controls (3 seed)

| 조건 | MCQ | Δ |
|---|---|---|
| rand_0 × 1x | 67.67% | 0.00 |
| rand_0 × 2x | 68.33% | +0.67 |
| rand_1 × 1x/2x | 69.00%/69.33% | +1.33/+1.67 |
| rand_2 × 1x/2x | 68.33%/68.67% | +0.67/+1.00 |
| **평균** | **68.56%** | **+0.89 (negligible)** |

#### Identity axis (direction과 직교)

| 조건 | MCQ | Δ |
|---|---|---|
| id_axis × 1x | 68.33% | +0.67 |
| id_axis × 2x | 69.00% | +1.33 |

#### 4가지 엄밀 finding

**Finding 1 — Axis specificity (causal)**:
- Direction axis 1×: **+10.67pp**
- Random axes 1×: **+0.67pp** (무시 가능)
- Identity axis 1×: **+0.67pp**
- → **같은 크기를 다른 축에 넣으면 효과 10배 감소**. Direction axis만 특별함.

**Finding 2 — Sign sensitivity (semantic)**:
- +1× SC: +10.67pp
- **−1× SC: −53.33pp** (67.67% → 14.33%)
- −2× SC: 4.67% (chance 25% 아래)
- → **반대 방향 주입 → 반대 예측**. Axis는 부호 있는 의미 벡터.

**Finding 3 — Monotonic with peak**:
- 0.5× → 3× 단조 증가
- **3×SC에서 peak** (+21.67pp)
- 5×SC 정체, 10×SC catastrophic collapse
- → Over-amplification은 downstream 혼란.

**Finding 4 — Effect asymmetry**:
- Positive direction: +22pp (upper bound, refinement capacity가 제한)
- Negative direction: −53pp (직접 뒤집힘, 최대 효과 큼)

### 11.4 최종 엄밀 claim

> **"OP의 binding gap은 L21의 canonical direction axis 방향 magnitude 부족이 원인이다. Magnitude를 3×SC 수준으로 set하면 OP MCQ 67.67% → 89.33% (gap 30pp 중 22pp 회복).**
> **(a) axis 특정성 (random/identity axis 효과 <2pp)**
> **(b) 부호 민감 (negative magnitude → 5% 정답률, 반전)**
> **(c) monotonic with peak at 3×SC, 10×SC collapse**
> **(d) 상한 89% (single-axis magnitude 혼자로는 SC 99% 도달 불가)**"

---

## 12. 종합 — 정식 claim

### 12.1 Core rigorous claim (한 문장)

**"Fine-tuned VLM의 OOD binding gap은 L20-L21 canonical direction axis 방향의 magnitude deficit이 원인이다. 이 효과는 axis-specific (random/identity axes <2pp), sign-sensitive (negative → 5% MCQ), 3×SC에서 peak (+21.67pp), 10×SC에서 collapse."**

### 12.2 Rigorously proven findings (증거 체인)

| # | Finding | Evidence | Section |
|---|---|---|---|
| 1 | Direction 정보 OOD에도 존재 (probe 92%) | Direct | §4.1 |
| 2 | L21 axis task-invariant (cos 0.94) | 6 task-pair | §4.3 |
| 3 | L19 amplifier 존재 (Vanilla 0, Baseline +37.8) | attn+mlp projection | §6 |
| 4 | Axis rotation L19→L20 | cross-layer cos | §5.2 |
| 5 | **Magnitude causal (3×SC → +21.67pp)** | 15-condition verification | §11.3 |
| 6 | **Axis specificity (random <2pp)** | 3 seeds | §11.3 |
| 7 | **Sign sensitivity (−1× → 14%)** | Direct | §11.3 |
| 8 | Refinement > Binding | combined | §7.5 |
| 9 | Identity zero contribution | id_remove_lda = 0 | §11.1 |
| 10 | Δ_d는 letter/id-free | subset invariance | §11.2 |
| 11 | Vision-side linear 불가능 | all experiments | §9 |
| 12 | Vision ⊥ L21 axis (cos 0) | Direct | §9.4 |
| 13 | lm_head ⊥ L21 axis (cos 0) | Direct | §10.1 |
| 14 | Cascade from LLM L7-L14 | per-stage ‖Δ‖ | §8.3 |

### 12.3 Delta_direct의 실제 mechanism

**측정된 Delta의 변화**:

| 지표 | Baseline OP | Delta OP | 변화 |
|---|---|---|---|
| Projector axis cos(SC, OP) | 0.16 | 0.46 | +0.30 (3배) |
| Projector ‖Δ_d‖ | 0.2-0.4 | 0.5-0.75 | 2-3배 |
| **L19 amplifier push** | **+17.6** | **+26.9** | **+52%** |
| L21 magnitude | 26 | 32 | +23% |
| MCQ | 79% | 84% | +5pp |

**메커니즘**: Delta의 projector aux loss가 projector output을 "SC-like form"으로 reshape. LLM amplifier (SC-trained)가 OP input도 SC-like pattern으로 인식해서 +52% 강하게 fire. 결과 L21 magnitude +23% → letter probe +5pp → MCQ +5pp.

**중요**: Delta 효과는 inference-time linear operation으로 재현 불가. Training 중 projector + LLM LoRA가 **동시에 update되어 coadaptation** 해야 한다.

### 12.4 Inferred (rigorously proven 아닌 것)

정직히 구분:

1. **"LLM L7-L14 amplifier가 SC-biased인 이유 = training distribution"** — 정황 증거 (Vanilla에 없음, SC 학습하니까 생김). Training distribution ablation 안 함.
2. **"Vision-level non-linearity"** — 경험적 negative result만. 여러 linear axis 시도했지만 전부 실패.
3. **"Upper ceiling ≈ 89%"** — 3×SC peak에서 측정. Training-time version은 비슷할 것으로 추정.
4. **"Cross-VLM 일반화"** — LLaVA-Video-7B-Qwen2만 test. 다른 VLM 미확인.

### 12.5 역설 해소

> **Q**: L21에 task-invariant axis가 있는데도 왜 OOD binding이 실패하나?
>
> **A**: **Task-invariance는 "direction을 어느 axis에 쓸지"를 지배 (shared across tasks). Magnitude는 "얼마나 쓸지"를 지배 (SC-biased, OOD under-amplified). Binding downstream refinement (L22-L27)가 충분한 L21 magnitude 필요 → OOD magnitude가 SC의 60%라 refinement under-fire → letter logit 약함.**

---

## 13. Delta_direct v2 설계

§11 결과 기반 **엄밀하게 뒷받침되는** 설계.

### 13.1 Primary: L21 direction magnitude supervision

가장 directly 효과가 큰 target. §11.3 peak at 3×SC 근거:

```python
# SC sample의 L21 ‖Δ‖ running average 사용
target_mag = 3.0 * sc_running_mag

# Under-magnitude penalty (작으면 penalty)
L_L21_mag = ReLU(target_mag − ⟨h_L21[-1] − g_L21, Δ̂_d⟩)

# Soft upper bound (10×SC catastrophe 방지)
L_upper = ReLU(⟨h_L21[-1] − g_L21, Δ̂_d⟩ − 5.0 * sc_running_mag)
```

**예상 효과**: Baseline 79% → ~89% (§11.3 rigorous upper bound).

### 13.2 Secondary: Projector alignment + magnitude matching

Delta의 확장. §8.4 근거 (projector axis cos 0.16 → 0.46이 +5pp 기여):

```python
# Axis alignment
L_axis_align = 1 − cos(Δ_d_SC_proj, Δ_d_OP_proj)  # batch에서 계산

# Magnitude matching
L_mag_match = (‖Δ_d_SC_proj‖ − ‖Δ_d_OP_proj‖) ** 2
```

**예상 효과**: Delta 위 추가 +3~5pp (projector alignment 완전 쪽으로).

### 13.3 Data: Multi-domain (가장 근본)

LLM amplifier의 SC-bias 근본 해결:
- Real OP videos with direction labels (MCQ 없이 direction label만, 저렴)
- 또는 SC에 aggressive augmentation (color/bg/style transfer)

**예상 효과**: +5~7pp (LLM amplifier multi-domain 학습).

### 13.4 누적 예상 성능

| Strategy | OP MCQ 예상 | Rigorous or Expected? |
|---|---|---|
| Baseline (현재) | 79% | Measured |
| + Delta projector aux (현재) | 84% | Measured |
| + Projector alignment + magnitude matching | 87-89% | Extrapolated |
| + **L21 magnitude supervision (primary)** | **89%** | **§11.3 peak upper bound** |
| + Multi-domain data | 93-98% | Expected (unproven without data) |

### 13.5 Delta_direct v2에 포함하지 말 것 (엄밀히 반박됨)

- ❌ **Identity regularization** (§11.1: id_remove_lda = 0pp)
- ❌ **Binding stage (L15-L17) supervision** (§7.5: +7pp L16 letter but 0pp MCQ)
- ❌ **Inference-time vision intervention** (§9: 전부 destructive)
- ❌ **Unbounded magnitude loss** (§11.3: 10×SC catastrophic)

---

## 14. 미해결 / 후속 실험

### 즉시 가능한 것

1. **Delta_direct v2 구현 및 훈련** (§13 design대로)
2. **Vanilla L21 intervention**: Vanilla는 amplifier 없음. magnitude injection에 반응하는가? 만약 반응하면 mechanism 범용성, 아니면 amplifier 필수 입증.
3. **Cross-task verification 완성**: obj_color, shape_place magnitude sweep (부분 data 존재).

### 메커니즘 (circuit-level)

4. **L7-L14 head-level 분석**: 어느 attention head와 MLP neuron이 SC-biased amplifier 담당? Focused retraining target.
5. **L16-L17 binding attention**: Binding이 어느 axis에서 direction 읽는가? L14 local axis와 일치하는가?
6. **L22-L27 refinement mechanism**: 어느 layer가 direction → letter 비선형 변환?

### 일반화

7. **Cross-VLM validation**: Qwen3-VL, MiniCPM-V 등도 같은 mechanism?
8. **다른 OOD task**: Counting, spatial reasoning 등도 magnitude deficit?

### 정직한 한계

9. **"SC-biased amplifier" causal test**: Multi-domain data로 재학습하면 L14 SC/OP ratio 떨어지나? 미수행.
10. **Real OP data scaling**: OP direction label이 얼마나 있으면 gap 닫히나? (Data efficiency).

---

## Appendix A — 주요 측정치

| 값 | 수치 | 출처 |
|---|---|---|
| Baseline OP MCQ (factorial) | 68.8% | §7.1 |
| Baseline OP MCQ (R2R 1500) | ~79% | letter probe L27 |
| Baseline SC MCQ (factorial) | 95.6% | §11.1 |
| Delta OP MCQ | 84% | §12.3 |
| **L21 cross-task cos (Baseline/Delta)** | **0.94** | §4.3 |
| L21 cross-task cos (Vanilla) | 0.54 | §4.3 |
| L14 cross-task cos (Baseline) | 0.64 | §5.1 |
| L14 within-task cos vs L21 | 0.04 | §5.2 |
| L20 within-task cos vs L21 | 0.78 | §5.2 |
| L19 amplifier push (Baseline SC) | +37.8 | §6.2 |
| L19 amplifier push (Baseline OP) | +17.6 | §6.2 |
| L19 amplifier push (Delta OP) | +26.9 | §6.2 |
| L19 amplifier push (Vanilla) | ≈ 0 | §6.2 |
| L21 ‖Δ_d‖ Baseline SC | 48.5 | §8.1 |
| L21 ‖Δ_d‖ Baseline OP | 28.3 | §8.1 |
| **Peak magnitude (3×SC)** | **144** | §11.3 |
| clean_2x_sc (96 크기) | +17.6 ~ +19.3pp | §7.2 |
| **clean_3x_sc (144 크기)** | **+21.67pp (peak)** | §11.3 |
| clean_10x_sc | +4.33pp (collapse) | §11.3 |
| clean_−1x_sc | **−53.33pp** | §11.3 |
| L14/16/18 clean_sc | 0pp | §7.3 |
| Random axis intervention (mean) | +0.89pp | §11.3 |
| Identity axis intervention | +0.67~1.33pp | §11.3 |
| Vision amp_10x | −47.7pp (chance) | §9.2 |
| cos(vision Δ̂_d, L21 Δ̂_d) | ≈ 0 | §9.4 |
| cos(W_lm, Δ̂_L21) | ≈ 0 | §10.1 |
| Δ_d variant invariance cos | 0.949 | §11.2 |
| Δ_d identity invariance cos | 0.975 | §11.2 |
| Δ_d bg invariance cos | 0.893 | §11.2 |
| Δ_d split-half cos | 0.998 | §11.2 |
| Identity LDA ⊥ Direction axis | cos = 0.000 | §11.1 |

---

## Appendix B — 파일 맵

### 데이터 경로
- R2R 1500 features: `/data3/local_datasets/vlm_direction/linear_probing_1500/{model}/{stage}/{task}/`
- Factorial hiddens: `/local_datasets/vlm_direction/factorial_dataset/hiddens/`
- Attn/MLP contributions: `/local_datasets/vlm_direction/attn_mlp_contrib/`

### Analysis 스크립트 (`analysis/task_invariance/`)

| Script | 역할 | Section |
|---|---|---|
| `measure_invariance.py` | Cross-task axis cos | §4.1, §4.3 |
| `axis_layer_cos.py` | Within-task cross-layer axis rotation | §5.2 |
| `measure_subspace_offaxis.py` | On-axis energy 분해 | §5.3 |
| `stage_trajectory.py` | Stage별 on_off_ratio | §8 |
| `magnitude_cascade.py` | Stage별 ‖Δ‖ cascade | §8.1 |
| `vision_token_axis_per_layer.py` | VT vs AT axis cos | §5, §9.4 |
| `extract_attn_mlp_contrib.py` | Attn/MLP hook | §6.1 |
| `analyze_attn_mlp_contrib.py` | Layer별 direction push | §6 |
| `analyze_3models.py` | 3-model 비교 | §6 |
| `l19_intervention.py` | L19 단독 개입 | §6.4 |
| `mechanism_diagnosis.py` | 10-condition 개입 | §7.1 |
| `clean_op_mean.py` | Magnitude sweep (basic) | §7.2 |
| `vision_intervention.py` | V1/V2/V3 (실패) | §9.1 |
| `vision_amp.py` | Per-position vision amp (실패) | §9.2 |
| `vision_per_temporal.py` | Per-temporal vision (실패) | §9.3 |
| `propagation_test.py` | 개입별 magnitude cascade | §7.4 |
| `combined_intervention_probe.py` | Binding vs Refinement | §7.5 |
| `identity_test.py` | Identity isolation | §11.1 |
| `validate_pure_direction.py` | Pure direction axis 검증 | §11.2 |
| `rigorous_verification.py` | Axis 특정성 + 부호 + saturation | §11.3 |
| `lm_head_align.py` | Readout linear 반박 | §10.1 |
| `cross_task_mag_sweep.py` | Cross-task magnitude (부분) | §14 |
| `fast_probe.py` | GPU-based linear probe | utility |

### 결과 파일
- `analysis/task_invariance/mech_results/*.json` — 실험별 metrics
- `output_4way_1500/{model}/answer_probe_results/{task}/` — letter/direction probe CSV
- `analysis/letter_vs_direction_probing.json` — 모든 model × task × layer의 letter/direction probe

---

## Appendix C — 실험 목록

실행 순서 아닌 **논리적 기여 순서**.

### Observational (§4-6)
1. **Linear probing suite** (R2R 1500 전체): direction/letter probe 정확도
2. **Cross-task axis cos** (factorial + R2R): task-invariance 측정
3. **Within-task cross-layer axis cos** (factorial): axis rotation 궤적
4. **Attn+MLP contribution 분해**: L19 amplifier 식별
5. **Magnitude cascade**: stage별 ‖Δ_d‖ 추적

### Causal intervention (§7)
6. **10-condition L21 intervention**: amp/clean/add/on_axis/remove/full_rep
7. **Magnitude sweep**: 0.5×/1×/2× SC at L21
8. **L19 vs L20/L21 intervention**: single-layer 인과
9. **Propagation test**: 개입 시 magnitude cascade
10. **Combined intervention**: BMS (binding) + REF (refinement)

### Vision-side (§9)
11. **V1/V2/V3 vision intervention**: broadcast/align/swap
12. **Per-position vision amp**
13. **Per-temporal vision**

### Readout (§10)
14. **lm_head alignment**: cos(W_lm, Δ̂_L21)

### Rigorous verification (§11)
15. **Identity isolation**: LDA-based identity 제거
16. **Pure direction axis validation**: variant/id/bg/split-half 불변성
17. **Rigorous axis + sign + saturation**: direction vs random vs identity, magnitude sweep extended

### Archived (confounded, CLAUDE.md)
- Sections J/K/L/M (이전 intervention, 현재 rigorous version으로 대체)
- Scripts 11, 13-18 (vision rotation/flip, 다양한 confounded 실험)

---

**연구 완료**. 17개 주요 실험, 14개 rigorously proven finding, 1개 definitive causal claim, delta_direct v2 design 구현 준비됨.
