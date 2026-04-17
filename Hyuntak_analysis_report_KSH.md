# Cross-Modal Information Flow Analysis — Detailed Report

VLM(Video Language Model)의 방향 인식 능력을 분석한 프로젝트의 전체 실험 및 분석 상세 보고서.

---

## 0. 프로젝트 개요

### 0.1 연구 동기

Vision-Language Model은 영상을 보고 자연어 질문에 답할 수 있다. 특히 "물체가 어느 방향으로 움직이는가?" 같은 **방향 인식(direction reasoning)** 은 VLM의 cross-modal 추론 능력을 평가하는 대표적 task다.

본 연구는 VLM 파이프라인 **Vision Encoder → Projector → LLM** 에서 direction 정보가 어떻게 흐르는지, fine-tuning이 이 흐름을 어떻게 바꾸는지 mechanistic level에서 분석한다.

### 0.2 모델 세팅

| 약칭 | 설명 | LoRA Path |
|------|------|-----------|
| `Vanilla` | LLaVA-Video-7B-Qwen2 (fine-tuning 없음) | — |
| `Baseline` | 4combo_v2 LoRA (MCQ loss만) | `...baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5` |
| `Delta` | 4combo_v2 + delta_direct auxiliary loss | `...delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5` |

### 0.3 Delta_direct 학습 구조

Delta 모델의 auxiliary loss 구조:

```
Vision Encoder → Projector → temporal delta (frame[t+1] - frame[t])
                                    ↓
                          Direction head ← auxiliary loss (projector만 업데이트)

                  Projector output → LLM → MCQ answer ← main loss (전체 업데이트)
```

- Projector output에서 각 frame pair의 temporal delta를 계산
- 이 delta를 direction head에 통과시켜 direction 예측
- Auxiliary loss는 **projector만** 업데이트 (direction head는 inference 시 제거)
- Main MCQ loss는 전체 파이프라인 업데이트

이 구조의 의도: projector가 temporal motion을 **direction으로 직접 encoding**하도록 유도.

### 0.4 Task 구성: R2R 4-way 1500

총 4개 task × direction당 1500 sample × 4 direction = task당 6000 sample.

| Task | Object | Background | 난이도 |
|------|--------|------------|--------|
| `shape_color` | 합성 도형 (원/사각형/삼각형 등) | 합성 단색 배경 | In-domain (가장 쉬움) |
| `obj_color` | 실제 객체 (이미지) | 합성 단색 배경 | 중간 OOD |
| `shape_place` | 합성 도형 | 실제 장소 배경 | 어려운 OOD |
| `obj_place` | 실제 객체 | 실제 장소 배경 | 가장 어려운 OOD |

Direction: Up / Down / Left / Right (4-class, chance = 25%).

### 0.5 Feature Shape & Pooling 규칙

| Stage | 저장 Shape | 설명 |
|-------|-----------|------|
| Vision Encoder | (N, 8, 1152) = (N, T, D_ve) | SigLIP 출력, spatial mean pool 적용 |
| After Projector | (N, 8, 3584) = (N, T, D_llm) | mm_projector + bilinear pool 후 spatial mean |
| Vision Token (layer l) | (N, 8, 3584) per layer | LLM decoder layer l의 vision position, spatial mean |
| Answer Token (layer l) | (N, 3584) per layer | LLM decoder layer l의 last token position |

**Pooling 규칙**: Vision feature probing 시 N(spatial) 축은 제거하고 T(temporal) 축은 보존. `--pool_spatial` 플래그로 저장 시점에 적용.

---

## 1. Linear Probing — Direction 정보의 존재 측정

### 1.1 Probing 방법론

각 stage의 hidden state를 꺼내 linear classifier로 4-class direction prediction:

```
feature (d-dim) → Linear(d → 4) → softmax → Up/Down/Left/Right
```

- **높은 정확도**: 해당 위치에 direction 정보가 linearly separable하게 존재
- **낮은 정확도**: 정보가 없거나 비선형적으로 encoding됨

### 1.2 Vision Pipeline Results (Direction 4-class probe accuracy)

#### shape_color (In-domain, 가장 쉬운 task)

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | 99.8% | 99.8% | 99.8% |
| After Projector | 96.5% | 94.4% | 99.9% |
| Vision Token L0 | 97.7% | 97.0% | 100.0% |
| Vision Token L3 | 96.9% | 97.5% | 99.8% |
| Vision Token L7 | 96.8% | 99.4% | 99.9% |
| Vision Token L10 | 99.4% | 99.2% | 99.8% |
| Vision Token L14 | 99.5% | 99.8% | 99.8% |
| Vision Token L18 | 99.0% | 99.7% | 99.8% |
| Vision Token L21 | 98.7% | 99.7% | 99.9% |
| Vision Token L24 | 97.4% | 99.8% | 99.8% |
| Vision Token L27 | 96.5% | 99.7% | 99.6% |

**해석**: 쉬운 task이므로 전체적으로 96%+. 모델 간 차이 미미. Vanilla조차 L27에서 96.5% 유지.

#### obj_color (중간 OOD)

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | 94.1% | 94.1% | 94.1% |
| After Projector | 87.3% | 88.7% | 97.7% |
| Vision Token L0 | 85.7% | 87.8% | 95.9% |
| Vision Token L3 | 87.1% | 90.1% | 98.4% |
| Vision Token L7 | 91.8% | 87.1% | 98.2% |
| Vision Token L10 | 91.2% | 91.7% | 97.2% |
| Vision Token L14 | 91.3% | 96.9% | 97.8% |
| Vision Token L18 | 92.3% | 97.2% | 98.8% |
| Vision Token L21 | 88.6% | 97.6% | 98.1% |
| Vision Token L24 | 81.1% | 97.6% | 97.0% |
| Vision Token L27 | 77.1% | 96.4% | 97.5% |

**해석**: Vanilla는 L18 (92.3%) → L27 (77.1%)로 **15%p 하락**. Fine-tuning은 이 하락을 방지.

#### shape_place (어려운 OOD)

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | 82.9% | 82.9% | 82.9% |
| After Projector | 74.6% | 73.9% | 86.5% |
| Vision Token L0 | 75.2% | 75.3% | 83.3% |
| Vision Token L3 | 77.6% | 75.7% | 86.3% |
| Vision Token L7 | 82.9% | 87.3% | 89.7% |
| Vision Token L10 | 86.7% | 92.4% | 92.2% |
| Vision Token L14 | 89.9% | 96.4% | 97.5% |
| Vision Token L18 | 85.3% | 95.6% | 97.2% |
| Vision Token L21 | 86.7% | 95.3% | 97.2% |
| Vision Token L24 | 81.6% | 95.7% | 96.7% |
| Vision Token L27 | 72.7% | 91.2% | 94.7% |

#### obj_place (가장 어려운 OOD)

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | 86.9% | 86.9% | 86.9% |
| After Projector | 74.6% | 75.4% | 84.8% |
| Vision Token L0 | 76.1% | 79.5% | 83.4% |
| Vision Token L3 | 79.7% | 76.8% | 82.5% |
| Vision Token L7 | 86.8% | 84.8% | 88.9% |
| Vision Token L10 | 82.8% | 88.1% | 89.6% |
| Vision Token L14 | 84.8% | 89.8% | 95.5% |
| Vision Token L18 | 84.8% | 93.3% | 96.1% |
| Vision Token L21 | 79.9% | 91.3% | 94.7% |
| Vision Token L24 | 73.8% | 88.0% | 90.5% |
| Vision Token L27 | 64.6% | 89.1% | 90.2% |

**가장 선명한 패턴**: Vanilla L14 (84.8%) → L27 (64.6%)로 **20%p 급락**. Baseline/Delta는 L27까지 90%+ 유지.

### 1.3 Answer Token Layer-wise Results

Answer token = prefill의 last position (다음 토큰을 생성하기 직전 위치). 이 위치의 hidden state를 각 layer에서 probe.

#### shape_color

| Layer | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| L0 | 25.0% | 25.0% | 25.0% |
| L3 | 96.7% | 96.2% | 99.5% |
| L7 | 96.6% | 97.3% | 99.3% |
| L10 | 95.4% | 97.9% | 98.8% |
| L14 | 97.0% | 99.7% | 99.8% |
| L18 | 95.4% | 99.9% | 100.0% |
| L21 | 93.8% | 99.8% | 99.9% |
| L24 | 93.9% | 99.9% | 99.9% |
| L27 | 89.9% | 99.9% | 99.9% |

#### obj_color

| Layer | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| L0 | 25.0% | 25.0% | 25.0% |
| L3 | 90.8% | 90.2% | 95.6% |
| L7 | 90.8% | 88.3% | 93.8% |
| L10 | 88.9% | 91.6% | 94.1% |
| L14 | 87.9% | 94.8% | 96.4% |
| L18 | 85.8% | 97.5% | 98.4% |
| L21 | 81.4% | 97.9% | 98.6% |
| L24 | 80.6% | 98.1% | 98.2% |
| L27 | 73.1% | 97.9% | 97.8% |

#### shape_place

| Layer | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| L0 | 25.0% | 25.0% | 25.0% |
| L3 | 79.8% | 74.6% | 77.3% |
| L7 | 79.9% | 76.0% | 79.6% |
| L10 | 76.9% | 87.7% | 89.8% |
| L14 | 83.1% | 93.3% | 94.7% |
| L18 | 76.8% | 95.2% | 97.1% |
| L21 | 72.7% | 95.6% | 97.3% |
| L24 | 72.4% | 95.8% | 97.1% |
| L27 | 65.3% | 95.4% | 96.1% |

#### obj_place

| Layer | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| L0 | 25.0% | 25.0% | 25.0% |
| L3 | 74.6% | 70.7% | 75.5% |
| L7 | 76.4% | 70.4% | 74.2% |
| L10 | 73.7% | 79.9% | 83.7% |
| L14 | 75.5% | 87.4% | 90.4% |
| L18 | 69.8% | 91.0% | 93.8% |
| L21 | 68.3% | 92.6% | 95.1% |
| L24 | 66.1% | 92.3% | 94.6% |
| L27 | 60.1% | 90.8% | 93.8% |

### 1.4 핵심 관찰

#### 관찰 1. Fine-tuning = 후반 layer의 direction 보존 (Vanilla의 희석 복구)

Vanilla는 L18~L27로 갈수록 direction probe가 급격히 떨어진다.

- obj_place L27 Vision Token: Vanilla 65% → Baseline 89% → Delta 90%
- obj_place L27 Answer Token: Vanilla 60% → Baseline 91% → Delta 94%

**해석**: Direction 정보는 vision encoder에서 이미 존재 (86.9%). Vanilla의 문제는 **LLM 후반부에서 이 정보를 버리는 것**. Fine-tuning은 이 소실을 방지.

#### 관찰 2. Delta의 고유 효과는 초반 단계에 집중

초반 stage에서 Delta만 유독 높음:

- After Projector (obj_place): Vanilla 75% / Baseline 75% / **Delta 85%**
- Answer Token L3 (obj_place): Vanilla 75% / Baseline 71% / **Delta 76%**
- Vision Token L0 (obj_place): Vanilla 76% / Baseline 80% / **Delta 83%**

**해석**: 
- Delta_direct auxiliary loss가 projector output을 **direct하게 direction-rich**하게 만듦
- Baseline은 projector output에선 거의 Vanilla와 같고, LLM 후반에서 보강
- 종착점은 비슷 (L27에서 Baseline ≈ Delta), 하지만 도달 경로가 다름

#### 관찰 3. Task 난이도 ∝ Fine-tuning의 복구 효과 (L27 기준)

| Task | Vision Token Δ (Baseline−Vanilla) | Answer Token Δ (Baseline−Vanilla) |
|------|---|---|
| shape_color | +3 | +10 |
| obj_color | +19 | +25 |
| shape_place | +18 | +30 |
| obj_place | +25 | +31 |

OOD가 어려울수록 Vanilla 희석 심함 → fine-tuning 복구량 큼.

**함의**: In-domain accuracy로 fine-tuning 효과를 과소평가하기 쉬움. Cross-task / OOD evaluation이 필수.

#### 관찰 4. Vanilla 패턴: 초반 급상승 → 후반 희석

Vanilla obj_place answer token trajectory:
- L3 (75%) → L14 (76%) → L18 (70%) → L21 (68%) → L27 (60%)

**구조**:
- L3에서 이미 direction 정보 도달
- 하지만 L18부터 **decline** — representation이 language generation 방향으로 수렴하면서 direction 정보가 소실
- **Fine-tuning의 핵심 기여 = 이 decline을 막는 것**

#### 관찰 5. Delta vs Baseline 차이는 주로 어려운 OOD에서

- shape_color: 두 모델 L27 모두 99.9% (포화)
- obj_place: Baseline 91% vs **Delta 94%** (+3%p), shape_place에서도 유사
- Delta의 auxiliary supervision이 어려운 OOD에서 +1~3%p 마진

#### 관찰 6. Vision token ≈ Answer token (direction signal 측면)

- Baseline obj_place L27: vision 89% vs answer 91% (거의 동등)
- 200-sample에서 관측된 "last token이 vision token보다 direction 훨씬 높다"는 **overfitting artifact**
- Fine-tuning은 vision token과 answer token 둘 다에서 direction을 보존

---

## 2. Cross-Task Probing — Direction Representation의 일반화

### 2.1 방법론

Train probe on task A, test on task B → 4×4 matrix per model.

- **Diagonal (A→A)**: in-task accuracy
- **Off-diagonal (A→B)**: cross-task transfer
- Off-diagonal이 높을수록 **direction-readable axis가 task 간 공통**

⚠️ "Identity가 벗겨졌다"는 뜻은 아님 — identity 정보는 orthogonal 차원에 공존 가능.

### 2.2 Answer Token Last Layer 결과

#### Vanilla (Transfer gap 25.5%p)

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **88.5%** | 49.0% | 32.1% | 33.3% |
| obj_color | 70.5% | **71.3%** | 29.4% | 33.0% |
| shape_place | 45.7% | 38.8% | **60.4%** | 43.9% |
| obj_place | 46.6% | 45.2% | 53.3% | **55.6%** |

#### Baseline (Transfer gap 5.6%p)

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.9%** | 92.7% | 83.4% | 72.4% |
| obj_color | 99.5% | **97.5%** | 85.7% | 81.1% |
| shape_place | 98.5% | 92.8% | **94.6%** | 86.3% |
| obj_place | 97.8% | 93.9% | 93.7% | **89.8%** |

#### Delta (Transfer gap 3.4%p)

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.9%** | 91.9% | 90.2% | 81.0% |
| obj_color | 99.9% | **97.5%** | 94.8% | 88.1% |
| shape_place | 99.7% | 94.2% | **97.1%** | 90.9% |
| obj_place | 98.3% | 95.4% | 96.7% | **92.9%** |

### 2.3 종합

| 모델 | Diagonal (in-task) | Off-diagonal (cross) | Transfer Gap |
|---|---|---|---|
| Vanilla | 68.9% | 43.4% | **25.5%p** |
| Baseline | 95.5% | 89.8% | 5.6%p |
| Delta | 96.9% | 93.4% | **3.4%p** |

### 2.4 핵심 해석

#### 해석 1. Probe가 진짜 direction을 본다 (task-shortcut 아님)

Baseline/Delta의 cross-task 90%+는 학습된 probe가 **object/background 무관하게 direction을 추출함**을 증명.

만약 probe가 "shape_color에서 빨간 네모가 오른쪽에 있으면 Right"같은 shortcut을 학습했다면, obj_place에서 작동 불가. 하지만 실제로 cross-task 90%+ → probe는 shape/background에 무관하게 direction을 decoding.

#### 해석 2. Fine-tuning = Direction axis의 task-invariant alignment

- **Vanilla**: direction-readable axis가 task마다 다른 방향 (shape_color → obj_place transfer 33%)
- **Baseline**: task 간 공통 axis로 정렬 (transfer gap 25.5 → 5.6%p)
- **Delta**: 추가 정렬 (3.4%p) — auxiliary direction supervision이 task-invariant axis 강화

#### 해석 3. Delta의 고유 기여는 OOD cross-transfer에서 선명

- obj_place → shape_color 같은 어려운 전이에서도 Delta가 Baseline보다 1~5%p 높음

---

## 3. Pipeline 전체 Cross-Task Matrix

모든 stage × 모든 layer × 3 모델에서 4×4 cross-task matrix 측정.

### 3.1 Vision Encoder (frozen, 3 모델 동일)

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.5%** | 80.4% | 38.5% | 39.9% |
| obj_color | 85.6% | **94.2%** | 26.8% | 30.6% |
| shape_place | 24.2% | 28.2% | **85.0%** | 78.7% |
| obj_place | 35.5% | 43.9% | 72.6% | **85.4%** |

- **Diag = 91.0% | Off = 48.7% | Gap = 42.3%p**

**패턴 분석**:
- Color backgrounds (shape_color ↔ obj_color) 간 전이: 85%+ (object type 달라도 OK)
- Place backgrounds (shape_place ↔ obj_place) 간 전이: 72-78%
- **Color ↔ Place 전이는 26-40%로 급락** → SigLIP은 background type에 크게 의존

### 3.2 After Projector

| 모델 | Diag | Off-diag | Gap |
|------|------|----------|-----|
| Vanilla | 87.9% | 50.0% | 38.0%p |
| Baseline | 88.8% | 50.2% | **38.6%p** |
| **Delta** | **99.3%** | **78.1%** | **15.2%p** |

**결정적 관찰**:
- **Vanilla vs Baseline 거의 동일** (gap 38 vs 39%p). MCQ loss만으로는 projector 수준에서 task-invariant axis 정렬 안 됨.
- **Delta만 projector 수준에서 큰 개선**. Auxiliary direction loss가 projector output을 task-invariant direction axis로 정렬.

**Delta projector의 구체적 개선 사례**:
- Shape-Color → Obj-Place 전이: Baseline 30.5% → **Delta 45.2%**
- Obj-Place → Shape-Color 전이: Baseline 48.1% → **Delta 94.3%**
- Shape-Place → Obj-Color 전이: Baseline 33.3% → **Delta 88.6%**

### 3.3 Vision Token & Answer Token Layer-wise

| 측정 | Vanilla | Baseline | Delta |
|------|---------|----------|-------|
| Vision token L0 off-diag | ~55% | ~50% | **~75%** |
| Vision token L14 off-diag | ~45% | ~75% | ~85% |
| Vision token L27 off-diag | ~45% | ~80% | ~85% |
| Answer token L14 off-diag | ~50% | ~75% | ~85% |
| Answer token L27 off-diag | ~40% | ~90% | ~95% |

**모델별 패턴**:
- **Vanilla**: Layer가 깊어져도 cross-task transfer 개선 없음. 오히려 후반에 약간 하락.
- **Baseline**: L10~L20 구간에서 **phase transition** — cross-task 50% → 80% 급상승. LLM attention이 task-invariant direction axis로 정렬하는 구간.
- **Delta**: **초기 layer부터 이미 높은 off-diag**. Projector에서 완성된 task-invariant direction axis가 LLM 전체에 전파.

---

## 4. Cross-Task Gap 원인 분해 (Entanglement 해석의 정정)

⚠️ 이전 "direction이 identity와 entangle되어 있다"는 표현은 **over-claim**. Cross-task accuracy 낮다고 바로 entanglement 결론 내릴 수 없음.

### 4.1 4가지 가설

| 가설 | 의미 | 검증 방법 |
|------|------|------|
| (a) Axis rotation | Task마다 direction 축이 다른 방향 | cos(probe axes) |
| (b) Scale 차이 | 같은 축, 분산 다름 | rescale experiment |
| (c) Bias shift | 같은 축, offset 다름 | center-rescale |
| (d) True entanglement | Direction이 identity와 같은 dimension 공유 | Fisher dims overlap |

### 4.2 측정 결과 (projector 기준)

| 가설 | 실측 | 결론 |
|------|------|------|
| (a) Axis rotation | Vanilla/Baseline cos = 0.27-0.39 (낮음) | **주 원인** |
| (b) Scale 차이 | Rescale Δ = +3-5%p | 보조 원인 (작음) |
| (c) Bias shift | — | 미측정 |
| (d) True entanglement | Fisher dims overlap ≈ 0% | **아님** |

### 4.3 정확한 표현

- **X**: "Direction이 identity와 entangle"
- **O**: "Direction-readable axis가 task마다 다른 방향으로 놓임 (task-specific orientation)"

#### 핵심 포인트

- Direction과 identity가 **같은 dimension을 공유하지 않음** (Fisher overlap 0)
- 대신 **identity context가 direction encoding의 축 자체를 결정**
- "어느 차원이 direction을 encoding하는가"가 identity에 따라 달라짐

### 4.4 비유

같은 "오른쪽"이라도:
- shape_color: "합성 도형의 centroid가 right-patch로 이동" → dim 100-200 일대에 encoding
- obj_place: "객체가 실제 배경 대비 flow" → 완전히 다른 dim에 encoding
- 두 encoding이 같은 차원을 공유하지 않지만, SigLIP 공간에서 **다른 axis**에 놓임

### 4.5 Fine-tuning이 하는 일

여러 identity의 서로 다른 direction subspace를 → **공통 direction axis**로 align/rotate.

- Delta projector: 이를 projector 수준에서 수행 (cos 0.27 → 0.67)
- Baseline LLM: L15~L20 attention에서 수행 (LLM이 identity-specific axis들을 universal axis로 rotate)
- Vanilla: 수행 못 함 (cross-task transfer 실패)

---

## 5. Direction Axis Alignment 분석

Cross-task gap의 원인을 3가지로 분해:

1. **축이 다름**: probe weight의 direction axis (Up-Down, Left-Right)가 task마다 다른 방향
2. **Scale 차이**: 축은 같아도 task별 feature scale/variance가 달라 probe 부작동
3. **Pure transfer**: 축+scale 다 맞췄을 때도 남는 gap

### 5.1 Method 1: Probe axis cosine similarity

각 task에서 학습된 probe의 weight로부터:
- `v_UD = W[Up] − W[Down]`
- `v_LR = W[Left] − W[Right]`

Task pair 간 cos(v_A, v_B) 측정.

### 5.2 Method 2: Cross-task eval with rescaling

Source probe를 target task에 적용할 때 target 자체의 mean/std로 z-norm. Rescale 효과 (Δ)가 scale 기여도.

### 5.3 After Projector 결과 (핵심 비교)

| 모델 | cos_UD | cos_LR | off_orig | off_rescale | Rescale Δ |
|------|--------|--------|----------|-------------|-----------|
| Vanilla | 0.37 | 0.27 | 49.7% | 53.2% | +3.4 |
| Baseline | 0.39 | 0.27 | 49.8% | 53.6% | +3.8 |
| **Delta** | **0.67** | **0.70** | **78.1%** | **83.2%** | +5.1 |

**해석**:
- **Vanilla/Baseline projector**: 축이 task마다 다름 (cos ~0.3). Rescale 효과도 작음 → scale 문제 아니고 **축 문제**.
- **Delta projector**: **축 정렬 급상승** (cos 0.27 → 0.70). Auxiliary loss가 projector 수준에서 task-agnostic direction axis 학습.

### 5.4 Layer-wise Trajectory

**Vanilla**: Vision encoder → Answer L27 내내 cos ~0.3-0.5. **끝까지 alignment 실패**.

**Baseline**:
- L0: cos_UD 0.39 (Vanilla와 동일)
- **L14에서 급상승**: 0.55 (LLM attention이 축 정렬 시작)
- L21 peak: 0.50-0.55
- L27 answer: cos=0.37 (낮음)이지만 cross-task acc=90% (높음)
  - → Probe weight 벡터는 달라지지만 **decision boundary 수준 alignment 확보**
  - **High-dim equivalence** — 다른 weight solution이 같은 classification 결과

**Delta**:
- After projector부터 cos=0.67 (이미 정렬)
- LLM layer들은 유지/미세조정만 수행

### 5.5 Scale Contribution (Rescale Δ) Trajectory

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | +10.8 | +10.8 | +10.8 |
| After Projector | +3.4 | +3.8 | +5.1 |
| Vision Token L14 | +12.3 | +9.6 | +13.0 |
| Answer L21 | +8.7 | **+1.5** | **+1.0** |
| Answer L27 | +5.5 | **+1.5** | **+0.8** |

**해석**:
- **Vision encoder에서 scale 기여도 큼** (+10.8%p) — raw SigLIP feature는 task별 scale 차이 있음
- **Answer L21+ Baseline/Delta에서 Δ=+1-2%p만** — representation이 이미 scale-invariant. 남은 차이는 "axis shift" 아님.

### 5.6 종합 해석

**Cross-task gap의 원인 분해표**:

| Stage / Model | 주 원인 | 근거 |
|---------------|--------|------|
| Vision Encoder (frozen) | 축+scale 둘 다 | cos~0.4, Δ=+11 |
| Vanilla projector/LLM | 축 불변 (미개선) | cos 0.3-0.5, transfer 약함 |
| Baseline LLM (L14+) | 축 부분 정렬 + decision boundary 수렴 | cos 0.5-0.55, 후반 acc 90% |
| **Delta projector** | **축 직접 정렬** | **cos 0.27 → 0.70** |

### 5.7 이것이 보여주는 것

#### 1. Baseline과 Delta의 alignment 방식이 다름

- **Baseline**: LLM attention이 layer를 거치며 **feature space에서 decision boundary를 align** (축 자체는 여전히 일부 다름)
- **Delta**: Projector가 **축 자체를 직접 align** (cos 0.67) → LLM은 보존/미세조정

#### 2. Vanilla의 근본 문제

축 정렬 실패. Feature space에 direction 정보는 있지만 **task마다 다른 축**에 encoding되어있고, 이를 unify하는 메커니즘 없음.

#### 3. Delta projector의 인과적 역할

Auxiliary direction supervision이 **task-agnostic direction axis**를 projector에 직접 새김.

Cross-combination 결과("Delta proj + Baseline LLM = 73%")와 충돌하지 않음:
- Delta projector의 axis = universal direction
- Baseline LLM은 Baseline projector의 **local (task-entangled)** axis를 읽도록 학습
- → Mismatch 발생

---

## 6. Position / Direction / Object Dimension 분석

Mean-pooled vision feature에 어떤 정보들이 어느 dimension에 encoding되는지 직접 측정.

### 6.1 Setup

- Feature: 1500 feature (pool_spatial=True)
- Target:
  - **Position**: `start_pos = [x, y]` 좌표 (메타데이터) — 1st frame feature로 regression
  - **Direction**: Up/Down/Left/Right — (last_frame − first_frame) delta feature로 classification
  - **Object**: shape / obj_class — mean-frame feature로 classification
- Top-50 dim: regression coefficient magnitude (position) / Fisher ratio (direction, object)

### 6.2 Within-task 결과 (Vision Encoder, D=1152, random overlap = 2.17)

| Task | Pos R² | Dir acc | Obj acc | P∩D | P∩O | D∩O |
|------|-------|---------|---------|-----|-----|-----|
| shape_color | 0.99 | 100% | 96% | **6** | 8 | 0 |
| obj_color | 0.85 | 99% | 89% | **7** | 8 | 3 |
| shape_place | 0.92 | 97% | 88% | **6** | 2 | 4 |
| obj_place | 0.81 | 95% | 81% | **8** | 4 | 3 |

**관찰**:
- **Position info mean-pool 후에도 보존** (R² 0.81-0.99) — SigLIP의 absolute positional embedding이 mean 뒤에도 "signature" 형태로 남음
- P∩D = 6-8/50 (random 2.17의 3-4배, weak). "Direction = position delta"는 수학적/setup상 거의 tautological — 확정적 증거는 아님.
- Direction/Object overlap은 random 수준 → identity는 별도 dim

### 6.3 Cross-task Position Dim Overlap (핵심 발견)

Vision Encoder, random = 2.17:

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 4 | 2 | 2 |
| obj_color | 4 | 50 | 6 | **10** |
| shape_place | 2 | 6 | 50 | 8 |
| obj_place | 2 | **10** | 8 | 50 |

**관찰**:
- Task A의 "position 담당 dim" ≠ Task B의 "position 담당 dim"
- 대부분 pair에서 **random 수준 overlap** (2-10 vs expected 2.17)
- **Position encoding 자체가 task-specific**

### 6.4 Direction Dim Cross-task Overlap

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 16 | 7 | 6 |
| obj_color | 16 | 50 | 5 | 10 |
| shape_place | 7 | 5 | 50 | **23** |
| obj_place | 6 | 10 | 23 | 50 |

- shape_place ↔ obj_place: **23/50** (두 task 모두 real place background 사용 → 공통 encoding dim 존재)
- 나머지 pair는 낮은 overlap

### 6.5 Object Dim Overlap

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 2 | **17** | 3 |
| obj_color | 2 | 50 | 1 | **22** |
| shape_place | 17 | 1 | 50 | 3 |
| obj_place | 3 | 22 | 3 | 50 |

- shape_color ↔ shape_place: **17/50** (same shape categories)
- obj_color ↔ obj_place: **22/50** (same obj_class categories)
- 같은 identity label space 공유하는 pair에서만 dim 공유 — 합리적

### 6.6 핵심 해석: Cross-task direction transfer 실패의 근본 메커니즘

```
SigLIP은 position을 encoding함 (R² 0.81-0.99)
  → 하지만 "어느 dim이 position 담당"인지 task마다 다름 (cross-task overlap random 수준)
  → Direction = position delta이므로 direction encoding dim도 task-specific
  → Cross-task direction probe 실패 (우리가 보던 현상)
```

**왜 position encoding이 task-specific한가**:
- SigLIP의 mean-pool "position signature"는 **어느 patch가 salient한지**의 함수
- shape_color: 합성 도형 pixel의 pattern
- obj_place: 실제 객체 pixel + place background pattern
- 시각적 content가 달라지면 "salient patch 위치"의 feature encoding도 달라짐
- 같은 dimension이 task에 따라 다른 의미로 쓰임

**"Identity-conditional direction encoding" 가설의 mechanistic evidence**:
- 단순히 "direction axis가 task마다 다르다" (현상)를 넘어
- **Position dim 자체가 task마다 다르게 encoded** (메커니즘)
- Direction이 position delta라는 수학적 관계가 이 task-specificity를 그대로 상속

### 6.7 Within-task Dim Sharing

Task마다 top-50 dim overlap (Vision Encoder, random = 2.17):

| Task | P∩D | P∩O | D∩O |
|------|-----|-----|-----|
| shape_color | **6** | 8 | 0 |
| obj_color | **7** | 8 | 3 |
| shape_place | **6** | 2 | 4 |
| obj_place | **8** | 4 | 3 |

**관찰**:
- **P∩D = 6-8/50** (random의 3-4배): 약한 overlap. Direction을 delta feature로 probe하니까 position encoding 공간과 자연히 일부 공유 — setup상 near-tautological.
- **Task별 P∩O 패턴이 특이**:
  - Color background (shape_color, obj_color): P∩O = 8/50 (object dim과 position dim 약간 겹침)
  - Place background (shape_place, obj_place): P∩O = 2-4/50 (random 수준)
  - → 단색 배경에선 "어느 patch가 salient한가"(=position)가 "어떤 object인가"(=object)와 연관됨
  - → 실제 배경에선 전체 화면이 salient하니 object와 position이 독립
- **D∩O = 0-4/50**: Direction과 identity는 **거의 별도 dim** (task 전반)

### 6.8 After Projector에서의 변화

After projector (D=3584, random=0.70):
- Within-task P∩D overlap이 **0-2로 급감**
- Cross-task도 대부분 0-3
- Projector가 position-direction을 분산 encoding? 또는 top-50이 3584d에선 너무 tight해서 noise level

통계적 해석 신뢰도 낮음 (random expectation 0.7 대비 측정치가 0-3이라 구분 어려움).

---

## 7. Letter vs Direction Probing — MCQ-probe gap의 실체

지금까지 "direction probe 91%인데 MCQ 79%, gap 12%p는 readout alignment 문제"로 해석했으나, **letter 자체를 직접 probing하면 그림이 달라짐**.

### 7.1 Setup

- Feature: 4-way extraction answer token per layer
- Target 1: direction (Up/Down/Left/Right, 4-class)
- Target 2: **letter** (sample별 정답 A/B/C/D, 4-class)
- MCQ 특성: 같은 direction이라도 sample별 candidate shuffle에 따라 letter 달라짐

### 7.2 Baseline shape_color (in-domain)

| Layer | Direction | Letter |
|-------|-----------|--------|
| L7 | 96.5% | 26.3% (chance) |
| L14 | 99.9% | 32.7% (chance) |
| **L21** | 99.7% | **97.2%** ← 급등 |
| L27 | 99.7% | 99.2% |

### 7.3 Baseline obj_place (어려운 OOD)

| Layer | Direction | Letter |
|-------|-----------|--------|
| L7 | 74.1% | 22.7% |
| L14 | 89.3% | 24.3% |
| **L21** | 92.1% | **71.9%** ← 급등 |
| L27 | 91.6% | 78.8% |

### 7.4 Delta — 전체 4 task

#### Delta shape_color (in-domain)

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L3 | 99.6% | 25.4% | +74.2 |
| L7 | 99.0% | 26.0% | +73.0 |
| L10 | 99.2% | 26.2% | +73.0 |
| L14 | 99.8% | 31.0% | +68.8 |
| L15 | 99.9% | 35.9% | +64.0 |
| **L16** | 99.9% | **69.9%** | +30.0 ← binding 급등 (Baseline보다 1 layer 빠름) |
| **L17** | 100.0% | **87.4%** | +12.6 |
| L18 | 100.0% | 92.7% | +7.3 |
| L20 | 99.8% | 95.6% | +4.2 |
| L27 | 99.9% | **99.0%** | +0.9 |

#### Delta obj_color

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L14 | 96.7% | 28.5% | +68.2 |
| L15 | 97.6% | 27.9% | +69.7 |
| **L16** | 97.9% | **49.4%** | +48.5 ← binding 시작 |
| **L17** | 98.3% | **73.2%** | +25.1 |
| L18 | 98.2% | 79.8% | +18.3 |
| L20 | 97.9% | 85.8% | +12.1 |
| L27 | 97.5% | **93.7%** | +3.8 |

#### Delta shape_place

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L14 | 95.4% | 27.8% | +67.6 |
| L15 | 96.1% | 29.2% | +66.9 |
| **L16** | 96.6% | **42.8%** | +53.7 |
| **L17** | 97.3% | **69.1%** | +28.2 |
| L18 | 97.1% | 73.3% | +23.8 |
| L20 | 97.1% | 82.1% | +14.9 |
| L27 | 97.0% | **89.7%** | +7.3 |

#### Delta obj_place (가장 어려운 OOD)

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L14 | 92.4% | 26.0% | +66.4 |
| L15 | 93.4% | 26.8% | +66.6 |
| **L16** | 94.8% | **38.1%** | +56.7 |
| **L17** | 94.3% | **57.2%** | +37.2 |
| L18 | 94.5% | 63.6% | +30.9 |
| L20 | 94.9% | 76.1% | +18.9 |
| L21 | 95.3% | 77.1% | +18.2 |
| L24 | 94.9% | 80.9% | +14.0 |
| L27 | 94.2% | **84.0%** | +10.2 |

### 7.5 Delta vs Baseline 비교 (L27 letter probe)

| Task | Baseline letter | Delta letter | Δ |
|------|-----------------|--------------|---|
| shape_color | 99.2% | **99.0%** | -0.2 |
| obj_color | — | **93.7%** | — |
| shape_place | — | **89.7%** | — |
| obj_place | 78.8% | **84.0%** | **+5.2** |

**Delta binding phase transition 특징**:
- Baseline보다 **1 layer 일찍 binding 시작** (Baseline L17 → Delta L16)
- OOD (obj_place) L27 letter: Baseline 79% → **Delta 84%** (+5%p)
- 모든 task에서 direction probe가 L3부터 이미 90%+ (projector ΔW가 direction axis 조기 정렬)
- Binding gap이 OOD에서도 Baseline보다 작음 (obj_place Delta 10.2 vs Baseline ~13)

### 7.6 Vanilla (모든 layer에서 letter ≈ chance)

- shape_color L27: dir 88.7%, letter 24.7% (chance)
- obj_place L27: dir 56.8%, letter 29.7% (chance)
- → Vanilla는 representation에 **letter를 전혀 encoding하지 않음**

### 7.7 핵심 발견 — 중요한 Reframing

#### 1. "Phase transition Layer 15-20"의 실체 = direction-letter binding

- **이전 분석**: L14 → L20에서 direction probe가 급등
- **재해석 (letter probe 추가)**: L14에서 direction은 이미 encoded. **L14 → L21에서 letter가 emerge** (binding 계산 완료)
- Phase transition은 direction 추출이 아닌 **direction-candidate-letter binding 학습**

#### 2. Fine-tuning의 진짜 학습 대상 = binding

- Direction 추출 능력은 Vanilla도 상당 (L14 obj_place 75%)
- 차이는 **letter 생성을 위한 binding 학습**
- Vanilla는 direction 있어도 letter 생성 못 함 (binding 부재)

#### 3. MCQ-probe gap의 주 원인 = binding 불완전성

- Direction probe L27: 91.6% (풍부)
- Letter probe L27: 78.8% (거의 완성)
- MCQ acc: 79% (letter probe와 거의 같음)
- → Gap의 대부분이 "representation에서 letter까지의 binding" 부분에서 옴

#### 4. OOD 난이도의 재해석

- shape_color: direction 99% = letter 99% (binding perfect)
- obj_place: direction 91% > letter 79% (binding 12%p 미완성)
- OOD에서 어려운 건 direction 추출이 아니라 **binding 학습 부족**

#### 5. Delta vs Baseline 비교 (auxiliary loss 기여)

- Delta: binding이 Baseline보다 **1 layer 일찍 시작** (L16 vs L17)
- Delta: OOD binding 완성도 더 높음 (obj_place L27 letter 84% vs 79%, +5%p)
- Direction 추출 속도: Delta L3에서 이미 95%+ (auxiliary loss로 projector 조기 정렬)
- Baseline/Delta 모두 phase transition 구간 동일 (L15-L20) → mechanism 같고 **Delta는 더 강하게 수행**

### 7.8 시사점

- Direction probe 높다 ≠ 모델이 정답 낼 수 있다
- **Letter probe = 실질적 capability 지표**
- Binding은 L14~L21의 LLM attention 층에서 이뤄짐 (candidate context 활용)

---

## 8. Cross-Combination 재해석

### 8.1 기존 결과

Cross-combination 실험 결과 ("Delta 효과는 LLM weight 변화가 핵심"):
- MCQ accuracy 관점에서만 측정됨 → Delta projector + Baseline LLM = 73% (하락)

### 8.2 새로운 증거와 종합

- Delta projector는 **task-invariant direction axis**를 실제로 만듦 (cross-task gap 15%p)
- 하지만 Baseline LLM은 이 "새로운 representation"을 활용할 훈련 없음 → co-adaptation mismatch로 성능 하락
- Baseline LLM은 **정상적으로 (entangled) projector output**을 받아 L15~L20에서 disentangle하도록 학습됨
- Delta LLM은 **이미 disentangled projector output**을 받아 추가 refinement만 수행

### 8.3 결론

→ **Projector와 LLM이 어떻게 "역할 분담"하는지가 task-specific**. 둘 중 하나만 바꾸면 역할 분담이 무너져 성능 하락.

---

## 9. Averaged Last Token Swap Intervention (Stage 1-a)

첫 번째 intervention 실험. 예상과 정반대 결과로 Q3 주범 가설 반전.

### 9.1 Setup

#### Controlled dataset (Stage 0)

- R2R_4way_1500 각 task를 canonical candidates=[Up,Right,Down,Left]로 통일
- 결정론적 letter 매핑: up=A, right=B, down=C, left=D
- 각 (task, direction) × 1500 samples
- 저장: `analysis/swap_intervention/canonical_R2R/{task}.json`

#### Averaged hidden 추출

- (model, task, direction)별 N=100 sample forward
- Decoder layer L ∈ {15,16,17,18,19,20,21} output의 last token hidden 저장
- 평균 → `h_avg[model][task][direction][L]` shape [hidden_dim]
- 저장: `analysis/swap_intervention/avg_hidden/{model}_{task}.pt`

#### Window swap intervention

- Target sample forward 시 decoder layer L ∈ {15..21} 모두에 forward_hook 등록
- Hook: `output[0][:, -1, :] = h_avg[source_task][direction_of_target][L]`
- Vision + text token은 target 그대로, **last token만 averaged source로 교체**
- L+1 이후 정상 forward, 최종 logit에서 A/B/C/D token ID 중 argmax로 letter 예측

#### 실험 조건

- 2 model (Vanilla, Baseline) × 4 source × 4 target = 32 cell
- 각 cell당 target 400 sample (100/direction), no-swap baseline과 함께 측정

### 9.2 결과

#### Vanilla: 전 조건 ~25% (chance). Swap Δ ≈ 0

- Canonical ordering에서 Vanilla는 letter bias로 헛답 (training 분포와 prompt 형식 다름)
- Swap 효과 없음 = **Vanilla는 binding state 자체 없음**

#### Baseline — 예상 반전

| Src → Tgt | no-swap acc | swap acc | Δ |
|----------|-------------|----------|---|
| any → shape_color (in-domain easiest) | 98.8% | 100% | +1.2 |
| any → obj_color | 93.0% | 99-100% | +7 |
| any → shape_place | 81.0% | 94-97% | +13~16 |
| **any → obj_place (hardest OOD)** | **77.5%** | **95-99%** | **+17~21** |

**Aggregate**:
- In-domain (4 diag cell): no_swap 87.6% → swap **96.9%** (Δ +9.4)
- Cross-domain (12 off cell): no_swap 87.6% → swap **98.0%** (Δ +10.4)

**Cross-domain swap이 in-domain만큼 또는 그 이상 효과적.**

### 9.3 obj_place Target 상세 분석

**Direction별 no-swap 정답률 (Baseline)**:
- Up: 76/100 (76%)
- Right: **92/100 (92%)** ← 가장 쉬움
- Down: 72/100 (72%)
- Left: 70/100 (70%)
- Overall: 77.5%

**Direction asymmetry 발견**: Right가 월등히 잘 됨. Up/Down/Left는 70-76% 구간.

**Source별 swap 결과 (target = obj_place)**:

| Source → obj_place | swap acc | Δ vs no-swap |
|-------------------|----------|--------------|
| shape_color → obj_place | **98.50%** | **+21.0%p** |
| obj_color → obj_place | 96.50% | +19.0%p |
| shape_place → obj_place | 95.50% | +18.0%p |
| **obj_place → obj_place (자기 swap)** | 94.75% | +17.3%p |

**흥미로운 발견**:
- **shape_color → obj_place가 최고** (98.50%) — 가장 다른 task의 prototype이 가장 효과적
- **자기 swap이 가장 낮음** (94.75%) — 같은 task 평균도 조금 noisy
- → "Task identity가 문제가 아니라 representation noise가 문제"의 증거

### 9.4 해석 — Q3 주범 가설 반전

**예상**: cross-domain swap이 깨지면 → binding identity-conditional → OOD gap 직접 mechanism
**실제**: cross-domain swap이 **오히려 OOD acc를 in-domain 수준으로 끌어올림**

#### 새로운 해석

1. **Binding axis는 진짜 task-invariant**. shape_color 평균 last token을 obj_place target에 주입해도 98%+ acc — identity와 무관하게 direction→letter mapping 작동.

2. **Averaged last token = universal direction-letter prototype**. 100 sample 평균으로 sample-specific noise는 cancel되고 공통 "direction binding state"만 남음.

3. **OOD drop의 실제 원인 = 개별 OOD sample의 last token state가 prototype에서 벗어난 high-variance 영역에 있음**.
   - Obj_place → obj_place 자기 평균 swap도 77.5 → 94.8 (+17): 같은 task 내 sample 간에도 individual이 average보다 noisy
   - Binding 정보는 있지만 개별 sample별로 일관되게 clean state로 추출되지 않음

4. **Identity-conditional binding은 반증됨** (Q3 주범이 그게 아님)

#### Reframe

OOD binding gap은 representation이 "identity에 종속되어서" 생기는 게 아니라, OOD sample이 Baseline이 학습한 prototype trajectory에서 멀리 있어 **noisy/high-variance state**로 귀결되기 때문. Prototype을 강제 주입하면 clean up됨.

### 9.5 Implementation Detail

#### Script 구조 (`analysis/swap_intervention/`)

- `01_canonicalize_dataset.py`: canonical JSON 생성
- `02_extract_avg_hidden.py`: averaged hidden 추출 (per (model, task))
- `03_swap_experiment.py`: no-swap baseline + window swap 측정 (per (model, source, target))
- `04_summarize.py`: 32 result JSON aggregate
- `run_extract_all.sh`, `run_swap_all.sh`: GPU 0-3 병렬 launcher

#### Hook 구현

```python
def make_hook(replacement_vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = replacement_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        return output
    return hook

for L in [15,16,17,18,19,20,21]:
    model.model.layers[L].register_forward_hook(make_hook(h_avg[L]))
```

- Qwen2 decoder layer의 forward output은 tuple (hidden, ...). Hidden tensor [0]만 수정
- Last position `[:, -1, :]`만 교체, 나머지 토큰은 그대로
- Hook은 각 target sample forward 전 등록, 후 해제

#### Letter prediction

- Final logit `out.logits[0, -1, :]`에서 A/B/C/D의 token ID만 restrict
- `tokenizer.encode("A"|" A", add_special_tokens=False)` 단일 token id 탐색
- Restrict subset에서 argmax → 예측 letter

#### Pair 통제

- Extraction: 각 direction의 첫 100 sample 사용
- Swap target: offset=200부터 100 sample/direction (extraction set과 overlap 방지)
- Canonical ordering으로 source/target 모두 identical prompt → vision + text token count 동일 → token position 정렬 자동 보장

#### GPU 병렬

- Extraction: 8 unit (2 model × 4 task) → GPU 0-3 × 2 unit 순차 (~15 min)
- Swap: 32 job → GPU 0-3 × 8 job 순차 round-robin (~2 hour)

---

## 10. Proper Intervention Suite — Exp 0/1/2 (Stage 1-b)

Section J (Stage 1-a)의 averaged last token swap은 "averaging 시점에 identity 이미 제거됨" 확인 후, Kang et al. 방법론 직접 적용한 3종 실험.

### 10.1 배경 — Section J의 한계

Stage 1-a averaged swap은:
- Source의 100-sample averaged last token을 target에 주입 → OOD acc 77 → 97% 회복
- 해석: "clean averaged prototype → noisy individual 교정"
- 한계: **averaging이 이미 identity 성분을 cancel시킴** → "binding identity-invariant" 증명이 tautological
- 또한 whole-hidden swap은 direction 외 component도 동시에 교체 → causal isolation 부재

Kang Alg.2와 Eq.3-5를 제대로 적용해야 두 가설 분리 가능:
- **가설 A**: binding function 자체가 identity-conditional (OOD에서 direction→letter mapping 깨짐)
- **가설 B**: direction encoding이 OOD에서 sample-level로 noisy (function은 OK)

### 10.2 Setup

#### Common dataset (Section J와 동일)

- `analysis/swap_intervention/canonical_R2R/{task}.json`
- Canonical candidates = [Up, Right, Down, Left], letter A=Up/B=Right/C=Down/D=Left
- 각 (task, direction) 1500 sample

#### Exp 0 — Readout Alignment (Kang Eq.8)

hidden direction axis vs lm_head readout axis의 cosine similarity 측정.

```
h_UD = h_avg(Up) - h_avg(Down)      ← per (model, task, layer) averaged hidden
h_LR = h_avg(Right) - h_avg(Left)
w_letter_UD = lm_head.weight[A_id] - lm_head.weight[C_id]
w_letter_LR = lm_head.weight[B_id] - lm_head.weight[D_id]
w_word_UD = lm_head.weight[Up_id] - lm_head.weight[Down_id]
w_word_LR = lm_head.weight[Right_id] - lm_head.weight[Left_id]

Metric: cos(h_axis, w_axis) per (model, task, layer 15-21, 4 metric)
```

Kang Eq.8: `logit(A) - logit(C) ≈ (w_A - w_C)·h`. Alignment가 높아야 MCQ 작동.

#### Exp 1 — Counterfactual Steering (Kang Alg.2)

Pure direction vector + norm-preserving additive intervention.

```
Δ(d)[L] = h_avg(d)[L] - (1/4) Σ_d' h_avg(d')[L]    ← identity cancel by grand mean

Forward 중 L=15..21 모든 layer에서 forward_hook:
  h_new[:, -1, :] = h_original[:, -1, :] - Δ(current_dir)[L] + Δ(opposite_dir)[L]
  
Opposite flip: Up↔Down, Left↔Right

Target sample forward 후 final logit의 A/B/C/D 중 argmax → predicted letter

측정:
  flip_to_target_rate: 답이 opposite direction의 letter로 바뀐 비율
  flip_unchanged_rate: 답이 원래 direction letter 유지
  flip_to_other_rate: 답이 perpendicular 방향 letter로 튐
```

#### Exp 2 — Per-sample Variance

평균 없이 sample별 last token 저장 (L=15..27), MCQ correctness와 함께.

```
각 sample i에 대해:
  h_i[L] saved per layer
  MCQ prediction (letter argmax) + correct label saved

계산:
  Δ(d)[L] = h_avg(d)[L] - h_avg(all)[L]    ← 샘플 기반 재계산
  per-sample alignment_i[L] = cos(h_i - h_avg_all, Δ(direction_i))
  per-sample deviation_i[L] = ||h_i - h_avg(direction_i)||
  per-sample projection_i[L] = (h_i - h_avg_all) · Δ(d) / ||Δ(d)||
  
Aggregate per (model, task, layer):
  align_mean, align_std
  align_corr: mean alignment among MCQ correct samples
  align_wrong: mean alignment among MCQ wrong samples
  dev_mean: mean deviation from direction prototype
```

#### Scale

- Exp 0: CPU, 기존 avg_hidden 재활용 + lm_head 로드. ~1분
- Exp 1: 2 model × 4 task × 200 sample/dir × 4 dir × 2 forward (no-steer + steer) = 12,800 forward. 4 GPU ~2시간
- Exp 2: 2 model × 4 task × 200 sample/dir × 4 dir = 6,400 forward + per-sample hidden 저장. 4 GPU ~25분

### 10.3 결과

#### Exp 0 — Readout alignment at L=21

| Model | Task | cos(h_UD, w_letter_UD) | cos(h_UD, w_word_UD) |
|-------|------|----|----|
| Vanilla | shape_color | 0.039 | 0.031 |
| Vanilla | obj_place | 0.036 | 0.046 |
| **Baseline** | shape_color | 0.024 | **0.107** |
| **Baseline** | obj_color | 0.018 | **0.109** |
| **Baseline** | shape_place | 0.033 | **0.100** |
| **Baseline** | obj_place | 0.019 | **0.098** |

**관찰**:
- Baseline의 word readout alignment 모든 task에서 0.098-0.109 (~30x random level in 3584-dim)
- Vanilla는 전부 noise level (0.02-0.04)
- Letter readout alignment는 둘 다 약함 → letter binding은 L21 이후에 완성
- Task간 차이 거의 없음 → **alignment 자체는 OOD에서도 유지**

#### Exp 1 — Counterfactual steering flip rates

| Model | Task | orig_acc | **flip_to_target** | unchanged | →other |
|-------|------|----------|----|-----|-----|
| Vanilla | shape_color | 28.4% | 25.9% | 22.8% | 51.3% |
| Vanilla | obj_color | 23.5% | 22.8% | 23.1% | 54.1% |
| Vanilla | shape_place | 26.6% | 25.2% | 25.9% | 48.9% |
| Vanilla | obj_place | 25.6% | 24.8% | 24.1% | 51.1% |
| **Baseline** | shape_color | 99.4% | **71.4%** | 0.2% | 28.4% |
| **Baseline** | obj_color | 93.8% | **75.1%** | 1.6% | 23.2% |
| **Baseline** | shape_place | 83.4% | **62.1%** | 1.9% | 36.0% |
| **Baseline** | obj_place | 76.5% | **71.1%** | 1.9% | 27.0% |

**관찰**:
- Vanilla: flip rate ≈ chance (25%). Δ(d) 자체가 encoded되어 있지 않음.
- Baseline: flip rate 62-75%, **OOD (obj_place) 71%**가 in-domain (71%)와 거의 동일.
- Kang 논문 보고 median 64.6% 대비 오히려 강함.
- 28-36% "→other": Δ steering이 perfect isolation이 아님 → direction 정보가 Δ 외 subspace에도 분산.

**Per-direction asymmetry (shape_color 기준)**:
```
  Up → Down flip:    94.5%   (매우 강함)
  Right → Left flip: 94.5%
  Down → Up flip:    28.5%   (chance 수준)
  Left → Right flip: 68.0%
```
Up/Right 방향이 Down/Left보다 훨씬 steerable. Direction encoding이 asymmetric.

#### Exp 2 — Per-sample variance at L=21

| Model | Task | acc | align_mean | align_std | align_corr | align_wrong | corr-wrong |
|-------|------|-----|----|----|----|----|----|
| Vanilla | shape_color | 28.9% | 0.224 | 0.362 | — | — | +0.175 |
| Vanilla | obj_place | 24.5% | 0.106 | 0.225 | — | — | -0.014 |
| **Baseline** | shape_color | 99.6% | **0.775** | **0.124** | 0.78 | 0.23 | **+0.551** |
| **Baseline** | obj_color | 92.0% | 0.662 | 0.198 | 0.70 | 0.29 | +0.409 |
| **Baseline** | shape_place | 83.5% | 0.612 | 0.234 | 0.66 | 0.37 | +0.401 |
| **Baseline** | obj_place | 78.6% | 0.576 | **0.235** | 0.64 | 0.40 | **+0.365** |

**관찰**:
- Baseline alignment 평균: in-domain 0.775 → OOD 0.576 (**−26%**)
- Baseline alignment std: in-domain 0.124 → OOD 0.235 (**+90%, 거의 2배**)
- MCQ 맞은 sample의 alignment가 틀린 sample보다 0.365~0.551 높음 → **per-sample Eq.8 mechanism 작동 증명**
- 전반적으로 OOD에서 sample들이 direction axis에 덜 집중, 들쭉날쭉

### 10.4 종합 — Q3 (OOD binding gap) 원인 확정

#### 가설 A (binding function identity-conditional) — 반증

- Exp 1에서 OOD (obj_place) flip rate 71.1%, in-domain 71.4%와 거의 동일
- Δ(d) 주입만으로 direction 답 flip 가능 → binding은 direction을 linear readout
- Function 자체는 OOD에서도 작동 ⇒ **A 틀림**

#### 가설 B (direction encoding OOD에서 sample-level noisy) — 확증

- Exp 2 OOD에서 alignment std 2배 증가, 평균 26% 감소
- 개별 sample alignment가 MCQ 결과와 강한 correlation (correct-wrong 차이 0.365)
- OOD sample들의 direction 표현이 axis에서 흩어져 있음 ⇒ **noisy input이 binding에 들어감**

#### Stage 1-a (Section J) 결과와의 통합

- Averaged swap OOD +21% 회복 = noisy individual을 clean prototype으로 교체 효과
- Exp 1 71% flip = direction component만으로도 binding이 작동
- Exp 2 alignment variance 큼 = noise의 정체 = sample-level deviation
- 세 실험이 **같은 mechanism의 다른 측면을 보여줌**

#### Fine-tuning의 역할

- **Vanilla**: Δ(d) 자체가 hidden에 encoded 안 됨 (Exp 2 align 0.1, Exp 1 flip 25% chance)
- **Baseline**: Δ(d) 형성 + linear readout mechanism 작동 (Exp 2 align 0.6-0.8, Exp 1 flip 62-75%)
- **Fine-tuning = "direction vector formation + lm_head 축으로의 정렬"을 동시 학습**

### 10.5 분석/해석

#### Q 구조에 대한 답

- **Q1 (direction encoding이 어떻게 바뀌나)**: Baseline에서 Δ(d) axis 형성, alignment 0.5-0.8 (vs Vanilla 0.1)
- **Q2 (binding이 어떻게 가능해지나)**: Direction axis를 linear readout. lm_head direction 쪽으로 정렬 (word readout cos 0.10 vs Vanilla 0).
- **Q3 (OOD drop 원인)**: **가설 B — direction encoding의 sample-level variance**. 회로는 OK, input 들쭉날쭉.
- **Q4 (개선 방향)**: **variance 출처 추적** 필요. Vision encoder? Projector? Early LLM layers? 어느 stage에서 OOD sample의 representation이 in-domain보다 흩어지는지.

#### 논문 narrative 후보

> Fine-tuning creates a "direction vector Δ(d) + linear readout" mechanism. This mechanism is identity-invariant: counterfactual steering flips answers equally well in OOD as in in-domain (71% vs 71%). However, OOD hurts via a different route — individual OOD samples have direction representations scattered more widely around Δ(d) (variance 2x). The binding function reads from a noisy input in OOD. The mechanism is intact; the representation going into it is unreliable.

#### Limitation

- Flip rate 100%가 아님 (28-36% → other): Δ(d)가 perfect direction isolation 아님
- Direction asymmetry (Up/Right vs Down/Left): encoding 비대칭 → 추가 분석 필요
- L21만 측정: L22-27 alignment 측정으로 letter binding 최종 완성 layer 파악 필요

### 10.6 Implementation Detail

#### Script 구조 (`analysis/swap_intervention/`)

- `05_readout_alignment.py` — Exp 0: lm_head + avg_hidden → cosine similarities (1회 실행)
- `06_extract_per_sample.py` — Exp 2: per-sample hidden 저장 (per model × task)
- `07_counterfactual_steering.py` — Exp 1: Kang Alg.2 additive intervention
- `08_analyze_variance.py` — Exp 2 post-processing
- `09_summarize_steering.py` — Exp 1 aggregate
- `10_plot_all.py` — 3개 figure 생성

#### Counterfactual hook

```python
def make_additive_hook(sub_vec, add_vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = h[:, -1, :] - sub_vec.to(h.device, h.dtype) + add_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        return output
    return hook

for L in [15,16,17,18,19,20,21]:
    sub = delta[current_dir][L]
    add = delta[opposite_dir][L]
    decoder_layers[L].register_forward_hook(make_additive_hook(sub, add))
```

#### Per-sample extraction (Exp 2)

```python
hiddens = np.zeros((n, len(LAYERS), hidden_dim), dtype=np.float16)
for i, L in enumerate(LAYERS):
    hiddens[idx, i] = out.hidden_states[L+1][0, -1, :].cpu().numpy()
```

#### Δ(d) 계산

```python
def compute_deltas(avg_hidden):
    deltas = {}
    for L in LAYERS:
        h_all = sum(avg_hidden[d][L] for d in DIRS) / len(DIRS)
        for d in DIRS:
            deltas.setdefault(d, {})[L] = avg_hidden[d][L] - h_all
    return deltas
```

#### Readout extraction

```python
W = model.lm_head.weight.detach().float().cpu()
tid_A = tokenizer.encode("A" or " A", add_special_tokens=False)[0]
letter_UD = W[tid_A] - W[tid_C]
```

#### 결과 파일

- `readout_alignment.json` (Exp 0)
- `per_sample/{model}_{task}.npz` (Exp 2 raw)
- `variance_summary.json` (Exp 2 processed)
- `steering_results/{model}_{task}.json` + `_summary.json` (Exp 1)
- `fig1_readout_alignment.png`, `fig2_variance_analysis.png`, `fig3_steering.png`

---

## 11. Research Questions & Answers

**Central framing**: Fine-tuning이 VLM direction reasoning의 두 현상 (task-invariant direction encoding + letter binding)을 어떻게/어디서 만들어내며, 이것이 OOD 성능에 어떻게 귀결되는가.

### 11.1 Q1 — Fine-tuning 후 direction encoding이 어떻게 바뀌었나?

**관찰**: Last token에 task-invariant direction info가 생김 (cross-task gap: Vanilla 25.5%p → Baseline 5.6%p)

**Sub-questions**:
- **(a) 어느 layer에서 task-invariance가 형성되는가?** — layer별 cross-task probing (answer token per layer). 이미 L0/L14/L27 있음; L3,7,10,18,21,24 추가 필요.
- **(b) 이 encoding이 identity-conditional인가?** — Design A, **in-domain swap**. Averaged source last token을 같은 task 다른 sample에 주입 → 답 유지율.

**Answer**: Baseline에서 Δ(d) axis 형성, alignment 0.5-0.8 (vs Vanilla 0.1).

### 11.2 Q2 — Fine-tuning 후 binding이 어떻게 가능해졌나?

**관찰**: Letter probe가 L15~21에서 chance → 70%+ 급등 (Vanilla는 전 layer chance)

**Sub-questions**:
- **(a) Q1의 task-invariance와 binding의 시간관계** — cross-task gap trajectory vs letter probe trajectory overlay. 두 현상이 같은 mechanism인지/순차인지 판별.
- **(b) 이 binding이 identity-conditional인가?** — Design A, **cross-domain swap**. Source task A에서 평균낸 last token을 task B target에 주입 → 답 유지율.

**Answer**: Direction axis를 linear readout. lm_head direction 쪽으로 정렬 (word readout cos 0.10 vs Vanilla 0).

### 11.3 Q3 — OOD에서 왜 떨어지나?

**답 (Section K 확정): 가설 (B) — direction encoding의 sample-level variance 증가.**

- Counterfactual steering flip rate: in-domain 71% ≈ OOD 71% → binding function은 OOD에서도 작동 ⇒ (a) 반증
- Per-sample alignment std: in-domain 0.124 → OOD 0.235 (2배) ⇒ (b) 확증
- MCQ 맞은/틀린 sample의 alignment 차이 0.365-0.551 → sample-level Eq.8 mechanism 작동
- **OOD sample들의 direction 표현이 axis에서 흩어짐 → noisy input으로 인한 readout 실패** (function은 OK)

Stage 1-a averaged swap의 denoising 효과와 일관됨: clean prototype 주입 시 77%→97% 회복.

### 11.4 Q4 — 어떻게 개선하나?

Q3 원인 = direction encoding의 sample-level variance. 개선 방향:

#### (i) Variance 출처 추적 (후속 실험 필요)

- Vision encoder 단계에서 OOD sample이 이미 분산 큰가?
- Projector 통과 시 variance 증폭되나?
- LLM early layer (L0-L14)에서 축적되나?
- → Stage-wise variance tracking 실험

#### (ii) Variance 직접 감소

- OOD 데이터로 augment fine-tuning
- Contrastive loss로 direction prototype 주변 tight clustering 강제
- Noise-robust training objective

#### (iii) Inference-time 보정

- Stage 1-a가 이미 보여줌: averaged prototype 주입으로 OOD 77→97% 회복
- 하지만 practical하진 않음 (inference 시 averaged hidden 필요)
- Steering-based: 최대 logit 방향으로 Δ(predicted_dir) 강화

#### (iv) 기존 관찰과의 통합

- Cross-combination 결과 (Section I): projector만 바꾸면 73% 하락 → projector-LLM co-adaptation 필요
- Delta가 작동하는 이유: auxiliary loss → projector 변화 → LLM input 분포 변화 → LoRA가 다른 방향으로 학습 (indirect effect)

---

## 12. 실험 계획 (intervention-centric, observational 지양)

### 12.1 Stage 0 — Controlled dataset ✅

- Canonical candidates=[Up,Right,Down,Left] 통일, letter={A,B,C,D} 결정론적 매핑
- 각 (task, direction)별 1500 sample 확보 → identity pool averaging 가능
- 저장: `analysis/swap_intervention/canonical_R2R/{task}.json`

### 12.2 Stage 1 — Q1(b) + Q2(b) + Q3 주범 판별 (Design A, in-domain vs cross-domain) ✅

**Averaged last token swap (Kang Eq.3 style)**:
- 각 (model, task, direction)별 100 sample에서 last token hidden 추출 → 평균
- Target forward 중 layer L=15~21 (binding window)에서 target last token을 averaged source로 교체 (window swap)
- Vision+text는 target 그대로 유지
- Pair: same-direction same-answer diff-identity

**실험 조건**: 2 model × 4 source × 4 target = 32 cell
- source=target → in-domain swap
- source≠target → cross-domain swap

**해석 매트릭스**:

| In-domain | Cross-domain | 결론 |
|-----------|-------------|------|
| 유지 | 유지 | Binding identity-free, OOD gap은 다른 mechanism |
| 유지 | 깨짐 | **Cross-domain binding이 identity-conditional → OOD gap 직접 mechanism** |
| 깨짐 | 깨짐 | Direction encoding이 identity-conditional (task-invariance 약함) |

### 12.3 Stage 2 — Q1(a) + Q2(a) 보강 (layer trajectory)

- Answer token layer-wise cross-task probing (누락 layer 채우기: L3, 7, 10, 18, 21, 24)
- Cross-task gap trajectory vs Letter probe trajectory overlay → Q2(a) 답

### 12.4 Stage 3 — Stage 1 결과에 따라 우선순위 조정

Q3 주범이 (a)라면:
- Direction encoding이 어느 layer/module에서 task-invariant해지는가 → attn vs MLP patching (Stage 2-a), head-level scan (Stage 2-b)
- LoRA grafting (Stage 3-b) → weight locus

Q3 주범이 (b)라면 (현재 확정):
- Binding이 어느 layer에서 일어나는가 (L15~21 중 어디) — sub-window localization
- Binding이 어느 module에서 일어나는가 — attn/MLP/head
- OOD에서 binding이 왜 깨지는가 — probe-circuit axis 불일치 (Stage 4-b)

### 12.5 Stage 4 — Q4 보강 (개선 방향 insight)

- Cross-combination 재실험 (Section I의 1500 sample 재검증)
- Delta projector의 ΔW SVD 분해 → task-invariance 학습의 기하

### 12.6 실행 순서

Stage 1이 Q1(b)+Q2(b)+Q3을 동시에 답. 결과에 따라 Stage 3 방향 결정.

### 12.7 Infra

- MCQ JSON candidates 4개로 재업로드 (완료)
- Delta 모델 4-way feature 추출 완료 확인 필요
- Cross-model validation (Qwen3-VL, future work)

---

## 13. 기술적 세부 사항

### 13.1 디렉토리 구조

#### Feature 저장

```
/data3/local_datasets/vlm_direction/linear_probing_1500/{model}/
  vision_encoder/{task}/features.npy
  after_projector/{task}/features.npy
  vision_token/{task}/features_layer_*.npy
  answer_token/{task}/features_layer_*.npy
```

#### 결과

```
output_1500/{model}/
  linear_probe_results/{task}/
  answer_probe_results/{task}/
```

### 13.2 핵심 Python 파일

| 파일 | 역할 |
|------|------|
| `linear_probing/extract_vision_features.py` | Vision feature 추출 (`--pool_spatial`로 N 축 제거) |
| `linear_probing/extract_answer_features.py` | Last token per-layer 추출 |
| `linear_probing/linear_probe.py` | GPU-only probing |

### 13.3 기술적 최적화

- GPU-only probing (`train_linear_probe_gpu`)
- Streaming mmap write (`FeatureWriter`, RAM 상수)
- TF32 활성화 (`torch.backends.cuda.matmul.allow_tf32 = True`)
- `generate()` → `prepare_inputs_labels_for_multimodal + forward` (generation loop 제거)
- Per-layer CPU sync → batched (GPU stack 후 단일 `.cpu()`)
- num_workers 16

### 13.4 Hook 주의사항

- **Qwen2 `self_attn` forward hook의 return값 반영 안 됨** — decoder layer hook 사용
- LLaVA `generate()` 출력은 생성된 토큰만 반환 (input 미포함)

### 13.5 Conda 환경

| 환경 | 용도 |
|------|------|
| `lmms_llavavideo` | Feature 추출 + probing + analysis 실행 |
| `llava_next` | 모델 weight 분석 (lm_head, LoRA delta 등) |
| `lmms_py311` | Qwen3-VL용 (향후 cross-model validation) |

### 13.6 실행 스크립트

```bash
# 1500 전체 (3모델 × 4task × GPU 0~3 병렬, 모델 간 순차)
bash scripts/linear_probing/run_R2R_1500_all.sh
```

---

## 14. 종합 결론

### 14.1 핵심 발견 Top 5

1. **Fine-tuning = 희석 방지**
   Direction 정보는 Vision Encoder에 이미 존재하고, 문제는 LLM 후반까지 보존하는 것. Fine-tuning은 새로운 정보를 만들지 않고, Vanilla에서 L18 이후 발생하는 희석을 방지.

2. **Phase transition의 실체는 letter binding**
   L14-L21에서 emerge하는 것은 direction이 아니라 **direction→letter 매핑**. Direction 정보는 L14 이전에 이미 있음. Fine-tuning의 진짜 학습 대상은 binding.

3. **Projector vs LLM 분업 차이**
   - **Baseline**: LLM-centric (L15-L20 decision boundary alignment)
   - **Delta**: Projector-first (axis 직접 align)
   - 같은 최종 accuracy이지만 mechanism이 다름. Cross-combination 시 mismatch로 성능 하락.

4. **OOD drop은 function 문제가 아닌 input noise 문제**
   Binding circuit은 OOD에서도 identity-invariant하게 작동. 단, 입력 direction representation의 sample-level variance가 2배. Mechanism intact, input unreliable.

5. **Cross-task gap의 진짜 mechanism**
   SigLIP position encoding이 task-specific → direction encoding도 task-specific → probe axis가 task마다 다른 방향. Entanglement 아닌 **axis rotation**. Fine-tuning은 이 축들을 universal axis로 align.

### 14.2 확정된 Mechanism

```
Fine-tuning이 만든 회로:
  h → Δ(d) projection → linear readout (lm_head) → letter
  
이 회로는 identity-invariant (OOD에서도 작동)
But: OOD sample들의 h가 Δ(d) axis 주변에 더 넓게 흩어짐 (noisy input)
  → Linear readout이 noisy input 받음 → 틀린 답
```

### 14.3 논문 narrative

> **Fine-tuning creates a "direction vector Δ(d) + linear readout" mechanism.** This mechanism is **identity-invariant**: counterfactual steering flips answers equally well in OOD as in-domain (71% vs 71%). However, OOD hurts via a **different route** — individual OOD samples have direction representations scattered more widely around Δ(d) (variance 2x). The **binding function reads from a noisy input** in OOD. **The mechanism is intact; the representation going into it is unreliable.**

### 14.4 앞으로의 방향

**Variance 출처 추적**: Vision encoder / Projector / early LLM 중 어디서 OOD sample의 representation이 in-domain보다 흩어지는지 stage별로 측정. 이 결과가 개선 전략을 결정:

- Variance가 vision encoder에서 시작 → vision-side augmentation
- Projector에서 증폭 → projector regularization
- Early LLM에서 축적 → early layer stabilization

**Inference-time steering**: Averaged prototype 주입으로 OOD 77→97% 회복 가능. 실용화 방향은 lightweight version (e.g., 방향별 small vector 저장).

**Cross-model validation**: Qwen3-VL 등 다른 VLM에서도 같은 mechanism이 관찰되는지 검증.
