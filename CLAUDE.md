# Cross-Modal Information Flow Analysis

VLM(Video Language Model)의 방향 인식 능력을 분석하는 프로젝트.
Vision encoder → Projector → LLM 파이프라인에서 direction 정보가 어떻게 흐르고, fine-tuning이 이를 어떻게 변화시키는지 분석.

---

## Conda 환경

| 환경 | 용도 |
|------|------|
| `lmms_llavavideo` | Feature 추출 + probing + analysis 실행 |
| `llava_next` | 모델 weight 분석 (lm_head, LoRA delta 등) |
| `lmms_py311` | Qwen3-VL용 (향후 cross-model validation) |

---

## 모델

| 약칭 | 설명 | LoRA Path |
|------|------|-----------|
| `Vanilla` | LLaVA-Video-7B-Qwen2 (fine-tuning 없음) | — |
| `Baseline` | 4combo_v2 LoRA (MCQ loss만) | `...baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5` |
| `Delta` | 4combo_v2 + delta_direct auxiliary loss | `...delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5` |

### delta_direct 학습 구조

```
Vision Encoder → Projector → temporal delta (frame[t+1] - frame[t])
                                    ↓
                          Direction head ← auxiliary loss (projector만 업데이트)

                  Projector output → LLM → MCQ answer ← main loss (전체 업데이트)
```

Inference 시 auxiliary head 제거.

---

## Task: R2R 4-way 1500

Direction당 1500 sample, 총 6000/task.

| Task | Object | Background | 난이도 |
|------|--------|------------|--------|
| shape_color | 합성 도형 | 합성 색상 | In-domain |
| obj_color | 실제 객체 | 합성 색상 | 중간 OOD |
| shape_place | 합성 도형 | 실제 장소 | 어려운 OOD |
| obj_place | 실제 객체 | 실제 장소 | 가장 어려운 OOD |

- **정답 direction**: Up/Down/Left/Right (4-class, chance=25%)

---

## Feature Shapes & Pooling

| Stage | 저장 Shape | 설명 |
|-------|-----------|------|
| Vision Encoder | (N, 8, 1152) = (N, T, D_ve) | SigLIP 출력, spatial mean pool |
| After Projector | (N, 8, 3584) = (N, T, D_llm) | mm_projector + bilinear pool 후 spatial mean |
| Vision Token (layer l) | (N, 8, 3584) per layer | LLM decoder layer l vision position, spatial mean |
| Answer Token (layer l) | (N, 3584) per layer | LLM decoder layer l last token |

**규칙:** Vision feature probing 시 N(spatial) 축은 날리고 T(temporal) 축 보존. `--pool_spatial` 플래그로 저장 시점에 적용.

---

## Linear Probing 결과 (1500-sample)

3모델(Vanilla / Baseline / Delta) × 4 task × vision pipeline + answer token 전부 완료.

### A. Vision Pipeline (Direction 4-class probe accuracy)

#### shape_color (in-domain)

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

### B. Answer Token (Layer-wise)

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

---

## 핵심 관찰

### 1. Fine-tuning = 후반 layer의 direction 보존 (Vanilla의 희석 복구)

Vanilla는 L18~L27로 갈수록 direction probe가 급격히 떨어짐. Fine-tuning은 이 희석을 막음.

**obj_place L27 Vision Token:** Vanilla 65% → Baseline 89% → Delta 90%
**obj_place L27 Answer Token:** Vanilla 60% → Baseline 91% → Delta 94%

### 2. Delta의 효과는 2군데에 집중 (projector + L19 amplifier)

**초반 (projector output, direction probe acc):**
- After Projector obj_place: Vanilla 75% / Baseline 75% / **Delta 85%**
- Vision Token L0 obj_place: Vanilla 76% / Baseline 80% / **Delta 83%**

**L19 direction amplifier** (N.3 후속 측정):
- SC L19 total push: Baseline +37.8 / **Delta +48.8** (+29%)
- OP L19 total push: Baseline +17.6 / **Delta +26.9** (+52%) ← OOD에서 Delta가 더 강함
- 특히 MLP 기여 강화 (Baseline 23 → Delta 32)

- Delta_direct auxiliary loss → projector direction-rich + LLM L19 amplifier 강화 (둘 다)
- **Direction probe L27 종착점은 비슷** (Baseline≈Delta), 단 **Letter probe L27 OP: Baseline 79% vs Delta 84% (+5pp)** — binding 완성도 차이

### 3. Task 난이도 ∝ Fine-tuning의 복구 효과 (L27 기준)

| Task | Vision Token Δ (Baseline-Vanilla) | Answer Token Δ (Baseline-Vanilla) |
|------|---|---|
| shape_color | +3 | +10 |
| obj_color | +19 | +25 |
| shape_place | +18 | +30 |
| obj_place | +25 | +31 |

OOD가 어려울수록 Vanilla 희석 심함 → fine-tuning 복구량 큼.

### 4. Vanilla 패턴: 초반 급상승 → 후반 희석

Vanilla obj_place answer token: L3 75%, L14 76%, L18 70%, L21 68%, L27 60%.
- L3에서 이미 direction 정보 도달
- 하지만 L18부터 **decline** — representation이 language generation 방향으로 수렴하면서 direction 정보가 소실됨
- Fine-tuning의 핵심 기여 = 이 decline을 막는 것

### 5. Delta vs Baseline 차이는 주로 어려운 OOD에서

- shape_color: 두 모델 L27 모두 99.9% (포화)
- obj_place: Baseline 91% vs **Delta 94%** (+3%p), shape_place에서도 유사
- Delta의 auxiliary supervision이 어려운 OOD에서 +1~3%p 마진

### 6. Vision token ≈ Answer token (direction signal 측면)

- Baseline obj_place L27: vision 89% vs answer 91% (거의 동등)
- 200-sample에서 관측된 "last token이 vision token보다 direction 훨씬 높다"는 overfitting artifact
- Fine-tuning은 vision token과 answer token 둘 다에서 direction을 보존

---

## 실행 스크립트

```bash
# 1500 전체 (3모델 × 4task × GPU 0~3 병렬, 모델 간 순차)
bash scripts/linear_probing/run_R2R_1500_all.sh
```

---

## 디렉토리 구조

### Feature 저장
```
/data3/local_datasets/vlm_direction/linear_probing_1500/{model}/
  vision_encoder/{task}/features.npy
  after_projector/{task}/features.npy
  vision_token/{task}/features_layer_*.npy
  answer_token/{task}/features_layer_*.npy
```

### 결과
```
output_1500/{model}/
  linear_probe_results/{task}/
  answer_probe_results/{task}/
```

---

## 핵심 Python 파일

| 파일 | 역할 |
|------|------|
| `linear_probing/extract_vision_features.py` | Vision feature 추출 (`--pool_spatial`로 N 축 제거) |
| `linear_probing/extract_answer_features.py` | Last token per-layer 추출 |
| `linear_probing/linear_probe.py` | GPU-only probing |

---

## 기술적 최적화

- GPU-only probing (`train_linear_probe_gpu`)
- Streaming mmap write (`FeatureWriter`, RAM 상수)
- TF32 활성화 (`torch.backends.cuda.matmul.allow_tf32 = True`)
- `generate()` → `prepare_inputs_labels_for_multimodal + forward` (generation loop 제거)
- Per-layer CPU sync → batched (GPU stack 후 단일 `.cpu()`)
- num_workers 16

## Hook 주의사항

- **Qwen2 `self_attn` forward hook의 return값 반영 안 됨** — decoder layer hook 사용
- LLaVA `generate()` 출력은 생성된 토큰만 반환 (input 미포함)

---

### C. Cross-Task Direction Probing (Last Layer Answer Token)

Train on task A, test on task B → 4×4 matrix per model. Off-diagonal이 높을수록 **direction-readable axis가 task 간 공통**임을 의미 (representation에 identity가 없다는 뜻은 아님 — identity는 orthogonal 차원에 공존 가능).

**Vanilla (Transfer gap 25.5%p):**

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **88.5%** | 49.0% | 32.1% | 33.3% |
| obj_color | 70.5% | **71.3%** | 29.4% | 33.0% |
| shape_place | 45.7% | 38.8% | **60.4%** | 43.9% |
| obj_place | 46.6% | 45.2% | 53.3% | **55.6%** |

**Baseline (Transfer gap 5.6%p):**

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.9%** | 92.7% | 83.4% | 72.4% |
| obj_color | 99.5% | **97.5%** | 85.7% | 81.1% |
| shape_place | 98.5% | 92.8% | **94.6%** | 86.3% |
| obj_place | 97.8% | 93.9% | 93.7% | **89.8%** |

**Delta (Transfer gap 3.4%p):**

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.9%** | 91.9% | 90.2% | 81.0% |
| obj_color | 99.9% | **97.5%** | 94.8% | 88.1% |
| shape_place | 99.7% | 94.2% | **97.1%** | 90.9% |
| obj_place | 98.3% | 95.4% | 96.7% | **92.9%** |

**종합:**

| | Diagonal (in-task) | Off-diagonal (cross) | Transfer Gap |
|---|---|---|---|
| Vanilla | 68.9% | 43.4% | **25.5%p** |
| Baseline | 95.5% | 89.8% | **5.6%p** |
| Delta | 96.9% | 93.4% | **3.4%p** |

**핵심 해석:**

1. **Probe가 진짜 direction을 본다 (task-shortcut 아님)**: Baseline/Delta의 cross-task 90%+는 학습된 probe가 object/background 무관하게 direction을 추출함을 증명.

2. **Fine-tuning = Direction axis의 task-invariant alignment**:
   - Vanilla: direction-readable axis가 task마다 다른 방향 (shape_color → obj_place transfer 33%)
   - Baseline: task 간 공통 axis로 정렬 (transfer gap 25.5 → 5.6%p)
   - Delta: 추가 정렬 (3.4%p) — auxiliary direction supervision이 task-invariant axis 강화
   - ※ "identity가 벗겨졌다"는 뜻 아님. Identity 정보는 별개 차원에 남아있을 수 있음 (MCQ 생성에 필요)

3. **Delta의 고유 기여는 OOD cross-transfer에서 선명**: obj_place → shape_color 같은 어려운 전이에서도 Delta가 Baseline보다 꾸준히 1-5%p 높음.

> Note: 이 결과는 8-way candidates 버전 feature (`linear_probing_1500/`) 기반. Direction label은 candidates 개수와 무관하므로 probing 결과는 유효. 4-way feature 재추출 후 재검증 예정.

시각화: `analysis/cross_task_probing_1500.png`

### D. Cross-Task Probing — Pipeline 전체 (Vision Encoder / After Projector / LLM 각 layer)

모든 stage × 모든 layer × 3 모델에서 4×4 cross-task matrix 측정.

#### Vision Encoder (frozen, 3 모델 동일)

| Train \ Test | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | **99.5%** | 80.4% | 38.5% | 39.9% |
| obj_color | 85.6% | **94.2%** | 26.8% | 30.6% |
| shape_place | 24.2% | 28.2% | **85.0%** | 78.7% |
| obj_place | 35.5% | 43.9% | 72.6% | **85.4%** |

**Diag=91.0% | Off=48.7% | Gap=42.3%p**

- Color backgrounds (shape_color ↔ obj_color) 간 전이는 85%+ (object type 달라도 OK)
- Place backgrounds (shape_place ↔ obj_place) 간 전이도 72-78%
- **Color ↔ Place 전이는 26-40%로 급락** — SigLIP은 background type에 크게 의존

#### After Projector

| 모델 | Diag | Off-diag | Gap |
|------|------|----------|-----|
| Vanilla | 87.9% | 50.0% | 38.0%p |
| Baseline | 88.8% | 50.2% | 38.6%p |
| **Delta** | **99.3%** | **78.1%** | **15.2%p** |

**핵심 발견:**
- **Vanilla vs Baseline 거의 동일** (gap 38 vs 39%p). MCQ loss만으로는 projector 수준에서 task-invariant axis 정렬 안 됨.
- **Delta만 projector 수준에서 큰 개선** (gap 39 → 15%p, off-diag 50 → 78%). Auxiliary direction loss가 projector output을 task-invariant direction axis로 정렬.

**Delta projector의 구체적 개선:**
- Shape-Color → Obj-Place 전이: Baseline 30.5% → **Delta 45.2%**
- Obj-Place → Shape-Color 전이: Baseline 48.1% → **Delta 94.3%**
- Shape-Place → Obj-Color 전이: Baseline 33.3% → **Delta 88.6%**

시각화: `analysis/cross_task_ve_ap_matrices.png`

#### Vision Token & Answer Token Layer-wise

| 측정 | Vanilla | Baseline | Delta |
|------|---------|----------|-------|
| Vision token L0 off-diag | ~55% | ~50% | **~75%** |
| Vision token L14 off-diag | ~45% | ~75% | ~85% |
| Vision token L27 off-diag | ~45% | ~80% | ~85% |
| Answer token L14 off-diag | ~50% | ~75% | ~85% |
| Answer token L27 off-diag | ~40% | ~90% | ~95% |

**패턴:**
- **Vanilla**: Layer가 깊어져도 cross-task transfer 개선 없음. 오히려 후반에 약간 하락.
- **Baseline**: L10~L20 구간에서 **phase transition** — cross-task 50% → 80% 급상승. LLM attention이 task-invariant direction axis로 정렬하는 구간.
- **Delta**: **초기 layer부터 이미 높은 off-diag**. Projector에서 완성된 task-invariant direction axis가 LLM 전체에 전파.

시각화: `analysis/cross_task_pipeline_vision.png`, `cross_task_pipeline_answer.png`, `cross_task_pipeline_comparison.png`

### E. Cross-task Gap 원인 분해 (Entanglement 해석의 정정)

> **주의**: 이전 "direction이 identity와 entangle되어 있다"는 표현은 **over-claim**. Cross-task accuracy 낮다고 바로 entanglement 결론 내릴 수 없음.

Cross-task acc 낮음을 설명할 수 있는 4가지 가설:
| 가설 | 의미 | 검증 |
|------|------|------|
| (a) Axis rotation | Task마다 direction 축이 다른 방향 | cos(probe axes) |
| (b) Scale 차이 | 같은 축, 분산 다름 | rescale experiment |
| (c) Bias shift | 같은 축, offset 다름 | center-rescale |
| (d) True entanglement | Direction이 identity와 같은 dimension 공유 | Fisher dims overlap |

**측정 결과 (projector 기준):**

| 가설 | 실측 | 결론 |
|------|------|------|
| (a) Axis rotation | Vanilla/Baseline cos = 0.27-0.39 (낮음) | **주 원인** |
| (b) Scale 차이 | Rescale Δ = +3-5%p | 보조 원인 (작음) |
| (c) Bias shift | — | 미측정 |
| (d) True entanglement | Fisher dims overlap ≈ 0% | **아님** |

**정확한 표현**: "Direction이 identity와 entangle" (X) → **"Direction-readable axis가 task마다 다른 방향으로 놓임 (task-specific orientation)"** (O)

- Direction과 identity가 **같은 dimension을 공유하지 않음** (Fisher overlap 0)
- 대신 **identity context가 direction encoding의 축 자체를 결정**
- 즉 "어느 차원이 direction을 encoding하는가"가 identity에 따라 달라짐

**비유:** 같은 "오른쪽"이라도
- shape_color: "합성 도형의 centroid가 right-patch로 이동" → dim 100-200 일대에 encoding
- obj_place: "객체가 실제 배경 대비 flow" → 완전히 다른 dim에 encoding
- 두 encoding이 같은 차원을 공유하지 않지만, SigLIP 공간에서 **다른 axis**에 놓임

**Fine-tuning이 하는 일:** 여러 identity의 서로 다른 direction subspace를 → **공통 direction axis**로 align/rotate.
- Delta projector: 이를 projector 수준에서 수행 (cos 0.27 → 0.67)
- Baseline LLM: L15~L20 attention에서 수행 (LLM이 identity-specific axis들을 universal axis로 rotate)
- Vanilla: 수행 못 함 (cross-task transfer 실패)

### F. Direction Axis Alignment 분석 — Cross-task gap의 원인 분해

Cross-task acc 낮은 원인을 3가지로 분해:
(1) **축이 다름** — probe weight의 direction axis (Up-Down, Left-Right)가 task마다 다른 방향
(2) **Scale 차이** — 축은 같아도 task별 feature scale/variance가 달라 probe 부작동
(3) **Pure transfer** — 축+scale 다 맞췄을 때도 남는 gap

**Method 1: Probe axis cosine similarity**
각 task에서 학습된 probe의 `v_UD = W[Up] - W[Down]`, `v_LR = W[Left] - W[Right]` 추출.
Task pair 간 cos(v_A, v_B) 측정.

**Method 2: Cross-task eval with rescaling**
Source probe를 target task에 적용할 때 target 자체의 mean/std로 z-norm. Rescale 효과 (Δ)가 scale 기여도.

#### After Projector (핵심 비교)

| 모델 | cos_UD | cos_LR | off_orig | off_rescale | Rescale Δ |
|------|--------|--------|----------|-------------|-----------|
| Vanilla | 0.37 | 0.27 | 49.7% | 53.2% | +3.4 |
| Baseline | 0.39 | 0.27 | 49.8% | 53.6% | +3.8 |
| **Delta** | **0.67** | **0.70** | **78.1%** | **83.2%** | +5.1 |

- **Vanilla/Baseline projector**: 축이 task마다 다름 (cos ~0.3). Rescale 효과도 작음 → scale 문제 아니고 **축 문제**.
- **Delta projector**: **축 정렬 급상승** (cos 0.27 → 0.70). Auxiliary loss가 projector 수준에서 task-agnostic direction axis 학습.

#### Layer-wise Trajectory

**Vanilla:** Vision encoder → Answer L27 내내 cos ~0.3-0.5. **끝까지 alignment 실패**.

**Baseline:** 
- L0: cos_UD 0.39 (Vanilla와 동일)
- **L14에서 급상승**: 0.55 (LLM attention이 축 정렬 시작)
- L21 peak: 0.50-0.55
- L27 answer: cos=0.37 (낮음)이지만 cross-task acc=90% (높음)
  - → Probe weight 벡터는 달라지지만 **decision boundary 수준 alignment 확보**
  - High-dim equivalence — 다른 weight solution이 같은 classification 결과

**Delta:**
- After projector부터 cos=0.67 (이미 정렬)
- LLM layer들은 유지/미세조정만 수행

#### Scale Contribution (Rescale Δ) Trajectory

| Stage | Vanilla | Baseline | Delta |
|-------|---------|----------|-------|
| Vision Encoder | +10.8 | +10.8 | +10.8 |
| After Projector | +3.4 | +3.8 | +5.1 |
| Vision Token L14 | +12.3 | +9.6 | +13.0 |
| Answer L21 | +8.7 | **+1.5** | **+1.0** |
| Answer L27 | +5.5 | **+1.5** | **+0.8** |

- **Vision encoder에서 scale 기여도 큼** (+10.8%p) — raw SigLIP feature는 task별 scale 차이 있음
- **Answer L21+ Baseline/Delta에서 Δ=+1-2%p만** — representation이 이미 scale-invariant. 남은 차이는 "axis shift" 아님.

#### 종합 해석

**Cross-task gap의 원인 분해표:**

| Stage / Model | 주 원인 | 근거 |
|---------------|--------|------|
| Vision Encoder (frozen) | 축+scale 둘 다 | cos~0.4, Δ=+11 |
| Vanilla projector/LLM | 축 불변 (미개선) | cos 0.3-0.5, transfer 약함 |
| Baseline LLM (L14+) | 축 부분 정렬 + decision boundary 수렴 | cos 0.5-0.55, 후반 acc 90% |
| **Delta projector** | **축 직접 정렬** | **cos 0.27 → 0.70** |

**이것이 보여주는 것:**

1. **Baseline과 Delta의 alignment 방식이 다름**
   - Baseline: LLM attention이 layer를 거치며 **feature space에서 decision boundary를 align** (축 자체는 여전히 일부 다름)
   - Delta: Projector가 **축 자체를 직접 align** (cos 0.67) → LLM은 보존/미세조정

2. **Vanilla의 근본 문제**: 축 정렬 실패. Feature space에 direction 정보는 있지만 **task마다 다른 축**에 encoding되어있고, 이를 unify하는 메커니즘 없음.

3. **Delta projector의 인과적 역할**: Auxiliary direction supervision이 **task-agnostic direction axis**를 projector에 직접 새김. Cross-combination 결과("Delta proj + Baseline LLM = 73%")와 충돌하지 않음:
   - Delta projector의 axis = universal direction
   - Baseline LLM은 Baseline projector의 **local (task-entangled)** axis를 읽도록 학습
   - Mismatch 발생

시각화: `analysis/direction_axis_alignment.png`

### G. Position / Direction / Object Dim 분석 — SigLIP에서의 encoding 구조

Mean-pooled vision feature에 어떤 정보들이 어느 dimension에 encoding되는지 직접 측정.

#### Setup
- Feature: 1500 feature (`linear_probing_1500/`, pool_spatial=True)
- Target:
  - **Position**: `start_pos = [x, y]` 좌표 (메타데이터) — 1st frame feature로 regression
  - **Direction**: Up/Down/Left/Right — (last_frame - first_frame) delta feature로 classification
  - **Object**: shape / obj_class — mean-frame feature로 classification
- Top-50 dim: regression coefficient magnitude (position) / Fisher ratio (direction, object)

#### Within-task 결과 (Vision Encoder, D=1152)

| Task | Pos R² | Dir acc | Obj acc | P∩D | P∩O | D∩O |
|------|-------|---------|---------|-----|-----|-----|
| shape_color | 0.99 | 100% | 96% | **6** | 8 | 0 |
| obj_color | 0.85 | 99% | 89% | **7** | 8 | 3 |
| shape_place | 0.92 | 97% | 88% | **6** | 2 | 4 |
| obj_place | 0.81 | 95% | 81% | **8** | 4 | 3 |

(random overlap expectation = 2.17)

**관찰:**
- **Position info mean-pool 후에도 보존** (R² 0.81-0.99) — SigLIP의 absolute positional embedding이 mean 뒤에도 "signature" 형태로 남음
- P∩D = 6-8/50 (random 2.17의 3-4배, weak). "Direction = position delta"는 수학적/setup상 거의 tautological — 확정적 증거는 아님.
- Direction/Object overlap은 random 수준 → identity는 별도 dim

#### Cross-task Position Dim Overlap (핵심 발견)

Vision Encoder (random=2.17):

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 4 | 2 | 2 |
| obj_color | 4 | 50 | 6 | **10** |
| shape_place | 2 | 6 | 50 | 8 |
| obj_place | 2 | **10** | 8 | 50 |

- Task A의 "position 담당 dim" ≠ Task B의 "position 담당 dim"
- 대부분 pair에서 **random 수준 overlap** (2-10 vs expected 2.17)
- **Position encoding 자체가 task-specific**

#### Direction Dim Cross-task Overlap

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 16 | 7 | 6 |
| obj_color | 16 | 50 | 5 | 10 |
| shape_place | 7 | 5 | 50 | **23** |
| obj_place | 6 | 10 | 23 | 50 |

- shape_place ↔ obj_place: **23/50** (두 task 모두 real place background 사용 → 공통 encoding dim 존재)
- 나머지 pair는 낮은 overlap

#### Object Dim Overlap

| | shape_color | obj_color | shape_place | obj_place |
|---|---|---|---|---|
| shape_color | 50 | 2 | **17** | 3 |
| obj_color | 2 | 50 | 1 | **22** |
| shape_place | 17 | 1 | 50 | 3 |
| obj_place | 3 | 22 | 3 | 50 |

- shape_color ↔ shape_place: **17/50** (same shape categories)
- obj_color ↔ obj_place: **22/50** (same obj_class categories)
- 같은 identity label space 공유하는 pair에서만 dim 공유 — 합리적

#### 핵심 해석: **Cross-task direction transfer 실패의 근본 메커니즘**

```
SigLIP은 position을 encoding함 (R² 0.81-0.99)
  → 하지만 "어느 dim이 position 담당"인지 task마다 다름 (cross-task overlap random 수준)
  → Direction = position delta이므로 direction encoding dim도 task-specific
  → Cross-task direction probe 실패 (우리가 보던 현상)
```

**왜 position encoding이 task-specific한가:**
- SigLIP의 mean-pool "position signature"는 **어느 patch가 salient한지**의 함수
- shape_color: 합성 도형 pixel의 pattern
- obj_place: 실제 객체 pixel + place background pattern
- 시각적 content가 달라지면 "salient patch 위치"의 feature encoding도 달라짐
- 같은 dimension이 task에 따라 다른 의미로 쓰임

**"Identity-conditional direction encoding" 가설의 mechanistic evidence:**
- 단순히 "direction axis가 task마다 다르다" (현상)를 넘어
- **Position dim 자체가 task마다 다르게 encoded** (메커니즘)
- Direction이 position delta라는 수학적 관계가 이 task-specificity를 그대로 상속

시각화 및 전체 결과: `analysis/position_direction_object_dims.json`

#### Within-task Dim Sharing (position/direction/object가 같은 dim 쓰는가)

Task마다 top-50 dim overlap (Vision Encoder, random=2.17):

| Task | P∩D | P∩O | D∩O |
|------|-----|-----|-----|
| shape_color | **6** | 8 | 0 |
| obj_color | **7** | 8 | 3 |
| shape_place | **6** | 2 | 4 |
| obj_place | **8** | 4 | 3 |

**P∩D = 6-8/50** (random의 3-4배): 약한 overlap. Direction을 delta feature로 probe하니까 position encoding 공간과 자연히 일부 공유 — setup상 near-tautological.

**Task별 P∩O 패턴이 특이:**
- **Color background (shape_color, obj_color)**: P∩O = 8/50 (object dim과 position dim 약간 겹침)
- **Place background (shape_place, obj_place)**: P∩O = 2-4/50 (random 수준)
- → 단색 배경에선 "어느 patch가 salient한가"(=position)가 "어떤 object인가"(=object)와 연관됨
- → 실제 배경에선 전체 화면이 salient하니 object와 position이 독립

**D∩O = 0-4/50**: Direction과 identity는 **거의 별도 dim** (task 전반).

Post-projector에선 overlap 0-2로 급감 (top-50 분석의 resolution 한계 or 분산 encoding).

시각화: `analysis/within_task_dim_sharing.png`, `analysis/within_task_dim_composition.png`

#### After Projector에서의 변화

After projector (D=3584, random=0.70):
- Within-task P∩D overlap이 **0-2로 급감**
- Cross-task도 대부분 0-3
- Projector가 position-direction을 분산 encoding? 또는 top-50이 3584d에선 너무 tight해서 noise level

통계적 해석 신뢰도 낮음 (random expectation 0.7 대비 측정치가 0-3이라 구분 어려움).

### H. Letter vs Direction Probing — MCQ-probe gap의 실체

지금까지 "direction probe 91%인데 MCQ 79%, gap 12%p는 readout alignment 문제"로 해석했으나, **letter 자체를 직접 probing하면 그림이 달라짐**.

#### Setup
- Feature: 4-way extraction (`linear_probing_4way_1500/`) answer token per layer
- Target 1: direction (Up/Down/Left/Right, 4-class)
- Target 2: **letter** (sample별 정답 A/B/C/D, 4-class)
- MCQ 특성: 같은 direction이라도 sample별 candidate shuffle에 따라 letter 달라짐

#### Baseline shape_color (in-domain)

| Layer | Direction | Letter |
|-------|-----------|--------|
| L7 | 96.5% | 26.3% (chance) |
| L14 | 99.9% | 32.7% (chance) |
| **L21** | 99.7% | **97.2%** ← 급등 |
| L27 | 99.7% | 99.2% |

#### Baseline obj_place (어려운 OOD)

| Layer | Direction | Letter |
|-------|-----------|--------|
| L7 | 74.1% | 22.7% |
| L14 | 89.3% | 24.3% |
| **L21** | 92.1% | **71.9%** ← 급등 |
| L27 | 91.6% | 78.8% |

#### Delta — 전체 4 task (1500 샘플, 전 layer)

**Delta shape_color (in-domain):**

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

**Delta obj_color:**

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L14 | 96.7% | 28.5% | +68.2 |
| L15 | 97.6% | 27.9% | +69.7 |
| **L16** | 97.9% | **49.4%** | +48.5 ← binding 시작 |
| **L17** | 98.3% | **73.2%** | +25.1 |
| L18 | 98.2% | 79.8% | +18.3 |
| L20 | 97.9% | 85.8% | +12.1 |
| L27 | 97.5% | **93.7%** | +3.8 |

**Delta shape_place:**

| Layer | Direction | Letter | Gap |
|-------|-----------|--------|-----|
| L14 | 95.4% | 27.8% | +67.6 |
| L15 | 96.1% | 29.2% | +66.9 |
| **L16** | 96.6% | **42.8%** | +53.7 |
| **L17** | 97.3% | **69.1%** | +28.2 |
| L18 | 97.1% | 73.3% | +23.8 |
| L20 | 97.1% | 82.1% | +14.9 |
| L27 | 97.0% | **89.7%** | +7.3 |

**Delta obj_place (가장 어려운 OOD):**

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

**Delta vs Baseline 비교 (L27 letter probe):**

| Task | Baseline letter | Delta letter | Δ |
|------|-----------------|--------------|---|
| shape_color | 99.2% | **99.0%** | -0.2 |
| obj_color | — | **93.7%** | — |
| shape_place | — | **89.7%** | — |
| obj_place | 78.8% | **84.0%** | **+5.2** |

**Delta binding phase transition 특징:**
- Baseline보다 **1 layer 일찍 binding 시작** (Baseline L17 → Delta L16)
- OOD (obj_place) L27 letter: Baseline 79% → **Delta 84%** (+5%p)
- 모든 task에서 direction probe가 L3부터 이미 90%+ (projector ΔW가 direction axis 조기 정렬)
- Binding gap이 OOD에서도 Baseline보다 작음 (obj_place Delta 10.2 vs Baseline ~13)

#### Vanilla (모든 layer에서 letter ≈ chance)

- shape_color L27: dir 88.7%, letter 24.7% (chance)
- obj_place L27: dir 56.8%, letter 29.7% (chance)
- → Vanilla는 representation에 **letter를 전혀 encoding하지 않음**

#### 핵심 발견

**1. "Phase transition Layer 15-20"의 실체 = direction-letter binding**
- 이전 분석: L14 → L20에서 direction probe가 급등
- 재해석 (letter probe 추가): L14에서 direction은 이미 encoded. **L14 → L21에서 letter가 emerge** (binding 계산 완료)
- Phase transition은 direction 추출이 아닌 **direction-candidate-letter binding 학습**

**2. Fine-tuning의 진짜 학습 대상 = binding**
- Direction 추출 능력은 Vanilla도 상당 (L14 obj_place 75%)
- 차이는 **letter 생성을 위한 binding 학습**
- Vanilla는 direction 있어도 letter 생성 못 함 (binding 부재)

**3. MCQ-probe gap의 주 원인 = binding 불완전성**
- Direction probe L27: 91.6% (풍부)
- Letter probe L27: 78.8% (거의 완성)
- MCQ acc: 79% (letter probe와 거의 같음)
- → Gap의 대부분이 "representation에서 letter까지의 binding" 부분에서 옴

**4. OOD 난이도의 재해석**
- shape_color: direction 99% = letter 99% (binding perfect)
- obj_place: direction 91% > letter 79% (binding 12%p 미완성)
- OOD에서 어려운 건 direction 추출이 아니라 **binding 학습 부족**

**5. Delta vs Baseline 비교 (auxiliary loss 기여)**
- Delta: binding이 Baseline보다 **1 layer 일찍 시작** (L16 vs L17)
- Delta: OOD binding 완성도 더 높음 (obj_place L27 letter 84% vs 79%, +5%p)
- Direction 추출 속도: Delta L3에서 이미 95%+ (auxiliary loss로 projector 조기 정렬)
- Baseline/Delta 모두 phase transition 구간 동일 (L15-L20) → mechanism 같고 **Delta는 더 강하게 수행**

#### 시사점

- Direction probe 높다 ≠ 모델이 정답 낼 수 있다
- **Letter probe = 실질적 capability 지표**
- **Binding phase transition = L16-L17** (Delta 기준 L15→L16→L17: 29% → 38% → 57% on OP). L18 이후는 refinement
- Section N.5에서 더 세밀하게 분석됨

시각화: `output_4way_1500/{model}/answer_probe_results/{task}/direction_vs_letter.png`,
      `analysis/letter_vs_direction_combined.png`

### I. Cross-Combination 재해석

이전 cross-combination 결과("Delta 효과는 LLM weight 변화가 핵심"):
- MCQ accuracy 관점에서만 측정됨 → Delta projector + Baseline LLM = 73% (하락)

**새로운 증거와 종합:**
- Delta projector는 **task-invariant direction axis** 를 실제로 만듦 (cross-task gap 15%p)
- 하지만 Baseline LLM은 이 "새로운 representation"을 활용할 훈련 없음 → co-adaptation mismatch로 성능 하락
- Baseline LLM은 **정상적으로 (entangled) projector output** 을 받아 L15~L20에서 disentangle하도록 학습됨
- Delta LLM은 **이미 disentangled projector output** 을 받아 추가 refinement만 수행

→ **Projector와 LLM이 어떻게 "역할 분담"하는지가 task-specific**. 둘 중 하나만 바꾸면 역할 분담이 무너져 성능 하락.

---

### Intervention Experiments — Archived (2026-04-16)

**상태:** Sections J/K/L/M 모두 confound 발견. 결과는 reusable 자산으로 보관, 해석은 archive. 새 framing은 Research Questions 섹션에 정리.

#### 살아있는 자산 (재사용)

| 자산 | 위치 | 용도 |
|---|---|---|
| Factorial dataset (5×5×4×20×4) | `/local_datasets/vlm_direction/factorial_dataset/` | 통제된 (obj, bg, dir, instance, mcq) |
| Per-layer last token hidden | `/local_datasets/vlm_direction/factorial_dataset/hiddens/` | 28 layer × 8000 sample/condition |
| Vision means (post-projector) | 같은 폴더 | Vision-level Δ 추출용 |
| R2R 1500 features | `/data3/local_datasets/vlm_direction/linear_probing_1500/` | Probing + 큰 pool Δ 추출용 |
| 학습된 letter/direction probes | `output_4way_1500/{model}/answer_probe_results/` | Representation level 측정용 |

#### 핵심 empirical (해석은 별도, 숫자는 신뢰)

**Per-sample variance @ L21 (Baseline):**
| Task | acc | align_mean | align_std |
|------|-----|-----------|-----------|
| shape_color | 99.6% | 0.775 | **0.124** |
| obj_place | 78.6% | 0.576 | **0.235** (2×) |

→ OOD에서 sample들이 direction axis 주변에서 흩어짐 (확정, confound 없음).

**Factorial L=20 single-layer intervention (full direction averaging):**
| Cond | obj_place | shape_color |
|------|-----------|-------------|
| no swap | 68.80% | 95.60% |
| direction-only avg (cond e) | **86.40%** (+17.6%p) | 99.80% |

→ L20이 binding hub. Direction signal cleanup만으로 OOD 회복 가능 (확정).

**Letter probe vs Direction probe (Baseline obj_place L27):** direction 92% vs letter 79% → binding gap 13%p.

#### Confounds (해석에서 빼야 할 것들)

| 실험 | 보고했던 해석 | 실제 confound |
|---|---|---|
| Section J: whole-swap +20%p 회복 | "binding state 직접 주입" | identity/direction/letter 동시 swap, isolation X |
| Section K Exp 1: flip 71% (obj_place) | "binding이 OOD에서도 작동" | canonical letter 고정 → direction+letter routing **bundled**. Pure direction swap의 증거 아님 |
| Section L: "obj category irrelevant" | within-obj averaging이 다 회복 | place + instance noise 동시 cancel → confounded (Section M에서 정정됨) |
| Vision Procrustes rotation (script 18) | axis 정렬로 회복 기대 | full D×D rotation → identity 등 다른 정보 파괴, acc 69→24% |
| Vision broadcast replace (Phase 6) | upstream 개입 | 모든 position 동일 vector → spatial 파괴, chance |
| Factorial variant-0 flip 20.5% (script 16) vs Section K 71% | "pool 작아서 약함" | 둘 다 bundled. Pool 차이 + bundling, isolation 평가 불가 |

#### 미해결 (다음 framing으로 이동)

1. **Direction이 L20 last token에서 linearly readable한가?** — letter output 말고 probe로 측정 필요
2. **OOD noise는 어느 stage에서 들어오나?** — vision/projector/early LLM cross-sample tracing 필요

→ Research Questions 섹션 Q-A / Q-B로 재정리.

#### 보존된 스크립트 위치

- Section J: `analysis/swap_intervention/{02_extract,03_swap}_*.py`
- Section K: `analysis/swap_intervention/{05_readout,06_extract_per_sample,07_counterfactual}_*.py` + figures
- Section L: `analysis/swap_intervention/{11_extract_per_identity,12_three_condition_steering}.py`
- Section M: `analysis/factorial_experiment/scripts/{01-08,10}_*.py` + `results/*.json`
- Discarded: `scripts/{11,13-18}_*.py` (variant-0 flip / vision rotation 류 — confounded)

#### Variant orderings (factorial dataset 참조용)

```
Variant 0: [Up, Right, Down, Left]   (Up=A)
Variant 1: [Right, Up, Left, Down]   (Up=B)
Variant 2: [Down, Left, Up, Right]   (Up=C)
Variant 3: [Left, Down, Right, Up]   (Up=D)
```
4 variant 평균 시 각 direction이 모든 letter에 1번씩 → letter cancel.

---

### N. Task-invariance geometry + L19 amplifier (2026-04-16)

**소스**: `analysis/task_invariance/` (observational analysis, 개입 없음). R2R 1500 cached features + 신규 attn/mlp contribution extraction.

#### N.1 Last-token direction encoding — 축 정렬 (task-invariance)

**Δ_d axis**: per (model, task, layer) 측정. Δ_d = h_avg(direction=d) - h_avg(all). Axis 단위벡터 cosine으로 task 간 정렬도.

**Cross-task Δ-axis cos (4 direction, 6 task-pair 평균)**:

| Layer | Vanilla | Baseline | Delta |
|---|---|---|---|
| L14 | 0.46 | 0.64 | 0.67 |
| L18 | 0.48 | 0.85 | 0.80 |
| **L21** | **0.54** | **0.94** | **0.94** ← task-invariance peak |
| L27 | 0.50 | 0.92 | 0.94 |

- Fine-tuning만 task-invariant axis 생성 (Vanilla 0.5 → Baseline 0.94)
- Baseline ≈ Delta (axis 자체는 동등)
- L21이 peak, 이후 decay (letter 축으로 전환)

**Task별 per-sample alignment** (L21, Baseline):

| Task | acc | align mean | align std |
|---|---|---|---|
| SC | 99.6% | 0.74 | **0.12** |
| OP | 78.6% | **0.57** | **0.24** (2×) |

→ OOD에서 axis는 같음. 하지만 개별 sample이 axis 주변으로 **2배 흩어짐**, projection magnitude 23% 약함.

**Subspace principal angles (4D → 3D span)**: L18까지 min cos 0.7-0.8, L20+에서 한 축이 수렴 (numerical rank 이슈 있음). Subspace는 거의 정렬됨.

#### N.2 On-axis / total energy ratio — stage trajectory

각 sample (h-g)의 총 에너지 중 Δ̂_d 방향 성분 비율.

**Answer token, IN-OOD gap (SC_ratio - OP_ratio)**:

| Layer | Vanilla | Baseline | Delta |
|---|---|---|---|
| L10-L14 | ≈0 | ≈0.03 | ≈0.05 |
| L18 | 0.01 | **+0.15** | **+0.14** |
| **L21** | +0.04 | **+0.18** | **+0.16** ← gap peak |
| L27 | +0.03 | +0.12 | +0.14 |

**Vision encoder / projector / LLM L0-L10**: IN-OOD 비슷 (gap inversely/flat).

→ **OOD gap은 L14-L21 구간에서 생성**. Upstream (vision/projector)은 원인 아님.

#### N.3 L19 direction amplifier (attn + mlp per-layer contribution)

Hook: 각 decoder layer의 self_attn.output, mlp.output의 last-token slice 캡처 (R2R 1500, 2000 sample/(model, task)).

**Metric**: `proj_L = ⟨module_out_L[-1], Δ̂_{d,L21}⟩` — 각 layer module이 L21 readout axis 방향으로 얼마나 push.

**L19 total direction push (attn + mlp)**:

| Model | SC (IN) | OP (OOD) | IN-OOD gap |
|---|---|---|---|
| Vanilla | **-0.3** | **+0.2** | ≈0 (amplifier 회로 부재) |
| Baseline | **+37.8** | **+17.6** | +20.2 |
| Delta | **+48.8** | **+26.9** | +21.9 |

**Module breakdown (Baseline SC L19)**: attn 15.0 / mlp 22.8. MLP 비중 60%.
**Delta SC L19**: attn 16.6 / mlp 32.3 — Delta는 MLP 더 강화.

**다른 layer**: L10-L18 모두 0 근처. L20이 2차 amplifier (L19의 1/3). L21 이후 negative projection (letter 축 전환).

→ **L19 = amplifier hub** (Vanilla에는 없음, fine-tuning이 생성).
→ **OOD acc gap = L19 amplifier under-fire** 만이 부분 설명. 완전 설명 아님 (N.5 참조).

#### N.4 L14 ↔ L21 axis rotation — 축 회전은 L19→L20 사이

같은 task 내 Δ_d axis를 layer별로 비교 (Baseline obj_place, reference = L21):

| Layer | cos with L21 axis |
|---|---|
| L14 | **0.04** (거의 직교) |
| L17 | 0.18 |
| L18 | 0.19 |
| L19 | 0.17 |
| **L20** | **0.78** ← 급등 |
| L21 | 1.00 (ref) |
| L27 | 0.53 |

→ **L14-L18의 direction 정보는 L21 canonical axis와 거의 직교**. Probe가 읽는 건 서로 다른 축이지만 정보는 linearly separable하므로 accuracy 높게 나옴.
→ **축 회전은 L19→L20 사이에서 일어남**. L19 push는 "이 회전을 수행하는 에너지".

#### N.5 Binding gap 위치 — L16-L17 letter phase transition

**측정 (letter probe, Delta per-layer)**:

| Layer | OP letter | SC letter | gap |
|---|---|---|---|
| L14 | 26% | 31% | 5pp |
| L15 | 27% | 36% | 9pp |
| **L16** | **38%** | **70%** | **32pp** ← gap 폭발 |
| L17 | 57% | 87% | 30pp |
| L18 | 64% | 93% | 29pp |
| L21 | 77% | 99% | 22pp |
| L27 | 84% | 99% | 15pp |

**Direction probe L14 OP=92% vs L21 OP=95%** → direction은 OOD에서도 거의 intact (최대 7pp gap).

**확정된 사실**:
- Letter gap 32pp는 **L15→L16** 한 layer에서 발생 (급등 지점)
- L19 amplifier push는 letter binding **이후** 작동 (temporal order)
- L17→L27에서 letter gap 30→15pp로 감소 (partial refinement)

**가설 (미검증, 추가 실험 필요)**:
- Root cause = L16-L17 attention heads가 direction→letter routing할 때 OOD에서 under-fire
- L14 direction axis (task-specific, L21 cos 0.04)에서 binding circuit이 동작하므로 OOD 분포 mismatch가 치명적
- L19-L20 amplifier는 secondary consolidation (canonical axis 정렬)로 partial rescue 제공

→ Head-level 측정 전까진 root cause claim은 가설 수준.

#### N.6 L20 intervention (+17pp)의 mechanism — entangled

Factorial cond-e "full direction avg" L20 injection = OOD 68.8→86.4%.

**주입되는 것** (8000 sample × 4 variant 평균):
- Direction d (유일한 fix 변수) — 남음
- Letter mapping — cancel (4 variant에서 d가 A/B/C/D 각각 1번)
- Obj/bg identity — smeared (25 cell 평균)
- Candidate 집합 info — 보존
- Attention pattern residue — 4 candidate uniform attend 상태로 평균

**+17pp의 효과 분해** (미해결):
- (i) Direction signal clarification (target의 noisy direction 대체)
- (ii) Letter bias neutralization (uniform across 4 letters)
- (iii) Identity smearing (factorial cond-d 기반 marginal)

Cond-b (id+bg 유지, instance만 cancel) = +17.2% → identity 기여 거의 0. **(i)과 (ii) 분리 못 함** — 두 효과 entangled.

→ "Clean direction만으로 +17pp" over-claim. 정확한 표현: **"Direction-enriched + letter-neutralized prototype 주입 효과. Direction 단독 기여 미분해"**.

#### N.7 종합 — 현재 이해

**Mechanism (tentative)**:
1. L3-L14: direction 정보 task-specific axis에 encoded. OOD probe 87% (정보 있음).
2. **L16-L17**: Letter binding attention — direction을 letter로 매핑. **OOD에서 under-fire → 32pp letter gap 생성** (root cause, 미측정).
3. L18: letter 부분 결정 (OP 64%).
4. **L19**: direction → L21 canonical axis 방향 대량 push (Baseline SC +37.8, OP +17.6).
5. **L19→L20 사이**: direction axis cos 0.17 → 0.78로 회전.
6. L21: direction signal peak, on-axis ratio max.
7. L21-L27: letter refinement on direction basis. OOD letter gap 32→20pp 감소 (불완전).
8. L27 → lm_head: 최종 output.

**Fine-tuning 효과**:
- Vanilla: L19 amplifier 회로 없음 (push 0)
- Baseline: L19 amplifier 생성 + axis 정렬 회로
- Delta: Baseline과 같은 mechanism, OOD에서 L19 amplifier 더 강하게 fire (push +52% vs Baseline)

**미해결**:
- L16-L17 letter binding attention의 head-level 구조 (누가 binding?)
- OOD에서 어떤 attention pattern이 어떻게 다른가 (확인 필요)
- Direction cleanup vs letter reset의 분리 기여도

#### N.8 L19 vs L20/L21 single-layer intervention — causality 검증

Factorial dataset 500 target (variant_id=0), direction prototype을 각 layer에 single-layer replace.

**obj_place**:

| Cond | acc | Δ vs no_swap |
|---|---|---|
| no_swap | 68.8% | baseline |
| L19 단독 | **76.6%** | **+7.8pp** |
| L20 단독 | 86.0% | **+17.2pp** |
| L21 단독 | **86.4%** | **+17.6pp** ← peak |
| L19-20 | 86.0% | +17.2pp |
| L18-19-20 | 86.0% | +17.2pp (L18 추가 기여 없음) |
| L19-20-21 | 86.4% | +17.6pp |

**shape_color**: 전부 99.0-99.8% (ceiling, 분해 의미 없음)

**해석**:
- **L19 단독 intervention은 L20/L21의 절반 효과** (+7.8 vs +17.2-17.6)
- L19 혼자는 binding hub 아님 확증 (N.5 해석 지지)
- L20/L21이 실질적 intervention sweet spot — direction signal이 canonical axis에 도착한 시점
- L18 추가 기여 0 → binding 단계(L16-L17)보다 뒤의 consolidation 단계에서 작동
- 이 실험은 **causal 증명 (no_swap → +17pp 회복)**. N.3의 관측적 L19 push 측정과 **complementary** (push는 attn+mlp 기여, intervention은 hidden state 전체 교체)

**중요 caveat**: 이 intervention도 Section N.6의 entanglement (direction cleanup + letter neutralization) 영향 있음. "L20/L21이 중요"는 확실하지만, "direction만의 효과"인지는 별도 분해 필요.

#### N.9 관련 파일

- `analysis/task_invariance/measure_invariance.py` — task-invariance metrics
- `measure_subspace_offaxis.py` — subspace + on-axis energy decomposition
- `stage_trajectory.py` — stage-wise on_off_ratio
- `extract_attn_mlp_contrib.py` — attn/mlp output extraction (hook)
- `analyze_attn_mlp_contrib.py`, `analyze_3models.py` — per-layer direction push analysis
- `axis_layer_cos.py` — within-task cross-layer axis rotation
- `l19_intervention.py` — L19 single-layer replacement (완료)

---

### O. Mechanism diagnosis — binding gap = late-layer magnitude deficit (2026-04-16)

**실험**: Last token 단일-layer hook + clean intervention. Factorial OP 500 target (variant_id=0), Baseline. 10 conditions × 500 sample = 5000 forwards.

#### O.1 Direction magnitude per layer (Baseline factorial)

| Layer | ‖Δ‖ OP (mean) | ‖Δ‖ SC | ratio SC/OP |
|---|---|---|---|
| L14 | 0.53 | 1.97 | 3.71× |
| L16 | 1.37 | 3.02 | 2.20× |
| L18 | 3.59 | 6.74 | 1.88× |
| L21 | 28.3 | 48.5 | 1.71× |

→ Direction signal magnitude는 전 layer에서 OP<SC. L21에서도 **OP는 SC의 60%**.

#### O.2 Intervention results (obj_place, baseline 68.80%)

| Condition | acc | Δ | 주입 방식 |
|---|---|---|---|
| no_swap | 68.80% | — | baseline |
| L21 amp_2x | 73.80% | **+5.0pp** | on-axis ×2 |
| L21 clean_sc | 78.80% | **+10.0pp** | remove own + set to SC magnitude |
| L21 add_canon | 80.40% | **+11.6pp** | add only, no removal |
| L21 on_axis | 78.60% | **+9.8pp** | keep only on-axis (off-axis nuke) |
| L21 remove_own | 52.40% | **-16.4pp** | control: remove direction |
| L21 full_rep | 86.40% | **+17.6pp** | Section M cond-e reference ✓ |
| L18 clean_sc | 68.80% | 0.0pp | |
| L16 clean_sc | 68.60% | -0.2pp | |
| L14 clean_sc | 68.80% | 0.0pp | |

#### O.3 3-가설 verdict

**H-mag (magnitude insufficiency)**: **확증 (dominant)**
- L21 amp_2x +5, clean_sc +10, add_canon +11.6pp
- Magnitude 증폭만으로 +5-12pp 회복
- Full replace (+17.6pp)와 add_canon (+11.6pp) 사이 6pp gap → magnitude가 **17.6pp 중 11.6pp (66%) 설명**

**H-noise (off-axis noise)**: **확증 (weaker)**
- L21 on_axis (off-axis 전부 제거) +9.8pp
- Off-axis 성분 noise가 있음 (제거 시 회복)
- 단 on_axis만 < add_canon (9.8 < 11.6) → off-axis 정보 중 일부는 유용 (letter routing 등)

**H-layer (early layer intervention)**: **반박**
- L14/L16/L18 clean_sc: **전부 0pp 효과**
- 이유: L14-L18 Δ axis는 L21 canonical과 cos 0.04-0.19 (N.4). Canonical 방향으로 주입해도 **L19-L20 rotation dynamics가 이를 덮어씀**
- Intervention sweet spot은 **L20-L21** (axis rotation 완료 후)

**Control (remove_own -16.4pp)**: 
- 52.4%로 추락 but chance (25%) 위
- → Baseline LLM이 OP's direction signal을 **실제로 읽고 있음** (68.8%가 direction 의존)
- Magnitude만 SC 수준으로 올려도 9-12pp 회복한다는 것은 **"정보가 약하게 있지만 충분한 magnitude가 아니다"**를 의미

#### O.4 Magnitude sweep — binding gap의 **전부가 magnitude** (정정)

500 samples, L21 intervention, on-axis magnitude를 다양한 값으로 set:

| Condition | Target magnitude | acc | Δ |
|---|---|---|---|
| clean_op_half | 14 (OP_mean × 0.5) | 63.60% | **-5.2pp** |
| no_swap | sample별 (mean 28) | 68.80% | baseline |
| clean_op_mean | 28 (OP_mean) | 71.20% | +2.4pp |
| clean_sc_mean | 48 (SC_mean) | 78.80% | +10.0pp |
| **clean_2x_sc** | **96 (2×SC_mean)** | **86.40%** | **+17.6pp** |
| full_rep (ref) | h_avg_OP[d] | 86.40% | +17.6pp |

**결정적 발견**:
1. **clean_2x_sc ≡ full_rep**: magnitude를 96으로 set하는 것만으로 **full_rep 재현 (+17.6pp 동일)**.
2. → **full_rep의 +17.6pp는 전부 magnitude 덕분**. 이전 "6pp residual = identity/letter content" 해석은 **틀렸음**.
3. **Monotonic no saturation**: 14→28→48→96 각각 ~8pp씩 회복. Magnitude-acc 선형에 가까움.
4. **Per-sample noise 기여 미미**: clean_op_mean (noise 제거만, magnitude 유지) = +2.4pp만.

**Paper claim (정정)**:
> "The OOD binding gap is **fully explained by a direction-signal magnitude deficit on the canonical direction axis at L20-L21**. Setting the on-axis projection to 2× the in-domain mean recovers OOD accuracy to the level of full hidden-state replacement (+17.6pp). Per-sample variance contributes negligibly (+2pp). The gap is not identity-specific, not letter-routing-specific, and not axis-alignment-specific — it is magnitude-only."

- **Binding gap mechanism**: OOD에서 direction signal이 canonical axis 방향으로 **충분히 amplified되지 않음** (SC는 평균 48, OP는 평균 28 — SC의 58%)
- **Fix**: magnitude를 최소 SC 수준(48)으로 올리면 +10pp, 2× (96)이면 +17.6pp (full recovery)

#### O.5 Design implication for delta_direct v2

**Training-time fix 방향**:
1. **L20-L21 direction magnitude supervision**: OP sample의 L20-L21 last token direction on canonical axis가 SC 분포에 가깝도록 loss
2. **Projector auxiliary**: projector output이 LLM late-layer canonical axis와 정렬된 direction signal을 강하게 내도록 (현 delta_direct + magnitude component)
3. **Early layer (L14-L18) intervention은 효과 없음** — 여기 supervision 걸어도 downstream rotation이 washing out. L19-L21 direct supervision 필요

**Inference-time fix (지식증명용)**:
- L21 add_canon (+11.6pp) = 매우 simple 증강. 배포 시 OOD detection + L21 magnitude boost 가능
- But identity-agnostic (어차피 direction label 필요)

#### O.6 Vision-level 대응 실험 (per-position amplification)

L21 intervention의 vision 버전 테스트 (projector output 각 position에서 on-axis scaling):

| Condition | OP acc | Δ |
|---|---|---|
| no_swap | 67.67% | — |
| amp_own_2x (vision) | 65.00% | **-2.7pp** |
| amp_own_5x | 38.00% | **-29.7pp** |
| amp_own_10x | 20.00% | **-47.7pp** (chance) |
| amp_in_2x (SC axis) | 60.67% | -7.0pp |
| push_in_mag (SC magnitude) | 64.00% | -3.7pp |
| clean_sc (remove own + SC add) | 45.67% | -22.0pp |

→ **전부 효과 없거나 파괴적**. L21과 정반대 패턴 (L21 amp_2x +5pp, vision amp_2x -2.7pp).

**해석**: 
- Vision-level "direction axis" (Δ̂_d_OP_vision)는 statistical axis — direction labels를 linear하게 분리하는 방향
- 하지만 **LLM attention은 이 축을 직접 읽지 않음**. LLM은 spatial/temporal pattern에서 direction을 재구성
- Projector output의 "direction axis 증폭"은 LLM이 기대하는 input distribution과 불일치 → 도움 안 됨
- **Delta projector는 단순 magnitude scaling이 아니라 LLM이 읽을 수 있는 form으로 reshape** — training 중 joint adaptation 필요

#### O.7 핵심 Paper Claim 정리

**Binding gap in OOD = pure magnitude deficit on canonical axis at L20-L21**:
1. **Location**: L20-L21 last token, canonical direction axis 방향 magnitude
2. **Not**: axis misalignment (L21 cos 0.94), direction absence (probe 92%), identity content, letter routing
3. **Magnitude-acc relationship**: monotonic, no saturation up to 2×SC (96)
4. **Fix sweet spot**: L20-L21 direction on-axis magnitude를 2×SC mean (96)으로 set → **+17.6pp (full recovery)**
5. **Vision-level intervention 불가능**: vision axis와 L21 canonical axis 직교 (cos ≈ 0). Vision magnitude 증폭은 -47pp (catastrophic).
6. **Delta의 real mechanism**: projector + LLM joint coadaptation으로 L21 magnitude 증폭 (Delta 자체의 L21 magnitude 더 큰지 측정 필요)

**Implication for delta_direct v2 training loss**:
- **Direct**: L20-L21 hidden direction on-axis magnitude supervision (OP samples가 SC level에 도달하도록)
- Projector aux loss는 LLM coadaptation 전제로만 작동
- Vision token magnitude 자체 supervision은 효과 없음 확증

#### O.8 Magnitude cascade — 왜 OOD에서 작아지나

Per-stage ‖Δ_d‖ mean (R2R 1500, 3 models × 4 tasks):

| Stage | Vanilla SC/OP | Baseline SC/OP | Delta SC/OP |
|---|---|---|---|
| Vision encoder (SigLIP, frozen) | 0.59× | 0.59× | 0.59× |
| After projector | 0.56× | 0.59× | **1.53×** |
| Vision token L7 | 0.70× | 0.81× | 1.55× |
| **Vision token L14** | 1.05× | **2.96×** | **3.85×** |
| Vision token L21 | 1.38× | 4.60× | 4.77× |
| Answer token L18 | 1.55× | 1.80× | 2.09× |
| **Answer token L21 (readout)** | 1.96× | **1.51×** | 1.46× |

**핵심 발견**:
1. **SigLIP/Projector에서는 OOD 문제 없음** (OP 오히려 큼)
2. **Delta projector만 SC-biased amplify** (1.53× 이미 projector에서)
3. **Baseline LLM L7→L14에서 SC-bias 생성** (0.81× → 2.96×)
4. **모든 모델 L14-L21 vision token에서 SC-biased amplification 누적** (ratio 4.6-4.8×)
5. L21 last token에서 일부 compensate (1.5× 정도), but OP는 SC의 66%만

**왜 LLM이 SC-biased amplify하는가 (mechanism 가설)**:
- LLM의 L14+ direction amplifier는 **"SC-like visual pattern"을 trigger로 작동**
- OP vision content는 SC training distribution과 다른 pattern → amplifier **under-fire**
- Result: vision token에 SC-biased magnitude 누적 → last token도 비례
- Delta's projector는 OP에 SC-like form 주입 → LLM이 약간 더 amplify (L14 Baseline 2.96× → Delta 3.85×), 하지만 여전히 gap 존재

**Implication**:
- Binding gap 근본 원인 = **LLM L7-L14의 SC-biased amplification**
- Projector-only intervention (Delta 같은) = partial fix (vision-level 1.53× amplify 기여)
- Full fix 위해선 **LLM's amplification circuit도 OP에 equally fire**해야 함 → training-time supervision 필수

#### O.9 미해결

- Cross-task (obj_color, shape_place)에서 magnitude sweep 동일한 pattern? (실행 중)
- Vanilla L21 intervention (Vanilla는 magnitude 작음, 회복 가능성)
- L7-L14 구간 직접 측정: 어느 head/MLP가 SC-bias amplify 담당?
- Training-time loss design: L20-L21 magnitude loss vs L14 amplifier re-train

---

## Research Questions (재정리, 2026-04-16)

Central framing: **Fine-tuning이 VLM direction reasoning의 두 현상 (task-invariant direction encoding + letter binding)을 어떻게/어디서 만들어내며, 이것이 OOD 성능에 어떻게 귀결되는가.**

### 확정된 사실 (Section A-N, 측정 기반)

**Representation geometry:**
1. Vision encoder/projector는 task-specific direction axis (cross-task cos 0.1), LLM이 canonical axis (L21 cos 0.94) 생성 — fine-tuning만의 효과 (Vanilla 0.54)
2. Direction은 OOD에서도 probe 92%로 readable — axis 자체는 intact
3. L14 direction axis는 L21 canonical axis와 cos 0.04로 거의 직교. 같은 task 내에서도 **axis 회전은 L19→L20 사이 (cos 0.17 → 0.78)**
4. OOD per-sample alignment: std 2배 (0.12 → 0.24), mean 23%↓

**Letter binding timing:**
5. Letter binding phase transition = **L15→L16 한 layer에서 32pp gap 폭발** (Baseline OP: L15 27%, L16 38%, L17 57%)
6. Direction gap은 동일 구간에서 최대 7pp — binding gap ≠ direction gap

**L19 direction amplifier (신규 측정):**
7. L19 attn+mlp가 last token에 L21 canonical axis 방향으로 대량 push: Vanilla 0, Baseline +37.8 SC / +17.6 OP, Delta +48.8/+26.9
8. 다른 layer 모두 0 근처 (L20 2차, L19의 1/3). **Vanilla에는 amplifier 회로 부재**
9. L19 attn:mlp ≈ 40:60 (Baseline). Delta는 MLP 강화 (32:68)

**Intervention causality:**
10. Single-layer prototype injection: L19 단독 +7.8pp, L20/L21 +17.2-17.6pp (factorial OP n=500)
11. L18 추가 기여 없음 → L20/L21이 intervention sweet spot

### 가설 (미검증)

- Binding gap의 root cause = L16-L17 attention heads의 OOD under-fire (head-level 미측정)
- L20/L21 intervention의 +17pp는 direction cleanup + letter bias neutralization entangled (분해 미측정)
- L19 amplifier는 letter binding 후 consolidation 단계 (temporal 순서 근거)

### 다음 실험 (우선순위 순)

**Q-N1. L16-L17 letter binding attention의 head-level 구조** — 최우선 (가설 검증)
- 각 head의 attention pattern: vision token / candidate token / last token에 얼마나 attend?
- Head output의 letter axis projection (lm_head letter vector 기준)
- IN vs OOD에서 head activation 차이 측정
- 결과로 "L16-L17 attention under-fire가 root cause"라는 가설 검증

**Q-N2. L20 intervention +17pp 분해** — direction vs letter reset 엉킨 것 풀기
- Construction A: Kang-style h - Δ(curr) + Δ(flip) (direction 조작, letter mapping 유지)
- Construction B: letter bias만 neutralize (direction 유지)
- 각각의 독립 기여도 측정

**Q-N3. 개선 방향 insight (Q-N1 결과 후 결정)**
- L16-L17 binding circuit을 OOD-robust하게 LoRA fine-tune
- L19 amplifier를 OOD에서도 IN 수준으로 fire하도록 supervision

### 보류

- Vision-side Kang-style direction prototype injection (Section M의 vision-side 연장)
- Cross-combination 재실험 (Section I)
- Delta projector ΔW SVD 분해
