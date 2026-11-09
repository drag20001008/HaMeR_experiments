# HaMeR / WiLoR 손 복원 실험 정리

두 갈래로 실험했다.
- **손 모양**: HaMeR backbone 학습 방식을 바꾼 실험(M1~M4)을 M0(pretrained HaMeR)와 10 epoch 기준으로 비교
- **손 위치(camera-space 배치)**: 예측한 손을 실제 카메라 좌표계에 놓는 방법(P0~P6)을 HaMeR와 WiLoR로 비교

아래 1장에서 데이터, 실험 이름, 방법 이름을 먼저 정리하고, 2장부터 결과를 적는다.

---

## 1. 실험과 방법 정리

### 1-1. 평가 데이터

| 데이터 | 샘플 수 | depth | 쓰인 곳 |
|---|---|---|---|
| FreiHAND eval | 3,960 | 없음 (RGB만) | 손 모양, HaMeR 배치 (P0, P2만) |
| HO3D_v3 official eval | 19,062 | 있음 | 손 모양, HaMeR 배치 |
| DexYCB official test | 78,768 | 있음 | 손 모양, HaMeR 배치 |
| DexYCB test onscreen | 70,595 | 있음 | WiLoR 배치 (21개 관절이 모두 화면 안에 있는 샘플) |

FreiHAND는 depth가 없어서 depth를 쓰는 방법(P3, P4)은 돌리지 않았다.

### 1-2. 손 모양 모델: HaMeR backbone 실험

공통 학습 설정
- 학습 데이터: DexYCB official train (406,888) + HO3D_v3 전체 (77,743) + FreiHAND 전체 (130,240)
- AdamW, lr 1e-4, weight decay 1e-4, grad clip 1.0, batch 32/GPU, 2 GPU, 10 epoch
- ViT backbone은 항상 pretrained HaMeR에서 시작

| 이름 | 한 줄 설명 | backbone (ViT) | decoder (MANO head) | 추가 모듈 |
|---|---|---|---|---|
| **M0 Pretrained** | pretrained HaMeR 그대로 | – | – | – |
| **M1 LoRA** | ViT를 LoRA로만 조정, decoder는 처음부터 학습 | 고정 + **LoRA** (r=8, α=16, qkv/proj/fc1/fc2) | **랜덤 초기화** | – |
| **M2 Full-FT** | ViT 전체를 학습, decoder는 처음부터 학습 | **전체 학습** | **랜덤 초기화** | – |
| **M3 β-adapter** | ViT 고정, decoder는 처음부터 학습 + 손 모양(β) 조건 | **고정** | **랜덤 초기화** | **β-adapter** |
| **M4 β-adapter + 2D head** | M3 + 2D keypoint head | **고정** | **랜덤 초기화** | **β-adapter + 2D head** |

- **β-adapter**: GT 손 모양 파라미터 β를 MLP로 임베딩해 ViT context token에 더하는 residual 모듈 (rank 64, 임베딩 128차원, 193,920 params, 0으로 초기화되어 학습 초기엔 영향 없음). 학습과 평가 때 GT β를 입력으로 받음.
- **2D head**: decoder의 global token에서 21개 관절의 2D 위치(crop 좌표)를 직접 예측하는 작은 MLP. 보조 loss(가중치 0.01)로 학습되며, P2(PnP)의 2D 입력으로 쓰임.
- 학습 파라미터 수: M1 44.7M, M2 670.4M.

### 1-3. 손 위치 방법: camera-space 배치 (P0~P6)

모든 방법에서 **손 모양(손목 기준 3D 관절)과 회전은 모델 출력 그대로** 두고, 카메라 좌표계의 **위치(translation)만** 구한다. 번호는 쓰는 정보가 늘어나는 순서다.

| 이름 | 방법 | 쓰는 정보 | 적용 |
|---|---|---|---|
| **P0** | 모델 출력 | 네트워크 translation (가상 focal 기준)을 원본 이미지 좌표로 변환 | HaMeR, WiLoR |
| **P1** | K 변환 | 같은 네트워크 출력을 실제 K로 변환 (손을 평면으로 가정) | WiLoR |
| **P2** | PnP | 3D 관절 + 2D 관절 + 실제 K, translation 3개만 선형 최소제곱 | HaMeR, WiLoR |
| **P3** | ICP | 손 마스크 안 depth point cloud에 ICP로 translation 맞춤 | HaMeR |
| **P4** | ICP → PnP | P3 위치에서 시작해 2D reprojection 오차 최소화 | HaMeR |
| **P5** | PnP + depth 보정 | P2 결과를 depth에 맞춰 시선 방향으로 평행이동 | WiLoR |
| **P6** | AGILE | ICP로 손 scale을 구한 뒤, scale 곱한 3D 관절로 PnP | WiLoR |

P2 입력: WiLoR는 GT β + WiLoR 손 자세로 만든 3D 관절을 사용. 2D는 M0는 HaMeR의 2D keypoint 출력, M4는 2D head 출력(비교용으로 GT 2D도 사용), WiLoR는 WiLoR 2D 출력.

**P5 depth 보정** (손 모양, 크기, 회전은 그대로)
1. P2 손 mesh를 렌더링해 픽셀별 예측 표면 depth z_pred(p)를 구함
2. 예측 손 ∩ 손 마스크 ∩ depth 유효 픽셀에서 Δ(p) = D(p) − z_pred(p)
3. m = median(Δ), 손 중심 c, 방향 r̂ = c/‖c‖
4. δ = m / r̂_z만큼 이동: P′ = P + δ·r̂

손 마스크는 두 가지로 비교: 실제 손 마스크(물체에 가려진 부분 제외), GT 손 윤곽(가려진 부분 포함).

**P6 AGILE**: WiLoR 손 mesh(예측 β)를 depth point cloud에 ICP로 맞춰 손 scale을 구하고(회전 고정), scale을 곱한 3D 관절 + WiLoR 2D + K로 PnP를 풀어 위치를 정하는 방식 (AGILE 논문 Sec 3.2). 논문 기재대로 재현, SAM3 손 마스크 사용.

### 1-4. 지표

- **PA-MPJPE**: 21개 관절을 Procrustes(회전, 스케일, 위치)로 정렬한 뒤의 평균 오차 → 손 모양 정확도
- **CS-MPJPE**: 정렬 없이 카메라 좌표계에서 잰 21개 관절 평균 오차 → 모양 + 위치 정확도
- **손목 오차**: 예측 손목 위치 − GT 손목 위치 (카메라 좌표계, z = depth). MAE, bias(부호 있는 평균), 3D 거리

---

## 2. 결과: 손 모양 (PA-MPJPE, mm, 평균 / 중앙값)

| 모델 | FreiHAND | HO3D eval | DexYCB test |
|---|---|---|---|
| **M0 Pretrained** | 6.39 / 5.07 | 6.84 / 5.48 | **4.34 / 3.45** |
| M1 LoRA | **6.02 / 4.92** | 6.80 / 5.55 | 6.03 / 5.06 |
| M2 Full-FT | 14.74 / 11.59 | 10.17 / 8.61 | 8.81 / 7.36 |
| M3 β-adapter | 6.53 / 5.25 | **6.55 / 5.38** | 6.09 / 5.08 |
| M4 β-adapter + 2D head | 6.62 / 5.28 | 6.81 / 5.55 | 6.06 / 5.09 |

### M0 대비 차이 (평균, mm, 음수 = 개선)

| 모델 | FreiHAND | HO3D eval | DexYCB test |
|---|---|---|---|
| M1 LoRA | **−0.37** | −0.04 | +1.69 |
| M2 Full-FT | +8.35 | +3.33 | +4.47 |
| M3 β-adapter | +0.14 | **−0.29** | +1.75 |
| M4 β-adapter + 2D head | +0.23 | −0.03 | +1.72 |

---

## 3. 결과: HaMeR 손 위치 (P0, P2, P3, P4)

비교 모델: M0, M4 (30 epoch).

### 3-1. 손목 위치 오차 (mm)

| 데이터 | 모델 | 방법 | MAE x / y / z | bias z | 3D 거리 평균 / 중앙값 |
|---|---|---|---|---|---|
| DexYCB test | M0 | P0 모델 출력 | 138.04 / 120.79 / 2563.87 | +2563.87 | 2574.09 / 2493.89 |
| | M0 | P2 PnP | 14.06 / 16.75 / 57.51 | +52.08 | 64.04 / 35.42 |
| | M0 | P4 ICP → PnP | 14.94 / 17.87 / 69.16 | −13.85 | 75.79 / 32.65 |
| | M4 | P0 모델 출력 | 139.36 / 120.98 / 2427.21 | +2427.21 | 2438.21 / 2441.25 |
| | M4 | **P2 PnP** | **7.13 / 7.23 / 29.00** | +15.90 | **32.13 / 22.47** |
| | M4 | P4 ICP → PnP | 10.53 / 10.58 / 54.68 | −33.80 | 58.41 / 29.17 |
| FreiHAND | M0 | P0 모델 출력 | 99.60 / 10.10 / 2534.69 | +2534.69 | – |
| | M0 | P2 PnP | 5.55 / 4.70 / 65.63 | +25.27 | – |
| | M4 | P0 모델 출력 | 98.02 / 9.43 / 2480.96 | +2480.96 | – |
| | M4 | **P2 PnP** | **4.17 / 3.89 / 46.54** | +0.60 | – |

### 3-2. CS-MPJPE (M4, mm, 평균 / 중앙값)

| 데이터 | P2 PnP (GT 2D) | P2 PnP (2D head) | P3 ICP | P4 ICP → PnP |
|---|---|---|---|---|
| FreiHAND | 50.99 / 39.40 | 47.55 / 41.87 | – | – |
| HO3D eval | 31.83 | 36.79 | 55.82 | 35.26 |
| DexYCB test | 34.01 | 32.62 | 83.30 | 58.74 |

손목 위치를 GT로 고정했을 때(손 모양 오차만 남음): FreiHAND 12.97, HO3D 15.57, DexYCB 11.40.

---

## 4. 결과: WiLoR 손 위치 (P0, P1, P2, P5, P6)

평가 데이터: DexYCB test onscreen (70,595).

| 방법 | CS-MPJPE 평균 / 중앙값 | bias x / y / z | MAE x / y / z | 손목 3D 거리 |
|---|---|---|---|---|
| P0 모델 출력 | 16,337.90 / 16,239.47 | −6.8 / +1.0 / +16,338.2 | 11.3 / 6.4 / 16,338.2 | 16,338.2 |
| P1 K 변환 | 49.39 / 40.28 | +0.1 / −1.4 / +3.1 | 5.5 / 5.1 / 46.3 | 48.0 |
| P2 PnP | 30.22 / 22.55 | +0.7 / −0.5 / +19.9 | 5.6 / 5.1 / 27.7 | 29.7 |
| P5 depth 보정 (GT depth + 실제 손 마스크) | 9.93 / 7.77 | +0.3 / −0.6 / +1.2 | 3.6 / 3.7 / 4.6 | 8.2 |
| **P5 depth 보정 (GT depth + GT 손 윤곽)** | **8.59 / 7.67** | +0.3 / −0.7 / +0.4 | 3.5 / 3.6 / 3.3 | **7.0** |
| P6 AGILE (GT depth + SAM3 손 마스크) | 310.66 / 186.68 | −1.9 / +2.1 / −300.4 | 41.2 / 34.0 / 302.2 | 309.4 |

---

## 5. 정리

**손 모양 (2장)**
- DexYCB에서는 네 실험 모두 M0보다 나쁨 (+1.7 ~ +4.5mm).
- FreiHAND에서는 M1(LoRA)만 개선 (−0.37mm), HO3D에서는 M3(β-adapter)만 뚜렷하게 개선 (−0.29mm).
- M2(Full-FT)는 모든 데이터에서 크게 나쁨.
- M3와 M4는 PA-MPJPE가 비슷함 → 2D head는 손 모양 정확도에는 거의 영향 없음.

**손 위치 (3, 4장)**
- P0(모델 출력)은 depth가 HaMeR 2.4~2.6m, WiLoR 16m씩 틀려 쓸 수 없음.
- P2(PnP)로 다시 풀면 수십 mm 수준이 되고, 남는 오차는 대부분 z.
- HaMeR에서는 M4(2D head) + P2가 M0 + P2보다 약 2배 좋음: DexYCB 손목 거리 32.13 vs 64.04mm, z MAE 29.00 vs 57.51mm.
- 2D head로 푼 P2가 GT 2D로 푼 P2와 비슷함 → 2D 정확도는 병목이 아님.
- depth를 ICP로 쓰는 P3, P4는 P2보다 오히려 나쁨.
- WiLoR: P0 16m → P1 48.0mm → P2 29.7mm (손목 거리).
- P5(depth 보정)는 z MAE를 27.7 → 3.3~4.6mm로, CS-MPJPE를 30.22 → 8.59~9.93mm로 줄임.
- P6(AGILE)는 ICP가 손 scale을 줄여(중앙값 0.75) 손이 카메라 쪽으로 당겨지고, CS-MPJPE 310.66mm로 P5보다 크게 나쁨.
