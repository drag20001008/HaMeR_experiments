# WiLoR camera-space 손 위치 복원: D_lfix → depth 보정 파이프라인

DexYCB s0 test (78,768장)에서 pretrained WiLoR로 손을 카메라 좌표계에 놓고, depth map으로 시선 방향 위치를 보정하는 파이프라인과 결과 정리.

- 좌표계: DexYCB 각 color 카메라 좌표계 (OpenCV: x 오른쪽, y 아래, z 광축 방향 depth), 단위 m (결과는 mm)
- 평가 subset: **onscreen** = GT 2D 관절 21개가 모두 이미지(640×480) 안에 있는 샘플 (70,595장, 89.6%)

---

## 1. D_lfix: depth 없이 손 놓기

1. **WiLoR 추론** (crop 입력): 손가락 관절 각도 θ, 전역 회전 R, 21개 관절 2D keypoint (crop 정규화 좌표).
   WiLoR의 예측 β와 camera head 출력(pred_cam)은 쓰지 않음.
2. **손 3D 모양**: MANO에 **GT β** (DexYCB 피험자별 손 모양) + WiLoR θ, R → 관절 21개 → 손목 기준 상대 좌표 `J_i`.
3. **왼손 처리 ("lfix")**
   - WiLoR는 왼손을 좌우 반전해 처리하므로 3D 예측, crop 2D, GT 3D의 x 부호를 되돌림 (unmirror).
   - DexYCB 왼손 GT β는 `MANO_LEFT.pkl`로 피팅됐는데 이 파일은 shapedirs x 성분이 반전되지 않은 버그가 있음
     → 왼손은 `shapedirs[:, 0, :] *= -1` 한 MANO로 생성.
4. **2D를 원본 이미지 좌표로**: `u_i = box_center + (crop 정규화 2D) × box_size`
5. **translation-only PnP**: 회전 고정, 21개 3D–2D 대응으로 `T`만 추정.
   `u_i = Π_K(J_i + T)`에 depth를 곱해 T에 대한 선형식(21×2개)으로 만들고 **선형 최소제곱 (closed-form, DLT 방식)**으로 풂. K는 DexYCB 실제 값.
6. **결과**: `P_i = J_i + T`

## 2. Depth 보정: 시선 방향 강체 평행이동

입력은 D_lfix 결과. 손 모양, 크기, 회전은 유지하고 위치만 바꿈.

1. **예측 손 렌더링**: D_lfix mesh를 실제 K로 래스터화 → 픽셀별 예측 표면 depth `z_pred(p)` (z-buffer, 보이는 앞면만).
2. **사용 픽셀**: 예측 손이 렌더링된 픽셀 ∩ 손 마스크 ∩ depth 유효.
3. **픽셀별 차이**: `Δ(p) = D(p) − z_pred(p)` (표면 대 표면 비교 → 관절–피부 offset 불필요). Δ > 0 이면 실제 손이 더 멂.
4. **depth 변화량**: `m = median_p Δ(p)` (경계에서 튀는 픽셀에 강건).
5. **방향**: 사용 픽셀들의 예측 표면 3D 점 평균 `c`, 시선 방향 `r̂ = c / ‖c‖`.
6. **거리와 적용**: `δ = m / r̂_z`, `P' = P + δ·r̂` (모든 관절·정점에 같은 벡터). c는 자기 시선 위에서만 움직이므로 손 중심의 2D 위치는 유지.
7. **(센서 depth용) 필터**: 사용 픽셀 < 1000개 또는 `q90(Δ) − q10(Δ) > 100mm`이면 보정하지 않고 D_lfix 유지.

**oracle δ\*** (상한): 같은 방식(예측 관절 21개 중심을 지나는 시선)으로, 이동량만 GT로 정함 — 예측 관절 평균 z가 GT 관절 평균 z와 같아지도록.

---

## 3. 결과

### 3-1. depth 보정 비교 (onscreen 70,595장, 손목 오차, mm)

erode 0, 전체 영역, 유효 픽셀 ≥ 1000, 집계 = median.

| 방법 | CS-MPJPE | bias x | bias y | bias z | MAE x | MAE y | MAE z | 3D 거리 \|e\| |
|---|---|---|---|---|---|---|---|---|
| D_lfix (depth 안 씀) | 30.22 | +0.7 | −0.5 | +19.9 | 5.6 | 5.1 | 27.7 | 29.7 |
| 센서 depth | 31.59 | −0.1 | +0.4 | +22.1 | 6.2 | 5.1 | 28.3 | 30.8 |
| 센서 depth + 필터 | 21.52 | +0.5 | −0.2 | +13.5 | 4.8 | 4.4 | 18.5 | 20.7 |
| GT depth, 실제 손 마스크 | 9.93 | +0.3 | −0.6 | +1.2 | 3.6 | 3.7 | 4.6 | 8.2 |
| GT depth, 실제 손 마스크 + 필터 | 10.24 | +0.3 | −0.6 | +1.4 | 3.6 | 3.7 | 5.0 | 8.5 |
| GT depth, GT 손 윤곽 | 8.59 | +0.3 | −0.7 | +0.4 | 3.5 | 3.6 | 3.3 | 7.0 |
| GT depth, GT 손 윤곽 + 필터 | 9.11 | +0.3 | −0.7 | +0.9 | 3.6 | 3.6 | 3.9 | 7.5 |
| oracle δ\* | 7.83 | +0.3 | −1.0 | +0.3 | 3.6 | 3.7 | 4.1 | 7.6 |

- 센서 depth: DexYCB RealSense depth. GT depth: GT MANO mesh를 렌더링한 depth (완벽한 depth 가정).
- 실제 손 마스크: DexYCB GT seg (`seg == 255`), 물체에 가려진 부분 제외. GT 손 윤곽: GT mesh 렌더링 영역 (가림 없음).
- 센서 depth는 GT 손 표면보다 평균 약 +9.7mm 멀게 찍히고 (카메라별 +1.7~+15.4mm), 꼬리가 김 (4.5%가 50mm 이상) → 보정량 오차 분산의 93%가 센서 depth–GT 라벨 불일치.

### 3-2. 집계 방식 비교 (onscreen, 필터 없음)

| depth 조건 | 집계 | CS-MPJPE | bias x | bias y | bias z | MAE x | MAE y | MAE z | \|e\| |
|---|---|---|---|---|---|---|---|---|---|
| D_lfix (depth 안 씀) | – | 30.22 | +0.7 | −0.5 | +19.9 | 5.6 | 5.1 | 27.7 | 29.7 |
| 센서 depth | median | 31.59 | −0.1 | +0.4 | +22.1 | 6.2 | 5.1 | 28.3 | 30.8 |
| | mean | 103.02 | −0.4 | +2.8 | +93.0 | 13.8 | 12.3 | 99.4 | 102.6 |
| GT depth, 실제 손 마스크 | median | 9.93 | +0.3 | −0.6 | +1.2 | 3.6 | 3.7 | 4.6 | 8.2 |
| | mean | 10.03 | +0.3 | −0.6 | +0.0 | 3.6 | 3.6 | 5.2 | 8.5 |
| GT depth, GT 손 윤곽 | median | 8.59 | +0.3 | −0.7 | +0.4 | 3.5 | 3.6 | 3.3 | 7.0 |
| | mean | 8.50 | +0.3 | −0.7 | −0.0 | 3.5 | 3.6 | 3.6 | 7.2 |
| oracle δ\* | – | 7.83 | +0.3 | −1.0 | +0.3 | 3.6 | 3.7 | 4.1 | 7.6 |

### 3-3. WiLoR 기준선 (depth 없음, onscreen)

| 방법 | translation 구하는 법 | CS-MPJPE | MAE x | MAE y | MAE z | bias z |
|---|---|---|---|---|---|---|
| A_demo | demo.py 방식 (가짜 focal) | 16,338 | 11.3 | 6.4 | 16,338 | +16,338 |
| B | WiLoR cam 출력을 실제 K로 변환 | 49.39 | 5.5 | 5.1 | 46.3 | +3.1 |
| C | WiLoR β + 선형 PnP | 26.52 | 5.1 | 4.8 | 23.9 | +11.0 |
| C_nounm | C에서 왼손 unmirror 안 함 (기존 eval 코드 버그) | 123.29 | 109.9 | 5.4 | 30.2 | +2.8 |
| D_lfix | GT β + 선형 PnP | 30.22 | 5.6 | 5.1 | 27.7 | +19.9 |
| D_gn_lfix | D_lfix + reprojection GN 정제 (AGILE L_joint) | 48.24 | 8.6 | 7.5 | 45.7 | +43.4 |
| E_lfix | GT β + GT 2D + 선형 PnP | 27.22 | 4.7 | 4.5 | 25.3 | −2.0 |
| E_gn_lfix | E_lfix + GN 정제 | 32.02 | 5.5 | 5.2 | 30.3 | +20.4 |

- WiLoR 2D는 GT보다 약 2% 작게 예측됨 (spread 비율 중앙값 0.980) → GT β(실제 크기) 손이 멀리 놓여 D_lfix z bias +20mm.
- reprojection 오차를 직접 최소화(GN)하면 depth가 오히려 나빠짐.

---

## 4. 코드와 결과 위치 (`/workspace/drag1008/wilor/`)

| 파일 | 내용 |
|---|---|
| `eval_dexycb_pnp_gtbeta.py` (+ `.sh`) | 기준선 평가 (A~E, lfix, GN) → `eval_dexycb_pnp_gtbeta_full/` |
| `extract_depth_anchor.py` (+ `.sh`) | depth 보정용 추출 (렌더링, 픽셀별 Δ 통계). `--depth_source sensor` → `depth_anchor_extract/`, `gt` → `depth_anchor_extract_gtdepth/` |
| `compare_depth_sources.py` | 3-1 표 |
| `analyze_depth_anchor.py`, `gate_depth_anchor.py`, `decompose_depth_anchor.py` | 후보 비교, 필터, 오차 요인 분해 |
| `check_gt_mesh.py`, `mano_data_dexycb/` | GT MANO mesh 재현 (smplx는 PCA 45개면 PCA를 끄므로 직접 디코딩, 왼손 중지 끝 vertex 445) |
| `viz_depth_samples.py`, `export_3d_samples.py`, `make_depth_correction_slide.py` | 시각화 (샘플 30개 비교, depth point 탑뷰, 방법 그림) |
