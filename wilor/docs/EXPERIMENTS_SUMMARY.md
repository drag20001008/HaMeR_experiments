# WiLoR / HaMeR 실험 결과 총정리 (2026-08 ~ 2026-11)

soda(`/workspace/drag1008/{wilor,hamer,hamer_prev}`), vegi(`/workspace/drag1008/{wilor,hamer,hamer_prev}`), nas2(`/nas2/lait/1000_Members/jaehoon/`)의 코드, slurm 로그, 결과 파일을 읽어서 정리한 문서 (2026-11-04).
가장 최근 작업(camera-space, depth 보정)의 자세한 내용은 같은 폴더의 `DEPTH_CORRECTION_PIPELINE.md` 참고.

단위: mm. 표의 값은 별도 표시가 없으면 평균, `a / b`는 평균 / 중앙값.

---

## 0. 먼저 알아야 할 버그와 주의점

| # | 내용 | 영향받는 결과 |
|---|---|---|
| 1 | **WiLoR `get_parameters()` 버그** (커밋 `c0645b5`, 2026-08-20 이전): optimizer에 backbone만 들어가 **refine_net이 학습되지 않음** | soda: baseline_v2, beta_v2 / vegi: beta_bbox, beta_bbox_realfocal, beta_bbox_scaleloss |
| 2 | **왼손 unmirror 누락** (`wilor/eval_metric_scale.py`, `hamer/eval_global_positioning.py`) | DexYCB의 translation, CS-MPJPE 전부 (test의 49.7%가 왼손). **PA-MPJPE는 영향 없음** |
| 3 | **HO3D v1 npz 관절 순서 오류** (`ho3d_annotations.npz`, `dae13a3`에서 수정) | v1 npz로 낸 HO3D PA ≈ 42mm, CS ≈ 63~70mm는 모두 무효 |
| 4 | **HaMeR 평가의 "CS-MPJPE"는 camera-space 지표가 아님**: 가짜 focal(5000) 기반 `pred_cam_t` 사용, 2.3~5.0m | `eval_backbone_experiments.py`, `hamer_prev/eval_hamer_baseline.py`의 CS 값 |
| 5 | **AGILE ICP 초기값 버그** (t=0에서 시작 → scale이 0으로 붕괴) | `hamer/analyze_translation_axiswise_dexycb_original_agile.py` (z bias −783mm, onscreen −826mm), `eval_icp_pnp_ho3d.py` v1 (438mm)도 같은 패턴 |
| 6 | **HaMeR 학습 DDP 의심** (`train_backbone_experiments.py`): DDP wrapper의 `forward`를 거치지 않고 `model.module.forward_step` 호출 → GPU 간 gradient가 합쳐지지 않았을 가능성. **아직 검증 안 됨** | HaMeR exp1~exp6 학습 전체 |
| 7 | **β 조건 실험은 평가 때 GT β를 넣음** (oracle shape) | WiLoR beta_v2/beta_bbox, HaMeR β-adapter 실험의 FreiHAND/DexYCB 수치 |
| 8 | **학습 데이터와 평가 데이터 겹침**: pretrained WiLoR/HaMeR 학습에 DexYCB train, HO3D train이 포함됨 | HO3D **train** split 평가(N=77,743)는 held-out이 아님. held-out은 HO3D official eval(N=19,062) |
| 9 | 평가는 모두 **GT keypoint 기반 bbox** 사용 (detector bbox 아님) | 전체 |

---

## 1. WiLoR

### 1-1. 학습 실험

공통: pretrained `wilor_final.ckpt`에서 시작, DexYCB `wilor_annotations_v3.npz` official s0 (train 406,888 / val 22,728), 4×GPU DDP, 최대 10 epoch. **검증 지표는 어디에도 저장되지 않음** (wandb에는 train loss만).

| 실험 | 서버 / 날짜 | 설정 | 버그 #1 | 결과 |
|---|---|---|---|---|
| v1 baseline / beta | soda 08-17~18 | v1 npz, subject split | 해당 | epoch 7~8에서 취소, 체크포인트·평가 없음 |
| **baseline_v2** | soda 08-18~19 | 기본 (β 예측) | **해당** | epoch 5에서 NFS/NCCL 오류로 중단. ckpt epoch 0~4 (`runs_baseline_v2/`) |
| **beta_v2** | soda 08-18~19 | `--beta_conditioning` (GT β 주입) | **해당** | epoch 5에서 중단. ckpt epoch 0~4 (`runs_beta_v2/`) |
| **refinenet_ft** | soda 08-20~22 | `--freeze_backbone --reinit_refine_net --lr 1e-4` (refine_net만 학습, 8.3M) | 수정 후 | **10 epoch 완료** (`runs_refinenet_ft/`) |
| beta_bbox_vegi | vegi 08-18~19 | `--beta_conditioning --bbox_info` | **해당** | epoch 9에서 취소. ckpt 0~8 |
| beta_bbox_scaleloss_vegi | vegi 08-18 | + `--scale_loss 0.05` | **해당** | epoch 0에서 취소 |
| beta_bbox_realfocal_vegi | vegi 08-18~19 | `--beta_conditioning --bbox_info --real_focal` | **해당** | 10 epoch 완료 |
| lora_refinenet | vegi 08-20 | ViT 고정 + LoRA r8, refine_net 재초기화, lr 1e-4 | 수정 후 | 10 epoch 완료 |

vegi 체크포인트는 2026-10-30에 nas2 `wilor_checkpoints/`로 옮김.

### 1-2. 평가 (`eval_metric_scale.py`)

**DexYCB s0 test** (N=78,768, †는 onscreen만 N=70,595). translation과 CS-MPJPE는 버그 #2로 **무효**라 참고만.

| 체크포인트 | PA-MPJPE | (translation, 무효) | (CS-MPJPE, 무효) |
|---|---|---|---|
| **pretrained WiLoR** | **4.93** / 4.38 | 132.39 | 132.99 |
| baseline_v2 e4 | 5.97 / 5.53 | 126.92 | 127.58 |
| beta_v2 e4 (GT β) | 6.12 / 5.76 | 133.50 | 133.91 |
| refinenet_ft epoch=4 † | 5.23 / 4.82 | 122.80 | 123.46 |
| refinenet_ft epoch=9 † | 5.46 / 5.07 | 123.52 | 124.22 |
| beta_bbox e4 / e8 (vegi, GT β) | 6.14 / 6.06 | 133.86 / 133.18 | 134.29 / 133.60 |
| realfocal e4 / e8 (vegi, GT β) | 6.09 / 6.10 | 132.80 / 134.41 | 133.46 / 134.99 |
| lora_refinenet epoch=4 / 9 † (vegi) | 5.76 / 5.76 | 124.58 / 122.60 | 125.38 / 123.44 |

→ **DexYCB PA-MPJPE는 pretrained(4.93)가 가장 좋고, 모든 finetuning이 나빠짐.** († 표시는 onscreen 부분집합이라 직접 비교 시 주의)

**HO3D_v3 official eval** (held-out, N=19,062, 전부 오른손이라 translation/CS도 유효)

| 체크포인트 | translation | CS-MPJPE | PA-MPJPE |
|---|---|---|---|
| **pretrained WiLoR** | **31.26** / 26.21 | **32.41** | **6.48** / 6.21 |
| baseline_v2 e4 | 31.57 | 32.48 | 7.45 |
| beta_v2 e4 (GT β) | 37.53 | 35.56 | 7.71 |
| beta_bbox e4 / e8 (vegi) | 37.01 / 37.59 | 35.60 / 35.97 | 7.80 / 7.89 |
| realfocal e4 / e8 (vegi) | 42.44 / 40.18 | 42.47 / 40.35 | 7.79 / 8.03 |
| refinenet_ft, lora_refinenet | **평가 안 함** | | |

**HO3D_v3 train** (`ho3d_annotations_v2.npz`, N=77,743, held-out 아님)

| 체크포인트 | translation | CS-MPJPE | PA-MPJPE |
|---|---|---|---|
| pretrained | 9.73 | 10.30 | 2.53 |
| baseline_v2 e4 | 22.68 | 23.37 | 5.98 |
| beta_v2 e4 | 20.40 | 20.33 | 5.74 |
| refinenet_ft epoch=4 / 9 | 11.41 / 11.05 | 12.65 / 12.37 | 4.11 / 4.43 |
| beta_bbox e4 (vegi) | 20.29 | 20.04 | 5.76 |
| realfocal e4 (vegi) | 29.19 | 29.48 | 5.75 |
| lora_refinenet epoch=4 / 9 (vegi) | 15.46 / 17.70 | 16.54 / 18.98 | 4.56 / 5.01 |

FreiHAND 평가는 WiLoR 쪽에서 하지 않음. PVE, F-score, AUC도 계산하지 않음.

### 1-3. 분석 (soda)
- **outlier 분석** (baseline_v2 e2, DexYCB test 앞 12,800장): translation 평균 134.2 / 중앙값 51.4, p99 715mm. 최악은 subject-03 한 시퀀스. PA-MPJPE와 translation 오차의 상관 0.053 (거의 무관). 버그 #2가 outlier의 상당 부분을 만든 것으로 보임.

### 1-4. camera-space와 depth 보정 (pretrained WiLoR, 2026-09-30 ~ 10-12)
자세한 내용은 `DEPTH_CORRECTION_PIPELINE.md`. DexYCB s0 test **onscreen** CS-MPJPE:

| 방법 | CS-MPJPE |
|---|---|
| demo.py (가짜 focal) | 16,338 |
| WiLoR cam 출력 + 실제 K | 49.39 |
| **WiLoR β + translation-only PnP (C)** | **26.52** |
| C, 왼손 unmirror 없음 (기존 eval 방식) | 123.29 |
| GT β + PnP, 왼손 shapedirs 보정 (D_lfix) | 30.22 |
| D_lfix + GN 정제 | 48.24 |
| GT β + GT 2D + PnP (E_lfix) | 27.22 |
| D_lfix + 센서 depth 보정 + 필터 | 21.52 |
| D_lfix + GT depth 보정 (실제 손 마스크) | **9.93** |
| oracle (시선 방향 이동 상한) | 7.83 |

- WiLoR 2D가 GT보다 약 2% 작게 나와 GT β를 쓰면 손이 멀리 놓임 (z bias +20mm).
- 센서 depth는 GT 손 표면보다 평균 +9.7mm 멀게 찍히고 꼬리가 김.

**AGILE 재현** (수정한 ICP, 논문 그대로, SAM3 마스크): ICP scale 중앙값 센서 0.326 / GT depth 0.750 (기준 0.996) → 손이 줄어들어 onscreen CS 542 / 311mm. 논문에 적힌 정보만으로는 재현되지 않음. L_joint 정제 버전은 scale 붕괴로 일부 NaN.

---

## 2. HaMeR

### 2-1. RayHand (hamer_prev, 2026-08-24~27)
frozen HaMeR ViT + RayFusion + HandDGP 스타일 decoder, 실제 K 사용 (CS 의미 있음). PA-MPJPE FreiHAND / HO3D / DexYCB:

| 변형 | e8 | e40 |
|---|---|---|
| v2 (film + ray) | 13.07 / 10.71 / 8.43 | 41.76 / 45.97 / 43.43 (발산) |
| nofilm | 13.65 / 14.49 / 11.16 | 11.30 / 13.86 / 9.08 (CS 80.0 / 70.4 / 156.6) |
| noray | 14.04 / 17.10 / 14.51 | 13.61 / 16.75 / 15.21 (CS 72.6 / 53.1 / 150.9) |

→ pretrained HaMeR보다 크게 나쁨.

### 2-2. Backbone 실험 (exp1~6, 2026-09-01~16)
공통: DexYCB train + HO3D_v3 전체 + FreiHAND 전체, AdamW lr 1e-4, bs 32, 50 epoch, 2 GPU (**버그 #6 의심**). 체크포인트는 nas2 `hamer_checkpoints/`.

| 실험 | 학습 서버 | backbone | decoder | 추가 | 마지막 ckpt |
|---|---|---|---|---|---|
| exp1_lora_random | soda | LoRA r8 | 랜덤 | – | epoch 013 (미완) |
| exp2_full_random | soda | 전체 학습 | 랜덤 | – | epoch 010 (미완) |
| exp3_freeze_random | vegi | 고정 | 랜덤 | – | 049 |
| exp4_freeze_pretrained | vegi | 고정 | pretrained | – | 049 |
| exp3_freeze_random_beta | vegi | 고정 | 랜덤 | β-adapter | 049 |
| exp5_freeze_beta_2d | vegi | 고정 | 랜덤 | β-adapter + 2D head | 049 |
| exp4_freeze_pretrained_beta | ? | 고정 | pretrained | β-adapter | 029 (미완, 평가 없음) |
| exp5_lora_beta_2d | ? | LoRA | 랜덤 | β-adapter + 2D head | 012 (미완, 평가 없음) |
| exp6_freeze_beta_2d | ? | 고정 | 랜덤 | β-adapter + 2D head | 028 (미완, 평가 없음) |

**PA-MPJPE** (`eval_backbone_experiments.py`, FreiHAND N=3,960 / HO3D eval N=19,062 / DexYCB test N=78,768; 결과 `nas2/.../hamer_eval_results/`). 파일 이름의 epoch은 1부터, ckpt는 0부터 셈.

| 모델 (ckpt) | FreiHAND | HO3D eval | DexYCB test |
|---|---|---|---|
| **pretrained HaMeR** | 6.39 / 5.07 | 6.84 / 5.48 | **4.34 / 3.45** |
| exp1 LoRA (008) | **6.02** / 4.92 | 6.80 | 6.03 |
| exp2 full (008) | 14.74 | 10.17 | 8.81 |
| exp3 (019) | 6.52 | 6.71 | 6.15 |
| exp4 (019) | 6.60 | 6.80 | 6.08 |
| exp3_beta (029, GT β) | 6.39 | 6.93 | 6.10 |
| exp5 (019, GT β) | 6.51 | **6.49** / 5.27 | 6.06 |
| exp5 (029, GT β) | 6.34 | 6.60 | **5.95** / 4.96 |
| exp5 (049, GT β) | 6.40 | 6.60 | 6.07 |

→ **DexYCB에서는 어떤 실험도 pretrained HaMeR(4.34)를 넘지 못함** (최고 5.95). FreiHAND와 HO3D에서는 약간 나은 경우가 있음 (FreiHAND 6.02 vs 6.39, HO3D 6.49 vs 6.84). β-adapter 결과는 GT β를 넣은 것 (버그 #7).

### 2-3. camera-space translation (HaMeR)
**global positioning** (`eval_global_positioning.py`, exp5, 선형 PnP). FreiHAND와 HO3D만 유효 (DexYCB는 버그 #2):

| ckpt | FreiHAND GT 2D / 예측 2D | HO3D GT 2D / 예측 2D |
|---|---|---|
| exp5 019 | 49.61 / 46.26 | 30.86 / 36.03 |
| exp5 029 | 50.99 / 47.55 | 31.27 / 36.70 |

손목 위치를 GT로 둔 root-relative MPJPE (029): FreiHAND 12.97, HO3D 15.57, DexYCB 11.40.

**ICP + PnP** (exp5 029로 추정, 체크포인트 기록 없음)

| 데이터 | PnP (GT 2D) | PnP (예측 2D) | ICP만 | ICP → PnP |
|---|---|---|---|---|
| HO3D eval (N=19,062, SAM3 손 마스크) | 31.83 | 36.79 | 55.82 | 35.26 |
| DexYCB test (N=78,768, GT seg, 왼손 처리 정상) | 34.01 | 32.62 | 83.30 | 58.74 |

**축별 손목 오차** (`analyze_translation_axiswise_*.py`, 2026-09-17, MAE x / y / z):

| 데이터 / 모델 | 방법 | MAE x / y / z | z bias | ‖e‖ 평균 |
|---|---|---|---|---|
| DexYCB, exp5 029 | PnP (예측 2D) | 7.13 / 7.23 / 29.00 | +15.90 | **32.13** |
| DexYCB, exp5 029 | ICP → PnP | 10.53 / 10.58 / 54.68 | −33.80 | 58.41 |
| DexYCB, pretrained HaMeR | PnP | 14.06 / 16.75 / 57.51 | +52.08 | 64.04 |
| DexYCB, pretrained HaMeR | ICP → PnP | 14.94 / 17.87 / 69.16 | −13.85 | 75.79 |
| DexYCB, pretrained HaMeR | AGILE (버그 #5, 무효) | 113.68 / 110.07 / 815.38 | −783.08 | 840.93 |
| FreiHAND, exp5 | PnP | 4.17 / 3.89 / 46.54 | +0.60 | – |
| FreiHAND, pretrained HaMeR | PnP | 5.55 / 4.70 / 65.63 | +25.27 | – |

→ camera-space에서는 **2D head를 단 exp5 + PnP가 pretrained HaMeR + PnP보다 확실히 좋음** (DexYCB ‖e‖ 32.1 vs 64.0).

### 2-4. 기타
- pytorch3d 렌더링 검증: exp5 + PnP 손의 렌더링 depth와 실측 depth 차이 평균 24.47mm, 마스크 IoU 0.54 (HO3D 10장). GT MANO로는 1.6~23.4mm (5장).
- 시각화 덤프: `nas2/.../hamer_eval_results/viz*_samples*.json`

---

## 3. 요약

1. **PA-MPJPE(손 모양) 기준으로는 pretrained가 가장 좋음.** WiLoR(DexYCB 4.93), HaMeR(DexYCB 4.34) 모두, 어떤 finetuning도 DexYCB에서 이를 넘지 못함. 단 WiLoR 학습 대부분은 버그 #1(refine_net 미학습), HaMeR 학습은 버그 #6(DDP) 가능성이 있어 결론을 단정하긴 이름.
2. **camera-space 위치는 translation을 실제 K로 다시 푸는 게 핵심.** 가짜 focal 출력은 무의미하고(수 m), translation-only PnP만으로 WiLoR 26.5mm(onscreen), HaMeR exp5 32.1mm(DexYCB 전체).
3. **depth를 쓰면 크게 개선.** 정확한 depth면 WiLoR 9.93mm(상한 7.83). 센서 depth는 센서–라벨 불일치 때문에 21.5mm에 그침.
4. **과거 DexYCB camera-space 수치 상당수가 왼손 버그(#2)로 무효.** 이후 결과(`eval_dexycb_pnp_gtbeta.py`, `eval_icp_pnp_dexycb.py`, axiswise)는 왼손을 정상 처리함.

## 4. 진행 중 (2026-10-28 ~)
- **CHOIR 최적화 테스트** 준비: SAM3 손 마스크(test 전체), SAM3 물체 마스크(GT bbox + text, `nas2/.../DexYCB/sam3_object_masks_gtbox/`), SAM 3D Objects 환경(soda, vegi 설치 완료). SAM3D 체크포인트(Hugging Face 로그인 필요)와 CHOIR `choir` 환경이 남음.
