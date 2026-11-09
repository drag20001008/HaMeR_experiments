# hand_experiments

HaMeR / WiLoR 기반 손 복원 실험 코드 모음. 결과 정리는 [RESULTS.md](RESULTS.md).

- **손 모양 모델 (M0~M4)**: HaMeR backbone 학습 방식 비교
- **손 위치 방법 (P0~P6)**: 예측한 손을 실제 카메라 좌표계에 배치하는 방법 비교 (HaMeR, WiLoR)

HaMeR 모델 수정은 `hamer_package_changes.patch`에 담았다. [HaMeR](https://github.com/geopavlakos/hamer)를 받은 뒤 레포 루트에서 `git apply hamer_package_changes.patch`로 적용한다.

| 파일 | 내용 |
|---|---|
| `hamer/models/components/lora.py` | M1 LoRA |
| `hamer/models/components/beta_context_adapter.py` | M3, M4 β-adapter |
| `hamer/models/heads/mano_head.py` | M4 2D head |
| `hamer/models/hamer.py` | backbone 모드(lora / full / freeze), decoder 초기화, β 조건, 2D loss |
| `hamer/configs_hydra/experiment/hamer_vit_transformer.yaml` | 학습 설정 |

---

## 폴더 구성

```
hand_experiments/
├── RESULTS.md                실험 설명 + 수치
├── hamer_package_changes.patch  HaMeR 패키지 수정 (아래 표의 5개 파일)
├── hamer_backbone/           M1~M4 학습 / PA-MPJPE 평가
├── hamer_camspace/           HaMeR 손 위치 (P0, P2, P3, P4)
├── hamer_viz/                HaMeR 샘플 덤프 (시각화용)
└── wilor/
    ├── data_prep/            DexYCB, HO3D → npz 변환
    ├── training/             WiLoR 학습 스크립트
    ├── eval/                 WiLoR PA/CS-MPJPE 평가
    ├── camspace/             WiLoR 손 위치 (P0, P1, P2, P5, P6)
    ├── viz/                  분포 그림, 샘플 시각화, 발표용 그림, 3D 뷰어
    ├── realsense/            RealSense 실시간 depth 보정 데모
    └── docs/                 WiLoR 실험 세부 문서
```

---

## 손 모양 모델 (M0~M4) → 스크립트

`hamer_backbone/train_backbone_experiments.py`가 학습, `hamer_backbone/eval_backbone_experiments.py`가 평가(FreiHAND / HO3D / DexYCB PA-MPJPE). sbatch 파일 이름에는 실행 당시 이름(exp*)이 남아 있다.

| 모델 | 학습 옵션 | 학습 sbatch | 평가 sbatch |
|---|---|---|---|
| M0 Pretrained | – | – | 각 평가 sbatch에 포함 |
| M1 LoRA | `--backbone_mode lora --decoder_init random` | `exp1_lora_random.sbatch` | `eval_exp1_lora_random_multi.sbatch` |
| M2 Full-FT | `--backbone_mode full --decoder_init random` | `exp2_full_random.sbatch` | `eval_exp2_full_random_multi.sbatch` |
| M3 β-adapter | `--backbone_mode freeze --decoder_init random` + β 조건 | `exp3_freeze_random_beta.sbatch` | `eval_exp3_freeze_random_beta_epoch10.sbatch` |
| M4 β-adapter + 2D head | M3 + 2D head | `exp5_freeze_beta_2d.sbatch` | `eval_exp5_freeze_beta_2d_epoch10.sbatch` |

`smoke_test.py`: 학습 전 모델 생성, forward 확인용.

---

## 손 위치 방법 (P0~P6) → 스크립트

### HaMeR (`hamer_camspace/`)

| 스크립트 | 내용 |
|---|---|
| `analyze_translation_axiswise_dexycb.py` | DexYCB 손목 오차 축별 (M4: P0 / P2 / P4) |
| `analyze_translation_axiswise_dexycb_original.py` | 같은 분석, M0 |
| `analyze_translation_axiswise_freihand.py`, `_freihand_original.py` | FreiHAND 손목 오차 (P0 / P2), M4 / M0 |
| `analyze_translation_axiswise_dexycb_original_agile.py` | AGILE 절차 첫 구현 (이후 `wilor/camspace/extract_agile.py`로 대체) |
| `eval_global_positioning.py` | CS-MPJPE, P2 (GT 2D / 2D head) |
| `eval_icp_pnp_dexycb.py`, `eval_icp_pnp_ho3d_v2.py` | CS-MPJPE, P2 / P3 / P4 (DexYCB, HO3D) |
| `eval_icp_pnp_ho3d.py` | HO3D 첫 버전 (ICP가 scale까지 조정) |
| `verify_pytorch3d_gt_mano.py`, `verify_pytorch3d_depth_ho3d.py` | depth 렌더러 좌표계 검증 |
| `sbatch/` | 위 스크립트 실행용 (DexYCB는 shard 3개 병렬) |

### WiLoR (`wilor/camspace/`)

| 스크립트 | 내용 |
|---|---|
| `eval_dexycb_pnp_gtbeta.py` (+ `.sh`) | P0 / P1 / P2 비교, CS-MPJPE와 손목 오차 |
| `solve_absolute_depth.py` | translation-only PnP 풀이 (closed-form) |
| `check_gt_mesh.py` | DexYCB 라벨로 GT MANO mesh 재현 확인 |
| `extract_depth_anchor.py` (+ `.sh`) | P5용 데이터 추출: P2 mesh와 GT mesh 렌더링, 픽셀별 depth 차이 (`--depth_source sensor\|gt`) |
| `analyze_depth_anchor.py` | P5 설정 탐색 (영역, erosion, 집계 방식, 최소 픽셀 수) |
| `decompose_depth_anchor.py`, `gate_depth_anchor.py` | P5 오차 요인 분해, 필터 |
| `compare_depth_sources.py` | P5 결과 비교 (GT depth + 실제 손 마스크 / GT 손 윤곽, 센서 depth) |
| `extract_agile.py` (+ `.sh`), `compare_agile.py` | P6 AGILE 재현과 비교 |

### WiLoR 기타

| 폴더 | 내용 |
|---|---|
| `wilor/data_prep/` | `dexycb_to_npz.py`, `ho3d_to_npz.py`, `ho3d_eval_to_npz.py` (평가용 npz 생성), `analyze_detections.py` |
| `wilor/training/` | `train_wilor.py` (β 조건 옵션), 학습 sh |
| `wilor/eval/` | `eval_metric_scale.py` (PA/CS-MPJPE), `analyze_outliers.py`, 체크포인트별 평가 sh |
| `wilor/viz/` | 손목 오차 분포, 앞/뒤 depth 오차, 2D 비교, 샘플 비교, P5 개념도와 발표 그림, 3D 뷰어 HTML 생성 |
| `wilor/realsense/` | RealSense D435i에서 P2 + P5 (SAM3 손 마스크, 센서 depth) |
| `wilor/docs/` | `EXPERIMENTS_SUMMARY.md` (WiLoR 실험 전체 목록), `DEPTH_CORRECTION_PIPELINE.md` (P2 + P5 파이프라인 상세) |

---

## 실행 환경

- HaMeR 스크립트: `hamer` conda env + HaMeR 설치 후 `hamer_package_changes.patch` 적용 (`hamer_camspace`, `wilor/camspace`의 렌더링 스크립트는 pytorch3d 필요)
- WiLoR 스크립트: `wilor` conda env + [WiLoR](https://github.com/rolpotamias/WiLoR) 설치. 손 위치 실험(`wilor/camspace`)은 원본 WiLoR 그대로 사용
- MANO 모델, 체크포인트, 데이터셋(DexYCB, HO3D_v3, FreiHAND)은 포함하지 않음

### WiLoR 학습에 쓴 패키지 수정 (코드는 포함하지 않음)

WiLoR는 CC-BY-NC-ND 라이선스라 수정한 WiLoR 코드는 이 레포에 넣지 않았다. `wilor/training/`의 학습은 원본 WiLoR에 아래 수정을 직접 적용한 상태에서 돌렸다.

- 학습용 dataset 로더 추가 (WiLoR 공개 코드에 학습 dataset이 없음)
- `get_parameters()`가 refine_net을 optimizer에서 빠뜨리던 버그 수정, `freeze_backbone` / `reinit_refine_net` 옵션
- GT β 조건 주입 (ViT, refinement net)
- bbox 정보(CLIFF 방식) 카메라 토큰 조건, scale loss, 실제 focal length 실험 옵션

---

## 라이선스

- 이 레포의 코드: MIT ([LICENSE](LICENSE))
- `hamer_package_changes.patch`는 [HaMeR](https://github.com/geopavlakos/hamer) 코드 일부를 포함한다. HaMeR: MIT License, Copyright (c) 2023 UC Regents, Georgios Pavlakos
- WiLoR, MANO, 각 데이터셋은 각자의 라이선스를 따른다
