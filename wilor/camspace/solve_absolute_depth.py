"""
21개 관절의 2D 예측 + pelvis-relative 3D 구조(MANO, beta 조건으로 실제 미터 단위)를
실제 카메라 intrinsics와 결합해서, pelvis의 절대 3D 위치(카메라 좌표계, 미터)를
최소제곱으로 복원하는 training-free solver.
"""
import torch


def solve_absolute_translation(
    pred_keypoints_2d_norm: torch.Tensor,  # (B, N, 2) WiLoR 정규화 좌표, [-0.5, 0.5] (= pixel/patch_width - 0.5)
    rel_keypoints_3d: torch.Tensor,        # (B, N, 3) pelvis-relative 3D (미터), MANO 출력
    box_center: torch.Tensor,              # (B, 2)  crop 중심 (원본 이미지 픽셀, cx,cy)
    box_size: torch.Tensor,                # (B,)    crop bbox 크기 (원본 이미지 픽셀, b)
    fx: torch.Tensor, fy: torch.Tensor,    # (B,)    실제 카메라 focal length
    px: torch.Tensor, py: torch.Tensor,    # (B,)    실제 카메라 principal point
) -> torch.Tensor:
    B, N, _ = pred_keypoints_2d_norm.shape

    # ---- Step 1: 정규화 좌표 -> 원본 이미지 픽셀로 되돌리기 (CLIFF식 역변환) ----
    # WiLoR 데이터 파이프라인 convention: norm = pixel_in_crop/patch_width - 0.5
    # => pixel_offset_from_crop_center = norm * box_size (patch_width는 상쇄되어 사라짐)
    center = box_center.view(B, 1, 2)
    b = box_size.view(B, 1, 1)
    u_orig = center + pred_keypoints_2d_norm * b                    # (B, N, 2)
    u, v = u_orig[..., 0], u_orig[..., 1]                          # (B, N) 각각

    x, y, z = rel_keypoints_3d[..., 0], rel_keypoints_3d[..., 1], rel_keypoints_3d[..., 2]

    fx_, fy_ = fx.view(B, 1), fy.view(B, 1)
    px_, py_ = px.view(B, 1), py.view(B, 1)

    # ---- Step 2: 관절마다 핀홀 투영 조건을 선형 방정식으로 세움 ----
    # fx*tx           - (u-px)*tz = (u-px)*z - fx*x
    #        fy*ty    - (v-py)*tz = (v-py)*z - fy*y
    zero = torch.zeros_like(u)
    row_x = torch.stack([fx_.expand(-1, N), zero, -(u - px_)], dim=-1)   # (B, N, 3)
    row_y = torch.stack([zero, fy_.expand(-1, N), -(v - py_)], dim=-1)   # (B, N, 3)

    rhs_x = (u - px_) * z - fx_ * x   # (B, N)
    rhs_y = (v - py_) * z - fy_ * y   # (B, N)

    # ---- Step 3: 21개 관절(x N) x 2개 방정식 -> 과결정 선형계 조립 ----
    A = torch.cat([row_x, row_y], dim=1)                         # (B, 2N, 3)
    b_vec = torch.cat([rhs_x, rhs_y], dim=1).unsqueeze(-1)        # (B, 2N, 1)

    # ---- Step 4: 최소제곱으로 t=(tx,ty,tz) 풀기 (미분 가능, 파라미터 없음) ----
    t = torch.linalg.lstsq(A, b_vec).solution.squeeze(-1)         # (B, 3)
    return t


def evaluate_batch(model, batch, pelvis_id: int = 0):
    """평가 스크립트에서 배치 하나 처리하는 예시."""
    with torch.no_grad():
        output = model.forward_step(batch, train=False)

    pred_keypoints_3d = output["pred_keypoints_3d"]                       # (B,21,3), raw MANO 출력
    rel_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id:pelvis_id+1, :]
    pred_keypoints_2d = output["pred_keypoints_2d"]                        # (B,21,2), 정규화 좌표

    t_pred = solve_absolute_translation(
        pred_keypoints_2d, rel_3d,
        batch["box_center"], batch["box_size"],
        batch["fx"], batch["fy"], batch["px"], batch["py"],
    )

    gt_pelvis_abs = batch["keypoints_3d_raw"][:, pelvis_id, :]  # DexYCB 원본 절대좌표 (미터)
    error_mm = (t_pred - gt_pelvis_abs).norm(dim=-1) * 1000.0
    return error_mm  # (B,) 샘플별 depth(위치) 오차, mm 단위
