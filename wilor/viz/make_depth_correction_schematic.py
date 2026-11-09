"""depth 방향 보정 (b) 방식 개념도 (실제 데이터 없음, 탑뷰 X-Z).
 픽셀 시선마다 Δ(p) = D(p) - z_pred(p)  ->  m = median Δ  ->  손 중심 c의 시선 r̂ = c/‖c‖ 로 δ = m / r̂_z 만큼 강체 평행이동.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
C_PRED, C_DEPTH, C_AFTER, C_RAY = "#2a78d6", "#1baf7a", "#eb6834", "#8f8e87"
OUT = "depth_anchor_extract_gtdepth/depth_correction_schematic.png"


def ellipse(center, a, b, ang, n=240):
    t = np.linspace(0, 2 * np.pi, n)
    p = np.stack([a * np.cos(t), b * np.sin(t)], -1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return p @ R.T + center


def inside(pts, center, a, b, ang):
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (pts - center) @ R.T
    return (q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 <= 1


def first_hit(d, center, a, b, ang):
    """카메라(원점)에서 방향 d로 나간 시선이 타원에 처음 닿는 점 (앞면)."""
    t = np.linspace(0, 20, 20001)
    pts = t[:, None] * d[None]
    hit = inside(pts, center, a, b, ang)
    return pts[np.argmax(hit)] if hit.any() else None


A, Bax, ANG = 1.9, 0.55, np.deg2rad(8)          # 손 모양 (탑뷰에서 본 손의 단면)
C_pred = np.array([0.8, 5.6])                     # 예측 손 중심 (PnP 결과, 실제보다 멂)
r_hat = C_pred / np.linalg.norm(C_pred)
C_true = C_pred - 1.1 * r_hat                    # 실제 손 = 같은 시선 위에서 더 가까움

# 픽셀 시선: 예측 손 앞면을 지나는 방향들
angs = np.linspace(np.arctan2(C_pred[0] - 1.75, C_pred[1] - 0.1), np.arctan2(C_pred[0] + 1.75, C_pred[1] + 0.1), 9)
dirs = np.stack([np.sin(angs), np.cos(angs)], -1)
pairs = []
for d in dirs:
    p = first_hit(d, C_pred, A, Bax, ANG)
    q = first_hit(d, C_true, A, Bax, ANG)
    if p is not None and q is not None:
        pairs.append((p, q))
pred_front = np.array([p for p, _ in pairs])
c = pred_front.mean(0)                             # 사용 픽셀의 예측 표면 3D 중심
m = np.median([q[1] - p[1] for p, q in pairs])     # Δ = D - z_pred (z 성분)
r = c / np.linalg.norm(c)
delta = m / r[1]
c2 = c + delta * r

fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=SURF)
ax.set_facecolor(SURF)
cam = np.array([0.0, 0.0])
# 픽셀 시선
for p, q in pairs:
    far = p + 0.6 * (p / np.linalg.norm(p))
    ax.plot([0, far[0]], [0, far[1]], color=GRID, lw=1.0, zorder=0)
# 예측 손 / 보정 후 손
ax.add_patch(Polygon(ellipse(C_pred, A, Bax, ANG), closed=True, fc=C_PRED, alpha=0.12, ec=C_PRED, lw=2, zorder=1))
ax.add_patch(Polygon(ellipse(C_pred + delta * r, A, Bax, ANG), closed=True, fc="none", ec=C_AFTER, lw=2.2, zorder=4))
# 픽셀별 Δ: 같은 시선 위 예측 표면 점 -> depth 점
for k, (p, q) in enumerate(pairs):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK2, lw=1.6, zorder=3)
    ax.plot(*p, "o", color=C_PRED, ms=7, zorder=5, mec=SURF, mew=1.2)
    ax.plot(*q, "o", color=C_DEPTH, ms=7, zorder=5, mec=SURF, mew=1.2)
p0, q0 = pairs[1]
ax.annotate(r"$\Delta(p) = D(p) - z_{\mathrm{pred}}(p)$", xy=((p0[0] + q0[0]) / 2, (p0[1] + q0[1]) / 2),
            xytext=(-3.5, 6.9), fontsize=13, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
# 중심 c를 지나는 시선과 이동 화살표
ray_end = c + 1.2 * r
ax.plot([0, ray_end[0]], [0, ray_end[1]], color=INK, lw=1.3, ls=(0, (6, 4)), zorder=2)
ax.annotate("", xy=c2, xytext=c, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6, mutation_scale=22), zorder=7)
ax.plot(*c, "o", color=C_PRED, ms=11, zorder=8, mec=INK, mew=1.4)
ax.plot(*c2, "o", color=C_AFTER, ms=11, zorder=8, mec=INK, mew=1.4)
ax.text(c[0] + 0.25, c[1] + 0.25, "c", fontsize=15, color=C_PRED, weight="bold")
ax.annotate(r"$\delta\,\hat r,\ \ \delta = m/\hat r_z$", xy=((c[0] + c2[0]) / 2, (c[1] + c2[1]) / 2), xytext=(3.0, 4.4),
            fontsize=14, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
ax.annotate(r"viewing ray  $\hat r = c/\|c\|$", xy=(0.55 * c[0], 0.55 * c[1]), xytext=(1.2, 2.1),
            fontsize=13, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
# 카메라
ax.plot(*cam, marker="^", ms=20, color=INK2, zorder=6)
ax.text(-0.35, -0.15, "camera", fontsize=13, color=INK2, va="center", ha="right")
# 범례 / 설명
handles = [plt.Line2D([], [], color=C_PRED, lw=2, label="predicted hand (PnP)"),
           plt.Line2D([], [], ls="", marker="o", ms=7, color=C_PRED, label=r"predicted surface  $z_{\mathrm{pred}}(p)$"),
           plt.Line2D([], [], ls="", marker="o", ms=7, color=C_DEPTH, label="depth  D(p)"),
           plt.Line2D([], [], color=C_AFTER, lw=2.2, label="corrected hand")]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=12, labelcolor=INK)
ax.text(-5.25, 2.2, "1.  per pixel p:  " r"$\Delta(p) = D(p) - z_{\mathrm{pred}}(p)$" "\n"
                   "2.  " r"$m = \mathrm{median}_p\ \Delta(p)$" "\n"
                   "3.  shift the whole hand along " r"$\hat r$:  " r"$P' = P + \delta\,\hat r$",
        fontsize=12, color=INK, linespacing=1.9, va="center")
ax.set_xlim(-5.3, 6.4)
ax.set_ylim(-0.6, 7.4)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.annotate("", xy=(6.1, 7.1), xytext=(6.1, 5.7), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
ax.text(6.2, 6.4, "depth Z", fontsize=11, color=INK2, rotation=90, va="center")
ax.set_title("Depth correction along the viewing ray (top view, schematic)", loc="left", fontsize=15, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=SURF)
print("saved", OUT, f"m={m:.3f} delta={delta:.3f}")
