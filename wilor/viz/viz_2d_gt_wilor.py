"""
eval_dexycb_pnp_gtbeta.py가 저장한 per_sample.npz에서 GT 2D(초록)와 WiLoR 2D(빨강)만 그린다.
  viz2d_random.png : 랜덤 샘플
  viz2d_worst.png  : WiLoR 2D 오차(21관절 평균 px) 최악 샘플
"""
import argparse
import os

import cv2
import numpy as np

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_hand(img, kp, color, thick):
    for a, b in BONES:
        cv2.line(img, tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int)), color, thick, cv2.LINE_AA)
    for p in kp:
        cv2.circle(img, tuple(np.round(p).astype(int)), thick + 1, color, -1, cv2.LINE_AA)


def make_tile(d, i, tile):
    img = cv2.imread(str(d["imgname"][i]))
    cx, cy = d["box_center"][i]
    half = 0.65 * float(d["box_size"][i])
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    # 이미지 밖은 검정: side x side 캔버스에 이미지와 겹치는 영역만 복사
    canvas = np.zeros((side, side, 3), np.uint8)
    H, W = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (tile, tile))
    s = tile / side
    offs = np.array([x0, y0], dtype=np.float32)
    draw_hand(crop, (d["gt2d"][i] - offs) * s, (0, 200, 0), 2)    # GT: green
    draw_hand(crop, (d["wilor2d"][i] - offs) * s, (0, 0, 255), 1)  # WiLoR: red
    t = f"#{int(d['idx'][i])} {'R' if d['right'][i] > 0.5 else 'L'}  2D {float(d['err2d'][i]):.1f}px"
    cv2.putText(crop, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(crop, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def save_grid(d, ids, path, cols, tile):
    tiles = [make_tile(d, i, tile) for i in ids]
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    legend = np.full((30, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(legend, "green: GT 2D   red: WiLoR 2D (crop->full image)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate([legend, grid], 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out_dir", default=None, help="기본값: npz가 있는 폴더")
    parser.add_argument("--n_random", type=int, default=18)
    parser.add_argument("--n_worst", type=int, default=12)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))

    d = np.load(args.npz, allow_pickle=True)
    n = len(d["idx"])
    rand_i = np.random.default_rng(args.seed).choice(n, size=min(args.n_random, n), replace=False)
    worst_i = np.argsort(-d["err2d"])[:args.n_worst]
    save_grid(d, rand_i, os.path.join(out_dir, "viz2d_random.png"), args.cols, args.tile)
    save_grid(d, worst_i, os.path.join(out_dir, "viz2d_worst.png"), args.cols, args.tile)
    print(f"N={n}  WiLoR 2D err mean={d['err2d'].mean():.2f}px median={np.median(d['err2d']):.2f}px  -> {out_dir}")


if __name__ == "__main__":
    main()
