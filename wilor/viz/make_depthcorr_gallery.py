import json
import os
import sys

import numpy as np

root = sys.argv[1]  # .../RealSense_D435i_Capture/depth_correction
S = json.load(open(os.path.join(root, "summary.json")))
p = ['<!doctype html><meta charset=utf-8><title>RealSense depth correction</title>',
     '<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}'
     'img{width:100%;max-width:1200px;display:block;margin:4px 0 18px}.no{outline:3px solid #c33}'
     'nav a{color:#9cf;margin-right:12px}label{margin-left:16px}</style>',
     '<h1>WiLoR (C: WiLoR beta + PnP) &rarr; sensor-depth correction</h1>',
     '<p>gray = sensor depth in SAM3 mask, red = before (PnP), green = after. Red outline = not corrected (filter).</p>',
     '<nav>' + ''.join(f'<a href="#{s}">{s}</a>' for s in S) + '</nav>',
     '<label><input type=checkbox onchange="document.querySelectorAll(\'.ok\').forEach('
     'e=>e.style.display=this.checked?\'none\':\'\')"> only not-corrected</label>']
for s, v in S.items():
    q = v["shift_mm_q10_q90"]
    p.append(f'<h2 id="{s}">{s}: corrected {v["hands_corrected"]}/{v["hands"]} hands, '
             f'shift median {v["shift_mm_median"]:.1f}mm (q10 {q[0]:.1f}, q90 {q[1]:.1f}), '
             f'frames without detection {v["frames_no_detection"]}</h2>')
    for r in (json.loads(x) for x in np.load(os.path.join(root, s, "results.npz"))["records"]):
        cls = "ok" if r["applied"] else "no"
        p.append(f'<img loading=lazy class={cls} src="{s}/viz/{r["frame"]}_h{r["hand"]}.jpg">')
open(os.path.join(root, "index.html"), "w").write("\n".join(p))
print("ok")
