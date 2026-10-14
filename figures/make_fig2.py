# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
def _find_data_dir():
    _e = _os.environ.get('RUMOR_DATA_DIR')
    if _e:
        return _e if _e.endswith(_os.sep) else _e + _os.sep
    _p = _os.path.abspath(__file__)
    for _ in range(6):
        _p = _os.path.dirname(_p)
        if _os.path.exists(_os.path.join(_p, 'rumor_ori_simple.xlsx')):
            return _p + _os.sep
    return _p + _os.sep
DATA_DIR = _find_data_dir()
_DIRNAME = _os.path.basename(DATA_DIR.rstrip('/\\'))
def _add_cn_font():
    for _f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyh.ttf',
               'C:/Windows/Fonts/simhei.ttf',
               '/System/Library/Fonts/PingFang.ttc',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
               '/usr/share/fonts/truetype/arphic/uming.ttc'):
        if _os.path.exists(_f):
            try:
                import matplotlib.font_manager as _fm
                _fm.fontManager.addfont(_f)
                return _f
            except Exception:
                pass
    return None



import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.stdout.reconfigure(encoding='utf-8')

D = DATA_DIR
smp = pd.read_excel(D + r"/rumor_ori_simple.xlsx")
sub = smp[smp['辟谣后扩散状态'] == '逆火反弹'].dropna(subset=['情绪唤醒度', '波峰个数']).copy()
y = sub['波峰个数'].astype(int).values
x = sub['情绪唤醒度'].astype(float).values
grp = np.where(y >= 3, '≥3', y.astype(str))
order = ['1', '2', '≥3']
data = [x[grp == g] for g in order]
ns = [len(d) for d in data]
means = [d.mean() for d in data]
print("n =", ns, " means =", [round(m, 2) for m in means])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.linewidth': 1.1, 'axes.edgecolor': '#333333',
    'xtick.direction': 'out', 'ytick.direction': 'out',
})

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)

bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.52, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#cfe1f2', edgecolor='#2b6ca3', linewidth=1.4),
                medianprops=dict(color='#12395e', linewidth=2.6),
                whiskerprops=dict(color='#2b6ca3', linewidth=1.4),
                capprops=dict(color='#2b6ca3', linewidth=1.4))

rng = np.random.default_rng(20260922)
for i, d in enumerate(data, start=1):
    jitter = rng.uniform(-0.11, 0.11, len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, s=22, color='#7fa8cf', alpha=0.38,
               edgecolors='none', zorder=2)

ax.plot([1, 2, 3], means, linestyle='--', color='#b23a2f', linewidth=2.0,
        marker='D', markersize=9, markerfacecolor='white', markeredgecolor='#b23a2f',
        markeredgewidth=1.8, zorder=5, label='Group mean')

for i, m in enumerate(means, start=1):
    ax.annotate('%.2f' % m, xy=(i, m), xytext=(i + 0.13, m + 0.30),
                fontsize=12.5, color='#b23a2f', fontweight='bold')

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['1 peak\n(n = %d)' % ns[0], '2 peaks\n(n = %d)' % ns[1],
                    '\u22653 peaks\n(n = %d)' % ns[2]], fontsize=13)
ax.set_xlabel('Number of attention peaks', fontsize=14, labelpad=10)
ax.set_ylabel('Emotional arousal (1\u20139 scale)', fontsize=14, labelpad=10)
ax.set_ylim(0.5, 9.5)
ax.set_yticks(range(2, 10, 2))
ax.yaxis.grid(True, linestyle=':', linewidth=0.7, color='#cccccc', zorder=0)
ax.set_axisbelow(True)

ax.text(0.985, 0.055,
        'Kruskal\u2013Wallis H = 31.113, df = 2, p < .001\nSpearman \u03c1 = 0.549, p < .001',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=11.5, color='#333333',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f7f7f7', edgecolor='#bbbbbb', linewidth=0.8))

leg = ax.legend(handles=[Line2D([0], [0], color='#b23a2f', linewidth=2.0, linestyle='--',
                                marker='D', markersize=9, markerfacecolor='white',
                                markeredgecolor='#b23a2f', label='Group mean')],
                loc='upper left', frameon=True, fontsize=12.5, framealpha=0.95)
leg.get_frame().set_edgecolor('#bbbbbb')

plt.tight_layout()
out = D + r"/_new_fig2.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(D + r"/_new_fig2_preview.png", dpi=110, bbox_inches='tight', facecolor='white')
from PIL import Image
im = Image.open(out)
print("输出:", out, im.size, "mode=", im.mode)
