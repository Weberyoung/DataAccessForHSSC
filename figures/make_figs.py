
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


import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

S = json.load(open(DATA_DIR + '_stats.json', encoding='utf-8'))
OE = ['Long-term Hope','Short-term Hope','Long-term Fear','Short-term Fear']
OS = ['Decay','Backfire','NoSigFluct']
C_DECAY='#4C72B0'; C_BACK='#C44E52'; C_NSF='#55A868'
COL = {'Decay':C_DECAY,'Backfire':C_BACK,'NoSigFluct':C_NSF}
LEG = {'Decay':'Decay','Backfire':'Backfire','NoSigFluct':'No Significant Fluctuation (NSF)'}
# Re-index JSON (serialized as {state:{emotion:val}}) into emotion-keyed dicts
pct_e  = {e:{s:S['fig1_pct'][s][e]  for s in OS} for e in OE}
cnt_e  = {e:{s:S['fig1_counts'][s][e] for s in OS} for e in OE}
groups = ['Professional','Non-professional']
p3_src = {src:{s:S['fig3_pct'][s][src] for s in OS} for src in groups}
resid_e= {e:{s:S['fig4_resid'][s][e] for s in OS} for e in OE}

# ============ FIGURE 1: 100% stacked bar ============
fig, ax = plt.subplots(figsize=(8.2,5.0))
pct = pct_e; cnt = cnt_e
x = np.arange(len(OE))
bottom = np.zeros(len(OE))
for s in OS:
    vals = [pct[OE[i]][s] for i in range(4)]
    ax.bar(x, vals, bottom=bottom, width=0.62, color=COL[s], edgecolor='white', linewidth=0.8, label=LEG[s])
    for i,v in enumerate(vals):
        if v >= 7:
            yc = bottom[i] + v/2
            ax.text(x[i], yc, f"{v:.1f}", ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
    bottom += np.array(vals)
# overall decay reference line
od = S['fig1_overall_decay']
ax.axhline(od, color='#333333', linestyle='--', linewidth=1.1, zorder=5)
ax.text(0, od+1.2, f"Overall decay rate = {od:.1f}%", fontsize=8.5, color='#333333', ha='left', va='bottom')
ax.set_ylim(0,100); ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Emotion Type')
ax.set_xticks(x); ax.set_xticklabels(OE)
ax.set_axisbelow(True); ax.yaxis.grid(True, color='#E5E5E5', linewidth=0.6)
ax.legend(loc='lower center', frameon=False, ncol=3, fontsize=9, bbox_to_anchor=(0.5,1.005))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
chi=S['fig1_chi']
ax.text(0.0, -0.13, f"Note: χ² = {chi[0]:.3f}, p = {chi[1]:.4f} (df = {chi[2]}).", transform=ax.transAxes, fontsize=8, color='#555555')
fig.tight_layout()
fig.savefig(DATA_DIR + 'Figure1_Emotion_StackedBar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============ FIGURE 2: boxplot of arousal ============
fig, ax = plt.subplots(figsize=(6.6,5.0))
g2 = S['fig2']
data = [np.array([S['fig2'][s]['q1']]*1) for s in OS]  # placeholder
boxes=[]
stats_list=[]
for s in OS:
    v=[]
    # We only stored summary; reconstruct approximate distribution not needed -> use summary for box via patch + errorbars
    pass
# Use summary stats to draw boxes manually
for i,s in enumerate(OS):
    m=g2[s]
    ax.bar(i, 0, color='none')  # ensure tick
    # draw box
    from matplotlib.patches import Rectangle
    x0=i-0.28; x1=i+0.28
    box=Rectangle((x0, m['q1']), 0.56, m['q3']-m['q1'], facecolor=COL[s], alpha=0.65, edgecolor=COL[s], linewidth=1.2, zorder=2)
    ax.add_patch(box)
    # median
    ax.hlines(m['median'], x0, x1, color='#1a1a1a', linewidth=1.6, zorder=4)
    # whiskers
    ax.vlines(i, m['min'], m['q1'], color=COL[s], linewidth=1.1, zorder=3)
    ax.vlines(i, m['q3'], m['max'], color=COL[s], linewidth=1.1, zorder=3)
    ax.hlines(m['min'], i-0.10, i+0.10, color=COL[s], linewidth=1.1, zorder=3)
    ax.hlines(m['max'], i-0.10, i+0.10, color=COL[s], linewidth=1.1, zorder=3)
    # mean marker
    ax.plot(i, m['mean'], marker='D', color='#1a1a1a', markersize=5, zorder=5)
    # label median & mean
    ax.text(i, m['max']+0.12, f"M={m['median']:.1f} / m={m['mean']:.1f}", ha='center', va='bottom', fontsize=8.5, color='#222222')
ax.set_xticks(range(3)); ax.set_xticklabels(['Decay','Backfire','No Significant\nFluctuation'])
ax.set_ylabel('Emotional Arousal (1–9)')
ax.set_xlabel('Post-refutation Diffusion Status')
ax.set_ylim(0,9.4)
ax.set_axisbelow(True); ax.yaxis.grid(True, color='#E5E5E5', linewidth=0.6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
H=S['fig2_H']
ax.text(0.02, 0.97, f"Kruskal–Wallis H = {H[0]:.3f}, p < 0.001", transform=ax.transAxes, fontsize=8.5, color='#333333', va='top')
fig.tight_layout()
fig.savefig(DATA_DIR + 'Figure2_Arousal_Boxplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============ FIGURE 3: grouped bar by source ============
fig, ax = plt.subplots(figsize=(7.2,5.0))
p3=p3_src
groups=['Professional','Non-professional']
gx=np.arange(len(groups)); w=0.26
for j,s in enumerate(OS):
    vals=[p3[groups[i]][s] for i in range(2)]
    pos=gx + (j-1)*w
    bars=ax.bar(pos, vals, width=w, color=COL[s], edgecolor='white', linewidth=0.8, label=LEG[s])
    for i,v in enumerate(vals):
        ax.text(pos[i], v+0.8, f"{v:.1f}", ha='center', va='bottom', fontsize=8.5, color='#222222')
ax.set_xticks(gx); ax.set_xticklabels(groups)
ax.set_ylabel('Percentage (%)'); ax.set_xlabel('Debunking Source')
ax.set_ylim(0, max(max(p3['Professional'].values()),max(p3['Non-professional'].values()))+8)
ax.set_axisbelow(True); ax.yaxis.grid(True, color='#E5E5E5', linewidth=0.6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=False, ncol=1, fontsize=9)
chi=S['fig3_chi']
ax.text(0.0,-0.13, f"Note: χ² = {chi[0]:.3f}, p = {chi[1]:.4f} (df = {chi[2]}). "
                  f"Decay gap (Professional − Non-professional) = 17.2 pp.", transform=ax.transAxes, fontsize=8, color='#555555')
fig.tight_layout()
fig.savefig(DATA_DIR + 'Figure3_DebunkingSource_GroupedBar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============ FIGURE 4: mosaic of emotion x state ============
fig, ax = plt.subplots(figsize=(8.2,6.0))
cnt=cnt_e
N=sum(cnt[e][s] for e in OE for s in OS)
rowtot={e:sum(cnt[e][s] for s in OS) for e in OE}
colwidth={e:rowtot[e]/N for e in OE}
resid=S['fig4_resid']
cmap=LinearSegmentedColormap.from_list('RdBu',['#2166AC','#F7F7F7','#B2182B'])
vmax=max(abs(resid_e[e][s]) for e in OE for s in OS)
# x positions
xedges=[0]; 
for e in OE: xedges.append(xedges[-1]+colwidth[e])
# draw cells: within each column, stack states by height = count/rowtot (column height=1)
yedges_per_col={}
for ci,e in enumerate(OE):
    x0=xedges[ci]; x1=xedges[ci+1]
    y=0.0
    for s in OS:  # order bottom->top same as stacked bar: Decay, Backfire, NSF
        h=cnt[e][s]/rowtot[e]
        r=resid_e[e][s]
        color=cmap(r/vmax)
        ax.add_patch(Rectangle((x0,y), x1-x0, h, facecolor=color, edgecolor='white', linewidth=1.4))
        # label count and %; narrow columns show count only to avoid crowding
        colw = x1 - x0
        if colw < 0.10:
            lab = f"{cnt[e][s]}"
        else:
            lab = f"{cnt[e][s]}\n({cnt[e][s]/rowtot[e]*100:.1f}%)"
        ax.text((x0+x1)/2, y+h/2, lab, ha='center', va='center', fontsize=7.5, color='#111111')
        y+=h
# axes
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xticks([(xedges[i]+xedges[i+1])/2 for i in range(4)]); ax.set_xticklabels(OE)
ax.set_yticks([1/6,3/6,5/6]); ax.set_yticklabels(['Decay','Backfire','No Significant Fluctuation'])
ax.set_xlabel('Emotion Type'); ax.set_ylabel('Post-refutation Diffusion Status')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
ax.tick_params(length=0)
sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
sm.set_array([])
cbar=fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label('Standardized Residual', fontsize=9)
cbar.ax.tick_params(labelsize=8)
chi=S['fig4_chi']
ax.text(0.0,-0.10, f"Note: χ² = {chi[0]:.3f}, p < 0.001. Cell area ∝ frequency; red = observed > expected, blue = observed < expected.", transform=ax.transAxes, fontsize=8, color='#555555')
fig.tight_layout()
fig.savefig(DATA_DIR + 'Figure4_Emotion_State_Mosaic.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("All 4 figures generated.")
