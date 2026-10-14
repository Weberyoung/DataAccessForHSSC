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



import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from scipy import stats

x = pd.read_excel(DATA_DIR + r'rumor_ori_simple.xlsx')
d = x.copy()
d['logtime'] = np.log(pd.to_numeric(d['辟谣及时性'], errors='coerce') + 1)
d['expert'] = (d['辟谣主体'] != '非专业人员').astype(int)   # 1 = professional
ds = d.dropna(subset=['辟谣后扩散状态新编码', '辟谣主体编码型', 'logtime']).copy()
ds['exp_x_log'] = ds['expert'] * ds['logtime']
ys = ds['辟谣后扩散状态新编码'].astype(int)
# 0 = 注意力弱化型(base) ; 1 = 注意力强化型 ; 2 = 注意力维持型
N = len(ds)

out = {'N': int(N)}
print('N =', N, '| 类别频数:', ys.value_counts().sort_index().to_dict())


def pack(m, name, llf0=None, k_extra=0):
    rec = {'name': name, 'llf': float(m.llf), 'aic': float(m.aic),
           'prs2': float(m.prsquared), 'rows': []}
    for g in m.params.columns:
        for v in m.params.index:
            rec['rows'].append({
                'outcome': int(g), 'var': v,
                'b': float(m.params.loc[v, g]), 'se': float(m.bse.loc[v, g]),
                'or': float(np.exp(m.params.loc[v, g])),
                'p': float(m.pvalues.loc[v, g]),
                'or_lo': float(np.exp(m.params.loc[v, g] - 1.96 * m.bse.loc[v, g])),
                'or_hi': float(np.exp(m.params.loc[v, g] + 1.96 * m.bse.loc[v, g]))})
    if llf0 is not None:
        lr = 2 * (m.llf - llf0)
        rec['lr'] = {'lr': float(lr), 'df': int(k_extra * len(m.params.columns)),
                     'p': float(stats.chi2.sf(lr, k_extra * len(m.params.columns)))}
    return rec


# Model 1: H3a —— 仅专业性
X1 = sm.add_constant(ds[['expert']].astype(float))
m1 = MNLogit(ys, X1).fit(disp=0)
out['m1'] = pack(m1, 'Model 1: source expertise (H3a)')

# Model 2: H3b —— 专业性 + log(延迟) + 交互
X2 = sm.add_constant(ds[['expert', 'logtime', 'exp_x_log']].astype(float))
m2 = MNLogit(ys, X2).fit(disp=0)
out['m2'] = pack(m2, 'Model 2: + expertise x log(timeliness) (H3b)',
                 llf0=m1.llf, k_extra=2)

for k in ['m1', 'm2']:
    m = out[k]
    print('\n===', m['name'], '| llf %.4f  AIC %.4f  McFadden R2 %.4f' % (m['llf'], m['aic'], m['prs2']))
    for r in m['rows']:
        print('  out=%d %-34s b=%+.3f SE=%.3f OR=%.3f [%.2f,%.2f] p=%.4f'
              % (r['outcome'], r['var'], r['b'], r['se'], r['or'], r['or_lo'], r['or_hi'], r['p']))
print('\nLR test for interaction block (H3b):', out['m2']['lr'])

json.dump(out, open(DATA_DIR + r'_results_44.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\nWROTE _results_44.json')
