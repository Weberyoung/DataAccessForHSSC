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



import sys, json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

D = DATA_DIR
smp = pd.read_excel(D + r"/rumor_ori_simple.xlsx")
sub = smp[smp['辟谣后扩散状态'] == '逆火反弹'].copy()
sub = sub.dropna(subset=['情绪唤醒度', '波峰个数'])
N = len(sub)
y_raw = sub['波峰个数'].astype(int)
x = sub['情绪唤醒度'].astype(float)

print("=" * 70)
print("样本：辟谣后扩散状态 = 逆火反弹，N = %d" % N)
print("=" * 70)

# 分组（>=3 合并）
grp = np.where(y_raw.values >= 3, '≥3', y_raw.values.astype(str))
sub['组'] = grp
order = ['1', '2', '≥3']

print("\n【A】分组描述统计（图2）")
rowsA = []
for g in order:
    v = x[sub['组'] == g]
    rowsA.append(dict(组=g, n=len(v), M=round(v.mean(), 2), SD=round(v.std(ddof=1), 2),
                      Mdn=round(v.median(), 2),
                      Q1=round(v.quantile(.25), 2), Q3=round(v.quantile(.75), 2),
                      min=round(v.min(), 2), max=round(v.max(), 2)))
    print("  峰数 %-3s n=%3d  M=%.2f SD=%.2f Mdn=%.2f (Q1=%.2f, Q3=%.2f) [%.2f, %.2f]"
          % (g, len(v), v.mean(), v.std(ddof=1), v.median(),
             v.quantile(.25), v.quantile(.75), v.min(), v.max()))

# Kruskal-Wallis
gs = [x[sub['组'] == g].values for g in order]
H, p_kw = stats.kruskal(*gs)
print("\n【B】Kruskal–Wallis H = %.3f, df = 2, p = %.4f" % (H, p_kw))
# Spearman（原始峰数）
rho, p_sp = stats.spearmanr(x.values, y_raw.values)
print("【C】Spearman ρ = %.3f, p = %.4f" % (rho, p_sp))
rho_g, p_sp_g = stats.spearmanr(x.values, pd.Series(grp).map({'1': 1, '2': 2, '≥3': 3}).values)
print("     （分组后 1/2/3）ρ = %.3f, p = %.4f" % (rho_g, p_sp_g))

# ---------- 表4 ----------
print("\n" + "=" * 70)
print("【D】计数回归：因变量 = 再激活峰数量，自变量 = 情绪唤醒度，N = %d" % N)
print("=" * 70)
mu, va = y_raw.mean(), y_raw.var(ddof=1)
print("均值 = %.4f  方差 = %.4f  方差/均值 = %.4f" % (mu, va, va / mu))
print("最小 = %d  最大 = %d" % (y_raw.min(), y_raw.max()))

X = sm.add_constant(x.values)

# 1) 负二项回归
nb = dict(b=np.nan, se=np.nan, irr=np.nan, p=np.nan, alpha=np.nan, ok=False)
try:
    m_nb = sm.NegativeBinomial(y_raw.values.astype(float), X).fit(disp=0, maxiter=500)
    b = m_nb.params[1]
    alpha = m_nb.params[2] if len(m_nb.params) > 2 else np.nan
    nb = dict(b=float(b), se=float(m_nb.bse[1]), irr=float(np.exp(b)),
              p=float(m_nb.pvalues[1]), alpha=float(alpha), ok=True,
              conv=bool(m_nb.mle_retvals.get('converged', False)))
    print("\n负二项回归: b=%.4f SE=%.4f IRR=%.4f p=%.4f alpha=%.4f converged=%s"
          % (nb['b'], nb['se'], nb['irr'], nb['p'], nb['alpha'], nb.get('conv')))
except Exception as e:
    print("\n负二项回归失败: %s" % e)

# 2) 泊松回归（模型标准误）
m_p = sm.GLM(y_raw.values.astype(float), X, family=sm.families.Poisson()).fit()
b_p = float(m_p.params[1])
se_model = float(m_p.bse[1])
p_model = float(m_p.pvalues[1])
print("泊松（模型SE）  b=%.4f SE=%.4f IRR=%.4f p=%.4f" % (b_p, se_model, np.exp(b_p), p_model))

# 3) HC1 稳健标准误
m_h = sm.GLM(y_raw.values.astype(float), X, family=sm.families.Poisson()).fit(cov_type='HC1')
se_hc1 = float(m_h.bse[1]); p_hc1 = float(m_h.pvalues[1])
print("泊松（HC1）     b=%.4f SE=%.4f IRR=%.4f p=%.4f" % (b_p, se_hc1, np.exp(b_p), p_hc1))

# 4) 准泊松标度标准误
chi2 = float(m_p.pearson_chi2); dfres = float(m_p.df_resid)
scale = chi2 / dfres
se_q = se_model * np.sqrt(scale)
z_q = b_p / se_q
p_q = 2 * (1 - stats.norm.cdf(abs(z_q)))
print("泊松（准泊松）  b=%.4f SE=%.4f IRR=%.4f z=%.3f p=%.4f  (scale=%.4f, Pearson χ²=%.3f, df=%.0f)"
      % (b_p, se_q, np.exp(b_p), z_q, p_q, scale, chi2, dfres))

# 5) 有序 Logistic（1/2/≥3 -> 1/2/3）
y_ord = pd.Series(grp).map({'1': 1, '2': 2, '≥3': 3}).values
try:
    m_o = OrderedModel(y_ord, x.values.reshape(-1, 1), distr='logit').fit(method='bfgs', disp=0)
    b_o = float(m_o.params[-1]); se_o = float(m_o.bse[-1]); p_o = float(m_o.pvalues[-1])
    or_o = float(np.exp(b_o))
    lo, hi = np.exp(b_o - 1.96 * se_o), np.exp(b_o + 1.96 * se_o)
    print("有序Logistic    b=%.4f SE=%.4f OR=%.4f p=%.4f 95%%CI[%.3f, %.3f]" % (b_o, se_o, or_o, p_o, lo, hi))
except Exception as e:
    b_o = se_o = p_o = or_o = lo = hi = np.nan
    print("有序Logistic 失败: %s" % e)

# IRR 95%CI（准泊松口径）
irr_lo, irr_hi = np.exp(b_p - 1.96 * se_q), np.exp(b_p + 1.96 * se_q)
print("\nIRR(准泊松) 95%% CI [%.3f, %.3f]" % (irr_lo, irr_hi))

out = dict(
    N=int(N),
    groups=rowsA,
    kw=dict(H=round(float(H), 3), df=2, p=round(float(p_kw), 4)),
    spearman=dict(rho=round(float(rho), 3), p=round(float(p_sp), 4)),
    spearman_grouped=dict(rho=round(float(rho_g), 3), p=round(float(p_sp_g), 4)),
    count=dict(mean=round(float(mu), 4), var=round(float(va), 4), ratio=round(float(va / mu), 4),
               min=int(y_raw.min()), max=int(y_raw.max()),
               dist={int(k): int(v) for k, v in y_raw.value_counts().sort_index().items()}),
    models=dict(
        nb=nb,
        poisson=dict(b=round(b_p, 4), se_model=round(se_model, 4), p_model=round(p_model, 4),
                     se_hc1=round(se_hc1, 4), p_hc1=round(p_hc1, 4),
                     se_quasi=round(float(se_q), 4), p_quasi=round(float(p_q), 4),
                     z_quasi=round(float(z_q), 3), scale=round(float(scale), 4),
                     irr=round(float(np.exp(b_p)), 4),
                     irr_lo=round(float(irr_lo), 3), irr_hi=round(float(irr_hi), 3)),
        ologit=dict(b=round(float(b_o), 4), se=round(float(se_o), 4), orv=round(float(or_o), 4),
                    p=round(float(p_o), 4), lo=round(float(lo), 3), hi=round(float(hi), 3)),
    ),
)
json.dump(out, open(D + r"/_results_43.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print("\n已写出 _results_43.json")
