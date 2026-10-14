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

import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

XLSX = DATA_DIR + r'rumor_ori_simple.xlsx'
df = pd.read_excel(XLSX, sheet_name='工作表1')

# ---- 因变量：波峰个数 ----
y = pd.to_numeric(df['波峰个数'], errors='coerce')
# ---- 自变量：情绪唤醒度 ----
ar = df['情绪唤醒度'].astype(str).str.replace('，', '.', regex=False).str.replace(',', '.', regex=False)
ar = pd.to_numeric(ar, errors='coerce')
mask = y.notna() & ar.notna()
y = y[mask].to_numpy(float); x = ar[mask].to_numpy(float)
n = len(y)
print(f'N = {n}')
print(f'均值 = {y.mean():.4f}  方差 = {y.var(ddof=1):.4f}  方差/均值 = {y.var(ddof=1)/y.mean():.4f}')
print('取值分布:', {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
print()

X = sm.add_constant(x)

# ================= 1. 标准 Poisson =================
p1 = Poisson(y, X).fit(disp=0)
# ================= 2. Poisson HC1 =================
p2 = Poisson(y, X).fit(disp=0, cov_type='HC1')
# ================= 3. Poisson HC3 =================
p3 = Poisson(y, X).fit(disp=0, cov_type='HC3')
# ---- Pearson 离散指数 ----
mu_hat = p1.predict(X)
pearson_chi2 = ((y - mu_hat) ** 2 / mu_hat).sum()
phi = pearson_chi2 / (n - X.shape[1])
print(f'[Poisson] Pearson 离散指数 phi = {phi:.4f}  (<1 即欠离散)')

def row(name, b, se, pv, extra=''):
    st = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else ''
    return dict(模型=name, beta_唤醒度=round(b, 4), SE=round(se, 4),
                z=round(b/se, 3), p=round(pv, 4), 显著性=st, 备注=extra)

rows = []
rows.append(row('Poisson (模型SE)', p1.params[1], p1.bse[1], p1.pvalues[1], 'Var=mu 假设'))
rows.append(row('Poisson-HC1', p2.params[1], p2.bse[1], p2.pvalues[1], '三明治稳健SE'))
rows.append(row('Poisson-HC3', p3.params[1], p3.bse[1], p3.pvalues[1], '小样本校正SE'))

# ================= 4. Quasi-Poisson（估计离散参数） =================
qse = p1.bse * np.sqrt(phi) if phi > 1 else p1.bse * np.sqrt(phi)
# 准泊松：SE_scaled = SE * sqrt(phi)，phi<1 时 SE 变小
from scipy import stats
qp_se = p1.bse * np.sqrt(phi)
zq = p1.params / qp_se
qp_pv = 2 * (1 - stats.norm.cdf(np.abs(zq)))
rows.append(row('Quasi-Poisson', p1.params[1], qp_se[1], qp_pv[1], f'SE x sqrt(phi), phi={phi:.3f}'))

# ================= 5. 广义泊松 GP（Consul 参数化, MLE） =================
# theta = mu(1-lam); Var = mu/(1-lam)^2 ; lam<0 -> 欠离散
def gp_nll(par):
    b0, b1, lam = par
    if abs(lam) >= 1: return 1e12
    mu = np.exp(b0 + b1 * x)
    theta = mu * (1 - lam)
    arg = theta + lam * y
    if np.any(arg <= 0) or np.any(theta <= 0): return 1e12
    ll = gammaln(np.maximum(theta, 1e-9)) * 0  # placeholder
    ll = np.log(theta) + (y - 1) * np.log(arg) - arg - gammaln(y + 1)
    return -ll.sum()

# 初值：Poisson 估计 + lam 略负
best = None
for lam0 in [-0.2, -0.1, 0.0, 0.1]:
    r = minimize(gp_nll, [p1.params[0], p1.params[1], lam0], method='L-BFGS-B',
                 bounds=[(-10, 10), (-5, 5), (-0.95, 0.95)])
    if best is None or r.fun < best.fun: best = r
gp = best
b0g, b1g, lamg = gp.x
# 数值Hessian求SE
def num_hess(f, p, h=1e-4):
    k = len(p); H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            pp = p.copy(); pp[i] += h; pp[j] += h; fpp = f(pp)
            pp = p.copy(); pp[i] += h; pp[j] -= h; fpm = f(pp)
            pp = p.copy(); pp[i] -= h; pp[j] += h; fmp = f(pp)
            pp = p.copy(); pp[i] -= h; pp[j] -= h; fmm = f(pp)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    return H
Hg = num_hess(gp_nll, gp.x)
covg = np.linalg.pinv(Hg)
seg = np.sqrt(np.diag(covg))
zg = b1g / seg[1]
pg = 2 * (1 - stats.norm.cdf(abs(zg)))
disp_gp = 1 / (1 - lamg) ** 2
rows.append(row('广义泊松 GP', b1g, seg[1], pg, f'lam={lamg:.4f}, Var/mu={disp_gp:.3f}'))

# ================= 6. COM-Poisson（Conway-Maxwell-Poisson, MLE） =================
# P(y)=nu^y/(y!)^theta / Z ; theta>1 欠离散
JMAX = 200
def cmp_logZ(nu, th):
    j = np.arange(0, JMAX + 1)
    terms = j * np.log(nu) - th * gammaln(j + 1)
    return logsumexp(terms)
def cmp_nll(par):
    b0, b1, gamma = par
    nu = np.exp(b0 + b1 * x)
    th = np.exp(gamma)
    lz = np.array([cmp_logZ(v, th) for v in nu])
    ll = y * np.log(nu) - th * gammaln(y + 1) - lz
    return -ll.sum()
bestc = None
for g0 in [0.0, 0.3, 0.7]:
    r = minimize(cmp_nll, [p1.params[0], p1.params[1], g0], method='L-BFGS-B',
                 bounds=[(-10, 10), (-5, 5), (-3, 3)])
    if bestc is None or r.fun < bestc.fun: bestc = r
bc = bestc
b0c, b1c, gc = bc.x; thc = np.exp(gc)
Hc = num_hess(cmp_nll, bc.x)
covc = np.linalg.pinv(Hc); sec = np.sqrt(np.diag(covc))
zc = b1c / sec[1]; pc = 2 * (1 - stats.norm.cdf(abs(zc)))
rows.append(row('COM-Poisson (CMP)', b1c, sec[1], pc, f'theta={thc:.3f} (>1欠离散)'))

# ================= 7. 有序Logit（把1/2/3当有序结果） =================
from statsmodels.miscmodels.ordinal_model import OrderedModel
om = OrderedModel(y.astype(int), x, distr='logit').fit(method='bfgs', disp=0)
# 参数: [beta(唤醒度), cut1, cut2]
b_om = om.params[0]; se_om = om.bse[0]; p_om = om.pvalues[0]
rows.append(row('有序Logit (ordinal)', b_om, se_om, p_om, '尊重有界1-3'))

# ================= 8. 二项(n=2)：K=峰数-1 ~ Binom(2,p) =================
K = (y - 1).astype(int)  # 0,1,2
def bin_nll(par):
    b0, b1 = par
    eta = b0 + b1 * x
    p = 1 / (1 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(K * np.log(p) + (2 - K) * np.log(1 - p) + gammaln(3) - gammaln(K + 1) - gammaln(2 - K + 1))
rb = minimize(bin_nll, [0.0, 0.0], method='L-BFGS-B')
Hb = num_hess(bin_nll, rb.x); covb = np.linalg.pinv(Hb); seb = np.sqrt(np.diag(covb))
zb = rb.x[1] / seb[1]; pb = 2 * (1 - stats.norm.cdf(abs(zb)))
rows.append(row('二项回归 n=2 (bounded)', rb.x[1], seb[1], pb, 'K=峰数-1∈{0,1,2}'))

# ================= 9. 对数似然 / AIC 对比 =================
def aic(nll, k): return 2 * k + 2 * nll
cmp_tab = [
    ('Poisson', p1.llf, 2),
    ('广义泊松 GP', -gp.fun, 3),
    ('COM-Poisson', -bc.fun, 3),
    ('有序Logit', om.llf, 3),  # beta + 2 cuts
    ('二项 n=2', -rb.fun, 2),
]

print()
print('=' * 96)
print('模型对比（因变量：波峰个数；自变量：情绪唤醒度）')
print('=' * 96)
print(pd.DataFrame(rows).to_string(index=False))
print()
print('-' * 96)
print('拟合优度（对数似然 / AIC，越小越好）')
print('-' * 96)
ft = pd.DataFrame([(nm, round(ll, 2), round(2 * k - 2 * ll, 2)) for nm, ll, k in cmp_tab],
                  columns=['模型', '对数似然', 'AIC'])
print(ft.to_string(index=False))
print()

# 各模型在均值处的条件分散度 Var/mu
mu_bar = y.mean()
print('-' * 96)
print('分散度（响应均值处的 Var/Mean，=1 为等散，<1 欠离散）')
print('-' * 96)
print(f'实际数据                : {y.var(ddof=1)/mu_bar:.4f}')
print(f'Poisson                 : {1.0:.4f} (恒定)')
print(f'广义泊松 GP             : {1/(1-lamg)**2:.4f}  (lam={lamg:.4f})')
print(f'COM-Poisson             : theta={thc:.4f} (>1 表示欠离散)')
print(f'负二项 NB               : >=1 (只能处理过度离散，故此处失效)')
