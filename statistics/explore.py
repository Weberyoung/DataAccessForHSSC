
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


import pandas as pd, numpy as np, json
from scipy import stats

df = pd.read_excel(DATA_DIR + 'rumor_ori.xlsx', sheet_name='工作表1')
df.columns = [str(c).replace('\n',' ').strip() for c in df.columns]
emo_col='谣言类型（备注）'; state_col='辟谣后状态新编码'; src_col='辟谣主体类型（备注）'; aro_col='唤醒度_人工'

emo_map={'长期希望型':'Long-term Hope','短期希望型':'Short-term Hope','长期恐惧型':'Long-term Fear','短期恐惧型':'Short-term Fear'}
state_map={0:'Decay',1:'Backfire',2:'NoSigFluct'}
src_map={'专业人员':'Professional','非专业人员':'Non-professional'}

# Full set (no arousal needed): N=431
df['Emotion']=df[emo_col].map(emo_map)
df['State']=df[state_col].map(state_map)
df['Source']=df[src_col].map(src_map)
d_full=df.dropna(subset=['Emotion','State','Source']).copy()
N_full=len(d_full)
# Arousal set: N=430
df['Arousal']=pd.to_numeric(df[aro_col],errors='coerce')
d_aro=df.dropna(subset=['Emotion','State','Arousal']).copy()

order_e=['Long-term Hope','Short-term Hope','Long-term Fear','Short-term Fear']
order_s=['Decay','Backfire','NoSigFluct']

def ctab(d, r, c):
    t=pd.crosstab(d[r],d[c]).reindex(index=order_e,columns=order_s)
    return t

# FIG1 (N=431)
ct=ctab(d_full,'Emotion','State'); ct_pct=ct.div(ct.sum(1),axis=0)*100
chi1,p1,dof1,_=stats.chi2_contingency(ct)
print("N_full=",N_full)
print("FIG1 counts\n",ct); print("FIG1 %\n",ct_pct.round(1))
print(f"FIG1 chi2={chi1:.3f} p={p1:.4f} dof={dof1} overall_decay={round((d_full['State']=='Decay').mean()*100,1)}")

# FIG2 (N=430)
g2={}
for s in order_s:
    v=d_aro.loc[d_aro['State']==s,'Arousal']
    g2[s]={'n':int(len(v)),'median':round(v.median(),2),'mean':round(v.mean(),2),'q1':round(v.quantile(.25),2),'q3':round(v.quantile(.75),2),'min':float(v.min()),'max':float(v.max())}
H,pp=stats.kruskal(*[d_aro.loc[d_aro['State']==s,'Arousal'] for s in order_s])
print("FIG2",g2,"H=",round(H,3),"p=",round(pp,4))

# FIG3 (N=431)
cs=pd.crosstab(d_full['Source'],d_full['State']).reindex(index=['Professional','Non-professional'],columns=order_s)
cs_pct=cs.div(cs.sum(1),axis=0)*100
chi3,p3,dof3,_=stats.chi2_contingency(cs)
prof_b=cs_pct.loc['Professional','Backfire']; nonp_b=cs_pct.loc['Non-professional','Backfire']
a,b,c,e=cs.loc['Non-professional','Backfire'],cs.loc['Non-professional','Decay'],cs.loc['Professional','Backfire'],cs.loc['Professional','Decay']
orv=(a*b)/(c*e)
print("FIG3 counts\n",cs); print("FIG3 %\n",cs_pct.round(1))
print(f"FIG3 chi2={chi3:.3f} p={p3:.4f} ppdiff={round(nonp_b-prof_b,1)} OR={round(orv,3)}")

# FIG4 (N=431)
ct4=ctab(d_full,'Emotion','State'); chi4,p4,dof4,exp=stats.chi2_contingency(ct4)
resid=(ct4-exp)/np.sqrt(exp)
print("FIG4 resid\n",resid.round(2))

out={'N_full':int(N_full),'N_aro':int(len(d_aro)),
 'state_counts':{'Decay':int((d_full['State']=='Decay').sum()),'Backfire':int((d_full['State']=='Backfire').sum()),'NoSigFluct':int((d_full['State']=='NoSigFluct').sum())},
 'emo_counts':d_full['Emotion'].value_counts().to_dict(),
 'src_counts':d_full['Source'].value_counts().to_dict(),
 'fig1_counts':ct.astype(int).to_dict(),'fig1_pct':ct_pct.round(1).to_dict(),
 'fig1_overall_decay':round((d_full['State']=='Decay').mean()*100,1),
 'fig1_chi':[round(chi1,3),round(p1,4),int(dof1)],
 'fig2':g2,'fig2_H':[round(H,3),round(pp,4)],
 'fig3_counts':cs.astype(int).to_dict(),'fig3_pct':cs_pct.round(1).to_dict(),
 'fig3_ppdiff':round(nonp_b-prof_b,1),'fig3_chi':[round(chi3,3),round(p3,4),int(dof3)],'fig3_OR':round(orv,3),
 'fig4_resid':resid.round(2).to_dict(),'fig4_chi':[round(chi4,3),round(p4,4)]}
json.dump(out,open(DATA_DIR + '_stats.json','w'),ensure_ascii=False,indent=2)
print("Saved. N_full=",out['N_full'],"N_aro=",out['N_aro'])
