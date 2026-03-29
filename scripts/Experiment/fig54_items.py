"""fig54_items.py — Figure 5.4: Item-Level Mean Comparison"""
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'; PURPLE='#9B85B5'; LIGHT='white'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']
nc,nt=len(ctrl),len(trt)

def pstar(p):
    if p<.001: return '***'
    if p<.01:  return '**'
    if p<.05:  return '*'
    return 'n.s.'

fig,axes=plt.subplots(1,2,figsize=(18,7)); fig.patch.set_facecolor(LIGHT)
w=0.38

ax=axes[0]; ax.set_facecolor(LIGHT)
items_m=['E_1','E_2','E_3','E_4','F_1','F_2','F_3','G_1','G_2','G_3','G_4','G_5']
labels_m=['E1','E2','E3','E4','F1','F2','F3','G1','G2','G3','G4','G5']
cm=[ctrl[i].mean() for i in items_m]; cm_se=[ctrl[i].sem() for i in items_m]
tm=[trt[i].mean()  for i in items_m]; tm_se=[trt[i].sem()  for i in items_m]
x=np.arange(len(items_m))
ax.bar(x-w/2,cm,w,color=BLUE,  alpha=1.0,edgecolor='white',lw=1.2,
       yerr=[1.96*s for s in cm_se],error_kw=dict(capsize=5,lw=1.5,color='#555'),label='Control')
ax.bar(x+w/2,tm,w,color=ORANGE,alpha=1.0,edgecolor='white',lw=1.2,
       yerr=[1.96*s for s in tm_se],error_kw=dict(capsize=5,lw=1.5,color='#555'),label='Treatment')
for start,end,lbl,col in [(0,3,'Sympathy\n(E)',ORANGE),(4,6,'Purchase\nIntention (F)',GREEN),(7,11,'FWA\n(G)',PURPLE)]:
    ax.axvspan(start-.5,end+.5,alpha=.07,color=col,zorder=0)
    ax.text((start+end)/2,5.3,lbl,ha='center',fontsize=10,color=col,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_m,fontsize=10)
ax.set_ylim(0,5.6); ax.set_ylabel('Mean Score (1–5)')
ax.set_title('(a)  Per-Item Means: Sympathy / PI / FWA',pad=10)
ax.legend(framealpha=.9); ax.axhline(3,color='gray',lw=1,ls=':',alpha=.5)

ax=axes[1]; ax.set_facecolor(LIGHT)
mc_cm=[ctrl['C_1'].mean(),ctrl['C_2'].mean(),ctrl['C_3'].mean()]
mc_tm=[trt['Q34_1'].mean(),trt['Q34_2'].mean(),trt['Q34_3'].mean()]
mc_cse=[ctrl['C_1'].sem(),ctrl['C_2'].sem(),ctrl['C_3'].sem()]
mc_tse=[trt['Q34_1'].sem(),trt['Q34_2'].sem(),trt['Q34_3'].sem()]
x2=np.arange(3)
ax.bar(x2-w/2,mc_cm,w,color=BLUE,  alpha=1.0,edgecolor='white',lw=1.2,
       yerr=[1.96*s for s in mc_cse],error_kw=dict(capsize=5,lw=1.5,color='#555'),label='Control')
ax.bar(x2+w/2,mc_tm,w,color=ORANGE,alpha=1.0,edgecolor='white',lw=1.2,
       yerr=[1.96*s for s in mc_tse],error_kw=dict(capsize=5,lw=1.5,color='#555'),label='Treatment')
for xi in x2:
    _,pv=stats.ttest_ind(trt[['Q34_1','Q34_2','Q34_3']].iloc[:,xi].dropna(),
                          ctrl[['C_1','C_2','C_3']].iloc[:,xi].dropna())
    ax.text(xi,max(mc_cm[xi],mc_tm[xi])+.25,pstar(pv),ha='center',fontsize=14,fontweight='bold',color='#333')
ax.set_xticks(x2); ax.set_xticklabels(['C1\n"Seems to talk"','C2\n"Human char."','C3\n"Communicating"'],fontsize=10)
ax.set_ylim(0,5.6); ax.set_ylabel('Mean Score (1–5)')
ax.set_title('(b)  Manipulation Check: Per-Item Means',pad=10)
ax.legend(framealpha=.9); ax.axhline(3,color='gray',lw=1,ls=':',alpha=.5)
ax.text(.98,.02,'*** p < .001',transform=ax.transAxes,ha='right',fontsize=10,color='#555')
plt.tight_layout(pad=2)
plt.savefig('fig54_items.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig54_items.png")
