"""fig59_items_ef.py — Figure 5.9: Item-Level Sympathy (E) and PI (F)"""
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; ORANGE='#E8956D'; LIGHT='#f8f9fc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']
nc,nt=len(ctrl),len(trt)

fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor(LIGHT)
w=0.38

ax=axes[0]; ax.set_facecolor(LIGHT)
e_items=['E_1','E_2','E_3','E_4']
e_labels=['E1\n"Distressed"','E2\n"Pitiful"','E3\n"Urge to\nhelp"','E4\n"Feel sorry"']
e_pvals=[0.0196,0.0017,0.0000,0.0004]
cm=[ctrl[i].mean() for i in e_items]; ce=[ctrl[i].sem() for i in e_items]
tm=[trt[i].mean()  for i in e_items]; te=[trt[i].sem()  for i in e_items]
x=np.arange(4)
ax.bar(x-w/2,cm,w,color=BLUE,  alpha=1.0,edgecolor='white',lw=1.3,
       yerr=[1.96*s for s in ce],error_kw=dict(capsize=6,lw=1.8,color='#444'),label=f'Control (n={nc})')
ax.bar(x+w/2,tm,w,color=ORANGE,alpha=1.0,edgecolor='white',lw=1.3,
       yerr=[1.96*s for s in te],error_kw=dict(capsize=6,lw=1.8,color='#444'),label=f'Treatment (n={nt})')
for xi,(c,t,pv) in enumerate(zip(cm,tm,e_pvals)):
    star='***' if pv<.001 else ('**' if pv<.01 else '*')
    ax.text(xi,max(c,t)+.28,star,ha='center',fontsize=14,fontweight='bold',color='#222')
ax.set_xticks(x); ax.set_xticklabels(e_labels,fontsize=10)
ax.set_ylim(1.5,4.5); ax.set_ylabel('Mean Score (1–5)')
ax.set_title('(a)  Sympathy (E): All items significant',pad=10)
ax.legend(framealpha=.9); ax.axhline(3,color='gray',lw=1,ls=':',alpha=.5)

ax=axes[1]; ax.set_facecolor(LIGHT)
f_items=['F_1','F_2','F_3']
f_labels=['F1\n"Would buy"','F2\n"Likely to\nchoose"','F3\n"Would seek\nout"']
f_pvals=[0.7218,0.4686,0.0078]
cm=[ctrl[i].mean() for i in f_items]; ce=[ctrl[i].sem() for i in f_items]
tm=[trt[i].mean()  for i in f_items]; te=[trt[i].sem()  for i in f_items]
x=np.arange(3)
ax.bar(x-w/2,cm,w,color=BLUE,  alpha=1.0,edgecolor='white',lw=1.3,
       yerr=[1.96*s for s in ce],error_kw=dict(capsize=6,lw=1.8,color='#444'),label=f'Control (n={nc})')
ax.bar(x+w/2,tm,w,color=ORANGE,alpha=1.0,edgecolor='white',lw=1.3,
       yerr=[1.96*s for s in te],error_kw=dict(capsize=6,lw=1.8,color='#444'),label=f'Treatment (n={nt})')
for xi,(c,t,pv) in enumerate(zip(cm,tm,f_pvals)):
    star='n.s.' if pv>=.05 else ('**' if pv<.01 else '*')
    col='#999' if star=='n.s.' else '#222'
    ax.text(xi,max(c,t)+.18,star,ha='center',fontsize=14,fontweight='bold',color=col)
ax.set_xticks(x); ax.set_xticklabels(f_labels,fontsize=10)
ax.set_ylim(1.5,4.8); ax.set_ylabel('Mean Score (1–5)')
ax.set_title('(b)  Purchase Intention (F): Only F3 significant',pad=10)
ax.legend(framealpha=.9); ax.axhline(3,color='gray',lw=1,ls=':',alpha=.5)
plt.tight_layout(pad=2)
plt.savefig('fig59_items_ef.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig59_items_ef.png")
