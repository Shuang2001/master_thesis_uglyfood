"""fig55_manip.py — Figure 5.5: Manipulation Check"""
import pandas as pd, numpy as np
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; ORANGE='#E8956D'; LIGHT='#f8f9fc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
df['manip_check']=np.where(df['condition']=='condition = treatment',
    df[['Q34_1','Q34_2','Q34_3']].mean(axis=1),df[['C_1','C_2','C_3']].mean(axis=1))
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']
mc_c=ctrl['manip_check'].dropna(); mc_t=trt['manip_check'].dropna()

fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor(LIGHT)
ax=axes[0]; ax.set_facecolor(LIGHT)
np.random.seed(0)
for xi,vals,col in [(1,mc_c,BLUE),(2,mc_t,ORANGE)]:
    ax.scatter(np.random.normal(xi,.07,len(vals)),vals,alpha=1.0,s=18,color=col,zorder=2)
    m,se=vals.mean(),vals.sem()
    ax.bar(xi,m,width=.45,color=col,alpha=1.0,edgecolor='white',lw=1.5,zorder=1)
    ax.errorbar(xi,m,yerr=1.96*se,fmt='none',color='#333',capsize=9,lw=2.2,zorder=3)
    ax.text(xi,m+1.96*se+.12,f'{m:.3f}',ha='center',fontsize=12,fontweight='bold',color=col)
y_br=4.8; ax.plot([1,2],[y_br,y_br],'k-',lw=1.8)
for xi in [1,2]: ax.plot([xi]*2,[y_br-.07,y_br],'k-',lw=1.8)
ax.text(1.5,y_br+.08,'t(198) = 7.469, p < .001***, d = 1.061',ha='center',fontsize=11)
ax.set_xlim(.4,2.6); ax.set_ylim(1,5.4)
ax.set_xticks([1,2]); ax.set_xticklabels(['Control\n(Descriptive)','Treatment\n(Anthropomorphic)'])
ax.set_ylabel('Manipulation Check Score (1–5)')
ax.set_title('(a)  Mean Score ± 95% CI + Individual Data',pad=10)

ax=axes[1]; ax.set_facecolor(LIGHT)
x_kde=np.linspace(1,5,300)
for vals,col,lbl in [(mc_c,BLUE,f'Control  (M={mc_c.mean():.3f}, SD={mc_c.std():.3f})'),
                      (mc_t,ORANGE,f'Treatment (M={mc_t.mean():.3f}, SD={mc_t.std():.3f})')]:
    kde=gaussian_kde(vals,bw_method=.5)
    ax.fill_between(x_kde,kde(x_kde),alpha=.35,color=col)
    ax.plot(x_kde,kde(x_kde),lw=2.8,color=col,label=lbl)
    ax.axvline(vals.mean(),color=col,lw=2,ls='--')
ax.axvline(3,color='gray',lw=1.5,ls=':',alpha=.6,label='Scale midpoint (3)')
ax.set_xlabel('Manipulation Check Score'); ax.set_ylabel('Density')
ax.set_title('(b)  Score Distribution (KDE)',pad=10); ax.legend(loc='upper left',framealpha=.9)
plt.tight_layout()
plt.savefig('fig55_manip.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig55_manip.png")
