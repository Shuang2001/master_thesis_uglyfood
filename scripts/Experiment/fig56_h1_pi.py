"""fig56_h1_pi.py — Figure 5.6: H1 Purchase Intention (n.s.)"""
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
df['purchase_int']=df[['F_1','F_2','F_3']].mean(axis=1)
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']
pi_c=ctrl['purchase_int'].dropna(); pi_t=trt['purchase_int'].dropna()
nc,nt=len(ctrl),len(trt)

fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor(LIGHT)
ax=axes[0]; ax.set_facecolor(LIGHT)
parts=ax.violinplot([pi_c,pi_t],positions=[1,2],showmedians=False,showextrema=False)
for body,col in zip(parts['bodies'],[BLUE,ORANGE]):
    body.set_facecolor(col); body.set_alpha(1.0); body.set_edgecolor(col); body.set_linewidth(1.5)
ax.boxplot([pi_c,pi_t],positions=[1,2],widths=.13,patch_artist=True,
           boxprops=dict(facecolor='white',linewidth=1.8),medianprops=dict(color='#222',linewidth=2.5),
           whiskerprops=dict(linewidth=1.5),capprops=dict(linewidth=1.5),
           flierprops=dict(marker='o',markersize=4,alpha=.4))
for xi,vals,col in [(1,pi_c,BLUE),(2,pi_t,ORANGE)]:
    ax.scatter([xi],[vals.mean()],s=90,color=col,zorder=5,marker='D')
    ax.text(xi,.62,f'M={vals.mean():.3f}\nSD={vals.std():.3f}',ha='center',fontsize=10,fontweight='bold',color=col)
ax.text(1.5,5.1,'t(198) = 1.679, p = .095 (n.s.)  |  d = 0.238',ha='center',fontsize=11,color='#555')
ax.set_xticks([1,2]); ax.set_xticklabels(['Control\n(Descriptive)','Treatment\n(Anthropomorphic)'])
ax.set_ylim(.5,5.5); ax.set_ylabel('Purchase Intention (1–5)')
ax.set_title('(a)  Violin + Box Plot',pad=10); ax.axhline(3,color='gray',lw=1.2,ls=':',alpha=.5)

ax=axes[1]; ax.set_facecolor(LIGHT)
x_kde=np.linspace(.5,5.5,300)
for vals,col,lbl in [(pi_c,BLUE,f'Control  n={nc} | M={pi_c.mean():.3f}'),
                      (pi_t,ORANGE,f'Treatment n={nt} | M={pi_t.mean():.3f}')]:
    kde=gaussian_kde(vals,bw_method=.45)
    ax.fill_between(x_kde,kde(x_kde),alpha=.35,color=col)
    ax.plot(x_kde,kde(x_kde),lw=2.8,color=col,label=lbl)
    ax.axvline(vals.mean(),color=col,lw=2,ls='--')
ax.axvline(3,color='gray',lw=1.5,ls=':',alpha=.5,label='Neutral (3)')
ax.set_xlabel('Purchase Intention Score'); ax.set_ylabel('Density')
ax.set_title('(b)  Score Distribution (KDE)',pad=10); ax.legend(framealpha=.9)
plt.tight_layout()
plt.savefig('fig56_h1_pi.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig56_h1_pi.png")
