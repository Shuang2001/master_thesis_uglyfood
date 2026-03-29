"""fig510_momediation.py — Figure 5.10: H3 Moderated Mediation"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'; LIGHT='#f8f9fc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
df['perc_quality']=df[['D_1','D_2','D_3','D_4']].mean(axis=1)
df['sympathy']=df[['E_1','E_2','E_3','E_4']].mean(axis=1)
df['purchase_int']=df[['F_1','F_2','F_3']].mean(axis=1)
df['fwa']=df[['G_1','G_2','G_3','G_4','G_5']].mean(axis=1)
df['condition_bin']=(df['condition']=='condition = treatment').astype(int)
df['pq_c']=df['perc_quality']-df['perc_quality'].mean()
df['fwa_c']=df['fwa']-df['fwa'].mean()
df['sym_c']=df['sympathy']-df['sympathy'].mean()
df['sym_x_fwa']=df['sym_c']*df['fwa_c']
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']
fwa_mean=df['fwa'].mean(); fwa_sd=df['fwa'].std()

Xh3=np.column_stack([np.ones(200),df['condition_bin'],df['sym_c'],df['fwa_c'],df['sym_x_fwa'],df['pq_c']])
bh3,_,_,_=lstsq(Xh3,df['purchase_int'].values,rcond=None)
n3,k3=200,6; yh3=df['purchase_int'].values
sig2h3=((yh3-Xh3@bh3)**2).sum()/(n3-k3)
seh3=np.sqrt(sig2h3*np.linalg.inv(Xh3.T@Xh3).diagonal())
th3_=bh3/seh3; ph3_=2*stats.t.sf(np.abs(th3_),df=n3-k3)
int_beta=bh3[4]; int_p=ph3_[4]

lev_names=['Low FWA\n(M−1SD)','Mean FWA','High FWA\n(M+1SD)']
cie_means=[-0.014,0.013,0.041]; cie_lo=[-0.109,-0.037,-0.004]; cie_hi=[0.072,0.069,0.111]
lev_cols=[BLUE,ORANGE,GREEN]

fig,axes=plt.subplots(1,3,figsize=(21,7)); fig.patch.set_facecolor(LIGHT)

ax=axes[0]; ax.set_facecolor(LIGHT)
sym_range=np.linspace(df['sympathy'].min(),df['sympathy'].max(),150)
sym_c_range=sym_range-df['sympathy'].mean()
for wv,col,lbl in [(-fwa_sd,BLUE,'Low FWA (M−1SD)'),(0,ORANGE,'Mean FWA'),(fwa_sd,GREEN,'High FWA (M+1SD)')]:
    pi_pred=bh3[0]+0.5*bh3[1]+bh3[2]*sym_c_range+bh3[3]*wv+bh3[4]*sym_c_range*wv
    ax.plot(sym_range,pi_pred,lw=2.8,color=col,label=lbl)
ax.set_xlabel('Sympathy Score (E)'); ax.set_ylabel('Predicted Purchase Intention (F)')
ax.set_title('(a)  Interaction Plot: Sympathy × FWA',pad=10)
ax.legend(loc='upper left',framealpha=.9); ax.set_ylim(2,5)
ax.text(.97,.05,f'M×W: β={int_beta:.3f}, p={int_p:.3f} (n.s.)',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=10,bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#aaa',alpha=.95))

ax=axes[1]; ax.set_facecolor(LIGHT)
for xi,(m,lo,hi,col) in enumerate(zip(cie_means,cie_lo,cie_hi,lev_cols),start=1):
    ax.bar(xi,m,width=.52,color=col,alpha=1.0,edgecolor='white',lw=1.5)
    ax.plot([xi]*2,[lo,hi],'k-',lw=2.5,zorder=3)
    for cap in [lo,hi]: ax.plot([xi-.18,xi+.18],[cap,cap],'k-',lw=2.5,zorder=3)
    ax.text(xi,hi+.005,f'[{lo:.3f},{hi:.3f}]',ha='center',fontsize=10,color='#333')
ax.axhline(0,color='black',lw=2,ls='--',alpha=.7)
ax.set_xticks([1,2,3]); ax.set_xticklabels([l.replace('\n',' ') for l in lev_names],fontsize=11)
ax.set_ylabel('Conditional Indirect Effect')
ax.set_title('(b)  Conditional Indirect Effects\n± 95% BC Bootstrap CI',pad=10)
ax.text(.5,-.16,'H3 not supported (all CIs include zero)',transform=ax.transAxes,
        ha='center',fontsize=11,fontstyle='italic',color='#CC3300')

ax=axes[2]; ax.set_facecolor(LIGHT)
for dd,col,lab in [(ctrl,BLUE,'Control'),(trt,ORANGE,'Treatment')]:
    d=dd.dropna(subset=['fwa','sympathy'])
    ax.scatter(d['fwa'],d['sympathy'],alpha=1.0,s=28,color=col,label=lab)
    z=np.polyfit(d['fwa'],d['sympathy'],1); xl=np.linspace(d['fwa'].min(),d['fwa'].max(),100)
    ax.plot(xl,np.poly1d(z)(xl),color=col,lw=2.5)
ax.set_xlabel('Food Waste Awareness (G)'); ax.set_ylabel('Sympathy (E)')
ax.set_title('(c)  FWA vs. Sympathy by Condition',pad=10); ax.legend(framealpha=.9)
plt.tight_layout(pad=1.5)
plt.savefig('fig510_momediation.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig510_momediation.png")
