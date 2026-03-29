"""fig513_jn.py — Figure 5.13: Johnson-Neyman Region of Significance"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORANGE='#E8956D'; PURPLE='#9B85B5'; LIGHT='#f8f9fc'
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
fwa_mean=df['fwa'].mean(); fwa_sd=df['fwa'].std()

Xa=np.column_stack([np.ones(200),df['condition_bin']])
ba,_,_,_=lstsq(Xa,df['sympathy'].values,rcond=None); path_a=ba[1]
Xh3=np.column_stack([np.ones(200),df['condition_bin'],df['sym_c'],df['fwa_c'],df['sym_x_fwa'],df['pq_c']])
bh3,_,_,_=lstsq(Xh3,df['purchase_int'].values,rcond=None)

np.random.seed(42); fwa_range=np.linspace(df['fwa'].min(),df['fwa'].max(),150)
w_c=fwa_range-fwa_mean; ie_by_w=np.zeros((3000,len(fwa_range)))
for bi in range(3000):
    idx=np.random.choice(200,200,replace=True)
    bd=df.iloc[idx].dropna(subset=['condition_bin','sympathy','purchase_int','fwa','pq_c'])
    Xa_=np.column_stack([np.ones(len(bd)),bd['condition_bin']])
    ba_,_,_,_=lstsq(Xa_,bd['sympathy'].values,rcond=None)
    sc=bd['sympathy'].values-bd['sympathy'].mean(); fc=bd['fwa'].values-fwa_mean
    Xb_=np.column_stack([np.ones(len(bd)),bd['condition_bin'],sc,fc,sc*fc,bd['pq_c']])
    bb_,_,_,_=lstsq(Xb_,bd['purchase_int'].values,rcond=None)
    for wi,wv in enumerate(w_c): ie_by_w[bi,wi]=ba_[1]*(bb_[2]+bb_[4]*wv)
ci_lo_jn=np.percentile(ie_by_w,2.5,axis=0); ci_hi_jn=np.percentile(ie_by_w,97.5,axis=0)
cond_ie_line=path_a*(bh3[2]+bh3[4]*w_c)

fig,ax=plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
ax.plot(fwa_range,cond_ie_line,color=PURPLE,lw=3,label='Conditional IE (a×b)')
ax.fill_between(fwa_range,ci_lo_jn,ci_hi_jn,alpha=.35,color=PURPLE,label='95% BC CI')
ax.axhline(0,color='black',lw=1.8,ls='--',alpha=.7,label='Zero (null)')
for val,lbl in [(fwa_mean-fwa_sd,'M−1SD'),(fwa_mean,'Mean'),(fwa_mean+fwa_sd,'M+1SD')]:
    ax.axvline(val,color=ORANGE,lw=1.6,ls=':',alpha=.7)
    ax.text(val,ci_lo_jn.min()-.008,lbl,ha='center',fontsize=10,color=ORANGE,va='top')
ax.set_xlabel('Food Waste Awareness (G)'); ax.set_ylabel('Conditional Indirect Effect')
ax.set_title('Johnson–Neyman Region of Significance',pad=10); ax.legend(framealpha=.9)
ax.text(.5,.05,'95% CI band spans zero throughout — H3 not supported at any FWA level',
        transform=ax.transAxes,ha='center',fontsize=11,fontstyle='italic',color='#CC3300',
        bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#CC3300',alpha=.9))
plt.tight_layout()
plt.savefig('fig513_jn.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig513_jn.png")
