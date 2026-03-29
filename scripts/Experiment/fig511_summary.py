"""fig511_summary.py — Figure 5.11: Summary Dashboard"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'
PURPLE='#9B85B5'; CYAN='#5FBFCF'; LIGHT='white'
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

def pstar(p):
    if p<.001: return '***'
    if p<.01: return '**'
    if p<.05: return '*'
    return 'n.s.'
def r2(X,y):
    b,_,_,_=lstsq(X,y,rcond=None)
    return 1-((y-X@b)**2).sum()/((y-y.mean())**2).sum()

Xh3=np.column_stack([np.ones(200),df['condition_bin'],df['sym_c'],df['fwa_c'],df['sym_x_fwa'],df['pq_c']])
yh3=df['purchase_int'].values

fig=plt.figure(figsize=(18,9)); fig.patch.set_facecolor('white')
gs=GridSpec(2,3,figure=fig,hspace=.42,wspace=.35)

ax=fig.add_subplot(gs[0,0]); ax.set_facecolor('white')
bars=ax.bar(['Manipulation\nCheck','Framing →\nSympathy'],[1.061,0.554],
            color=[CYAN,GREEN],alpha=1.0,edgecolor='white',lw=1.5,width=.45)
for bar,d in zip(bars,[1.061,0.554]):
    ax.text(bar.get_x()+bar.get_width()/2,d+.02,f'd={d:.3f}',ha='center',fontsize=12,fontweight='bold')
ax.axhline(.2,color='gray',lw=1.5,ls=':',alpha=.6,label='Small (d=.2)')
ax.axhline(.5,color=ORANGE,lw=1.5,ls='--',alpha=.7,label='Medium (d=.5)')
ax.axhline(.8,color='red',lw=1.5,ls='-.',alpha=.6,label='Large (d=.8)')
ax.set_ylim(0,1.35); ax.set_ylabel("Cohen's d")
ax.legend(fontsize=9,framealpha=.9); ax.set_title("(a)  Effect Sizes (Cohen's d)",pad=10)

ax=fig.add_subplot(gs[0,1]); ax.set_facecolor('white')
path_names=["Path a\nX → M","Path b\nM → Y|X,PQ","Direct c'\nX → Y|M,PQ"]
betas_fp=[0.566,0.063,0.034]; ses_fp=[0.145,0.038,0.080]; pvals_fp=[0.0001,0.0940,0.6710]
cols_fp=[GREEN if p<.05 else '#BBBBBB' for p in pvals_fp]
for yi,b,se,p,col in zip([3,2,1],betas_fp,ses_fp,pvals_fp,cols_fp):
    ci_l,ci_h=b-1.96*se,b+1.96*se
    ax.plot([ci_l,ci_h],[yi,yi],color=col,lw=4,alpha=1.0,solid_capstyle='round')
    ax.scatter([b],[yi],s=120,color=col,zorder=3)
    ax.text(max(ci_h,.01)+.03,yi,f'β={b:.3f} {pstar(p)}',va='center',fontsize=11,color=col,fontweight='bold')
ax.axvline(0,color='black',lw=1.8,ls='--',alpha=.7)
ax.set_yticks([3,2,1]); ax.set_yticklabels(path_names,fontsize=11)
ax.set_xlabel('Unstandardised β (± 95% CI)')
ax.set_title('(b)  Path Coefficients Forest Plot',pad=10); ax.set_xlim(-.3,.9)

ax=fig.add_subplot(gs[0,2]); ax.set_facecolor('white')
r2_vals=[r2(np.column_stack([np.ones(200),df['condition_bin']]),yh3),
         r2(Xh3[:,:5],yh3), r2(Xh3,yh3)]
bars_r2=ax.bar(['Model 1\nX → Y','Model 2\nX+M+W+PQ','Model 3\n+M×W'],
               r2_vals,color=[ORANGE,GREEN,PURPLE],alpha=1.0,edgecolor='white',lw=1.5,width=.5)
for bar,v in zip(bars_r2,r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+.005,f'R²={v:.3f}',ha='center',fontsize=12,fontweight='bold')
ax.set_ylim(0,.45); ax.set_ylabel('R² (Explained Variance)')
ax.set_title('(c)  Model R² by Regression',pad=10)

ax=fig.add_subplot(gs[1,:]); ax.set_facecolor('white'); ax.axis('off')
rows=[
    ['H1','Framing → Purchase Intention','t(198)=1.679','d=0.238','p=.095','✗ Not Supported','#CC3300'],
    ['H2','Indirect via Sympathy','a×b=0.037','CI=[−0.005, 0.099]','—','✗ Not Supported','#CC3300'],
    ['H3','FWA moderates path b','β=0.070','IMM=0.041','p=.186','✗ Not Supported','#CC3300'],
    ['MC','Manipulation check','t(198)=7.469','d=1.061','p<.001','✓ Supported',GREEN],
]
col_x=[.02,.10,.50,.63,.77,.87]
for cx,lbl in zip(col_x,['Hyp.','Content','Statistic','Effect / CI','p-value','Verdict']):
    ax.text(cx,.92,lbl,transform=ax.transAxes,fontsize=12,fontweight='bold',color='white',va='center',
            bbox=dict(boxstyle='round,pad=0.3',fc='#333',ec='none'))
for i,(h,content,stat,eff,pv,verdict,vc) in enumerate(rows):
    y=.70-i*.18; bg='#ffffff' if i%2==0 else '#f0f4f8'
    ax.add_patch(plt.Rectangle((0,y-.07),1,.16,transform=ax.transAxes,facecolor=bg,edgecolor='none',zorder=0))
    for cx,v in zip(col_x,[h,content,stat,eff,pv,verdict]):
        ax.text(cx,y,v,transform=ax.transAxes,fontsize=11,va='center',
                color=vc if v==verdict else '#222',fontweight='bold' if v==verdict else 'normal')
ax.set_title('(d)  Hypothesis Testing Summary',fontsize=14,fontweight='bold',pad=12)
plt.savefig('fig511_summary.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Saved: fig511_summary.png")
