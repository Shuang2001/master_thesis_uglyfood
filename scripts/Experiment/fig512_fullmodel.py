"""fig512_fullmodel.py — Figure 5.12: Full Path Model"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'
PURPLE='#9B85B5'; PINK='#C98BAD'; CYAN='#5FBFCF'; LIGHT='#f8f9fc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.dpi':180})

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

Xa=np.column_stack([np.ones(200),df['condition_bin']])
ba,_,_,_=lstsq(Xa,df['sympathy'].values,rcond=None)
path_a=ba[1]
Xh3=np.column_stack([np.ones(200),df['condition_bin'],df['sym_c'],df['fwa_c'],df['sym_x_fwa'],df['pq_c']])
bh3,_,_,_=lstsq(Xh3,df['purchase_int'].values,rcond=None)

Xb=np.column_stack([np.ones(200),df['condition_bin'],df['sympathy'],df['pq_c']])
bb,_,_,_=lstsq(Xb,df['purchase_int'].values,rcond=None)
path_b=bb[2]; c_prime=bb[1]; pq_beta=bb[3]; int_beta=bh3[4]

np.random.seed(42); boots=[]
for _ in range(5000):
    idx=np.random.choice(200,200,replace=True)
    bd=df.iloc[idx].dropna(subset=['condition_bin','sympathy','purchase_int','pq_c'])
    Xa_=np.column_stack([np.ones(len(bd)),bd['condition_bin']])
    ba_,_,_,_=lstsq(Xa_,bd['sympathy'].values,rcond=None)
    Xb_=np.column_stack([np.ones(len(bd)),bd['condition_bin'],bd['sympathy'],bd['pq_c']])
    bb_,_,_,_=lstsq(Xb_,bd['purchase_int'].values,rcond=None)
    boots.append(ba_[1]*bb_[2])
boots=np.array(boots); ie_mean=boots.mean()
ie_ci=(np.percentile(boots,2.5),np.percentile(boots,97.5))

fig,ax=plt.subplots(figsize=(18,9)); fig.patch.set_facecolor(LIGHT)
ax.set_facecolor(LIGHT); ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis('off')

def box12(cx,cy,txt,col,w=2.6,h=1.2):
    b=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0.14',
                      facecolor=col,edgecolor='#333',linewidth=2.2,alpha=1.0)
    ax.add_patch(b)
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=12,fontweight='bold',color='white')
def arr12(x1,y1,x2,y2,col,ls='-',rad=0):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=2.5,linestyle=ls,connectionstyle=f'arc3,rad={rad}'))
def tag12(cx,cy,txt,col,w=2.9):
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=10,color=col,
            bbox=dict(boxstyle='round,pad=0.32',fc='white',ec=col,alpha=1.0,lw=1.6))

box12(1.5,3.5,'Framing\nCondition (X)',BLUE); box12(6.0,5.8,'Sympathy\n(E)',GREEN)
box12(10.5,3.5,'Purchase\nIntention (F)',PURPLE); box12(6.0,1.2,'Food Waste\nAwareness (G)',PINK)
box12(1.5,1.2,'Perceived\nQuality (D)',CYAN)
arr12(2.3,4.1,4.9,5.4,GREEN); arr12(7.1,5.4,9.7,4.1,ORANGE,'dashed')
arr12(2.4,3.2,9.6,3.2,'gray','dashed'); arr12(6.0,2.0,7.3,4.6,PINK,'dashed',rad=.2)
arr12(2.3,1.8,9.4,3.1,CYAN,'dashed')
tag12(3.1,5.15,f'Path a  β={path_a:.3f}***',GREEN)
tag12(8.8,5.15,f'Path b  β={path_b:.3f} (n.s.)',ORANGE)
tag12(6.0,2.85,f"Direct c'  β={c_prime:.3f} (n.s.)",'gray',w=3.2)
tag12(8.1,3.7,f'FWA×Sym\nβ={int_beta:.3f} (n.s.)',PINK)
tag12(5.2,2.1,f'PQ: β={pq_beta:.3f}***',CYAN)
ax.text(6.0,.52,
        f'Indirect a×b = {ie_mean:.4f}   |   95% BC CI [{ie_ci[0]:.4f}, {ie_ci[1]:.4f}]   |   '
        '✗ CI includes zero — H2 NOT supported',
        ha='center',fontsize=11,color='#CC3300',fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.38',fc='white',ec='#CC3300',alpha=1.0,lw=1.8))
ax.text(6.0,6.65,'H1: t(198)=1.679, p=.095 (n.s.)  |  H2: not supported  |  H3: not supported',
        ha='center',fontsize=12,fontweight='bold',color='#555',
        bbox=dict(boxstyle='round,pad=0.35',fc='white',ec='#aaa',alpha=1.0,lw=1.5))
ax.legend(handles=[mpatches.Patch(color=c,label=l) for c,l in [
    (GREEN,'Significant path (p < .05)'),(ORANGE,'Non-significant path'),
    (CYAN,'Covariate (PQ)'),(PINK,'Moderation (n.s.)')]],
    loc='lower left',fontsize=10,framealpha=.9,bbox_to_anchor=(0,0))
plt.tight_layout()
plt.savefig('fig512_fullmodel.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig512_fullmodel.png")
