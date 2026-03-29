"""fig53_reliability.py — Figure 5.3: Scale Reliability & Correlation Matrix"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'
PURPLE='#9B85B5'; PINK='#C98BAD'; CYAN='#5FBFCF'; LIGHT='white'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
df['manip_check']=np.where(df['condition']=='condition = treatment',
    df[['Q34_1','Q34_2','Q34_3']].mean(axis=1),df[['C_1','C_2','C_3']].mean(axis=1))
df['perc_quality']=df[['D_1','D_2','D_3','D_4']].mean(axis=1)
df['sympathy']=df[['E_1','E_2','E_3','E_4']].mean(axis=1)
df['purchase_int']=df[['F_1','F_2','F_3']].mean(axis=1)
df['fwa']=df[['G_1','G_2','G_3','G_4','G_5']].mean(axis=1)
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']

def cronbach(d):
    dd=d.dropna(); n=dd.shape[1]
    return round((n/(n-1))*(1-dd.var(ddof=1).sum()/dd.sum(axis=1).var(ddof=1)),3)

mc_c=ctrl[['C_1','C_2','C_3']].copy()
mc_t=trt[['Q34_1','Q34_2','Q34_3']].copy(); mc_t.columns=['C_1','C_2','C_3']
alphas=[cronbach(pd.concat([mc_c,mc_t])),cronbach(df[['D_1','D_2','D_3','D_4']]),
        cronbach(df[['E_1','E_2','E_3','E_4']]),cronbach(df[['F_1','F_2','F_3']]),
        cronbach(df[['G_1','G_2','G_3','G_4','G_5']])]
alabs=['MC (C)\n3 items','Perc. Quality (D)\n4 items','Sympathy (E)\n4 items',
       'Purch. Intent. (F)\n3 items','FWA (G)\n5 items']
acols=[CYAN,PURPLE,ORANGE,GREEN,PINK]

fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor(LIGHT)
ax=axes[0]; ax.set_facecolor(LIGHT)
bars=ax.barh(alabs,alphas,color=acols,alpha=1.0,edgecolor='white',lw=1.5,height=.55)
ax.axvline(.70,color='red',ls='--',lw=2,alpha=.7,label='α = .70 (minimum)')
ax.axvline(.80,color=ORANGE,ls=':',lw=2,alpha=.7,label='α = .80 (good)')
for bar,val,col in zip(bars,alphas,acols):
    ax.text(val+.01,bar.get_y()+bar.get_height()/2,f'α = {val:.3f}',
            va='center',fontsize=11,fontweight='bold',color=col)
ax.set_xlim(0,1.1); ax.set_xlabel("Cronbach's α")
ax.legend(framealpha=.9); ax.set_title("(a)  Internal Consistency (Cronbach's α)",pad=10); ax.invert_yaxis()

ax=axes[1]; ax.set_facecolor(LIGHT)
key=['manip_check','perc_quality','sympathy','purchase_int','fwa']
clabs=['MC\n(C)','Perc.\nQuality (D)','Sympathy\n(E)','Purch.\nIntent. (F)','FWA\n(G)']
corr=df[key].corr().values
im=ax.imshow(corr,cmap='RdYlGn',vmin=-1,vmax=1,aspect='auto')
cbar=plt.colorbar(im,ax=ax,shrink=.80,pad=.03); cbar.set_label('Pearson r',fontsize=11)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(clabs,fontsize=10); ax.set_yticklabels(clabs,fontsize=10)
for i in range(5):
    for j in range(5):
        v=corr[i,j]; fc='white' if abs(v)>.45 else '#222'
        ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=12,fontweight='bold',color=fc)
ax.set_title('(b)  Inter-Scale Correlation Matrix',pad=10)
plt.tight_layout(pad=1.5)
plt.savefig('fig53_reliability.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig53_reliability.png")
