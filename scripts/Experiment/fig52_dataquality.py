"""fig52_dataquality.py — Figure 5.2: Data Quality & Variable Relationships"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import stats; from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; ORANGE='#E8956D'; LIGHT='white'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold','xtick.labelsize':11,
    'ytick.labelsize':11,'legend.fontsize':10,'figure.dpi':180,
    'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--'})

df=pd.read_csv('survey.csv')
df['sympathy']=df[['E_1','E_2','E_3','E_4']].mean(axis=1)
df['purchase_int']=df[['F_1','F_2','F_3']].mean(axis=1)
df['fwa']=df[['G_1','G_2','G_3','G_4','G_5']].mean(axis=1)
ctrl=df[df['condition']=='condition = control']
trt=df[df['condition']=='condition = treatment']

def plab(p): return 'p < .001' if p<.001 else f'p = {p:.3f}'

fig,axes=plt.subplots(1,3,figsize=(21,7)); fig.patch.set_facecolor(LIGHT)

ax=axes[0]; ax.set_facecolor(LIGHT)
dur_c=ctrl['Duration (in seconds)'].clip(upper=1200)
dur_t=trt['Duration (in seconds)'].clip(upper=1200)
bins=np.linspace(100,1200,40)
ax.hist(dur_c,bins=bins,alpha=1.0,color=BLUE,  density=True,edgecolor='white',label=f'Control  M={dur_c.mean():.0f}s')
ax.hist(dur_t,bins=bins,alpha=1.0,color=ORANGE,density=True,edgecolor='white',label=f'Treatment M={dur_t.mean():.0f}s')
xd=np.linspace(100,1200,300)
for vals,col in [(dur_c,BLUE),(dur_t,ORANGE)]:
    if vals.nunique()>1: ax.plot(xd,gaussian_kde(vals,bw_method=.25)(xd),color=col,lw=2.5)
ax.axvline(120,color='red',lw=2,ls='--',alpha=.7,label='120 s threshold')
ax.set_xlabel('Completion Time (seconds)'); ax.set_ylabel('Density')
ax.set_title('(a)  Completion Time Distribution',pad=10); ax.legend(framealpha=.9)

for ax,(xvar,yvar,title) in zip(axes[1:],[
    ('sympathy','purchase_int','(b)  Sympathy vs. Purchase Intention'),
    ('fwa','purchase_int','(c)  FWA vs. Purchase Intention')]):
    ax.set_facecolor(LIGHT)
    for dd,col,lab in [(ctrl,BLUE,'Control'),(trt,ORANGE,'Treatment')]:
        d=dd.dropna(subset=[xvar,yvar])
        ax.scatter(d[xvar],d[yvar],alpha=1.0,s=28,color=col,label=lab)
        z=np.polyfit(d[xvar],d[yvar],1); xl=np.linspace(d[xvar].min(),d[xvar].max(),100)
        ax.plot(xl,np.poly1d(z)(xl),color=col,lw=2.5)
        rv,pv=stats.pearsonr(d[xvar],d[yvar])
        ypos=.93 if col==BLUE else .83
        ax.text(.03,ypos,f'{lab}: r={rv:.3f}, {plab(pv)}',transform=ax.transAxes,
                fontsize=10,color=col,bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=col,alpha=.9))
    xlabs={'sympathy':'Sympathy (E)','fwa':'Food Waste Awareness (G)'}
    ax.set_xlabel(xlabs[xvar]); ax.set_ylabel('Purchase Intention (F)')
    ax.set_title(title,pad=10); ax.legend(framealpha=.9)

plt.tight_layout(pad=2)
plt.savefig('fig52_dataquality.png',dpi=180,bbox_inches='tight',facecolor=LIGHT)
plt.close(); print("Saved: fig52_dataquality.png")
