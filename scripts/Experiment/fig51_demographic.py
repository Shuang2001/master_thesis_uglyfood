"""  fig51_demographics.py  —  run with survey.csv in same folder  """
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import gaussian_kde
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#5B8DB8'; GREEN='#6AAF6A'; ORANGE='#E8956D'
PURPLE='#9B85B5'; PINK='#C98BAD'; CYAN='#5FBFCF'
WHITE='#FFFFFF'

def set_rc():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':12,'axes.titlesize':13,'axes.titleweight':'bold',
        'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
        'figure.dpi':180,'axes.grid':True,'grid.alpha':0.2,'grid.linestyle':'--',
        'axes.facecolor':WHITE,'figure.facecolor':WHITE,
    })

def save(name):
    plt.savefig(name, dpi=180, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    plt.close(); print(f'Saved: {name}')

def pstar(p):
    if p<.001: return '***'
    if p<.01:  return '**'
    if p<.05:  return '*'
    return 'n.s.'

def plab(p): return 'p < .001' if p<.001 else f'p = {p:.3f}'

df = pd.read_csv('survey.csv')
df['manip_check'] = np.where(df['condition']=='condition = treatment',
    df[['Q34_1','Q34_2','Q34_3']].mean(axis=1), df[['C_1','C_2','C_3']].mean(axis=1))
df['perc_quality'] = df[['D_1','D_2','D_3','D_4']].mean(axis=1)
df['sympathy']     = df[['E_1','E_2','E_3','E_4']].mean(axis=1)
df['purchase_int'] = df[['F_1','F_2','F_3']].mean(axis=1)
df['fwa']          = df[['G_1','G_2','G_3','G_4','G_5']].mean(axis=1)
df['condition_bin']= (df['condition']=='condition = treatment').astype(int)
df['pq_c']  = df['perc_quality'] - df['perc_quality'].mean()
df['fwa_c'] = df['fwa']          - df['fwa'].mean()
df['sym_c'] = df['sympathy']     - df['sympathy'].mean()
df['sym_x_fwa'] = df['sym_c'] * df['fwa_c']
ctrl = df[df['condition']=='condition = control']
trt  = df[df['condition']=='condition = treatment']
nc, nt = len(ctrl), len(trt)
fwa_mean = df['fwa'].mean(); fwa_sd = df['fwa'].std()

set_rc()
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=WHITE)
panels = [
    {'ax':axes[0,0],'col':'A1','title':'(a)  Gender (A1)',
     'vals':[1,2,3,4],'labels':['Female\n(1)','Male\n(2)','Other\n(3)','Prefer not\nto say (4)']},
    {'ax':axes[0,1],'col':'A2','title':'(b)  Age Group (A2)',
     'vals':[1,2,3,4,5],'labels':['18–25','26–35','36–45','46–55','55+']},
    {'ax':axes[1,0],'col':'A3','title':'(c)  Monthly Disposable Income (A3)',
     'vals':[1,2,3],'labels':['< 6,000 CNY\n(1)','6,000–14,000\nCNY (2)','> 14,000 CNY\n(3)']},
    {'ax':axes[1,1],'col':'A4','title':'(d)  Weekly Fresh Produce\nPurchase Frequency (A4)',
     'vals':[1,2,3,4],'labels':['Almost\nnever (1)','1–2 times\n(2)','3–4 times\n(3)','5+ times\n(4)']},
]
w = 0.38
for p in panels:
    ax = p['ax']; ax.set_facecolor(WHITE)
    vc_c=ctrl[p['col']].value_counts(); vc_t=trt[p['col']].value_counts()
    cp=[vc_c.get(v,0)/nc*100 for v in p['vals']]
    tp=[vc_t.get(v,0)/nt*100 for v in p['vals']]
    x=np.arange(len(p['vals']))
    bc=ax.bar(x-w/2,cp,w,color=BLUE,  alpha=1.0,edgecolor=WHITE,lw=1.5,label=f'Control (n={nc})')
    bt=ax.bar(x+w/2,tp,w,color=ORANGE,alpha=1.0,edgecolor=WHITE,lw=1.5,label=f'Treatment (n={nt})')
    for bar in list(bc)+list(bt):
        h=bar.get_height()
        if h>=2: ax.text(bar.get_x()+bar.get_width()/2,h+0.8,f'{h:.1f}%',
                         ha='center',va='bottom',fontsize=8.5,color='#333')
    ax.set_xticks(x); ax.set_xticklabels(p['labels'],fontsize=10)
    ax.set_ylabel('Percentage (%)'); ax.set_title(p['title'],pad=10)
    ax.set_ylim(0,max(max(cp),max(tp))*1.25); ax.legend(framealpha=1.0,loc='upper right')
plt.tight_layout(pad=2.5)
save('fig51_demographics.png')
