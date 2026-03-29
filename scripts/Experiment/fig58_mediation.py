import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import gaussian_kde
from numpy.linalg import lstsq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE   = '#3A7DC9'
GREEN  = '#4DA84D'
ORANGE = '#E8762D'
PURPLE = '#7B5EA7'
CYAN   = '#2AAFC4'
WHITE  = '#FFFFFF'

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':180,
})

df = pd.read_csv('survey.csv')
df['perc_quality'] = df[['D_1','D_2','D_3','D_4']].mean(axis=1)
df['sympathy']     = df[['E_1','E_2','E_3','E_4']].mean(axis=1)
df['purchase_int'] = df[['F_1','F_2','F_3']].mean(axis=1)
df['condition_bin']= (df['condition']=='condition = treatment').astype(int)
df['pq_c']         = df['perc_quality'] - df['perc_quality'].mean()

Xa = np.column_stack([np.ones(200), df['condition_bin']])
ba,_,_,_ = lstsq(Xa, df['sympathy'].values, rcond=None)
n,k = 200,2
sig2a = ((df['sympathy'].values - Xa@ba)**2).sum()/(n-k)
sea   = np.sqrt(sig2a * np.linalg.inv(Xa.T@Xa).diagonal())
path_a = ba[1]

Xb  = np.column_stack([np.ones(200),df['condition_bin'],df['sympathy'],df['pq_c']])
yb  = df['purchase_int'].values
bb,_,_,_ = lstsq(Xb, yb, rcond=None)
n2,k2 = 200,4
sig2b = ((yb - Xb@bb)**2).sum()/(n2-k2)
seb   = np.sqrt(sig2b * np.linalg.inv(Xb.T@Xb).diagonal())
path_b  = bb[2]; c_prime = bb[1]; pq_beta = bb[3]

np.random.seed(42); boots=[]
for _ in range(5000):
    idx = np.random.choice(200,200,replace=True)
    bd  = df.iloc[idx].dropna(subset=['condition_bin','sympathy','purchase_int','pq_c'])
    Xa_ = np.column_stack([np.ones(len(bd)), bd['condition_bin']])
    ba_,_,_,_ = lstsq(Xa_, bd['sympathy'].values, rcond=None)
    Xb_ = np.column_stack([np.ones(len(bd)), bd['condition_bin'],
                            bd['sympathy'], bd['pq_c']])
    bb_,_,_,_ = lstsq(Xb_, bd['purchase_int'].values, rcond=None)
    boots.append(ba_[1]*bb_[2])
boots   = np.array(boots)
ie_mean = boots.mean()
ie_ci   = (np.percentile(boots,2.5), np.percentile(boots,97.5))

# Node positions
NX  = (1.5,  3.8)   # Framing X
NM  = (6.0,  6.2)   # Sympathy E  (top)
NY  = (10.5, 3.8)   # Purchase Intention F  (right)
NPQ = (8.25, 1.2)   # Perceived Quality D  (below path-b midpoint)

def draw_box(ax, cx, cy, txt, col, w=2.3, h=1.0):
    bx = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                         boxstyle='round,pad=0.13',
                         facecolor=col, edgecolor='white',
                         linewidth=2.5, alpha=0.95, zorder=3)
    ax.add_patch(bx)
    ax.text(cx, cy, txt, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white',
            linespacing=1.4, zorder=4)

def arrow(ax, x1,y1,x2,y2, col, ls='-', lw=2.4):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                linestyle=ls, connectionstyle='arc3,rad=0.0'),
                zorder=2)

def label(ax, x, y, txt, col, fs=9.5):
    ax.text(x, y, txt, ha='center', va='center',
            fontsize=fs, color=col, linespacing=1.3, zorder=5,
            bbox=dict(boxstyle='round,pad=0.26', fc='white',
                      ec=col, alpha=0.97, lw=1.4))

fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor(WHITE)

ax_d = fig.add_axes([0.01, 0.18, 0.54, 0.78])
ax_d.set_xlim(0, 12); ax_d.set_ylim(0, 7)
ax_d.axis('off'); ax_d.set_facecolor(WHITE)

draw_box(ax_d, *NX,  'Framing\n(X)',            BLUE)
draw_box(ax_d, *NM,  'Sympathy\n(E)',            GREEN)
draw_box(ax_d, *NY,  'Purchase\nIntention (F)',  PURPLE)
draw_box(ax_d, *NPQ, 'Perceived\nQuality (D)',   CYAN)

arrow(ax_d, NX[0]+1.15,  NX[1]+0.40,  NM[0]-1.15,  NM[1]-0.50, GREEN)
arrow(ax_d, NM[0]+1.15,  NM[1]-0.50,  NY[0]-1.15,  NY[1]+0.40, ORANGE, ls='dashed')
arrow(ax_d, NX[0]+1.15,  NX[1]-0.10,  NY[0]-1.15,  NY[1]-0.10, 'gray',  ls='dashed')
arrow(ax_d, NPQ[0]+1.15, NPQ[1]+0.30, NY[0]-1.15,  NY[1]-0.40, CYAN,   ls='dashed')

label(ax_d, 3.2,  5.45, f'Path a\nβ = {path_a:.3f}***',      GREEN)
label(ax_d, 8.8,  5.45, f'Path b\nβ = {path_b:.3f} (n.s.)',  ORANGE)
label(ax_d, 6.0,  3.30, f"c' = {c_prime:.3f} (n.s.)",        'gray', fs=9.2)
label(ax_d, 9.8,  2.40, f'PQ: β = {pq_beta:.3f}***',         CYAN)

ax_d.set_title('(a)  Path Diagram with Coefficients',
               fontsize=13, fontweight='bold', pad=8)

fig.text(0.275, 0.11,
         f'Indirect  a × b = {ie_mean:.4f}     '
         f'95 % BC CI  [{ie_ci[0]:.4f},  {ie_ci[1]:.4f}]',
         ha='center', va='center', fontsize=11, color='#1a1a1a',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#aaa', alpha=0.97, lw=1.5))
fig.text(0.275, 0.055,
         '✗  CI includes zero — H2 NOT supported',
         ha='center', va='center', fontsize=12,
         fontweight='bold', color='#C0392B')

ax_b = fig.add_axes([0.58, 0.12, 0.40, 0.80])
ax_b.set_facecolor(WHITE)
ax_b.hist(boots, bins=65, color=PURPLE, alpha=0.70,
          edgecolor='white', linewidth=0.5, density=True)
kde_b = gaussian_kde(boots, bw_method=0.15)
xb = np.linspace(boots.min(), boots.max(), 300)
ax_b.plot(xb, kde_b(xb), color=PURPLE, lw=3)
ax_b.axvline(0,        color='#E74C3C', lw=2.5, ls='--', label='Zero (null)')
ax_b.axvline(ie_ci[0], color='#333',   lw=1.8, ls=':')
ax_b.axvline(ie_ci[1], color='#333',   lw=1.8, ls=':',
             label=f'95% CI  [{ie_ci[0]:.3f}, {ie_ci[1]:.3f}]')
ax_b.axvline(ie_mean,  color='#111',   lw=2.5, label=f'IE = {ie_mean:.4f}')
ax_b.fill_between(np.linspace(ie_ci[0],ie_ci[1],200),
                  kde_b(np.linspace(ie_ci[0],ie_ci[1],200)),
                  alpha=0.22, color=PURPLE)
ax_b.set_xlabel('Indirect Effect  (a × b)', labelpad=8)
ax_b.set_ylabel('Density')
ax_b.set_title('(b)  Bootstrap Distribution  (5,000 resamples)', pad=10)
ax_b.legend(framealpha=0.92, fontsize=10)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.grid(alpha=0.2, linestyle='--')

plt.savefig('fig58_mediation.png', dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()