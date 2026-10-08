"""Plot teacher learning curve + three-config closed-loop comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'D:\fly\doomfly')
OUT = ROOT/'outputs'/'doom'/'distill-v1'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. teacher curve
log = json.loads((OUT.parent/'teacher-v1'/'teacher-log.json').read_text())
steps = [r['step'] for r in log]; kills = [r['kills_mean'] for r in log]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, kills, '-o', ms=4, lw=1.5, color='#2b6cb0')
ax.axhline(1.0, ls='--', c='#999', lw=1)
ax.text(steps[0], 1.15, '作者 BCI 基线 1.0', fontsize=9, color='#666')
ax.set_title('PPO 老师学习曲线 (combat_survival, 3 局均值击杀)')
ax.set_xlabel('训练步数'); ax.set_ylabel('平均击杀 / 局')
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/'teacher_curve.png', dpi=110); plt.close(fig)

# 2. three-config comparison
ev = json.loads((OUT/'evaluation.json').read_text())
import collections
agg = collections.defaultdict(list)
for r in ev: agg[r['config']].append(r)
cfgs = ['baseline', 'random', 'distilled']
mk = [np.mean([r['kills'] for r in agg[c]]) for c in cfgs]
mt = [np.mean([r['tics'] for r in agg[c]]) for c in cfgs]
mh = [np.mean([r['health'] for r in agg[c]]) for c in cfgs]
labels = ['作者 BCI\n(baseline)', '随机读出层\n(random)', '蒸馏读出层\n(distilled)']
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
bar1 = axes[0].bar(labels, mk, color=['#a0aec0', '#a0aec0', '#38a169'])
axes[0].set_title('平均击杀 / 局'); axes[0].set_ylim(0, 2.2)
for b, v in zip(bar1, mk): axes[0].text(b.get_x()+b.get_width()/2, v+0.06, f'{v:.2f}', ha='center', fontsize=10)
bar2 = axes[1].bar(labels, mt, color=['#a0aec0', '#a0aec0', '#38a169'])
axes[1].axhline(210, ls='--', c='#c53030', lw=1); axes[1].text(2.4, 213, '满局 210 tics', fontsize=8, color='#c53030', ha='center')
axes[1].set_title('平均存活 (tics/210)'); axes[1].set_ylim(0, 230)
for b, v in zip(bar2, mt): axes[1].text(b.get_x()+b.get_width()/2, v+3, f'{v:.0f}', ha='center', fontsize=10)
bar3 = axes[2].bar(labels, mh, color=['#a0aec0', '#a0aec0', '#38a169'])
axes[2].set_title('局末健康值'); axes[2].set_ylim(-30, 110)
for b, v in zip(bar3, mh): axes[2].text(b.get_x()+b.get_width()/2, v+3, f'{v:.0f}', ha='center', fontsize=10)
fig.suptitle('果蝇脑闭环对比 (3 seeds × 210 tics, combat_survival)', y=1.03)
fig.tight_layout(); fig.savefig(OUT/'three_config_compare.png', dpi=110, bbox_inches='tight')
print('plots saved:', OUT/'teacher_curve.png', OUT/'three_config_compare.png')
