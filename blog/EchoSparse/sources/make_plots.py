#!/usr/bin/env python3
"""Plots from documented observations only. Inputs: ../data/*.json (copied from the plot-data manifest)."""
import json, os, math, html
ROOT = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(ROOT)
SANS = "Inter, 'Helvetica Neue', Arial, 'PingFang SC', 'Noto Sans CJK SC', sans-serif"
INK, VIO, TEAL, MUTE, BG, GOLD, GREY = '#f4f4f1', '#b9a3ff', '#45d0c1', '#9a9ea6', '#000000', '#f2b544', '#7c8089'
def esc(t): return html.escape(t)
class S:
    def __init__(s, w, h): s.w, s.h, s.o = w, h, []
    def a(s, x): s.o.append(x)
    def text(s, x, y, t, fs=13, col=INK, anc='start', bold=False, ital=False, rot=None):
        tr = f' transform="rotate({rot} {x} {y})"' if rot else ''
        s.a(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{col}" text-anchor="{anc}"' + (' font-weight="700"' if bold else '') + (' font-style="italic"' if ital else '') + f'{tr}>{esc(t)}</text>')
    def out(s, path):
        open(path, 'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s.w} {s.h}" width="{s.w}" height="{s.h}" font-family="{SANS}"><rect width="{s.w}" height="{s.h}" fill="{BG}"/>' + '\n'.join(s.o) + '</svg>')

STR = {
 'zh': dict(t1='相邻去噪步的 LSE 变化（Wan2.1-1.3B，稠密 FA4，5 个提示词）', s1='展示来源报告列出的 7 个汇总点；连线仅引导阅读，不是拟合或插值。单位：自然对数',
            x1='去噪步（从 0 起）', y1='|LSE(t) − LSE(t−1)|（nats，对数坐标）', med='中位数', p99='99 分位',
            t2='每个去噪步的两路 CFG transformer forward 耗时（Wan2.1-1.3B，B200）', s2='三轮的中位数；浅色带为三轮观察到的最小–最大范围（n=3），不是置信区间',
            x2='去噪步（从 1 起）', y2='耗时 (ms)', dense='稠密 FA4', sp='历史单均值 LGSF，density 0.10', init='第 10 步：全块初始化', phase='稀疏阶段从第 11 步开始'),
 'en': dict(t1='Adjacent-step LSE change (Wan2.1-1.3B, dense FA4, 5 prompts)', s1='Seven step summaries reported in the source are shown; the line is a guide, not a fit or interpolation. Unit: natural log',
            x1='denoising step (zero-based)', y1='|LSE(t) − LSE(t−1)| (nats, log scale)', med='median', p99='99th percentile',
            t2='Two-CFG transformer-forward time per denoising step (Wan2.1-1.3B, B200)', s2='Median of 3 rounds; shaded band is the observed min–max range (n=3), not a confidence interval',
            x2='denoising step (one-based)', y2='time (ms)', dense='dense FA4', sp='historical single-mean LGSF, density 0.10', init='step 10: all-block init', phase='sparse phase begins at step 11'),
}
lse = json.load(open(os.path.join(B, 'data/lse_drift.json')))
lat = json.load(open(os.path.join(B, 'data/denoising_step_latency.json')))
for lg, T in STR.items():
    # ---- LSE drift, log y
    s = S(1000, 480); s.text(30, 34, T['t1'], 18, bold=True); s.text(30, 56, T['s1'], 12.5, MUTE, ital=True)
    x0, x1, y0, y1 = 100, 840, 400, 90
    X = lambda v: x0 + (x1-x0)*v/50
    lo, hi = math.log10(0.03), math.log10(4.0)
    Y = lambda v: y0 - (y0-y1)*(math.log10(v)-lo)/(hi-lo)
    for g in (0.05, 0.1, 0.2, 0.5, 1, 2):
        s.a(f'<line x1="{x0}" x2="{x1}" y1="{Y(g)}" y2="{Y(g)}" stroke="#202226"/>'); s.text(x0-8, Y(g)+4, str(g), 12, MUTE, 'end')
    for v in (0, 10, 20, 30, 40, 49):
        s.a(f'<line x1="{X(v)}" x2="{X(v)}" y1="{y0}" y2="{y0+5}" stroke="{MUTE}"/>'); s.text(X(v), y0+22, str(v), 12, MUTE, 'middle')
    s.a(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="{MUTE}"/>')
    s.text((x0+x1)/2, y0+46, T['x1'], 13, INK, 'middle'); s.text(24, (y0+y1)/2, T['y1'], 13, INK, 'middle', rot=-90)
    for key, col, lab in (('p99_abs_delta_lse_nats', GOLD, T['p99']), ('median_abs_delta_lse_nats', TEAL, T['med'])):
        pts = [(X(d['step_index']), Y(d[key]), d[key]) for d in lse]
        s.a(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a,b,_ in pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-opacity=".55"/>')
        for a, b, v in pts:
            s.a(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{col}"/>'); s.text(a, b-10, f'{v:g}', 11.5, col, 'middle')
        s.text(x1+8, pts[-1][1]+4, lab, 12.5, col, bold=True)
    s.out(os.path.join(B, f'assets/fig5_lse_drift_{lg}.svg'))
    # ---- per-step latency
    s = S(900, 500); s.text(30, 34, T['t2'], 18, bold=True); s.text(30, 56, T['s2'], 12.5, MUTE, ital=True)
    x0, x1, y0, y1 = 100, 840, 420, 100
    X = lambda n: x0 + (x1-x0)*(n-1)/49
    Y = lambda v: y0 - (y0-y1)*v/700
    for g in (0, 100, 200, 300, 400, 500, 600):
        s.a(f'<line x1="{x0}" x2="{x1}" y1="{Y(g)}" y2="{Y(g)}" stroke="#202226"/>'); s.text(x0-8, Y(g)+4, str(g), 12, MUTE, 'end')
    for n in (1, 10, 20, 30, 40, 50):
        s.text(X(n), y0+22, str(n), 12, MUTE, 'middle')
    s.text((x0+x1)/2, y0+46, T['x2'], 13, INK, 'middle'); s.text(24, (y0+y1)/2, T['y2'], 13, INK, 'middle', rot=-90)
    for arm, col, lab in ((lat[0], GREY, T['dense']), (lat[1], TEAL, T['sp'])):
        p = arm['points']
        up = ' '.join(f"{X(q['step_number']):.1f},{Y(q['max_ms']):.1f}" for q in p); dn = ' '.join(f"{X(q['step_number']):.1f},{Y(q['min_ms']):.1f}" for q in reversed(p))
        s.a(f'<polygon points="{up} {dn}" fill="{col}" fill-opacity=".2"/>')
        s.a(f'<polyline points="{" ".join(f"{X(q["step_number"]):.1f},{Y(q["median_ms"]):.1f}" for q in p)}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    s.a(f'<line x1="{X(10.5)}" x2="{X(10.5)}" y1="{y1-10}" y2="{y0}" stroke="{VIO}" stroke-dasharray="5 4" stroke-width="1.5"/>')
    s.text(X(10.5)+8, y1-2, T['phase'], 12.5, VIO, bold=True)
    s.text(X(10)-12, Y(lat[1]['points'][9]['median_ms'])+4, T['init'], 12, GOLD, 'end')
    s.text(x1, Y(lat[0]['points'][-1]['median_ms'])-12, f"{T['dense']}  ≈{lat[0]['points'][-1]['median_ms']:.0f} ms", 12.5, MUTE, 'end', bold=True)
    s.text(x1, Y(lat[1]['points'][-1]['median_ms'])+22, f"{T['sp']}  ≈{lat[1]['points'][-1]['median_ms']:.0f} ms", 12.5, TEAL, 'end', bold=True)
    s.out(os.path.join(B, f'assets/fig6_step_latency_{lg}.svg'))
print('plots ok')
