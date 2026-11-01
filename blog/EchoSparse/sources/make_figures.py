#!/usr/bin/env python3
"""Generate editable draw.io sources (.drawio) and SVG exports for the EchoSparse blog figures.
Usage: python3 make_figures.py   (run from anywhere; paths are relative to this file)"""
import json, os, html
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), 'assets')
SANS = "Inter, 'Helvetica Neue', Arial, 'PingFang SC', 'Noto Sans CJK SC', sans-serif"
INK, VIO, TEAL, GOLD, MUTE, BG = '#f4f4f1', '#b9a3ff', '#45d0c1', '#f2b544', '#9a9ea6', '#000000'
BLUE, CELL, HAIR, FAINT, AMBD, GREY = '#7ea9ff', '#141518', '#202226', '#3c3f45', '#a88230', '#7c8089'

class Fig:
    def __init__(s, w, h): s.w, s.h, s.items = w, h, []
    def rect(s, x, y, w, h, fill='#141518', stroke='#3c3f45', text='', fs=14, color=INK, r=8, dash=False, bold=False, sw=1.5):
        s.items.append(('rect', x, y, w, h, fill, stroke, text, fs, color, r, dash, bold, sw))
    def text(s, x, y, text, fs=14, color=INK, anchor='start', bold=False, italic=False):
        s.items.append(('text', x, y, text, fs, color, anchor, bold, italic))
    def arrow(s, x1, y1, x2, y2, color=INK, dash=False, label=''):
        s.items.append(('arrow', x1, y1, x2, y2, color, dash, label))
    # ---- SVG
    def svg(s):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s.w} {s.h}" width="{s.w}" height="{s.h}" font-family="{SANS}">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>',
             f'<rect width="{s.w}" height="{s.h}" fill="{BG}"/>']
        for it in s.items:
            if it[0] == 'rect':
                _, x, y, w, h, fill, stroke, t, fs, col, r, dash, bold, sw = it
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (' stroke-dasharray="5 4"' if dash else '') + '/>')
                if t:
                    lines = t.split('\n'); y0 = y + h/2 - (len(lines)-1)*fs*0.6 + fs*0.35
                    for i, ln in enumerate(lines):
                        o.append(f'<text x="{x+w/2}" y="{y0+i*fs*1.2}" font-size="{fs}" fill="{col}" text-anchor="middle"' + (' font-weight="700"' if bold else '') + f'>{html.escape(ln)}</text>')
            elif it[0] == 'text':
                _, x, y, t, fs, col, anc, bold, ital = it
                for i, ln in enumerate(t.split('\n')):
                    o.append(f'<text x="{x}" y="{y+i*fs*1.25}" font-size="{fs}" fill="{col}" text-anchor="{anc}"' + (' font-weight="700"' if bold else '') + (' font-style="italic"' if ital else '') + f'>{html.escape(ln)}</text>')
            else:
                _, x1, y1, x2, y2, col, dash, lab = it
                o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2" marker-end="url(#ah)"' + (' stroke-dasharray="6 4"' if dash else '') + '/>')
                if lab: o.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-6}" font-size="12" fill="{col}" text-anchor="middle">{html.escape(lab)}</text>')
        o.append('</svg>'); return '\n'.join(o)
    # ---- draw.io
    def drawio(s, name):
        cells, n = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>'], 2
        def esc(t): return html.escape(t, quote=True).replace('\n', '&lt;br&gt;')
        for it in s.items:
            if it[0] == 'rect':
                _, x, y, w, h, fill, stroke, t, fs, col, r, dash, bold, sw = it
                st = f'rounded=1;arcSize={int(r*100/min(w,h))};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={fs};fontColor={col};fontStyle={1 if bold else 0};' + ('dashed=1;' if dash else '')
                cells.append(f'<mxCell id="{n}" value="{esc(t)}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
            elif it[0] == 'text':
                _, x, y, t, fs, col, anc, bold, ital = it
                lines = t.count('\n') + 1; w = 340; hh = fs*1.3*lines
                xx = x if anc == 'start' else (x - w/2 if anc == 'middle' else x - w)
                al = {'start': 'left', 'middle': 'center', 'end': 'right'}[anc]
                st = f'text;html=1;strokeColor=none;fillColor=none;align={al};verticalAlign=middle;fontSize={fs};fontColor={col};fontStyle={(1 if bold else 0)|(2 if ital else 0)};whiteSpace=wrap;'
                cells.append(f'<mxCell id="{n}" value="{esc(t)}" style="{st}" vertex="1" parent="1"><mxGeometry x="{xx}" y="{y-fs}" width="{w}" height="{hh}" as="geometry"/></mxCell>')
            else:
                _, x1, y1, x2, y2, col, dash, lab = it
                st = f'endArrow=classic;html=1;strokeColor={col};strokeWidth=2;' + ('dashed=1;' if dash else '')
                cells.append(f'<mxCell id="{n}" value="{esc(lab)}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
            n += 1
        return (f'<mxfile host="app.diagrams.net"><diagram name="{name}" id="{name}"><mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" page="1" background="#000000" pageWidth="{s.w}" pageHeight="{s.h}"><root>'
                + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>')
    def save(s, base):
        open(os.path.join(OUT, base + '.svg'), 'w').write(s.svg())
        open(os.path.join(ROOT, base + '.drawio'), 'w').write(s.drawio(base))

T = {
 'zh': dict(
  f1_title='同一份统计量，在三个地方起作用', f1_prev='第 t−1 步', f1_prev1='LSE  a_i\n（softmax 的缩放）', f1_prev2='块质量图  M_ab\n（块之间的相对重要性）',
  f1_cur='第 t 步', f1_c1='① 固定 offset\n用 a_i 代替行最大值', f1_c2='② 选择分辨率\n按 M 排序：exact / refined / mean', f1_c3='③ 补偿\n落选块以 (k̃, ṽ, n)\n参与同一个 softmax',
  f1_k='融合 attention kernel', f1_k1='exact  m=64', f1_k2='refined  m=4', f1_k3='mean  m=1', f1_k4='同一个分子、同一个分母', f1_out='输出 õ  +  新的 LSE、新的块质量', f1_next='成为第 t+1 步的 LSE 与块质量',
  f1_note='LSE 提供缩放，块质量提供相对重要性；二者都在同一趟计算里顺带产生。',
  f2_title='packing 只重排“同一组项”', f2_a='nonpacked：三个循环', f2_b='grouped packing：两条紧凑流',
  f2_l1='exact 块列表', f2_l2='扫描全部块均值\n（exact / refined 的列被 mask）', f2_l3='refined 列表\n每块一个 id，展开为 4 个代表列', f2_p1='exact 块列表（TMA）', f2_p2='mean 流：一个 id 一列', f2_p3='refined 流：一个 id 展开 4 列',
  f2_eq='项集合、n_a、offset、统一归一化都不变\n⇒ 实数算术下输出相同', f2_legend=['exact', 'refined', 'mean', '被 mask 的无效列'],
  f3_title='三级软件流水的装载 / 计算时间线（示意，非实测）', f3_load='搬运 (TMA)', f3_comp='计算 (MMA)', f3_slot='槽位', f3_t='时间 →',
  f3_n1='槽位 0 被第 0 块用完后才能装入第 3 块', f3_n2='三个槽位 ≠ 三组 warp，也不是三个 kernel'),
 'en': dict(
  f1_title='One set of statistics, used in three places', f1_prev='step t−1', f1_prev1='LSE  a_i\n(softmax scaling)', f1_prev2='block-mass map  M_ab\n(relative importance of blocks)',
  f1_cur='step t', f1_c1='① fixed offset\na_i replaces the row max', f1_c2='② choose resolution\nrank by M: exact / refined / mean', f1_c3='③ compensate\nskipped blocks join the same\nsoftmax as (k̃, ṽ, n)',
  f1_k='fused attention kernel', f1_k1='exact  m=64', f1_k2='refined  m=4', f1_k3='mean  m=1', f1_k4='one numerator, one denominator', f1_out='output õ  +  new LSE, new block masses', f1_next='becomes the LSE and block masses of step t+1',
  f1_note='LSE supplies the scaling, block mass supplies relative importance; both fall out of the same pass.',
  f2_title='Packing only rearranges the same terms', f2_a='nonpacked: three loops', f2_b='grouped packing: two compact streams',
  f2_l1='exact block list', f2_l2='sweep over all block means\n(exact / refined columns masked out)', f2_l3='refined list\none block id, expanded to 4 representative columns', f2_p1='exact block list (TMA)', f2_p2='mean stream: one id, one column', f2_p3='refined stream: one id expands to 4 columns',
  f2_eq='same term set, n_a, offset and common normalization\n⇒ same output in real arithmetic', f2_legend=['exact', 'refined', 'mean', 'masked column'],
  f3_title='Load / compute timeline of a three-stage software pipeline (schematic, not measured)', f3_load='load (TMA)', f3_comp='compute (MMA)', f3_slot='slot', f3_t='time →',
  f3_n1='tile 3 can enter slot 0 only after tile 0 is consumed', f3_n2='three slots ≠ three warp groups, not three kernels'),
}
C_EX, C_RF, C_MN, C_MASK = BLUE, GOLD, AMBD, CELL

def fig1(L, lg):
    f = Fig(1000, 520)
    f.text(30, 40, L['f1_title'], 22, INK, bold=True)
    f.rect(30, 80, 270, 70, CELL, VIO, L['f1_prev1'], 14); f.rect(30, 170, 270, 70, CELL, VIO, L['f1_prev2'], 14)
    f.text(30, 72, L['f1_prev'], 13, MUTE, italic=True)
    f.text(360, 72, L['f1_cur'], 13, MUTE, italic=True)
    f.rect(360, 80, 290, 70, CELL, TEAL, L['f1_c1'], 14); f.rect(360, 170, 290, 70, CELL, TEAL, L['f1_c2'], 14); f.rect(360, 260, 290, 70, CELL, TEAL, L['f1_c3'], 14)
    f.arrow(300, 115, 360, 115, VIO); f.arrow(300, 205, 360, 205, VIO); f.arrow(300, 205, 360, 290, VIO, dash=True)
    f.rect(700, 80, 270, 250, '#0b0c0e', HAIR, '', 12)
    f.text(835, 108, L['f1_k'], 14, INK, 'middle', bold=True)
    for i, (t, c) in enumerate([(L['f1_k1'], BLUE), (L['f1_k2'], GOLD), (L['f1_k3'], AMBD)]):
        f.rect(725, 125 + i*50, 220, 38, c, c, t, 14, '#000', r=6, bold=True)
    f.rect(725, 275, 220, 40, 'none', INK, L['f1_k4'], 13, INK, r=6, dash=True)
    f.arrow(650, 115, 700, 150, TEAL); f.arrow(650, 205, 700, 205, TEAL); f.arrow(650, 295, 700, 265, TEAL)
    f.rect(360, 390, 610, 56, CELL, TEAL, L['f1_out'], 15, bold=True)
    f.arrow(835, 330, 835, 390, INK)
    f.arrow(360, 418, 165, 418, VIO, dash=True); f.arrow(165, 418, 165, 245, VIO, dash=True)
    f.text(165, 440, L['f1_next'], 13, VIO, 'middle')
    f.text(30, 495, L['f1_note'], 13, MUTE, italic=True)
    return f

def fig2(L, lg):
    f = Fig(1000, 560)
    f.text(30, 40, L['f2_title'], 22, INK, bold=True)
    f.text(30, 82, L['f2_a'], 15, TEAL, bold=True)
    def row(y, label, cells):
        f.text(30, y+22, label, 12, MUTE)
        x = 330
        for kind in cells:
            c = {'e': C_EX, 'r': C_RF, 'm': C_MN, 'x': C_MASK}[kind]
            f.rect(x, y, 34, 30, c, FAINT if kind == 'x' else c, '', r=4, dash=(kind == 'x'), sw=1.2); x += 38
    row(100, L['f2_l1'], 'eeee' + 'eeeeee')
    row(150, L['f2_l2'].split('\n')[0], 'xmxxmxxxmxmxxxm')
    f.text(30, 187, L['f2_l2'].split('\n')[1], 11, MUTE)
    row(215, L['f2_l3'].split('\n')[0], 'rrrr' 'rrrr' 'xxxx' 'xxxx')
    f.text(30, 252, L['f2_l3'].split('\n')[1], 11, MUTE)
    f.text(30, 312, L['f2_b'], 15, TEAL, bold=True)
    row(330, L['f2_p1'], 'eeee' + 'eeeeee')
    row(380, L['f2_p2'], 'mmmmm')
    row(430, L['f2_p3'], 'rrrr' 'rrrr')
    f.text(970, 322, '示意：列数为示意，非实际规模' if lg == 'zh' else 'schematic: column counts are illustrative', 11, MUTE, 'end', italic=True)
    f.rect(30, 490, 940, 50, CELL, TEAL, L['f2_eq'], 14, bold=True)
    x = 520
    for t, c in zip(L['f2_legend'], [C_EX, C_RF, C_MN, C_MASK]):
        f.rect(x, 70, 18, 18, c, FAINT if c == CELL else c, '', r=3, sw=1); f.text(x+24, 85, t, 12, INK); x += 100 if lg == 'zh' else 108
    return f

def fig3(L, lg):
    f = Fig(1000, 470)
    f.text(30, 40, L['f3_title'], 20, INK, bold=True)
    X0, U = 150, 100   # time unit = 100px
    cols = [BLUE, TEAL, GOLD]
    f.text(30, 135, L['f3_load'], 15, INK, bold=True); f.text(30, 245, L['f3_comp'], 15, INK, bold=True)
    # loads: tiles 0..3, issue times 0,0.4,0.8 (prologue), tile3 issued when tile0 compute ends (t=2.5)
    loads = [(0, 0.0, 1.5), (1, 0.4, 1.5), (2, 0.8, 1.5), (3, 2.5, 1.5), (4, 3.5, 1.5)]
    for i, (tile, t0, d) in enumerate(loads):
        slot = tile % 3
        f.rect(X0 + t0*U, 85 + (i % 3)*0 + [0, 28, 56, 0, 28][i], d*U, 24, cols[slot], cols[slot], f'T{tile} → {L["f3_slot"]} {slot}', 12, '#000', r=5, sw=1)
    comps = [(0, 1.5), (1, 2.5), (2, 3.5), (3, 4.5), (4, 5.5)]
    for tile, t0 in comps:
        slot = tile % 3
        f.rect(X0 + t0*U, 215, 1.0*U, 40, cols[slot], cols[slot], f'T{tile}', 14, '#000', r=5, bold=True, sw=1)
    f.arrow(X0, 300, X0 + 6.3*U, 300, MUTE); f.text(X0 + 6.3*U, 322, L['f3_t'], 12, MUTE, 'end')
    f.rect(X0 + 2.5*U, 150, 380, 32, CELL, VIO, L['f3_n1'], 12, r=6, dash=True)
    f.arrow(X0 + 2.6*U, 150, X0 + 2.6*U, 111, VIO, dash=True)
    f.rect(X0, 395, 600, 36, CELL, TEAL, L['f3_n2'], 13, r=6, bold=True)
    return f

for lg, L in T.items():
    for name, fn in (('fig1_feedback', fig1), ('fig2_packing', fig2), ('fig3_pipeline', fig3)):
        fn(L, lg).save(f'{name}_{lg}')

# ---- bar chart from the evidence JSON (complete GPU call, median of 20 CUDA events)
ev = json.load(open(os.path.join(os.path.dirname(ROOT), 'evidence', 'h100_attention_benchmark.json')))
c = ev['complete_ms']
bars = [('fa4', c['fa4']), ('base', c['base']), ('split_tma_stage', c['split_tma_stage'])]
data = {'source': 'evidence/h100_attention_benchmark.json (complete_ms)', 'unit': 'ms', 'statistic': 'median of 20 CUDA-event runs, with p10/p90',
        'bars': [{'key': k, 'median': v['median'], 'p10': v['p10'], 'p90': v['p90'], 'n': v['n']} for k, v in bars]}
json.dump(data, open(os.path.join(os.path.dirname(ROOT), 'data', 'h100_complete_ms.json'), 'w'), indent=1)
LAB = {'zh': ['稠密 FA4', 'nonpacked\nEchoSparse', 'grouped + TMA\n+ 三级流水'], 'en': ['dense FA4', 'nonpacked\nEchoSparse', 'grouped + TMA\n+ 3 stages']}
TIT = {'zh': 'H100 单次 attention GPU 流程耗时（合成 BF16，L=75600，H=40，D=128）', 'en': 'H100 single-attention GPU pipeline time (synthetic BF16, L=75600, H=40, D=128)'}
SUB = {'zh': '20 次 CUDA event 中位数，须线为 p10–p90；倍数相对 dense FA4', 'en': 'median of 20 CUDA-event runs, whiskers p10–p90; multipliers vs dense FA4'}
for lg in ('zh', 'en'):
    f = Fig(900, 460); f.text(30, 38, TIT[lg], 18, INK, bold=True); f.text(30, 62, SUB[lg], 13, MUTE, italic=True)
    base_y, top_y, vmax = 380, 100, 180.0
    sc = (base_y - top_y) / vmax
    for gv in (0, 50, 100, 150):
        y = base_y - gv*sc
        f.items.append(('arrow', 90, y, 860, y, HAIR, False, '')) if False else None
        f.rect(90, y-0.5, 770, 1, HAIR, HAIR, '', 0, sw=0); f.text(80, y+4, str(gv), 12, MUTE, 'end')
    for i, (k, v) in enumerate(bars):
        x = 150 + i*250; h = v['median']*sc; col = [GREY, BLUE, TEAL][i]
        f.rect(x, base_y-h, 120, h, col, col, '', 4)
        f.rect(x+59, base_y-v['p90']*sc, 2, (v['p90']-v['p10'])*sc, INK, INK, '', 0, sw=0)
        f.text(x+60, base_y-h-26, f"{v['median']:.2f} ms", 16, INK, 'middle', bold=True)
        if i: f.text(x+60, base_y-h-8-2+0, '', 1)
        ratio = c['fa4']['median']/v['median']
        if i: f.text(x+60, base_y-h/2+5, f'{ratio:.2f}×', 18, '#000', 'middle', bold=True)
        for j, ln in enumerate(LAB[lg][i].split('\n')): f.text(x+60, base_y+22+j*16, ln, 13, INK, 'middle')
    f.save(f'fig4_h100_bars_{lg}')
print('ok')
