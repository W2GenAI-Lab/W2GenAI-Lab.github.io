/* EchoSparse code walkthrough. Original implementation; no runtime dependencies. */
(() => {
  'use strict';

  const ONLINE = [
    ['scale', 'scale2 = log2(e) / sqrt(D)'],
    ['m', 'm = tl.full((BM,), -float("inf"), tl.float32)'],
    ['l', 'l = tl.zeros((BM,), tl.float32)'],
    ['acc', 'acc = tl.zeros((BM, Dv), tl.float32)'],
    ['loop', 'for k, v, log_count in kv_tiles:'],
    ['qk', '    s2 = tl.dot(q, tl.trans(k)) * scale2'],
    ['count', '    s2 += log_count[None, :]'],
    ['max', '    block_max = tl.max(s2, axis=1)'],
    ['mnew', '    m_new = tl.maximum(m, block_max)'],
    ['alpha', '    alpha = tl.exp2(m - m_new)'],
    ['p', '    p = tl.exp2(s2 - m_new[:, None])'],
    ['lsum', '    l = l * alpha + tl.sum(p, axis=1)'],
    ['oscale', '    acc = acc * alpha[:, None]'],
    ['pv', '    acc = tl.dot(p.to(tl.bfloat16), v, acc)'],
    ['mupdate', '    m = m_new'],
    ['out', 'out = acc / l[:, None]'],
    ['lse', 'new_lse2 = m + tl.log2(l)'],
  ].map(([id, text]) => ({ id, text }));

  // Each edit runs on the same line objects; removed rows collapse in place.
  const EDITS = [
    { kind: 'insert', id: 'prior', before: 'm', text: 'prior2 = tl.load(PREV_LSE2 + rows * H + head)', label: 'prior', duration: 1.8 },
    { kind: 'replace', id: 'p', text: '    p = tl.exp2(s2 - prior2[:, None])', label: 'exponent', duration: 2.0 },
    { kind: 'replace', id: 'lsum', text: '    l += tl.sum(p, axis=1)', label: 'denominator', duration: 1.8 },
    { kind: 'remove', id: 'oscale', label: 'output', duration: 1.5 },
    { kind: 'remove', id: 'mupdate', label: 'maximum', duration: 1.1 },
    { kind: 'remove', id: 'alpha', label: 'maximum', duration: 1.2 },
    { kind: 'remove', id: 'mnew', label: 'maximum', duration: 1.2 },
    { kind: 'remove', id: 'max', label: 'maximum', duration: 1.4 },
    { kind: 'remove', id: 'm', label: 'maximum', duration: 1.2 },
    { kind: 'replace', id: 'lse', text: 'new_lse2 = prior2 + tl.log2(l)', label: 'lse', duration: 1.8 },
  ];
  let end = 1.5;
  for (const edit of EDITS) {
    edit.start = Number(end.toFixed(2));
    edit.end = Number((end + edit.duration).toFixed(2));
    end = Number((edit.end + 0.5).toFixed(2));
  }
  const DURATION = Number((end + 1.5).toFixed(2));
  const POINTS = [0, ...EDITS.map(edit => edit.end), DURATION];
  const clamp = n => Math.max(0, Math.min(1, n));
  const smooth = n => { const t = clamp(n); return t * t * (3 - 2 * t); };
  const escape = text => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function apply(lines, edit) {
    const index = lines.findIndex(line => line.id === edit.id);
    if (edit.kind === 'insert') {
      lines.splice(lines.findIndex(line => line.id === edit.before), 0, { id: edit.id, text: edit.text, height: 1, opacity: 1 });
    } else if (edit.kind === 'remove') {
      lines.splice(index, 1);
    } else {
      lines[index].text = edit.text;
    }
  }

  function changedPart(before, after) {
    let left = 0, right = 0;
    while (left < before.length && left < after.length && before[left] === after[left]) left++;
    while (right < before.length - left && right < after.length - left && before[before.length - 1 - right] === after[after.length - 1 - right]) right++;
    return { left, oldMiddle: before.slice(left, before.length - right), newMiddle: after.slice(left, after.length - right), suffix: right ? before.slice(-right) : '' };
  }

  function frameAt(time) {
    const t = Math.max(0, Math.min(DURATION, time));
    const lines = ONLINE.map(line => ({ ...line, height: 1, opacity: 1 }));
    let active = null;
    let phase = 'online';
    for (const edit of EDITS) {
      if (t < edit.start) break;
      phase = edit.label;
      if (t >= edit.end) {
        apply(lines, edit);
        const modified = lines.find(line => line.id === edit.id);
        if (modified) modified.fresh = Math.max(0, 1 - (t - edit.end) / 1.2);
        continue;
      }
      const u = (t - edit.start) / edit.duration;
      if (edit.kind === 'insert') {
        const grow = smooth(u / 0.2);
        const n = Math.floor(clamp((u - 0.14) / 0.75) * edit.text.length);
        const row = { id: edit.id, text: edit.text.slice(0, n), height: grow, opacity: grow, fresh: 1, caret: n };
        lines.splice(lines.findIndex(line => line.id === edit.before), 0, row);
      } else {
        const row = lines.find(line => line.id === edit.id);
        if (edit.kind === 'remove') {
          row.cut = [row.text.search(/\S/), row.text.length];
          row.caret = row.text.length;
          row.height = 1 - smooth((u - 0.46) / 0.54);
          row.opacity = 1 - smooth((u - 0.3) / 0.65);
        } else {
          const diff = changedPart(row.text, edit.text);
          if (u < 0.26) {
            row.cut = [diff.left, diff.left + diff.oldMiddle.length];
            row.caret = diff.left + diff.oldMiddle.length;
          } else if (u < 0.52) {
            const n = Math.ceil((1 - clamp((u - 0.26) / 0.26)) * diff.oldMiddle.length);
            row.text = row.text.slice(0, diff.left) + diff.oldMiddle.slice(0, n) + diff.suffix;
            row.caret = diff.left + n;
            row.cut = [diff.left, diff.left + n];
          } else {
            const n = Math.floor(clamp((u - 0.52) / 0.42) * diff.newMiddle.length);
            row.text = row.text.slice(0, diff.left) + diff.newMiddle.slice(0, n) + diff.suffix;
            row.add = [diff.left, diff.left + n];
            row.caret = diff.left + n;
          }
        }
      }
      active = edit;
      break;
    }
    if (t >= EDITS[EDITS.length - 1].end) phase = 'fixed';
    return { lines, phase, active, time: t };
  }

  function textOf(lines) { return lines.map(line => line.text).join('\n'); }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { frameAt, duration: DURATION, online: textOf(ONLINE), fixed: textOf(frameAt(DURATION).lines), points: POINTS };
  }
  if (typeof document === 'undefined') return;

  const LABELS = {
    zh: {
      online: 'Online softmax', fixed: 'Fixed offset', prior: '读取上一去噪步的 LSE', exponent: '固定指数偏移',
      denominator: '移除分母重缩放', output: '移除输出重缩放', maximum: '移除 max 与 alpha', lse: '更新本步 LSE',
      play: '播放', pause: '暂停', replay: '重播', previous: '上一步', next: '下一步', progress: '代码动画进度', reduced: '减少动态效果',
    },
    en: {
      online: 'Online softmax', fixed: 'Fixed offset', prior: 'Load the previous LSE', exponent: 'Fix the exponent offset',
      denominator: 'Remove denominator rescaling', output: 'Remove output rescaling', maximum: 'Remove max and alpha', lse: 'Update the current LSE',
      play: 'Play', pause: 'Pause', replay: 'Replay', previous: 'Previous edit', next: 'Next edit', progress: 'Code animation progress', reduced: 'Reduced motion',
    },
  };

  function syntax(text) {
    const classes = Array(text.length).fill('');
    for (const [pattern, name] of [[/\b(for|in)\b/g, 'kw'], [/\btl\.\w+/g, 'fn'], [/\b(?:None|float|log2|sqrt|e|D|BM|Dv)\b/g, 'constant']]) {
      for (const match of text.matchAll(pattern)) {
        for (let i = match.index; i < match.index + match[0].length; i++) classes[i] = name;
      }
    }
    return classes;
  }

  function lineHTML(line) {
    const colors = syntax(line.text);
    let result = '', run = '', last = '';
    const flush = () => { if (run) result += last ? `<span class="${last}">${run}</span>` : run; run = ''; };
    for (let i = 0; i < line.text.length; i++) {
      let cls = colors[i];
      if (line.cut && i >= line.cut[0] && i < line.cut[1]) cls += ' cut';
      if (line.add && i >= line.add[0] && i < line.add[1]) cls += ' add';
      if (cls !== last) { flush(); last = cls; }
      run += escape(line.text[i]);
    }
    flush();
    if (line.caret != null && line.height > 0.6) result += `<span class="lse-code-caret" style="left:${line.caret}ch" aria-hidden="true"></span>`;
    return result || '\u200b';
  }

  function init(host) {
    if (host.dataset.motionReady) return;
    host.dataset.motionReady = 'true';
    const lang = host.dataset.lang === 'zh' ? 'zh' : 'en';
    const L = LABELS[lang];
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
    const panels = document.createElement('div');
    panels.className = 'lse-code-pair';
    panels.innerHTML = `<div class="lse-code-panel"><div class="lse-code-head">${L.online}</div><div class="lse-code-window" tabindex="0" aria-label="${L.online}"><code class="lse-code-lines lse-code-base"></code></div></div><div class="lse-code-panel lse-code-changing"><div class="lse-code-head lse-code-mode">${L.online}</div><div class="lse-code-window" tabindex="0" aria-label="${L.fixed}"><code class="lse-code-lines lse-code-live"></code></div></div>`;
    const controls = document.createElement('div');
    controls.className = 'lse-code-controls';
    controls.innerHTML = `<div class="lse-code-buttons"><button type="button" class="lse-code-toggle">${L.play}</button><button type="button" class="lse-code-back" aria-label="${L.previous}">‹</button><button type="button" class="lse-code-next" aria-label="${L.next}">›</button><button type="button" class="lse-code-replay" aria-label="${L.replay}">↻</button></div><output class="lse-code-status" aria-live="polite"></output><input class="lse-code-progress" type="range" min="0" max="${DURATION}" step="0.05" value="0" aria-label="${L.progress}">`;
    host.append(panels, controls);
    host.classList.add('is-enhanced');
    const base = panels.querySelector('.lse-code-base');
    const live = panels.querySelector('.lse-code-live');
    const mode = panels.querySelector('.lse-code-mode');
    const output = controls.querySelector('output');
    const toggle = controls.querySelector('.lse-code-toggle');
    const range = controls.querySelector('input');
    const prev = controls.querySelector('.lse-code-back');
    const next = controls.querySelector('.lse-code-next');
    const rows = new Map();
    let playing = false, userPaused = false, visible = false, resumeOnVisible = false;
    let time = reduce.matches ? DURATION : 0, origin = 0, animation = 0, lastPhase = '';
    base.innerHTML = ONLINE.map((line, i) => `<span class="lse-code-line"><span class="lse-code-number" aria-hidden="true">${i + 1}</span><span class="lse-code-text">${lineHTML(line)}</span></span>`).join('');

    function draw() {
      const state = frameAt(time);
      const keep = new Set(state.lines.map(line => line.id));
      for (const [id, record] of rows) {
        if (!keep.has(id)) { record.el.remove(); rows.delete(id); }
      }
      let number = 0;
      for (const line of state.lines) {
        let record = rows.get(line.id);
        if (!record) {
          const el = document.createElement('span'); el.className = 'lse-code-line';
          el.innerHTML = '<span class="lse-code-number" aria-hidden="true"></span><span class="lse-code-text"></span>';
          record = { el, num: el.firstChild, code: el.lastChild, html: '' };
          rows.set(line.id, record);
        }
        const at = live.children[number];
        if (at !== record.el) live.insertBefore(record.el, at || null);
        record.num.textContent = String(++number);
        record.el.style.height = `${line.height * 1.7}em`;
        record.el.style.opacity = String(line.opacity);
        record.el.style.setProperty('--fresh', String(line.fresh || 0));
        const html = lineHTML(line);
        if (record.html !== html) { record.code.innerHTML = html; record.html = html; }
      }
      if (state.phase !== lastPhase) {
        output.textContent = L[state.phase]; lastPhase = state.phase;
      }
      mode.textContent = state.phase === 'online' ? L.online : state.phase === 'fixed' ? L.fixed : `${L.online} → ${L.fixed}`;
      range.value = String(time);
      range.setAttribute('aria-valuetext', `${L[state.phase]}, ${Math.round(time / DURATION * 100)}%`);
      prev.disabled = time <= 0;
      next.disabled = time >= DURATION;
      toggle.disabled = reduce.matches;
      toggle.textContent = reduce.matches ? L.reduced : playing ? L.pause : time >= DURATION ? L.replay : L.play;
    }

    function stop(manual) {
      playing = false;
      if (manual) { userPaused = true; resumeOnVisible = false; }
      cancelAnimationFrame(animation);
      draw();
    }
    function tick(now) {
      if (!playing) return;
      time = Math.min(DURATION, (now - origin) / 1000);
      draw();
      if (time >= DURATION) { playing = false; draw(); return; }
      animation = requestAnimationFrame(tick);
    }
    function play(manual) {
      if (reduce.matches || playing) return;
      if (manual) userPaused = false;
      if (time >= DURATION) time = 0;
      playing = true; origin = performance.now() - time * 1000;
      draw(); animation = requestAnimationFrame(tick);
    }
    function seek(value) { stop(true); time = Math.max(0, Math.min(DURATION, value)); draw(); }
    toggle.addEventListener('click', () => playing ? stop(true) : play(true));
    controls.querySelector('.lse-code-replay').addEventListener('click', () => {
      seek(0); if (!reduce.matches) play(true);
    });
    range.addEventListener('input', () => seek(Number(range.value)));
    prev.addEventListener('click', () => seek([...POINTS].reverse().find(point => point < time - 0.08) ?? 0));
    next.addEventListener('click', () => seek(POINTS.find(point => point > time + 0.08) ?? DURATION));
    const onReduced = () => { if (reduce.matches) { stop(true); time = DURATION; draw(); } else draw(); };
    if (reduce.addEventListener) reduce.addEventListener('change', onReduced);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) { resumeOnVisible = playing; if (playing) stop(false); }
      else if (visible && resumeOnVisible && !userPaused && time < DURATION) play(false);
    });
    draw();
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(entries => {
        visible = entries[0].isIntersecting;
        if (!visible) { resumeOnVisible = playing; if (playing) stop(false); }
        else if (!reduce.matches && !document.hidden && !userPaused && time < DURATION) play(false);
      }, { threshold: 0.15 }).observe(host);
    }
  }

  for (const host of document.querySelectorAll('.lse-code-motion')) init(host);
})();
