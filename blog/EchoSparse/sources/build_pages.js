// Called by blog/_src/build.py with POST metadata and canonical Markdown sources.
// A standalone copy can also rebuild from its local post.json and Markdown files.
const fs = require('fs'), path = require('path');
const MD = require('markdown-it'), texmath = require('markdown-it-texmath'), katex = require('katex');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2), sourceArg = args.indexOf('--source-dir');
const sourceDir = sourceArg >= 0 ? path.resolve(args[sourceArg + 1]) : root;
const metadataPath = path.join(root, 'post.json');
const post = args.includes('--metadata-stdin') ? JSON.parse(fs.readFileSync(0, 'utf8'))
  : fs.existsSync(metadataPath) ? JSON.parse(fs.readFileSync(metadataPath, 'utf8'))
  : { authors: { en: 'EchoSparse Team', zh: 'EchoSparse Team' }, default_language: 'en' };
const defaultLanguage = post.default_language || 'en';
if (!['en', 'zh'].includes(defaultLanguage)) throw new Error('Unsupported default_language');
const routes = defaultLanguage === 'en' ? { en: 'index.html', zh: 'zh.html' } : { en: 'en.html', zh: 'index.html' };
const escapeHtml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const highlight = (str, lang) => {
  const esc = str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return lang === 'python' ? esc.replace(/#[^\n]*/g, c => `<span class="c">${c}</span>`) : esc;
};
const md = new MD({ html: true, typographer: false, highlight }).use(texmath, { engine: katex, delimiters: 'dollars', katexOptions: { throwOnError: false } });
const REPO = post.code || 'https://github.com/W2GenAI-Lab/EchoSparse-Attn';
const L = {
  zh: { file: 'echosparse_zh.md', out: routes.zh, lang: 'zh-CN', other: routes.en, otherName: 'English', toc: '目录', play: '播放', replay: '重播', foot: 'EchoSparse Team · W2GenAI Lab', alt: '原始 Canvas 动画', by: '作者' },
  en: { file: 'echosparse_en.md', out: routes.en, lang: 'en', other: routes.zh, otherName: '中文', toc: 'Contents', play: 'Play', replay: 'Replay', foot: 'EchoSparse Team · W2GenAI Lab', alt: 'Original Canvas animation', by: 'By' },
};
const PUBLIC_URL = 'https://w2genai-lab.github.io/blog/EchoSparse/';
const CSS = `
@font-face{font-family:"Inter";font-weight:400;font-style:normal;font-display:swap;src:url(assets/fonts/inter-latin-400-normal.woff2) format("woff2")}
@font-face{font-family:"Inter";font-weight:600;font-style:normal;font-display:swap;src:url(assets/fonts/inter-latin-600-normal.woff2) format("woff2")}
:root{--bg:#000;--ink:#f4f4f1;--ink2:#d4d5d1;--dim:#9a9ea6;--faint:#3c3f45;--hair:#202226;--cell:#141518;--teal:#45d0c1;--blue:#7ea9ff;--amber:#f2b544;--violet:#b9a3ff}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--bg);color:var(--ink2);font-family:"Inter","DejaVu Sans",system-ui,-apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:17px;line-height:1.85;-webkit-font-smoothing:antialiased}
body:lang(zh-CN){line-height:1.95;letter-spacing:.01em}
a{color:var(--teal);text-decoration:none;border-bottom:1px solid rgba(69,208,193,.35)}a:hover{border-bottom-color:var(--teal)}
.top{display:flex;justify-content:space-between;align-items:center;max-width:1180px;margin:0 auto;padding:22px 24px;font-size:14px;color:var(--dim)}
.top a{border:0;color:var(--dim)}.top a:hover{color:var(--ink)}.top .brand{color:var(--ink);font-weight:600;letter-spacing:.02em}
.top nav{display:flex;gap:22px}
.layout{max-width:1180px;margin:0 auto;padding:0 24px 80px;display:grid;grid-template-columns:minmax(0,1fr);gap:48px}
@media(min-width:1180px){.layout{grid-template-columns:210px minmax(0,1fr)}}
aside.toc{font-size:14px;line-height:1.5}
@media(min-width:1180px){aside.toc{position:sticky;top:24px;align-self:start;max-height:calc(100vh - 48px);overflow:auto;padding-top:8px}}
aside.toc summary{cursor:pointer;color:var(--dim);letter-spacing:.08em;text-transform:uppercase;font-size:12px;margin-bottom:10px}
@media(min-width:1180px){aside.toc summary{pointer-events:none;list-style:none}aside.toc summary::-webkit-details-marker{display:none}}
aside.toc ol{list-style:none;margin:0;padding:0}aside.toc li{margin:0}
aside.toc a{display:block;padding:6px 0 6px 12px;border:0;border-left:1px solid var(--hair);color:var(--dim)}
aside.toc a:hover,aside.toc a.on{color:var(--ink);border-left-color:var(--teal)}
article{min-width:0}
article>*{max-width:700px;margin-left:auto;margin-right:auto}
@media(min-width:1180px){article>*{margin-left:0!important}}
h1{font-size:clamp(30px,4.6vw,46px);line-height:1.18;font-weight:600;color:var(--ink);letter-spacing:-.015em;margin:12px 0 28px;max-width:860px}
article h2{font-size:26px;line-height:1.3;font-weight:600;color:var(--ink);margin-top:3.2em;margin-bottom:.8em;padding-top:1.1em;border-top:1px solid var(--hair);letter-spacing:-.005em}
.byline{color:var(--dim);font-size:15px;margin-bottom:2.2em!important}.byline strong{color:var(--ink2)}
article p{margin-top:1.05em;margin-bottom:1.05em}strong{color:var(--ink);font-weight:600}em{color:var(--ink2)}
ul{padding-left:1.3em}li{margin:.5em 0}li::marker{color:var(--faint)}
article{overflow-wrap:anywhere}code{overflow-wrap:anywhere;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.86em;background:var(--cell);border:1px solid var(--hair);border-radius:5px;padding:.08em .38em;color:var(--ink)}
.katex{color:var(--ink);font-size:1.06em}.katex-display{margin:1.4em 0;overflow-x:auto;overflow-y:hidden;padding:.3em 0}
section.eqn,.katex-display{max-width:700px}
figure{max-width:980px!important;margin:2.4em auto}figure img{display:block;width:100%;height:auto;border-radius:6px;border:1px solid var(--hair)}
figure.gif img{border-radius:6px}
figcaption{font-size:14.5px;line-height:1.7;color:var(--dim);max-width:700px;margin:.9em auto 0}
@media(min-width:1180px){figure{margin-left:0}figcaption{margin-left:0}}
figcaption em{font-style:normal;color:var(--dim)}
table{border-collapse:collapse;width:100%;max-width:700px;margin-top:1.8em;margin-bottom:1.8em;font-size:15.5px;font-variant-numeric:tabular-nums}
th,td{padding:.7em .8em;border-bottom:1px solid var(--hair);text-align:left}th{color:var(--dim);font-weight:400;font-size:13.5px;border-bottom:1px solid var(--faint)}
.tbl{max-width:700px;overflow-x:auto;margin-top:1.8em;margin-bottom:1.8em}.tbl table{margin:0;min-width:520px}
td .katex,th .katex{font-size:1em}.results tr:last-child td{color:var(--teal)}
pre{max-width:700px;background:var(--cell);border:1px solid var(--hair);border-radius:6px;padding:14px 16px;margin-top:1.6em;margin-bottom:1.6em;overflow-x:auto;font-size:13px;line-height:1.6;-webkit-overflow-scrolling:touch}
pre code{background:none;border:0;padding:0;font-size:inherit;color:var(--ink2);white-space:pre;overflow-wrap:normal;word-break:normal}
pre .c{color:var(--dim)}
article h3{font-size:19px;line-height:1.4;font-weight:600;color:var(--ink);margin-top:2.2em;margin-bottom:.6em}
.anim{max-width:980px!important;margin:2.4em auto}
.anim video{display:block;width:100%;aspect-ratio:16/9;background:#000;border:1px solid var(--hair);border-radius:6px;object-fit:contain}
a:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:3px}
footer{max-width:1180px;margin:0 auto;padding:30px 24px 60px;border-top:1px solid var(--hair);color:var(--dim);font-size:14px;display:flex;flex-wrap:wrap;gap:8px 24px;justify-content:space-between}
@media(max-width:700px){body{font-size:16px}h2{font-size:22px}.top{padding:16px}.layout{padding:0 16px 60px}.katex-display{font-size:.92em}th,td{padding:.6em .4em;font-size:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
`;
for (const [lg, T] of Object.entries(L)) {
  let src = fs.readFileSync(path.join(sourceDir, T.file), 'utf8');
  const sourceTitle = (src.match(/^# (.+)\r?\n/) || [])[1];
  const title = post.title?.[lg] || sourceTitle || 'EchoSparse';
  const author = post.authors?.[lg] || post.authors?.en || 'EchoSparse Team';
  src = src.replace(/^# .+\r?\n/, '').replace(/^\s*(?:作者|By)\s+\*\*[^*\n]+\*\*\s*\r?\n/, '').trim();
  let html = md.render(src);
  // headings -> ids + toc
  const toc = []; let n = 0;
  html = html.replace(/<h2>(.*?)<\/h2>/g, (_, t) => { const id = 's' + (++n); toc.push([id, t.replace(/<span class="katex-mathml">.*?<\/math><\/span>/g, '')]); return `<h2 id="${id}">${t}</h2>`; });
  html = html.replace(/<table>/g, '<div class="tbl"><table>').replace(/<\/table>/g, '</table></div>');
  html = html.replace(/<pre><code class="language-python">[\s\S]*?<\/code><\/pre>/, code => `<section class="lse-code-motion" data-lang="${lg}">${code}</section>`);
  const h1 = md.renderInline(title);
  // figures: <p><img></p><p><em>caption</em></p>
  html = html.replace(/<p>(<img [^>]*>)<\/p>\s*<p><em>([\s\S]*?)<\/em><\/p>/g, (_, img, cap) => `<figure>${img}<figcaption>${cap}</figcaption></figure>`);
  html = html.replace(/<p>(<img [^>]*H100[^>]*>|<img [^>]*fig4_h100[^>]*>)<\/p>/g, (_, img) => `<figure>${img}</figure>`);
  const anim = `<figure class="anim"><video id="anim" controls playsinline preload="metadata" poster="assets/echosparse-video-poster.png" aria-label="${T.alt}"><source src="assets/echosparse-v13-1080p.mp4" type="video/mp4"><a href="assets/echosparse-v13-1080p.mp4">MP4</a></video><figcaption>${lg === 'zh' ? '原版演示（早期实验）' : 'Original demonstration (early experiments)'}</figcaption></figure>`;
  html = html.replace(/<p><a href="assets\/echosparse-v13-1080p\.mp4"><img [^>]*echosparse-video-poster\.png[^>]*><\/a><\/p>|<figure><img [^>]*echosparse-explainer\.gif[^>]*><figcaption>[\s\S]*?<\/figcaption><\/figure>|<p><img [^>]*echosparse-explainer\.gif[^>]*><\/p>/, anim);
  html = html.replace(/<h1>.*?<\/h1>/, '');
  const page = `<!doctype html><html lang="${T.lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="author" content="${escapeHtml(author)}"><meta name="description" content="${escapeHtml(post.dek?.[lg] || '')}"><link rel="canonical" href="${PUBLIC_URL}${T.out === 'index.html' ? '' : T.out}"><link rel="alternate" hreflang="en" href="${PUBLIC_URL}${routes.en === 'index.html' ? '' : routes.en}"><link rel="alternate" hreflang="zh-CN" href="${PUBLIC_URL}${routes.zh === 'index.html' ? '' : routes.zh}"><title>${h1.replace(/<[^>]+>/g,'')}</title>
<link rel="stylesheet" href="assets/katex/katex.min.css"><link rel="stylesheet" href="assets/lse-code-motion.css"><style>${CSS}</style></head><body>
<header class="top"><a class="brand" href="${REPO}" target="_blank" rel="noopener">EchoSparse</a><nav><a href="${REPO}" target="_blank" rel="noopener">GitHub</a><a href="${T.other}" hreflang="${lg === 'zh' ? 'en' : 'zh'}">${T.otherName}</a></nav></header>
<div class="layout"><aside class="toc"><details open><summary>${T.toc}</summary><ol>${toc.map(([id, t]) => `<li><a href="#${id}">${t}</a></li>`).join('')}</ol></details></aside>
<article><h1>${h1}</h1><p class="byline">${T.by} <strong>${escapeHtml(author)}</strong></p>${html}</article></div>
<footer><span>${T.foot}</span><a href="${REPO}" target="_blank" rel="noopener">${REPO.replace('https://','')}</a></footer>
<script src="assets/lse-code-motion.js" defer></script><script>
const links=[...document.querySelectorAll('aside.toc a')],secs=links.map(a=>document.querySelector(a.getAttribute('href')));
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.remove('on'));links[secs.indexOf(e.target)].classList.add('on')}}),{rootMargin:'-20% 0px -70% 0px'});secs.forEach(s=>io.observe(s));
if(window.matchMedia('(max-width:1179px)').matches)document.querySelector('aside.toc details').removeAttribute('open');
</script></body></html>`;
  fs.writeFileSync(path.join(root, T.out), page);
  fs.writeFileSync(path.join(root, T.file), `# ${title}\n\n${T.by} **${author}**\n\n${src}\n`);
  console.log('wrote', T.out, toc.length, 'sections');
}
fs.writeFileSync(metadataPath, JSON.stringify(post, null, 2) + '\n');
for (const [lg, alias] of [['en', 'en.html'], ['zh', 'zh.html']]) {
  if (routes[lg] !== alias) fs.copyFileSync(path.join(root, routes[lg]), path.join(root, alias));
}
