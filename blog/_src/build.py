"""Build the static blog pages for w2genai-lab.github.io/blog/ from the Markdown sources.

  python blog/_src/build.py .     # from the repository root: rewrites blog/index.html and blog/EchoSparse/index.html
"""
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

HERE = Path(__file__).resolve().parent
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500"
         "&family=Noto+Sans+SC:wght@400;500;700&display=swap")
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"

POST = dict(
    slug="EchoSparse",
    default_language="en",
    date="2026-11-01",
    date_text=dict(en="November 1, 2026", zh="2026 年 11 月 1 日"),
    authors=dict(en="EchoSparse Team", zh="EchoSparse Team"),
    title=dict(en="EchoSparse: Letting Last Step's Statistics Steer This Step, and Packing Sparse Attention Tightly",
               zh="EchoSparse：让上一步的统计量为下一步指路，再把稀疏 attention 排得更紧"),
    dek=dict(en="EchoSparse reuses LSE and block-mass statistics across denoising steps, combines exact and "
                "representative terms in one weighted softmax, and organizes packed work for TMA and GPU pipelining.",
             zh="EchoSparse 在去噪步之间复用 LSE 与块质量统计量，让精算 token 和代表项共用一个加权 softmax，"
                "再通过 packed 布局、TMA 与 GPU 流水线组织计算。"),
    teaser_alt=dict(en="EchoSparse on Wan2.1-14B: 1.70× faster than FlashAttention-4 on a B200, closer to the dense "
                       "video than Sol-Attn and PISA",
                    zh="EchoSparse 在 Wan2.1-14B 上：B200 上比 FlashAttention-4 快 1.70 倍，比 Sol-Attn 和 PISA 更接近"
                       "稠密结果"),
    code="https://github.com/W2GenAI-Lab/EchoSparse-Attn",
)

UI = dict(
    en=dict(blog="Blog", lab="Lab", code="Code", by="", read="Read", toggle="中文", footer="W2GenAI Lab"),
    zh=dict(blog="博客", lab="实验室", code="代码", by="", read="阅读", toggle="EN", footer="W2GenAI Lab"),
)

CSS = r"""
:root{--bg:#fbfbfd;--surface:#ffffff;--text:#1d1d24;--text-2:#4b4b57;--muted:#6e6e7a;--rule:#e6e6ec;
--accent:#5a5fd6;--accent-soft:#eef0ff;--code-bg:#f4f4f8;--shadow:0 1px 2px rgba(20,20,40,.06),0 8px 24px rgba(20,20,40,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0f1015;--surface:#16171e;--text:#e9e9ef;--text-2:#c2c2cc;
--muted:#93939f;--rule:#2a2b35;--accent:#9da2ff;--accent-soft:#1e2040;--code-bg:#1b1c24;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);color-scheme:dark}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);
font:17px/1.75 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Noto Sans SC","Microsoft YaHei",sans-serif;
font-feature-settings:"cv11","ss01";-webkit-font-smoothing:antialiased}
:lang(zh-CN) body,article:lang(zh-CN){line-height:1.85}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline;text-underline-offset:3px}
.site{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 86%,transparent);
backdrop-filter:saturate(1.4) blur(10px);-webkit-backdrop-filter:saturate(1.4) blur(10px);border-bottom:1px solid var(--rule)}
.site-in{max-width:1080px;margin:0 auto;display:flex;align-items:center;gap:16px;height:56px;
padding-left:max(20px,env(safe-area-inset-left,0px));padding-right:max(20px,env(safe-area-inset-right,0px))}
.brand{font-weight:700;color:var(--text);letter-spacing:-.01em}
.brand span{color:var(--muted);font-weight:500}
.site nav{margin-left:auto;display:flex;align-items:center;gap:18px;font-size:14px}
.site nav a{color:var(--text-2)}
.lang{border:1px solid var(--rule);background:var(--surface);color:var(--text);border-radius:999px;
font:500 13px/1 inherit;padding:7px 12px;cursor:pointer}
.lang:hover{border-color:var(--accent);color:var(--accent)}
main{padding-left:max(20px,env(safe-area-inset-left,0px));padding-right:max(20px,env(safe-area-inset-right,0px))}
.post{max-width:720px;margin:0 auto;padding:56px 0 24px}
.kicker{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 14px}
h1{font-size:clamp(28px,4.4vw,40px);line-height:1.18;letter-spacing:-.022em;margin:0 0 18px;font-weight:700;text-wrap:balance}
.dek{font-size:19px;line-height:1.65;color:var(--text-2);margin:0 0 22px}
.byline{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;font-size:14px;color:var(--muted);
padding-bottom:28px;border-bottom:1px solid var(--rule)}
.byline strong{color:var(--text);font-weight:600}
.btn{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--rule);background:var(--surface);
color:var(--text);border-radius:999px;padding:5px 12px;font-weight:500}
.btn:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}
.btn svg{width:15px;height:15px}
figure{margin:32px 0}
figure img,picture img{display:block;width:100%;height:auto;border-radius:10px}
.hero,figure.wide,.post picture{display:block;width:min(1040px,calc(100vw - 40px));margin-left:50%;transform:translateX(-50%)}
.hero{margin-top:32px;margin-bottom:8px}
.hero img{box-shadow:var(--shadow);background:#fff}
.post picture{margin-top:28px;margin-bottom:20px}
.post picture img{background:var(--surface);border:1px solid var(--rule)}
figcaption{font-size:14px;color:var(--muted);margin-top:10px;text-align:center;line-height:1.6}
h2{font-size:24px;line-height:1.3;letter-spacing:-.015em;margin:52px 0 14px;font-weight:700;text-wrap:balance}
h3{font-size:19px;margin:36px 0 10px;font-weight:600}
p,ul,ol{margin:0 0 18px}
li{margin:4px 0}
li::marker{color:var(--muted)}
strong{font-weight:600}
code{font:500 .86em/1.5 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code-bg);
padding:.12em .38em;border-radius:5px}
pre{background:var(--code-bg);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;overflow-x:auto;
font-size:14px;line-height:1.6;margin:22px 0}
pre code{background:none;padding:0;font-size:inherit}
.table{overflow-x:auto;margin:22px 0}
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th{font-weight:600;text-align:left;color:var(--text);border-bottom:1.5px solid var(--text);padding:8px 12px 8px 0}
td{border-bottom:1px solid var(--rule);padding:8px 12px 8px 0;color:var(--text-2)}
tr:last-child td{border-bottom:none}
.arithmatex{overflow-x:auto;overflow-y:hidden}
div.arithmatex{margin:20px 0;padding:6px 0}
.katex{font-size:1.06em}
.site-foot{max-width:720px;margin:56px auto 0;padding:24px 20px 48px;border-top:1px solid var(--rule);
font-size:14px;color:var(--muted);display:flex;gap:16px;flex-wrap:wrap}
.site-foot a{color:var(--muted)}
[hidden]{display:none!important}
.list{max-width:720px;margin:0 auto;padding:64px 0 24px}
.list h1{margin-bottom:8px}
.list .intro{color:var(--text-2);margin:0 0 36px}
.entry{display:block;padding:26px 0;border-top:1px solid var(--rule);color:inherit}
.entry:hover{text-decoration:none}
.entry:hover .entry-title{color:var(--accent)}
.entry-meta{font-size:13px;color:var(--muted);margin-bottom:6px;letter-spacing:.02em}
.entry-title{font-size:22px;line-height:1.3;font-weight:700;letter-spacing:-.012em;margin:0 0 8px;transition:color .15s}
.entry-dek{color:var(--text-2);margin:0;font-size:16px;line-height:1.7}
@media (max-width:600px){body{font-size:16px}.dek{font-size:17px}.site nav{gap:12px}.site nav a.hide-sm{display:none}
.post{padding-top:36px}h2{font-size:21px;margin-top:40px}}
"""

GITHUB_SVG = ('<svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 '
              '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
              '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78'
              '-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27'
              '.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75'
              '-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8'
              '-8-8z"/></svg>')

LANG_JS = r"""
(function(){
  var KEY='w2genai-blog-lang';
  function get(){try{return localStorage.getItem(KEY)}catch(e){return null}}
  function set(v){try{localStorage.setItem(KEY,v)}catch(e){}}
  var q=(location.search.match(/[?&]lang=(en|zh)/)||[])[1];
  var lang=q||get()||'en';
  function apply(l){
    document.querySelectorAll('[data-lang]').forEach(function(el){el.hidden=el.getAttribute('data-lang')!==l});
    document.documentElement.lang=l==='zh'?'zh-CN':'en';
    var t=document.querySelector('[data-title-'+l+']');
    if(t)document.title=t.getAttribute('data-title-'+l);
    var b=document.querySelector('.lang');
    if(b)b.textContent=l==='zh'?'EN':'中文';
    lang=l;
  }
  apply(lang);
  var b=document.querySelector('.lang');
  if(b)b.addEventListener('click',function(){var n=lang==='zh'?'en':'zh';set(n);apply(n)});
})();
"""

KATEX_JS = r"""
document.addEventListener('DOMContentLoaded',function(){
  if(window.renderMathInElement){document.querySelectorAll('.arithmatex').forEach(function(el){
    renderMathInElement(el,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],
      throwOnError:false});});}
});
"""


def md_to_html(text):
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "codehilite", "pymdownx.arithmatex"],
                            extension_configs={"codehilite": {"css_class": "hl", "guess_lang": False},
                                               "pymdownx.arithmatex": {"generic": True}})
    return out.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")


def syntax_css():
    light = HtmlFormatter(style="friendly").get_style_defs(".hl")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".hl")
    dark = "\n".join("  " + line for line in dark.splitlines())
    return (light + "\n.hl, .hl pre{background:var(--code-bg)!important}\n"
            "@media (prefers-color-scheme:dark){\n" + dark + "\n.hl, .hl pre{background:var(--code-bg)!important}}\n")


def head(title_en, title_zh, description, canonical, math):
    katex = (f'<link rel="stylesheet" href="{KATEX}/katex.min.css">\n'
             f'<script defer src="{KATEX}/katex.min.js"></script>\n'
             f'<script defer src="{KATEX}/contrib/auto-render.min.js"></script>\n'
             f'<script>{KATEX_JS}</script>\n') if math else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title_en)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="W2GenAI Lab">
<meta property="og:title" content="{html.escape(title_en)}">
<meta property="og:description" content="{html.escape(description)}">
<meta property="og:url" content="{canonical}">
<meta name="twitter:card" content="summary_large_image">
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
{katex}<style>{CSS}{syntax_css()}</style>
</head>
<body data-title-en="{html.escape(title_en)}" data-title-zh="{html.escape(title_zh)}">
<header class="site"><div class="site-in">
  <a class="brand" href="/blog/">W2GenAI Lab <span>/ <span data-lang="en">Blog</span><span data-lang="zh" hidden>博客</span></span></a>
  <nav>
    <a class="hide-sm" href="/"><span data-lang="en">Lab</span><span data-lang="zh" hidden>实验室主页</span></a>
    <a class="hide-sm" href="https://github.com/W2GenAI-Lab">GitHub</a>
    <button class="lang" type="button" aria-label="Switch language">中文</button>
  </nav>
</div></header>
"""


def foot():
    return f"""<footer class="site-foot">
  <span>© 2026 W2GenAI Lab</span>
  <a href="/">w2genai-lab.github.io</a>
  <a href="https://github.com/W2GenAI-Lab">GitHub</a>
</footer>
<script>{LANG_JS}</script>
</body>
</html>
"""


def post_page():
    p = POST
    parts = [head(p["title"]["en"] + " · W2GenAI Lab", p["title"]["zh"] + " · W2GenAI Lab", p["dek"]["en"],
                  f"https://w2genai-lab.github.io/blog/{p['slug']}/", math=True), "<main>\n"]
    for lang in ("en", "zh"):
        body = md_to_html((HERE / "src" / f"echosparse_{lang}.md").read_text())
        hidden = "" if lang == "en" else " hidden"
        kicker = "Research note" if lang == "en" else "研究笔记"
        parts.append(f"""<article class="post" data-lang="{lang}" lang="{'en' if lang == 'en' else 'zh-CN'}"{hidden}>
  <p class="kicker">{kicker} · {p['date_text'][lang]}</p>
  <h1>{html.escape(p['title'][lang])}</h1>
  <p class="dek">{html.escape(p['dek'][lang])}</p>
  <div class="byline"><strong>{p['authors'][lang]}</strong><span>W2GenAI Lab</span>
    <a class="btn" href="{p['code']}">{GITHUB_SVG}{UI[lang]['code']}</a></div>
  <figure class="hero"><img src="assets/teaser.png" alt="{html.escape(p['teaser_alt'][lang])}"></figure>
{body}
</article>
""")
    parts.append("</main>\n")
    parts.append(foot())
    return "".join(parts)


def index_page():
    p = POST
    parts = [head("Blog · W2GenAI Lab", "博客 · W2GenAI Lab", "Research notes from W2GenAI Lab.",
                  "https://w2genai-lab.github.io/blog/", math=False), "<main>\n"]
    for lang in ("en", "zh"):
        hidden = "" if lang == "en" else " hidden"
        title = "Blog" if lang == "en" else "博客"
        intro = ("Research notes from W2GenAI Lab." if lang == "en" else "W2GenAI Lab 的研究笔记。")
        parts.append(f"""<section class="list" data-lang="{lang}"{hidden}>
  <h1>{title}</h1>
  <p class="intro">{intro}</p>
  <a class="entry" href="{p['slug']}/">
    <div class="entry-meta">{p['date_text'][lang]}</div>
    <h2 class="entry-title">{html.escape(p['title'][lang])}</h2>
    <p class="entry-dek">{html.escape(p['dek'][lang])}</p>
  </a>
</section>
""")
    parts.append("</main>\n")
    parts.append(foot())
    return "".join(parts)


def main():
    out = Path(sys.argv[1]).resolve() / "blog"
    post = out / POST["slug"]
    (post / "assets").mkdir(parents=True, exist_ok=True)
    if (HERE / "assets").is_dir():           # figures already in blog/EchoSparse/assets are left as they are
        for src in (HERE / "assets").iterdir():
            shutil.copy2(src, post / "assets" / src.name)
    (out / "index.html").write_text(index_page(), encoding="utf-8")
    renderer = post / "sources" / "build_pages.js"
    if renderer.is_file():
        env = os.environ.copy()
        node_paths = [str(HERE / "node_modules")]
        if env.get("NODE_PATH"):
            node_paths.append(env["NODE_PATH"])
        env["NODE_PATH"] = os.pathsep.join(node_paths)
        subprocess.run(
            ["node", str(renderer), "--source-dir", str(HERE / "src"), "--metadata-stdin"],
            input=json.dumps(POST, ensure_ascii=False), text=True, encoding="utf-8",
            cwd=post, env=env, check=True,
        )
    else:                                   # legacy checkout without the custom article renderer
        (post / "index.html").write_text(post_page(), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
