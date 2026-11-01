# EchoSparse blog

By EchoSparse Team. Publication target: https://w2genai-lab.github.io/blog/EchoSparse/ .

- `index.html`: English, the default page. `en.html` is a compatible English alias.
- `zh.html`: 中文版。
- `echosparse_en.md` and `echosparse_zh.md`: complete Markdown exports.
- `assets/`: figures, local fonts and math assets, the GIF and the original Canvas animation.
- `sources/`: editable draw.io figures and rendering scripts; `data/` and `evidence/` retain the plotted observations.

## 离线打开

将整个文件夹复制到本地并解压，双击 `index.html` 看英文，或双击 `zh.html` 看中文。请保留 `assets/` 等子目录；阅读不需要服务器、联网或安装依赖。

## Edit and rebuild in the website repository

Edit `blog/_src/src/echosparse_en.md` and `echosparse_zh.md` for the article. Edit `POST` near the beginning of
`blog/_src/build.py` for the authors, title, description and default language. From the repository root:

```bash
pip install markdown pymdown-extensions pygments
npm install --prefix blog/_src
python3 blog/_src/build.py .
```

This regenerates the blog listing, English default, Chinese page, English alias, metadata snapshot and Markdown
exports. Do not edit generated HTML directly. The latest upstream RTX 5090 light/dark result charts are retained
unchanged; the article uses `assets/results-rtx5090-dark.png`.

## Standalone copy

`post.json` is the exported metadata snapshot. A standalone copy can render its local Markdown with
`node sources/build_pages.js` when `markdown-it`, `markdown-it-texmath` and `katex` are installed. The full
website workflow above remains the canonical authoring route.
