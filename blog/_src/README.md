# Blog sources

`blog/index.html` and `blog/EchoSparse/index.html` are generated from the Markdown in `src/` (English and Chinese)
by `build.py`. Jekyll does not publish this folder (its name starts with an underscore).

To change the post, edit `src/echosparse_en.md` and `src/echosparse_zh.md` (title, summary and byline are in `POST`
at the top of `build.py`), then from the repository root:

```bash
pip install markdown pymdown-extensions pygments
python blog/_src/build.py .
```

Figures live in `blog/EchoSparse/assets/`. Math is written as `$$...$$` (display) or `\(...\)` (inline) and rendered
with KaTeX in the browser.
