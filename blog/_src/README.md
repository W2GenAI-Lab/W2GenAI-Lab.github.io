# Blog sources

`build.py` generates `blog/index.html` and the EchoSparse article from the English and Chinese Markdown in
`src/`. Jekyll does not publish this folder (its name starts with an underscore).

Edit `src/echosparse_en.md` and `src/echosparse_zh.md` for the article body. Edit `POST` at the beginning of
`build.py` for its title, summary, byline and default language. The current byline is **EchoSparse Team** and
`default_language="en"`. Do not edit generated HTML directly: the next build replaces it.

From the repository root, install the authoring dependencies and build:

```bash
pip install markdown pymdown-extensions pygments
npm install --prefix blog/_src
python3 blog/_src/build.py .
```

Python retains the blog listing generator and invokes `blog/EchoSparse/sources/build_pages.js` for the article,
passing `POST` as JSON on stdin and `src/` as the source directory. This preserves the article's black Canvas
style and keeps its author metadata and language selection tied to `POST`.

The article opens in English at `blog/EchoSparse/blog.html`. Chinese is at `zh.html`; `index.html` and `en.html`
redirect to the English page. Figures, the original animation, local fonts and KaTeX assets live in
`blog/EchoSparse/assets/`. Write math as `$...$` inline or `$$...$$` for display. The article renderer compiles
math during the build, so the generated pages do not need a network request to render it.
