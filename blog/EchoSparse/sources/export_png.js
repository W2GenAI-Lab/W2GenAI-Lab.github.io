// node export_png.js <assets dir> : renders every fig*.svg to a 2x PNG next to it (needs `playwright` + chromium)
const { chromium } = require('playwright'); const fs = require('fs'), path = require('path');
(async () => {
  const dir = process.argv[2]; const b = await chromium.launch();
  for (const f of fs.readdirSync(dir).filter(f => /^fig.*\.svg$/.test(f))) {
    const svg = fs.readFileSync(path.join(dir, f), 'utf8'); const m = svg.match(/viewBox="0 0 (\d+) (\d+)"/);
    const w = +m[1], h = +m[2]; const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await p.goto('file://' + path.resolve(dir, f)); await p.waitForTimeout(300);
    await p.screenshot({ path: path.join(dir, f.replace('.svg', '.png')) }); await p.close();
  }
  await b.close();
})();
