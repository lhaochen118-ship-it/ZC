// Render every out/*.html (one SVG each) to a transparent PNG at 2x, plus an opaque preview on the theme's page colour.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const dir = process.argv[2];
  const out = process.argv[3];
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1200, height: 800 } });
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.html'))) {
    await page.goto('file://' + path.resolve(dir, f));
    await page.evaluate(() => document.fonts.ready);
    const svg = await page.$('svg');
    const base = f.replace('.html', '');
    await svg.screenshot({ path: path.join(out, base + '.png'), omitBackground: true });
    // preview with page background for visual checks
    const bg = base.endsWith('dark') ? '#0d1117' : '#ffffff';
    await page.evaluate((c) => { document.body.style.background = c; }, bg);
    await svg.screenshot({ path: path.join(out, 'preview_' + base + '.png') });
  }
  await browser.close();
  console.log('rendered');
})();
