// Render the page written by what-goes-in-a-trade-off-cell.py to a 2x PNG with headless Chromium.
// Usage: node what-goes-in-a-trade-off-cell.mjs <page.html> <out.png>
// Fails if any text box overlaps another or runs off the 1080 x 1350 canvas.
let playwright;
try {
  playwright = await import('playwright');
} catch {
  playwright = await import('/opt/node22/lib/node_modules/playwright/index.mjs');
}
const [page_path, out] = process.argv.slice(2);
const browser = await playwright.chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 2 });
await page.goto('file://' + page_path);
await page.evaluate(() => document.fonts.ready);
const problems = await page.evaluate(() => {
  const sel = '.num,.pill,.df,.dots,.note,.card,.chip.single,.aside,.colh,.legt,h1,.sub,.panel-h,.panel-s,.foot';
  const boxes = [...document.querySelectorAll(sel)].map(e => ({ t: e.textContent.trim().slice(0, 24), r: e.getBoundingClientRect() }));
  const out = [];
  for (const b of boxes) if (b.r.left < 20 || b.r.right > 1060 || b.r.bottom > 1340) out.push(`off canvas: ${b.t}`);
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i].r, c = boxes[j].r;
    if (Math.min(a.right, c.right) - Math.max(a.left, c.left) > 1 && Math.min(a.bottom, c.bottom) - Math.max(a.top, c.top) > 1)
      out.push(`overlap: ${boxes[i].t} / ${boxes[j].t}`);
  }
  return out;
});
await page.screenshot({ path: out });
await browser.close();
if (problems.length) { console.error(problems.join('\n')); process.exit(1); }
console.log('wrote', out);
