export default async function run(page, ui) {
  await page.waitForSelector('input[aria-label="Customer name"]', { timeout: 90000 });
  await page.waitForFunction(() => document.body.innerText.includes('Run assessment'), null, { timeout: 90000 });
  await page.waitForTimeout(2000);
  const out = {};
  // sample the value digits rendered in number inputs
  out.spinners = await page.evaluate(() =>
    [...document.querySelectorAll('input[type="number"], input[role="spinbutton"], input[kind="NumberInputElement"]')]
      .slice(0, 8).map(el => ({ label: el.getAttribute('aria-label'), val: el.value }))
  );
  out.lang = await page.evaluate(() => document.documentElement.lang || '(none)');
  out.dir = await page.evaluate(() => getComputedStyle(document.body).direction);
  await page.screenshot({ path: 'numcheck.png' });
  return out;
}
