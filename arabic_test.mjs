export default async function run(page, ui) {
  await page.waitForSelector('input[aria-label="Customer name"]', { timeout: 90000 });
  await page.waitForFunction(() => document.body.innerText.includes('Run assessment'), null, { timeout: 90000 });
  await page.waitForTimeout(2000);
  const out = {};

  // Type Arabic-Indic digits into Annual income and check the pipeline accepts them
  const income = page.getByRole('textbox', { name: 'Annual income' });
  await income.fill('٦٠٠٠٠');           // Arabic-Indic 60000
  await page.waitForTimeout(1500);
  const loan = page.getByRole('textbox', { name: 'Loan amount' });
  await loan.fill('١٠٠٠٠');             // Arabic-Indic 10000
  await page.waitForTimeout(2500);

  await page.getByRole('button', { name: /Run assessment/ }).click();
  await page.waitForFunction(() => document.body.innerText.includes('Decision'), null, { timeout: 60000 });
  await page.waitForTimeout(4000);
  await page.evaluate(() => { document.querySelector('main').scrollTop = 0; });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'arabic_result.png' });
  const text = await page.evaluate(() => document.body.innerText);
  out.assessmentRan = text.includes('Decision');
  out.lti = (text.match(/LTI\s*\n?\s*([\d.]+%)/) || [])[1];
  // English digit inputs display check
  out.digitSample = await page.evaluate(() =>
    [...document.querySelectorAll('input')].slice(0, 6).map(el => el.value).join(' | '));
  return out;
}
