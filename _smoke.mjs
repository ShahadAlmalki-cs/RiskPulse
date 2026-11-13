export default async function run(page) {
  await page.waitForSelector('input[aria-label="Customer name"]', { timeout: 90000 });
  await page.getByRole('button', { name: /Run assessment/ }).click();
  await page.waitForFunction(() => document.body.innerText.includes('Recommended next steps'), null, { timeout: 60000 });
  await page.waitForTimeout(2000);

  const state = await page.evaluate(() => {
    const text = document.body.innerText;
    const heading = document.querySelector('.rp-hero-title');
    return {
      title: document.title,
      decision: heading ? heading.textContent.trim() : null,
      hasScores: text.includes('Model Default Probability') && text.includes('Blended Risk Score'),
      hasPerformance: text.includes('Model performance'),
      hasNextSteps: text.includes('Recommended next steps'),
      historyRows: document.querySelectorAll('[data-testid="stDataFrame"] canvas').length,
    };
  });

  await page.screenshot({ path: 'C:/Users/shaha/OneDrive/سطح المكتب/RiskPulse/_run_shot.png', fullPage: true });
  return state;
}
