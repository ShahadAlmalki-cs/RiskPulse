export default async function run(page, ui) {
  await page.getByRole('button', { name: 'Run assessment' }).click()
  await page.waitForTimeout(4000)
  await page.screenshot({ path: 'C:/Users/shaha/OneDrive/سطح المكتب/RiskPulse/_shot_light.png', fullPage: true })
  const body = await page.evaluate(() => document.body.innerText.slice(0, 1200))
  return { body }
}
