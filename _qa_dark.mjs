export default async function run(page, ui) {
  // Switch Streamlit to the dark theme through its own settings menu.
  await page.getByRole('button', { name: /Deploy|Settings|menu/i }).first().click().catch(() => { })
  await page.waitForTimeout(800)
  const snap = await ui.snapshot()
  return { snap }
}
