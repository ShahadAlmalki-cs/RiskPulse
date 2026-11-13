export default async function run(page) {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.reload();
  await page.waitForSelector('input[aria-label="Customer name"]', { timeout: 90000 });
  await page.waitForTimeout(1000);
  return page.evaluate(() => {
    const out = {};
    const app = document.querySelector('.stApp');
    const key = Object.keys(app).find(k => k.startsWith('__reactFiber$'));
    let node = key ? app[key] : null;
    const candidates = [];
    for (let i = 0; i < 120 && node; i++) {
      const p = node.memoizedProps;
      if (p && p.theme && typeof p.theme === 'object' && p.theme.colors) {
        const t = p.theme, c = t.colors;
        candidates.push({
          keys: Object.keys(t).slice(0, 14),
          secondaryBg: c.secondaryBg, bgColor: c.bgColor, bodyText: c.bodyText,
          primary: c.primary, borderColor: c.borderColor,
          grayText: c.grayText60 || c.grayText70,
          sidebarBg: t.sidebar && t.sidebar.colors ? t.sidebar.colors.bgColor : null,
        });
      }
      node = node.return;
    }
    out.count = candidates.length;
    out.themes = candidates.slice(0, 3);
    // Full colour key set from the first theme, to mirror faithfully.
    node = key ? app[key] : null;
    for (let i = 0; i < 120 && node; i++) {
      const p = node.memoizedProps;
      if (p && p.theme && p.theme.colors) {
        out.allKeys = Object.keys(p.theme.colors);
        out.sidebarColors = p.theme.sidebar && p.theme.sidebar.colors ? p.theme.sidebar.colors : null;
        out.generic = Object.keys(p.theme).filter(k => !['colors', 'sidebar', 'spacing', 'radii', 'sizes', 'fontSizes', 'fontWeights', 'fonts', 'genericFonts'].includes(k));
        break;
      }
      node = node.return;
    }
    return out;
  });
}
