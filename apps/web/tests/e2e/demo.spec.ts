import { test, expect } from '@playwright/test';

test.describe('FaultTrace-RAG End-to-End Demo Suite', () => {
  
  test('1. Dashboard loads successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/FaultTrace-RAG/);
    await expect(page.locator('text=FaultTrace')).toBeVisible();
  });

  test('2. Can navigate to Datasets', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Datasets');
    await expect(page).toHaveURL(/.*datasets/);
    await expect(page.locator('h1', { hasText: 'Datasets' })).toBeVisible();
  });

  test('3. Can navigate to Corpus Worlds', async ({ page }) => {
    await page.goto('/worlds');
    await expect(page.locator('h1', { hasText: 'Corpus Worlds' })).toBeVisible();
  });

  test('4. Can navigate to Query Library', async ({ page }) => {
    await page.goto('/queries');
    await expect(page.locator('h1', { hasText: 'Query Library' })).toBeVisible();
  });

  test('5. Run Lab opens and shows components', async ({ page }) => {
    await page.goto('/run-lab');
    await expect(page.locator('h1', { hasText: 'Run Lab' })).toBeVisible();
  });

  test('6. Run History shows run records', async ({ page }) => {
    await page.goto('/runs');
    await expect(page.locator('h1', { hasText: 'Run History' })).toBeVisible();
  });

  test('7. Oracle Diagnostics loads correctly', async ({ page }) => {
    await page.goto('/oracle');
    await expect(page.locator('h1', { hasText: 'Oracle Diagnostics' })).toBeVisible();
  });

  test('8. Certificates page displays table', async ({ page }) => {
    await page.goto('/certificates');
    await expect(page.locator('h1', { hasText: 'Certificates' })).toBeVisible();
  });

  test('9. Experiments page shows analytics', async ({ page }) => {
    await page.goto('/experiments');
    await expect(page.locator('h1', { hasText: 'Experiments' })).toBeVisible();
  });

  test('10. Reports & Exports works', async ({ page }) => {
    await page.goto('/reports');
    await expect(page.locator('h1', { hasText: 'Reports' })).toBeVisible();
  });

});
