import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    }
  ],
  webServer: [
    {
      command: 'python -m uvicorn faulttrace_api.main:app --host 127.0.0.1 --port 8000 --app-dir ../api',
      url: 'http://127.0.0.1:8000/api/v1/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
      gracefulShutdown: { signal: 'SIGINT', timeout: 500 },
    },
    {
      command: 'node node_modules/next/dist/bin/next start',
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
      gracefulShutdown: { signal: 'SIGINT', timeout: 500 },
      env: {
        HOSTNAME: '127.0.0.1',
        PORT: '3000',
        INTERNAL_API_URL: 'http://127.0.0.1:8000',
      },
    },
  ],
});
