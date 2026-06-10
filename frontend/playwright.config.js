import { defineConfig, devices } from "@playwright/test";

// Target site: live by default, override with BASE_URL for local dev.
const BASE_URL = process.env.BASE_URL || "https://ocr-evaluator.onrender.com";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000, // generous: free-tier backend may cold-start
  expect: { timeout: 15_000 },
  fullyParallel: false, // share the warmed-up backend; avoid hammering free tier
  workers: 1,
  retries: 1, // one retry absorbs a cold-start blip
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: BASE_URL,
    headless: true,
    actionTimeout: 20_000,
    navigationTimeout: 60_000,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
