// End-to-end UI tests for the NARA CER/WER evaluator.
//
// Runs against a deployed (or local) frontend. The site ships a demo dataset
// that includes NAID 111406406, so the full flow works without uploading CSVs.
//
// Usage:
//   cd frontend
//   npm install
//   npx playwright install chromium
//   BASE_URL=https://ocr-evaluator.onrender.com npx playwright test
//   # or against local dev:  npm run dev  (then) BASE_URL=http://localhost:5173 npx playwright test
//
// The backend is on a free tier and may cold-start (~50s on the first request),
// so evaluation waits use generous timeouts.

import { test, expect } from "@playwright/test";

const NAID = "111406406";
const EVAL_TIMEOUT = 90_000; // allow for backend cold start

async function loadAndEvaluate(page, naid = NAID) {
  await page.goto("/");
  await expect(page.locator(".brand")).toHaveText(/NARA/);
  const input = page.locator(".naid-field input");
  await input.fill(naid);
  await page.getByRole("button", { name: "View record" }).click();
  // metrics appear once the backend responds
  await expect(page.locator(".metrics")).toBeVisible({ timeout: EVAL_TIMEOUT });
}

test.describe("NARA CER/WER UI", () => {
  test("TC1: page loads with header, intro, demo data, and controls", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/OCR|CER|WER/i);
    await expect(page.locator(".brand")).toContainText("CER / WER");
    await expect(page.locator(".naid-field input")).toBeVisible();
    await expect(page.getByRole("button", { name: "View record" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Random" })).toBeVisible();
    // four normalization toggles
    for (const t of ["Ignore case", "Ignore punctuation", "Ignore space", "Ignore newline"]) {
      await expect(page.getByText(t, { exact: true })).toBeVisible();
    }
    // demo dataset banner + two CSV pickers
    await expect(page.locator(".dataset-status")).toContainText(/Demo data loaded|Full dataset/);
    await expect(page.getByText("naid_transcriptions.csv")).toBeVisible();
    await expect(page.getByText("ocr_extraction.csv")).toBeVisible();
  });

  test("TC2: evaluate 111406406 shows correct scores", async ({ page }) => {
    await loadAndEvaluate(page);
    const cards = page.locator(".metric");
    await expect(cards).toHaveCount(4);
    // labels present
    await expect(page.locator(".metric-label", { hasText: "CER" })).toBeVisible();
    await expect(page.locator(".metric-label", { hasText: "WER" })).toBeVisible();
    await expect(page.getByText("Char Accuracy")).toBeVisible();
    await expect(page.getByText("Word Accuracy")).toBeVisible();
    // every metric value is a percentage
    const values = await page.locator(".metric-value").allInnerTexts();
    expect(values).toHaveLength(4);
    for (const v of values) expect(v).toMatch(/^\d+(\.\d+)?%$/);
    // breakdown line names the record
    await expect(page.locator(".breakdown")).toContainText(`NAID ${NAID}`);
    await expect(page.locator(".breakdown")).toContainText(/correct/);
  });

  test("TC3: side-by-side diff shows green and red highlighting", async ({ page }) => {
    await loadAndEvaluate(page);
    await expect(page.locator(".compare-col h3", { hasText: "Ground Truth" })).toBeVisible();
    await expect(page.locator(".compare-col h3", { hasText: "OCR Result" })).toBeVisible();
    // there is matched (green) and error (red) content
    await expect(page.locator(".seg-match").first()).toBeVisible();
    await expect(page.locator(".seg-error").first()).toBeVisible();
    expect(await page.locator(".seg-match").count()).toBeGreaterThan(0);
    expect(await page.locator(".seg-error").count()).toBeGreaterThan(0);
    // green is actually green, red is actually red
    const greenColor = await page.locator(".seg-match").first().evaluate(
      (el) => getComputedStyle(el).backgroundColor
    );
    const redColor = await page.locator(".seg-error").first().evaluate(
      (el) => getComputedStyle(el).backgroundColor
    );
    expect(greenColor).not.toBe(redColor);
  });

  test("TC4: blue 'moved' reordered content is present", async ({ page }) => {
    await loadAndEvaluate(page);
    // 111406406 has reordered content (DISTRICT OF GEORGIA) -> blue moved segments
    await expect(page.locator(".seg-moved").first()).toBeVisible({ timeout: EVAL_TIMEOUT });
    expect(await page.locator(".seg-moved").count()).toBeGreaterThan(0);
    await expect(page.locator(".breakdown")).toContainText(/moved/);
  });

  test("TC5: Word <-> Character toggle switches highlighting", async ({ page }) => {
    await loadAndEvaluate(page);
    const wordBtn = page.locator(".mode-toggle button", { hasText: "Word" });
    const charBtn = page.locator(".mode-toggle button", { hasText: "Character" });
    await expect(wordBtn).toHaveClass(/active/);
    await charBtn.click();
    await expect(charBtn).toHaveClass(/active/);
    // still renders highlighted content in character mode
    await expect(page.locator(".seg-match").first()).toBeVisible({ timeout: EVAL_TIMEOUT });
  });

  test("TC6: Horizontal align shows aligned rows + yellow fillers + legend", async ({ page }) => {
    await loadAndEvaluate(page);
    await page.getByText("Horizontal align", { exact: true }).click();
    await expect(page.locator(".aligned")).toBeVisible({ timeout: EVAL_TIMEOUT });
    await expect(page.locator(".align-row").first()).toBeVisible();
    expect(await page.locator(".align-row").count()).toBeGreaterThan(5);
    // yellow filler cells appear (one side blank to align)
    expect(await page.locator(".align-filler").count()).toBeGreaterThan(0);
    // blue moved rows
    expect(await page.locator(".align-row.moved").count()).toBeGreaterThan(0);
    // legend present
    await expect(page.locator(".align-legend")).toContainText(/filler/);
  });

  test("TC7: ignore toggles change the score live", async ({ page }) => {
    await loadAndEvaluate(page);
    // CER is the first metric card; target it precisely (avoid the "1 − CER" hint)
    const cer = () => page.locator(".metric").nth(0).locator(".metric-value").innerText();
    const before = await cer();
    await page.getByText("Ignore punctuation", { exact: true }).click();
    // CER should change (and the request re-runs)
    await expect.poll(async () => await cer(), { timeout: EVAL_TIMEOUT }).not.toBe(before);
  });

  test("TC8: Random loads a record", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Random" }).click();
    await expect(page.locator(".metrics")).toBeVisible({ timeout: EVAL_TIMEOUT });
    await expect(page.locator(".naid-field input")).not.toHaveValue("");
  });

  test("TC9: invalid NAID shows an error, not a crash", async ({ page }) => {
    await page.goto("/");
    await page.locator(".naid-field input").fill("000000000");
    await page.getByRole("button", { name: "View record" }).click();
    await expect(page.locator(".error")).toBeVisible();
    await expect(page.locator(".error")).toContainText(/not found/i);
    await expect(page.locator(".metrics")).toHaveCount(0);
  });

  test("TC10: empty NAID is handled gracefully", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "View record" }).click();
    await expect(page.locator(".error")).toBeVisible();
  });
});
