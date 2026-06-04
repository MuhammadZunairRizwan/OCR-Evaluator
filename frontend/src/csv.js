import Papa from "papaparse";

/**
 * Parse a NARA CSV into a Map<naId, text>.
 *
 * Files are large (40-65 MB) with multi-line quoted text fields, so we parse
 * in a Web Worker and accumulate rows chunk-by-chunk to keep the UI responsive
 * and avoid building a giant intermediate array.
 *
 * The text column is auto-detected as the first header that isn't `naId`
 * (i.e. `transcriptionText` or `extracted_text`).
 */
export function parseNaraCsv(file, onProgress) {
  return new Promise((resolve, reject) => {
    const map = new Map();
    let textCol = null;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      chunk: (results) => {
        for (const row of results.data) {
          if (textCol === null) {
            const keys = Object.keys(row);
            textCol =
              keys.find((k) => k.toLowerCase() !== "naid") ?? keys[1] ?? null;
          }
          const id = row.naId ?? row.naid;
          if (id != null && String(id).trim() !== "") {
            map.set(String(id).trim(), textCol ? row[textCol] ?? "" : "");
          }
        }
        if (onProgress) onProgress(map.size);
      },
      complete: () => resolve(map),
      error: (err) => reject(err),
    });
  });
}
