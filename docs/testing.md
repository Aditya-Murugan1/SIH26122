# Testing

`pytest -q` exercises report ingestion, extraction, candidate evidence, review confirmation, audit writes, Time Agent reuse, CPM availability, and malformed file errors.

The benchmark generator creates 280 activities over all seven disciplines and 60 reports with terminology substitutions, OCR-like noise, missing IDs, near duplicates, and unmatched cases. Fifteen percent (9 cases) is placed in a separate held-out file. `scripts/run_benchmark.py` prints measured auto-match precision, silent errors, routing rates, Recall@5, and latency. The current corpus is synthetic and intentionally small; it is not a claim about field accuracy.

The test database is isolated through `PROGRESSSYNC_DB`. No benchmark metric is stored as a claim in source; run output is authoritative.

Quality commands: `python -m py_compile backend/app.py`, `node --check frontend/app.js`, `ruff check backend scripts tests`, `npx eslint frontend/app.js`, and `pytest -q`.
