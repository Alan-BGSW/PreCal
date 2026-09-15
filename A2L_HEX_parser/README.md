# A2L / HEX / CDFX Calibration Pipeline — Web App

Streamlit wrapper around `parser.py` (A2L+HEX → JSON) and `pipeline.py`
(multi-source CDFX update, extracted from `preCal_v1.ipynb`).

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>.

## How it works (matches the notebook flow)

1. **Upload destination CDFX** — the file that will be updated in place.
2. **Upload sources** — any mix of `.CDFX`, `.json`, and `.a2l` + `.hex`.
   Order in the uploader = priority (first source to provide a label wins).
3. **Parse A2L + HEX → JSON** — auto-detected pairs (same stem, or the
   single unique pair in the upload) are passed through
   `CalibrationExtractor` from `parser.py`. The produced JSONs are
   prepended to the source list.
4. **Run pipeline** — calls `pipeline.run_pipeline(...)` which is the
   exact orchestrator from the notebook's second code cell.
5. **Download** — updated CDFX, `needs_attention_labels.xlsx`, and the
   process log.

## Deploy

- **Streamlit Community Cloud** — push to GitHub, point app at `app.py`.
- **Docker** — `docker build -t cal-pipeline . && docker run -p 8501:8501 cal-pipeline`.

## Files

| File                     | Role                                               |
| ------------------------ | -------------------------------------------------- |
| `app.py`                 | Streamlit UI                                       |
| `pipeline.py`            | CDFX/JSON → CDFX update logic (from notebook)      |
| `parser.py`              | A2L + HEX → JSON (unchanged)                       |
| `preCal_v1.ipynb`        | Original notebook (kept for reference / debugging) |
| `requirements.txt`       | Python deps                                        |
| `.streamlit/config.toml` | 500 MB upload cap                                  |
| `Dockerfile`             | Optional container build                           |

The notebook is **not** modified — `pipeline.py` is a clean extraction of
its functions so the UI and the notebook can coexist.
