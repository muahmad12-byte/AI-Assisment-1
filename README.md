# AI-Assisment-1

# 📄 AI Resume ATS Analyzer

Upload a resume (PDF, DOCX or TXT) and get an **ATS score (0-100)**, a category
breakdown, problems found, concrete improvements, missing keywords and rewritten
bullet points. Optionally paste a job description to get a **job-match score**.

Built with **Streamlit** + **Google Gemini Flash** (`google-genai` SDK).

## Features
- ATS score with 5 category scores (formatting, keywords, impact, structure, readability)
- Strengths, issues (by severity) and prioritized improvements with examples
- Missing keywords and before/after bullet rewrites
- Optional job description matching
- Shows the text an ATS would actually extract from your file
- Downloadable JSON report

## Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then provide it in one of these ways:

**Option A - environment variable**
```bash
export GEMINI_API_KEY="your_key"      # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
```

**Option B - Streamlit secrets file** (`.streamlit/secrets.toml`, never commit this)
```toml
GEMINI_API_KEY = "your_key"
```

**Option C** - just paste the key into the sidebar when the app opens.

Start the app:
```bash
streamlit run app.py
```

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | - | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini Flash model name |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (keep `.streamlit/secrets.toml` out of it via `.gitignore`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

## Notes
- Scanned/image-only PDFs have no extractable text; the app will warn you (real ATS systems struggle with these too).
- The ATS score is an AI estimate, not the output of any specific commercial ATS.
- Resumes are sent to Google's Gemini API for analysis. Don't upload sensitive documents you can't share.
- Max file size: 5 MB.
