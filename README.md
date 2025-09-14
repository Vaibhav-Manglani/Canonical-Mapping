# Schema Mapper (Streamlit App)

A Streamlit-based tool to map messy CSV/Excel headers to a canonical schema.

## Features

- Upload CSV/Excel
- Enter Google API Key for LLM Access
- Suggests Data Preprocessing Options
- Suggests AI-powered mapping validation (If Google API Key verified)
- Suggests canonical names
- Lets user approve or override suggestions interactively
- Saves approved mappings in `history.json` for reuse
- Download mapped CSV or history.json

## Run

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Video Link with Demo - https://drive.google.com/file/d/1pT2kOk0919O2OqkNl3isLMJ36u9lF_GX/view?usp=sharing
