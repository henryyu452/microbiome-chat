# microbiome-chat

Chat over a Supabase-backed corpus of microbiome modifier rollup spreadsheets.

## One-time setup

1. Copy `.env.example` to `.env` and fill in your Supabase + Anthropic credentials.
2. Create a virtualenv and install deps:
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Windows)
   source .venv/bin/activate         (Mac/Linux)
   pip install -r requirements.txt
   ```

## Loading data

Put your `*_rollup.xlsx` files anywhere (e.g. a `data/` folder). Then:

```
python load.py data/saccharomyces_boulardii_rollup.xlsx
python load.py data/perilla_seed_products_rollup.xlsx
```

Or load every rollup file in a folder at once:

```
python load.py --all data/
```

To check parsing without touching the database:

```
python load.py --dry-run data/saccharomyces_boulardii_rollup.xlsx
```

Re-loading the same file is safe: it deletes the existing rows for that
modifier (cascading through all child tables) and re-inserts.

## Running the chat app (added in phase 4)

```
streamlit run app.py
```
