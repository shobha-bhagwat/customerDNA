CustomerDNA 🧬 — GenAI-Powered Persona Search & Customer Insights (Streamlit)

CustomerDNA turns natural-language personas into deterministic data filters and crisp customer insights. Business users type prompts like “female customers over 40 with affinity to Apparel”; the app returns the exact rows, plus an LLM persona summary and behavior radar.

What This Solves:

1. Fast audience discovery without SQL or dashboard clicks
2. Deterministic, auditable results (LLM proposes filters; pandas enforces logic)
3. Works across many columns; schema is inferred from the dataset

How It Works (Architecture)

                ┌───────────────────────────┐
                │  CSV/Excel Dataset        │
                └─────────────┬─────────────┘
                              │  load + normalize (cache)
                              ▼
                     ┌───────────────┐
                     │ Schema Builder│  ← col names, types, samples
                     └───────┬───────┘
                             │  persona text
                             ▼
                 ┌─────────────────────────┐
                 │ LLM (OpenAI Responses)  │ ───────────┬─────────────────► Persona summary for a single Customer ID  (LLM text generation)
                 │ Persona → JSON filter   │
                 └───────────┬─────────────┘
                             │  normalize/validate
                             ▼
                    ┌─────────────────┐
                    │  Pandas Engine  │  ← strict evaluator (AND/OR/NOT, ops)
                    └───┬─────────────┘
                        │         
                        │         
                        ▼
                ┌──────────────────────┐
                │ Streamlit UI         │
                │ Tables, Radar chart  │
                └──────────────────────┘



Why it’s different:

1. Deterministic, auditable results. The model never selects rows; it only proposes filters. Your code enforces logic and equality.

2. Scales to many columns. The schema is derived from the dataset—no hard-coded field mapping.

3. Secure by design. The full dataset stays local; only a minimal schema and the persona are sent to the model.

4. Production-minded. Caching, defensive parsing, normalized columns, and modular functions make it easy to extend.

Troubleshooting

Getting many/zero rows? Log the raw and normalized spec to verify shape/columns.

If your Responses SDK rejects temperature, send via extra_body={"temperature": 0}.

License

MIT. Contributions and stars welcome!
