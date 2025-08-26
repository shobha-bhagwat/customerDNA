"""
CustomerDNA — Modular, scalable Streamlit app to search customners who match a persona or retrieve individual customer profile.
Powered by GPT-5-mini

Requirements (pip): streamlit, pandas, numpy, openpyxl, matplotlib, streamlit-option-menu, openai>=1.0.0
Run: streamlit run app.py
"""
from __future__ import annotations

import os
import re
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import json
import streamlit as st
from matplotlib import pyplot as plt
from streamlit_option_menu import option_menu


try:
    # OpenAI python SDK v1+
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# =========================
# Constants & Configuration
# =========================
APP_TITLE = "CustomerDNA"
PAGE_LAYOUT = "wide"
DATA_PATH = "GuestDNA_Sample_data1.xlsx"

# Desired columns to display from persona search
DISPLAY_COLUMNS: List[str] = [
    "customer_id",
    "age",
    "ethnicity",
    "gender",
    "segment",
    "last_shopped",
    "lifetime_value($)",
    "revenue($)",
    "avg_basket($)"
]

# For radar chart; we support either spelling variant in data
RADAR_KEYS_MAP: Dict[str, str] = {
    "Price Sensitivity": "price_sensitivity",
    "Promo Responsiveness": "promo_responsiveness",
    "Marketing Responsiveness": "response_to_discounts",
    # Support both spellings in source data
    "Return Behavior": "return_behavior",
    "Return Behaviour": "return_behaviour",
}

LEVEL_MAP = {"low": 1, "medium": 2, "high": 3}

# ======================
# App Setup & Utilities
# ======================

def set_page_config() -> None:
    st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)


def load_css() -> None:
    """Inject custom CSS for styling."""
    st.markdown(
        """
        <style>
        .stApp { margin-top: -55px; }
        section[data-testid="stSidebar"] { margin-top: 100px; width: 550px !important; }
        body { background-color: #ffefea; }
        .customerdna-title { font-size: 56px; font-weight: 900; color: #CC0000; font-family: 'Helvetica', sans-serif; text-align: center; letter-spacing: -1px; }
        .dna-icon { font-size: 56px; vertical-align: middle; margin-left: 10px; }
        .section-title { font-size: 30px; font-weight: bold; color: #CC0000; margin: 10px 0; }
        .persona-box { border: 3px solid #CC0000; padding: 20px; margin-top: 20px; margin-bottom: 50px; background-color: #fff9f8; }
        .persona-text { font-size: 20px; font-weight: bold; text-align: Center; color: #CC0000; }
        .not-found { font-size: 24px; color: red; text-align: center; margin-top: 50px; }
        .check { font-size: 20px; color: green; }
        .cross { font-size: 20px; color: red; }
        </style>
        <div class="customerdna-title">Customer<span style='font-weight: bold;'>DNA</span> <span class="dna-icon">🧬</span></div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df2.columns]
    return df2


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = _read_excel(path)
    df = _normalize_columns(df)
    return df


@st.cache_data(show_spinner=False)
def build_corpus_for_genai(df: pd.DataFrame) -> str:
    """Create a compact, model-friendly corpus describing each customer on one line."""
    lines: List[str] = [
        "The following rows describe individual customers using 'column: value' pairs.",
        "Columns: " + ", ".join(df.columns),
    ]
    for _, row in df.iterrows():
        parts = []
        for col, val in row.dropna().items():
            parts.append(f"{col}: {val}")
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines) + "\n"


def get_openai_client() -> Optional[OpenAI]:
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# =====================
# LLM / Parsing Helpers
# =====================

def _strip_code_fences(text: str) -> str:
    # Remove ``` blocks and stray backticks
    text = re.sub(r"```[a-zA-Z0-9]*\n", "", text)
    text = text.replace("```", "")
    return text.strip()


def parse_markdown_table_to_df(md: str) -> pd.DataFrame:
    """Parses a pipe table into a DataFrame.
    Drops Unnamed columns created by leading/trailing pipes. Tolerates no separator row.
    """
    if not md:
        return pd.DataFrame()

    raw = _strip_code_fences(md).replace("\\n", "\n").strip()

    # Ensure header separator exists for Markdown tables
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    # Detect separator rows like | --- | :--- | ---: | :---: |
    sep_re = re.compile(r'^\s*\|?\s*(?::?-+:?\s*\|)+\s*(?::?-+:?\s*)\|?\s*$')

    header = lines[0]
    rest = lines[1:]

    # If next line is a separator, drop it; otherwise we keep data as-is.
    if rest and sep_re.match(rest[0]):
        data_lines = [header] + rest[1:]
    else:
        data_lines = [header] + rest  # don't inject a separator; pandas doesn't need it

    # Also drop any stray separator lines that might appear later (defensive)
    data_lines = [ln for ln in data_lines if not sep_re.match(ln)]

    table_text = "\n".join(data_lines)

    # Parse & treat first row as header
    df = pd.read_csv(StringIO(table_text), sep=r"\s*\|\s*", engine="python", header=0, dtype=str)

    # Drop helper columns from leading/trailing pipes
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Strip whitespace
    for c in df.columns:
        df[c] = df[c].str.strip()

    # Drop rows that are just dashes/colons (if any slipped through)
    is_sep_row = df.apply(lambda r: all(bool(re.fullmatch(r"[-:\.]*", (str(x) or '').strip())) for x in r), axis=1)
    df = df[~is_sep_row]

    return df

def infer_schema(df: pd.DataFrame, max_samples_per_col: int = 8) -> Dict:
    schema = []
    for col in df.columns:
        # infer type
        if pd.api.types.is_numeric_dtype(df[col]):
            typ = "number"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            typ = "date"
        elif pd.api.types.is_bool_dtype(df[col]):
            typ = "boolean"
        else:
            typ = "string"
        # light sampling of unique values to help the model map phrases to values
        try:
            samples = (
                df[col]
                .dropna()
                .astype(str).str.strip()
                .map(lambda s: s[:40])
                .unique()[:max_samples_per_col]
                .tolist()
            )
        except Exception:
            samples = []
        schema.append({"name": col, "type": typ, "samples": samples})
    # also provide a normalized->actual name map, so we can accept "Frequency of visits" vs "frequency_of_visits"
    norm_map = {normalize_name(c): c for c in df.columns}
    return {"columns": schema, "normalized_map": norm_map}

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower().replace("-", "_").replace(" ", "_"))


def persona_to_spec(persona: str, schema: Dict, client) -> Dict:
    system = (
        "You convert natural-language personas into STRICT boolean filters over a tabular dataset.\n"
        "Use ONLY the provided column names exactly as given. \n"
        "Default combinator is AND unless the persona clearly requests OR/either.\n"
        "Operators allowed: '=', '!=', '>', '>=', '<', '<=', 'in', 'contains', 'startswith', 'endswith', 'between'.\n"
        "Strings are case-insensitive. 'over N years old' => age > N; 'at least N' => age >= N.\n"
        "Return ONLY JSON in this exact shape:\n"
        "{\"op\":\"and\",\"filters\":[{\"col\":\"<column>\",\"op\":\"=\",\"value\":\"<value>\"}, ...]}"
    )

    # keep prompt small: send just name/type/samples
    cols_brief = [{"name": c["name"], "type": c["type"], "samples": c["samples"]} for c in schema["columns"]]
    user = json.dumps({
        "persona": persona,
        "schema": {"columns": cols_brief}
    })


    resp = client.responses.create(
        model="gpt-5-mini",
        input=[{"role":"system","content":system},{"role":"user","content":user}],
    )

    return json.loads(resp.output_text)


def map_col_to_actual(col: str, norm_map: Dict[str, str]) -> str | None:
    norm = normalize_name(col)
    return norm_map.get(norm)

def to_datetime_if_possible(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return series  # leave as-is

def eval_condition(s: pd.Series, op: str, value):
    # normalize strings
    if pd.api.types.is_object_dtype(s):
        left = s.astype(str).str.strip().str.casefold()
    else:
        left = s

    if op in (">", ">=", "<", "<=", "between"):
        # try datetime compare first, else numeric
        s_dt = to_datetime_if_possible(s)
        if pd.api.types.is_datetime64_any_dtype(s_dt):
            if op == "between":
                lo, hi = value
                lo = pd.to_datetime(lo)
                hi = pd.to_datetime(hi)
                return (s_dt >= lo) & (s_dt <= hi)
            rhs = pd.to_datetime(value)
            return {">": s_dt > rhs, ">=": s_dt >= rhs, "<": s_dt < rhs, "<=": s_dt <= rhs}[op]
        # numeric
        s_num = pd.to_numeric(s, errors="coerce")
        if op == "between":
            lo, hi = value
            return (s_num >= float(lo)) & (s_num <= float(hi))
        return {">": s_num > float(value), ">=": s_num >= float(value),
                "<": s_num < float(value), "<=": s_num <= float(value)}[op]

    # string-ish / equality ops
    if op == "=":
        rhs = str(value).strip().casefold()
        return left == rhs
    if op == "!=":
        rhs = str(value).strip().casefold()
        return left != rhs
    if op == "in":
        vals = [str(v).strip().casefold() for v in (value if isinstance(value, list) else [value])]
        return left.isin(vals)
    if op == "contains":
        rhs = str(value).strip().casefold()
        return left.str.contains(re.escape(rhs), na=False)
    if op == "startswith":
        rhs = str(value).strip().casefold()
        return left.str.startswith(rhs, na=False)
    if op == "endswith":
        rhs = str(value).strip().casefold()
        return left.str.endswith(rhs, na=False)

    # unsupported op -> all False
    return pd.Series(False, index=s.index)

def normalize_filter_spec(spec):
    # Accept top-level list → implicit AND
    if isinstance(spec, list):
        return {"op": "and", "filters": [normalize_filter_spec(s) for s in spec]}

    if not isinstance(spec, dict):
        return {"op": "and", "filters": []}

    # Canonicalize boolean groups
    if "op" in spec and "filters" in spec:
        return {"op": spec["op"].lower(),
                "filters": [normalize_filter_spec(f) for f in spec["filters"]]}

    if "and" in spec:
        return {"op": "and", "filters": [normalize_filter_spec(f) for f in spec["and"]]}
    if "or" in spec:
        return {"op": "or", "filters": [normalize_filter_spec(f) for f in spec["or"]]}
    if "not" in spec:
        return {"op": "not", "filters": [normalize_filter_spec(spec["not"])]}

    # Otherwise assume it's a leaf condition; normalize operator synonyms
    op_map = {"equals": "=", "is": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    if "op" in spec:
        spec["op"] = op_map.get(str(spec["op"]).lower(), str(spec["op"]).lower())
    return spec

def apply_spec(df: pd.DataFrame, spec: Dict, norm_map: Dict[str, str]) -> pd.DataFrame:
    """
    spec JSON examples:
      {"op":"and","filters":[{"col":"gender","op":"=","value":"Female"},
                            {"col":"age","op":">","value":40},
                            {"col":"category_affinity","op":"=","value":"Apparel"}]}
    Also supports nested groups and 'not'.
    """
    def eval_group(node) -> pd.Series:
        if "op" not in node:  # allow implicit AND: {"filters":[...]}
            return eval_group({"op":"and", "filters": node.get("filters", [])})

        op = node["op"].lower()
        fs = node.get("filters", [])
        if op == "not":
            assert len(fs) == 1, "'not' must have exactly one child"
            return ~eval_group(fs[0])

        masks = []
        for f in fs:
            if "col" in f:  # condition
                actual = map_col_to_actual(f["col"], norm_map)
                if not actual or actual not in df.columns:
                    # unknown column -> ignore this condition (neutral element for AND/OR)
                    masks.append(pd.Series(False, index=df.index))
                    continue
                m = eval_condition(df[actual], f["op"], f.get("value"))
                masks.append(m.fillna(False))
            else:
                # nested group
                masks.append(eval_group(f))

        if not masks:
            return pd.Series(True, index=df.index)  # neutral for AND

        if op == "and":
            out = masks[0]
            for m in masks[1:]:
                out &= m
            return out
        if op == "or":
            out = masks[0]
            for m in masks[1:]:
                out |= m
            return out

        # unknown -> all True (neutral)
        return pd.Series(True, index=df.index)

    mask = eval_group(spec if isinstance(spec, dict) else {"op":"and","filters": spec})
    return df.loc[mask]


# =====================
# Visualization Helpers
# =====================

def _to_level(v) -> int:
    if pd.isna(v):
        return 1
    if isinstance(v, (int, float)):
        # Clamp 1..3
        return int(max(1, min(3, round(float(v)))))
    s = str(v).strip().lower()
    return LEVEL_MAP.get(s, 1)


def plot_radar(characteristics: Dict[str, float | str | int]) -> plt.Figure:
    """Create a polar radar chart for characteristics."""
    labels = list(characteristics.keys())
    values = [_to_level(characteristics[k]) for k in labels]
    # Close the loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Low", "Medium", "High"], fontsize=7)
    return fig


# =====================
# Render Sections
# =====================

def render_persona_tab(df: pd.DataFrame, corpus: str) -> None:
    st.subheader("Search Customer Persona")
    persona = st.text_area(
        "Enter your target customer persona",
        height=150,
        max_chars=400,
        placeholder="Describe your target customer persona here...",
    )

    if st.button("Generate Matching Customers"):
        if not persona.strip():
            st.warning("Please enter a persona description first.")
            return

        client = get_openai_client()
        if not client:
            st.error("OpenAI client not configured.")
            return

        schema = infer_schema(df)

        with st.spinner("Analyzing persona with GenAI..."):
            #df_out, err = llm_persona_to_customers(persona=persona, corpus=corpus, client=client)
            spec_raw = persona_to_spec(persona, schema, client)


        # Apply locally and show only your chosen columns
        spec = normalize_filter_spec(spec_raw)
        result = apply_spec(df, spec, schema["normalized_map"])

        if result.empty:
            st.info("No matching customers found.")
        else:
            st.dataframe(result[DISPLAY_COLUMNS], use_container_width=True)

        #st.dataframe(df_out, use_container_width=True)


def _safe_get(ser: pd.Series, key: str, default: str = "-") -> str:
    if key in ser and pd.notna(ser[key]):
        return str(ser[key])
    return default


def render_demographics(guest_row: pd.Series) -> None:
    st.markdown('<div class="section-title">📋 Customer Demographics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='font-size:30px;'>
        <ul>
            <li><b>Customer Age:</b> {_safe_get(guest_row, 'age')} yrs</li>
            <li><b>Gender:</b> {_safe_get(guest_row, 'gender')}</li>
            <li><b>Region:</b> {_safe_get(guest_row, 'region')}</li>
            <li><b>Ethnicity:</b> {_safe_get(guest_row, 'ethnicity')}</li>
            <li><b>Has Membership:</b> {_safe_get(guest_row, 'has_membership')}</li>
            <li><b>Visit Frequency:</b> {_safe_get(guest_row, 'frequency_of_visits')}</li>
            <li><b>Last Shopped:</b> {_safe_get(guest_row, 'last_shopped')}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_behavior(guest_row: pd.Series) -> None:
    st.markdown('<div class="section-title">📊 Customer Behaviour Indicators</div>', unsafe_allow_html=True)

    # Find a valid return key
    return_key = "return_behavior" if "return_behavior" in guest_row.index else (
        "return_behaviour" if "return_behaviour" in guest_row.index else None
    )

    characteristics = {
        "Price Sensitivity": guest_row.get("price_sensitivity"),
        "Promo Responsiveness": guest_row.get("promo_responsiveness"),
        "Marketing Responsiveness": guest_row.get("response_to_discounts"),
        "Return Behavior": guest_row.get(return_key) if return_key else None,
    }

    fig = plot_radar(characteristics)
    st.pyplot(fig)


def render_persona_summary(df: pd.DataFrame, guest_id: str, corpus_all: str) -> None:
    client = get_openai_client()

    # Build a short prompt focused on a single guest id
    prompt = f"""
Generate a concise 2–3 line persona summary for customer_id={guest_id}. Include: in-store vs online mix, preferred fulfillment method, notable life-stage hints (baby/pet/school), responsiveness to BOGO/discounts, price/promo/speed sensitivities, cross-sell ideas, and an indicative churn probability.
Return plain text, no markdown tables.
"""

    if not client:
        st.info("OpenAI not configured; skipping persona generation.")
        return

    try:
        with st.spinner("Generating persona summary with GenAI..."):
            resp = client.responses.create(  # type: ignore
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": "You write crisp business summaries."},
                    {"role": "user", "content": corpus_all + "\n\n" + prompt},
                ],
            )
            summary_text = resp.output_text
    except Exception as e:
        st.error(f"Failed to generate persona summary: {e}")
        return

    st.markdown('<div class="section-title">🧠 Customer Persona</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="persona-box"><div class="persona-text">{summary_text}</div></div>
        """,
        unsafe_allow_html=True,
    )


def render_economics(guest_row: pd.Series) -> None:
    st.markdown('<div class="section-title">💰 Customer Economics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='font-size:30px;'>
        <ul>
            <li><b>Lifetime Value:</b> ${_safe_get(guest_row, 'lifetime_value')}</li>
            <li><b>Revenue Till Date:</b> ${_safe_get(guest_row, 'revenue')}</li>
            <li><b>Average Basket Size:</b> ${_safe_get(guest_row, 'avg_basket')}</li>
            <li><b>Customer Segment:</b> {_safe_get(guest_row, 'segment')}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _check(val: bool) -> str:
    return "<span class='check'>&#10004;</span>" if bool(val) else "<span class='cross'>&#10006;</span>"


def render_assortment(guest_row: pd.Series) -> None:
    st.markdown('<div class="section-title">🛍️ Assortment Preferences</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style='font-size:25px;'>
            <ul>
                <li><b>Preferred Size:</b> {_safe_get(guest_row, 'size_preference')}</li>
                <li><b>Category Affinity:</b> {_safe_get(guest_row, 'category_affinity')}</li>
                <li><b>Style Affinity:</b> {_safe_get(guest_row, 'style_affinity')}</li>
                <li><b>Collaborations clicked:</b> {_safe_get(guest_row, 'collaborations_clicked')}</li>
                <li><b>Abandonment patterns:</b> {_safe_get(guest_row, 'abandonment_patterns')}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style='font-size:30px;'>
            <ul>
                <li><b>Clean Beauty</b> {_check(guest_row.get('clean_beauty_affinity', False))}</li>
                <li><b>Vegan</b> {_check(guest_row.get('vegan/organic_food_preference', False))}</li>
                <li><b>Cruelty-free</b> {_check(guest_row.get('cruelty_free_preference', False))}</li>
                <li><b>Allergy Sensitive</b> {_check(guest_row.get('allergy_sensitive_selections', False))}</li>
                <li><b>Environment friendly</b> {_check(guest_row.get('refill_or_eco_friendly_products', False))}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_summary_tab(df: pd.DataFrame, corpus_all: str) -> None:
    st.subheader("Search Customer")
    guest_id_input = st.text_input("Search with Customer ID")
    search_btn = st.button("Search")

    if not search_btn:
        return

    sub = df[df["customer_id"].astype(str) == str(guest_id_input).strip()]
    if sub.empty:
        st.markdown('<div class="not-found">No Customer Found</div>', unsafe_allow_html=True)
        return

    guest_row = sub.iloc[0]

    c1, c2 = st.columns(2, gap="small")
    with c1:
        render_demographics(guest_row)
    with c2:
        render_behavior(guest_row)

    render_persona_summary(df, str(guest_id_input).strip(), corpus_all)

    c3, c4 = st.columns(2, gap="small")
    with c3:
        render_economics(guest_row)
    with c4:
        render_assortment(guest_row)


# ==============
# Main App Entry
# ==============

def render_header_nav() -> str:
    selected_tab = option_menu(
        menu_title=None,
        options=["Enter a Customer Persona", "Customer Summary"],
        icons=["person", "clipboard-data"],
        orientation="horizontal",
        default_index=1,
        styles={
            "container": {"padding": "0", "background-color": "#ffefea"},
            "icon": {"color": "#CC0000", "font-size": "18px"},
            "nav-link": {
                "font-size": "18px",
                "font-weight": "bold",
                "text-align": "center",
                "--hover-color": "#ffe6e6",
                "color": "#CC0000",
            },
            "nav-link-selected": {"background-color": "#CC0000", "color": "white"},
        },
    )
    return selected_tab


def main() -> None:
    set_page_config()
    load_css()

    # Load data & cache corpus
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    except Exception as e:  # pragma: no cover
        st.error(f"Failed to load data: {e}")
        st.stop()

    corpus_all = build_corpus_for_genai(df)

    selected_tab = render_header_nav()

    if selected_tab == "Enter a Customer Persona":
        render_persona_tab(df, corpus_all)
    else:  # "Customer Summary"
        render_summary_tab(df, corpus_all)


if __name__ == "__main__":
    main()
