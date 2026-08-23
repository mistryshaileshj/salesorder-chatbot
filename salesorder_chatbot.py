"""
Sales-order analytics chatbot — natural language -> T-SQL -> result.
Data source: Microsoft SQL Server view `Vw_SalesOrder_dash` (live query).
Input: type a question, or record one with the microphone. Spoken input is
transcribed by Groq Whisper (reuses GROQ_API_KEY, no extra credentials) and
placed in the chat box for you to review/edit before sending.
Note: the mic needs Streamlit >= 1.36 and a secure context (https:// or localhost).

Setup:
    pip install -r requirements.txt
    # requires a SQL Server ODBC driver on the host, e.g.
    #   "ODBC Driver 17 for SQL Server" or "ODBC Driver 18 for SQL Server"
    # put your secrets in .streamlit/secrets.toml (see secrets.toml.example):
    #   GROQ_API_KEY = "gsk_..."
    #   [mssql]
    #   server="...", database="...", user="...", password="...", driver="..."
    streamlit run sales_analytics_chatbot.py
"""

import re
import urllib.parse

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from groq import Groq

TABLE = "Vw_SalesOrder_dash"
GROQ_MODEL = "openai/gpt-oss-120b"   # VERIFY current IDs at console.groq.com/docs/models
# Speech-to-text (voice input). Groq's ASR API is OpenAI-compatible; turbo is the
# fastest/cheapest Whisper. Alternatives: "whisper-large-v3", "distil-whisper-large-v3-en".
GROQ_STT_MODEL = "whisper-large-v3-turbo"

# ---------------------------------------------------------------------------
# Database connection (Microsoft SQL Server via SQLAlchemy + pyodbc)
# Credentials live in .streamlit/secrets.toml under the [mssql] table.
# ---------------------------------------------------------------------------
# @st.cache_resource
# def get_engine():
#     cfg = st.secrets["mssql"]
#     driver = cfg.get("driver", "ODBC Driver 17 for SQL Server")
#     # `server` may be "host" or "host,port" or "host\\instance".
#     odbc = (
#         f"DRIVER={{{driver}}};"
#         f"SERVER={cfg['server']};"
#         f"DATABASE={cfg['database']};"
#         f"UID={cfg['user']};"
#         f"PWD={cfg['password']};"
#         f"TrustServerCertificate=yes;"
#     )
#     url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc)
#     return create_engine(url, pool_pre_ping=True)


# ── Database connection ───────────────────────────────────────────────────────
# Remote MSSQL source. Credentials are kept in one place below. For production,
# prefer moving them into .streamlit/secrets.toml (see the note at the bottom of
# this file) instead of hard-coding them here.
# DB_CONFIG = {
#     "server":   "162.4.2.219",
#     "port":     5051,
#     "database": "sandune",
#     "username": "vrs",
#     "password": "vrs0108",
#     "driver":   "ODBC Driver 18 for SQL Server",
#     # Table/view that holds the sales-order rows (must expose the schema below).
#     "table":    "tbl_SalesOrder_dash",
# }

DB_CONFIG = {
    "server":   st.secrets["SO_DB_SERVER"],
    "port":     st.secrets["SO_DB_PORT"],
    "database": st.secrets["SO_DB_DATABASE"],
    "username": st.secrets["SO_DB_USERNAME"],
    "password": st.secrets["SO_DB_PASSWORD"],
    "driver":   st.secrets["SO_DB_DRIVER"],
    # Table/view that holds the sales-order rows (must expose the schema below).
    "table":    st.secrets["SO_DB_TABLE"],
}

@st.cache_resource
def get_engine():
    """Create a cached SQLAlchemy engine for the remote MSSQL server.

    Uses st.cache_resource (not cache_data) so the connection pool is reused
    across reruns instead of being re-created every time.
    """
    url = URL.create(
        "mssql+pyodbc",
        username=DB_CONFIG["username"],
        password=DB_CONFIG["password"],   # URL.create escapes special chars safely
        host=DB_CONFIG["server"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        query={
            "driver": DB_CONFIG["driver"],
            "TrustServerCertificate": "yes",   # needed for most internal/self-signed servers
            "Encrypt": "no",
        },
    )
    return create_engine(url, pool_pre_ping=True)

def run_query(sql: str) -> pd.DataFrame:
    """Execute a read-only SELECT and return a DataFrame.
    exec_driver_sql sends the string straight to the DBAPI, so colons in
    literals aren't mistaken for bind parameters."""
    with get_engine().connect() as conn:
        res = conn.exec_driver_sql(sql)
        rows = res.fetchall()
        cols = list(res.keys())
    df = pd.DataFrame.from_records(rows, columns=cols)
    # pyodbc returns SQL Server decimal/money/numeric as Python Decimal objects,
    # which land in pandas as `object` dtype and are NOT seen as numeric — so the
    # chart logic mistakes a measure (e.g. SUM(amount)) for a dimension. Convert
    # any object column that is fully numeric back to real numbers.
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            # only replace if every non-null original value converted cleanly
            # (leaves genuine text dimensions like brand/item untouched)
            if df[c].notna().any() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_view_columns() -> list[str]:
    """Introspect the actual columns of the view so the model only ever
    references real column names."""
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_NAME = '{TABLE}' ORDER BY ORDINAL_POSITION"
    )
    df = run_query(sql)
    return df["COLUMN_NAME"].tolist() if not df.empty else []

# ---------------------------------------------------------------------------
# Semantic layer
# ---------------------------------------------------------------------------
SEMANTIC_LAYER_TMPL = """
Table `{table}` — one row per sales-order line item (size-level detail).

Key columns (fixed business meaning):
- doc_dt       : sales-order date. USE THIS for any period, "per day", "monthly",
                 "over time", or date-range filter. CAST(doc_dt AS DATE) is the day key.
- doc_no       : sales-order number. One order spans many line rows, so a COUNT of
                 orders/sales orders must be COUNT(DISTINCT doc_no), never COUNT(*).
- qty          : order quantity. Use SUM(qty) for quantity / units.
- balqty       : balance / pending quantity (ordered but not yet delivered).
                 Use SUM(balqty) for "balance qty", "pending qty", "pending
                 quantity", "balance", "outstanding qty".
- initqty      : short / cancelled quantity. Use SUM(initqty) for "short qty",
                 "short quantity", "cancelled qty", "cancelled quantity",
                 "cancellation".
- packingqty   : despatch / packing / delivery quantity (already sent out).
                 Use SUM(packingqty) for "despatch qty", "dispatch qty",
                 "packing qty", "delivery qty", "delivered quantity",
                 "despatched quantity".
- amount       : order line amount (the money measure). Use SUM(amount) for
                 value / revenue / sales.
- DocDtlSz_Id  : unique record id (one per line row). COUNT(DISTINCT DocDtlSz_Id)
                 = number of detail/line records.

Every other column in the view is a dimension you may GROUP BY or filter on.
The full, authoritative column list for `{table}` is:
{columns}
Use those exact names. NEVER invent a column that is not in that list.

METRIC CONVENTIONS (follow exactly):
- "quantity" / "qty" / "units" / "sold" / "selling"      -> SUM(qty)
- "balance" / "pending" / "outstanding" /
  "balance qty" / "pending qty"                          -> SUM(balqty)
- "short" / "cancelled" / "cancellation" /
  "short qty" / "cancelled qty"                          -> SUM(initqty)
- "despatch" / "dispatch" / "packing" / "delivery" /
  "delivered" / "packing qty" / "despatch qty"           -> SUM(packingqty)
- "amount" / "value" / "revenue" / "sales" / "total"     -> SUM(amount)
- "orders" / "how many orders" / "number of orders" /
  "count of sales orders" / "invoices"                   -> COUNT(DISTINCT doc_no)
- "average order value" / "average order" / "avg order"  (per ORDER)
      -> SUM(amount) / COUNT(DISTINCT doc_no)
         NOT AVG(amount): that averages line items, not orders.
- "records" / "line items"                               -> COUNT(DISTINCT DocDtlSz_Id)
Use SUM by default. Only average when the user says "average".
Default to sales quantity (SUM(qty)) for "top / best / highest selling".

DATES & PERIODS:
- Dates in questions are day-first: dd/MM/yyyy. So 03/04/2024 = 3 April 2024.
- "for the period X to Y" -> filter with
      CAST(doc_dt AS DATE) BETWEEN 'yyyy-mm-dd' AND 'yyyy-mm-dd'
- For "per month" / "monthly", group by FORMAT(doc_dt, 'yyyy-MM').
"""

FEW_SHOTS = f"""
Q: How many sales orders in total
SQL: SELECT COUNT(DISTINCT doc_no) AS orders FROM {TABLE};

Q: Total quantity and total amount
SQL: SELECT SUM(qty) AS qty, SUM(amount) AS amount FROM {TABLE};

Q: Sales orders per month
SQL: SELECT FORMAT(doc_dt, 'yyyy-MM') AS month, COUNT(DISTINCT doc_no) AS orders
     FROM {TABLE}
     GROUP BY FORMAT(doc_dt, 'yyyy-MM') ORDER BY month;

Q: Total amount per day for the period 01/04/2024 to 30/06/2024
SQL: SELECT CAST(doc_dt AS DATE) AS day, SUM(amount) AS amount
     FROM {TABLE}
     WHERE CAST(doc_dt AS DATE) BETWEEN '2024-04-01' AND '2024-06-30'
     GROUP BY CAST(doc_dt AS DATE) ORDER BY day;

Q: Average order value per month
SQL: SELECT FORMAT(doc_dt, 'yyyy-MM') AS month,
            SUM(amount) / COUNT(DISTINCT doc_no) AS avg_order_value
     FROM {TABLE}
     GROUP BY FORMAT(doc_dt, 'yyyy-MM') ORDER BY month;

-- For grouping examples, replace <dimension>/<dim1>/<dim2> with REAL column
-- names from the column list above.

Q: Top 5 <dimension> by quantity
SQL: SELECT TOP 5 <dimension>, SUM(qty) AS qty
     FROM {TABLE}
     GROUP BY <dimension> ORDER BY qty DESC;

Q: Top 10 <dimension> by amount for the period 01/01/2024 to 31/03/2024
SQL: SELECT TOP 10 <dimension>, SUM(amount) AS amount
     FROM {TABLE}
     WHERE CAST(doc_dt AS DATE) BETWEEN '2024-01-01' AND '2024-03-31'
     GROUP BY <dimension> ORDER BY amount DESC;

Q: Quantity by <dim1> and <dim2>  (two dimensions -> stacked chart)
SQL: SELECT <dim1>, <dim2>, SUM(qty) AS qty
     FROM {TABLE}
     GROUP BY <dim1>, <dim2> ORDER BY qty DESC;

Q: Top 5 <dim1> by quantity, split by <dim2>  (top-N on the PRIMARY dim via a CTE)
SQL: WITH top_dim AS (
       SELECT TOP 5 <dim1>
       FROM {TABLE}
       GROUP BY <dim1> ORDER BY SUM(qty) DESC
     )
     SELECT s.<dim1>, s.<dim2>, SUM(s.qty) AS qty
     FROM {TABLE} s
     WHERE s.<dim1> IN (SELECT <dim1> FROM top_dim)
     GROUP BY s.<dim1>, s.<dim2> ORDER BY s.<dim1>, qty DESC;
"""

def build_system_prompt() -> str:
    cols = get_view_columns()
    col_block = ", ".join(cols) if cols else "(column list unavailable)"
    semantic = SEMANTIC_LAYER_TMPL.format(table=TABLE, columns=col_block)
    # "customer" is the business term for the `party` dimension.
    if any(str(c).lower() == "party" for c in cols):
        semantic += (
            '\nCUSTOMER: the customer is the `party` column. Treat "customer" / '
            '"customers" in a question as the `party` dimension (e.g. GROUP BY party).\n'
        )
    return f"""You translate a question about sales orders into ONE Microsoft SQL Server
(T-SQL) SELECT statement. Output ONLY the SQL — no prose, no markdown fences.

{semantic}

Examples:
{FEW_SHOTS}

Rules:
- This is T-SQL (Microsoft SQL Server). Use SELECT TOP N for "top/highest",
  NOT LIMIT. There is no LIMIT clause in SQL Server.
- Exactly one statement, must start with SELECT or WITH.
- Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/EXEC/MERGE/GRANT.
- Always add a sensible TOP N for "top/highest" style questions.
- When the user names two dimensions (e.g. "by A and B"), GROUP BY both and
  SELECT both plus the measure, so the result draws as a stacked chart.
- CRITICAL for a "top N" with two dimensions: apply TOP to the PRIMARY dimension
  via a CTE (top N of dim1 by the measure), then return ALL rows of dim2 for
  those. NEVER put TOP on the (dim1, dim2) combination — that returns the top N
  pairs, repeats some dim1 values, and produces a broken partial stack.
- Reference ONLY columns from the column list above. If the question is
  impossible with the schema, return exactly: SELECT 'unanswerable' AS note;
"""

# ---------------------------------------------------------------------------
# SQL guardrail
# ---------------------------------------------------------------------------
_BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|exec|execute|merge|"
    r"grant|revoke|backup|restore|shutdown|waitfor|attach|copy)\b",
    re.IGNORECASE,
)

def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise ValueError("Query must start with SELECT or WITH.")
    if _BLOCKED.search(sql):
        raise ValueError("Query contains a forbidden keyword.")
    return sql

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def _extract_sql(text: str) -> str:
    """Pull a single SQL statement out of the model reply, tolerating a
    reasoning preamble, code fences, or trailing prose."""
    if not text:
        raise ValueError("Empty response from model.")
    # prefer a fenced code block if present
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    # take everything from the first SELECT / WITH onward (drops any preamble)
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    # cut at the first semicolon (drops any trailing explanation)
    return text.split(";")[0].strip()

def generate_sql(question: str) -> str:
    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=2048,            # headroom so reasoning can't crowd out the SQL
        reasoning_effort="low",     # gpt-oss: minimize reasoning for this simple task
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ],
    )
    msg = resp.choices[0].message
    # content is normally the SQL; if empty (reasoning used the budget), the SQL
    # is almost always still inside the reasoning text — fall back to that.
    text = (msg.content or "").strip() or (getattr(msg, "reasoning", "") or "").strip()
    return validate_sql(_extract_sql(text))

# ---------------------------------------------------------------------------
# Speech-to-text (voice input)
#   Sends recorded microphone audio to Groq's Whisper endpoint and returns the
#   transcribed text. Reuses the same Groq client / GROQ_API_KEY as the SQL
#   generation above, so no extra credentials or packages are required.
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe WAV audio bytes (from st.audio_input) to text via Groq Whisper."""
    resp = get_client().audio.transcriptions.create(
        file=("voice.wav", audio_bytes),   # (filename, bytes) — st.audio_input yields WAV
        model=GROQ_STT_MODEL,
        response_format="text",            # returns a plain string, not JSON
        # language="en",                   # uncomment to force English recognition
    )
    # With response_format="text" the SDK returns a str; be defensive either way.
    return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()

# ---------------------------------------------------------------------------
# Charting
#   1 measure + 1 dimension  -> bar, ordered by value, data labels
#   1 measure + 2 dimensions -> STACKED bar (2nd dim = color) with legend
# ---------------------------------------------------------------------------
# Column display names shown on charts (axis titles, legend, hover) and in
# table headers. Raw SQL aliases / view columns are converted to proper case;
# the `party` dimension is presented as "Customer" per business naming.
def _pretty(name: str) -> str:
    """Return a proper-case, human-friendly label for a column / alias."""
    key = str(name).strip().lower()
    if key == "party":
        return "Customer"
    return str(name).replace("_", " ").title()

def _round_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Round every numeric measure to a whole number (no decimals shown)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(0).astype("Int64")   # nullable int keeps NaNs
    return out

def _split_cols(df: pd.DataFrame):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dim = [c for c in df.columns if c not in num]
    return num, dim

def chartable(df: pd.DataFrame) -> bool:
    num, dim = _split_cols(df)
    return len(num) == 1 and len(dim) in (1, 2) and len(df) > 0

def render_chart(df: pd.DataFrame, key: str = None):
    num, dim = _split_cols(df)
    measure = num[0]
    d = df.copy()
    for c in dim:                       # dims must be categorical strings so
        d[c] = d[c].fillna("Unknown").astype(str)   # plotly stacks them
    x = dim[0]
    order = (d.groupby(x)[measure].sum()
               .sort_values(ascending=False).index.tolist())
    labels = {c: _pretty(c) for c in d.columns}   # proper-case display names
    if len(dim) == 1:
        fig = px.bar(d, x=x, y=measure, text_auto=",.0f",   # data labels: no decimals
                     category_orders={x: order}, labels=labels)
        fig.update_traces(textposition="outside")
    else:
        color = dim[1]
        fig = px.bar(d, x=x, y=measure, color=color, barmode="stack",
                     text_auto=",.0f", category_orders={x: order}, labels=labels)
        fig.update_layout(legend_title_text=_pretty(color))   # proper-case legend title
    fig.update_layout(
        xaxis_title=_pretty(x), yaxis_title=_pretty(measure),
        height=650, margin=dict(l=50, r=20, t=30, b=50),
        showlegend=True,                 # remove legend (incl. 2-dimension charts)
        font=dict(color="black"),         # axis titles + tick labels in black
    )
    fig.update_yaxes(tickformat=",.0f")   # measures: whole numbers, no decimals
    st.plotly_chart(fig, use_container_width=True, key=key)

def _show_table(df: pd.DataFrame):
    """Display a table with a 1-based Sr. No column instead of the 0-based index.
    Measures are rounded to whole numbers and headers shown in proper case."""
    disp = _round_measures(df)                                  # no decimals on measures
    disp = disp.rename(columns={c: _pretty(c) for c in disp.columns})   # proper-case headers
    disp.insert(0, "Sr. No", range(1, len(disp) + 1))
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Totals for the "Table only" view
#   1 dimension    -> a Grand Total row.
#   2+ dimensions  -> HIERARCHICAL sub-totals: a sub-total after each group at
#                     EVERY grouping level, then a Grand Total. e.g. grouping by
#                     Brand > Region > Item gives an Item block per Region, a
#                     Region sub-total, a Brand sub-total (over its Regions), and
#                     finally the Grand Total. Each group therefore gets its own
#                     total at its own level.
# Only additive measures (SUM of qty / amount / counts) are totalled. A measure
# whose name implies an average or ratio (e.g. avg_order_value) can't be summed
# meaningfully, so its total cells are left blank instead of showing a wrong sum.
# ---------------------------------------------------------------------------
_NON_ADDITIVE = ("avg", "average", "mean", "ratio", "per ")

# total_level tags on each output row, consumed by the table styler:
#   DETAIL_LEVEL -> a normal data row (no shading)
#   0, 1, 2, ... -> a sub-total for the dimension at that index (0 = first dim)
#   GRAND_LEVEL  -> the single Grand Total row
DETAIL_LEVEL = -1
GRAND_LEVEL = -2

# Row background per total level: Grand Total darkest, then progressively lighter
# for deeper (higher-index) sub-totals so the hierarchy reads at a glance.
_GRAND_SHADE = "#cdd7e6"
_SUBTOTAL_SHADES = ["#dbe2ee", "#e4eaf3", "#eaeff6", "#eff2f7"]

def _total_shade(level: int) -> str:
    if level == GRAND_LEVEL:
        return _GRAND_SHADE
    return _SUBTOTAL_SHADES[min(level, len(_SUBTOTAL_SHADES) - 1)]

def _is_additive(measure: str) -> bool:
    k = str(measure).lower()
    return not any(t in k for t in _NON_ADDITIVE)

def _sum_measures(block: pd.DataFrame, measures: list) -> dict:
    """SUM each additive measure over `block`; blank (NA) for non-additive ones."""
    return {m: (block[m].sum() if _is_additive(m) else pd.NA) for m in measures}

def _build_totals(df: pd.DataFrame, dims: list, measures: list):
    """Return (augmented_df, is_total_mask, total_level).

    Detail rows keep their original (SQL) order within their block. For 2+
    dimensions the data is nested hierarchically and a sub-total row is emitted
    at EVERY grouping level: after each innermost block, then after its parent,
    up to the first dimension, and finally one Grand Total. Groups at each level
    are ordered biggest-first by the first additive measure, so every block sits
    directly under its own rows even when the SQL interleaved them.

    total_level[i] tags row i: DETAIL_LEVEL for data rows, the dimension index
    (0 = first dim) for a sub-total at that level, GRAND_LEVEL for the grand
    total — used by the styler to shade each total by its depth."""
    d = df.copy()
    for c in dims:
        d[c] = d[c].astype(object)          # allow string labels / blanks in dim cells
    rows, is_total, total_level = [], [], []

    def _ordered_values(block, dim):
        """Distinct values of `dim` in `block`, biggest additive measure first."""
        additive = [m for m in measures if _is_additive(m)]
        if additive:
            return (block.groupby(dim)[additive[0]].sum()
                         .sort_values(ascending=False).index.tolist())
        return list(pd.unique(block[dim]))    # no additive measure -> first-seen order

    def emit_detail(block):
        for _, r in block.iterrows():
            rows.append(r.to_dict()); is_total.append(False)
            total_level.append(DETAIL_LEVEL)

    def recurse(block, level):
        """Walk one grouping level: recurse into children, then sub-total this
        group. The last dimension is the leaf whose rows are the detail rows."""
        dim = dims[level]
        is_leaf = (level == len(dims) - 1)
        for v in _ordered_values(block, dim):
            sub_block = block[block[dim] == v]
            if is_leaf:
                emit_detail(sub_block)
            else:
                recurse(sub_block, level + 1)
                sub = {c: "" for c in d.columns}
                sub[dim] = f"{v} \u2014 Total"        # e.g. "West — Total"
                sub.update(_sum_measures(sub_block, measures))
                rows.append(sub); is_total.append(True)
                total_level.append(level)

    if len(dims) == 1:
        emit_detail(d)
    else:
        recurse(d, 0)

    grand = {c: "" for c in d.columns}
    grand[dims[0]] = "Grand Total"
    grand.update(_sum_measures(d, measures))
    rows.append(grand); is_total.append(True); total_level.append(GRAND_LEVEL)

    out = pd.DataFrame(rows, columns=d.columns)
    for m in measures:                        # whole numbers, keep <NA> for ratios
        out[m] = pd.to_numeric(out[m], errors="coerce").round(0).astype("Int64")
    return out, is_total, total_level

def _show_table_with_totals(df: pd.DataFrame, dims: list, measures: list, key=None):
    """Table view with sub-total / grand-total rows, shaded and bold."""
    aug, is_total, total_level = _build_totals(df, dims, measures)
    disp = aug.rename(columns={c: _pretty(c) for c in aug.columns})
    # Sr. No: number the detail rows only; total rows (any level) get a blank.
    srno, n = [], 0
    for t in is_total:
        if t:
            srno.append("")
        else:
            n += 1
            srno.append(n)
    disp.insert(0, "Sr. No", srno)

    measure_labels = [_pretty(m) for m in measures]

    def _highlight(row):
        lvl = total_level[row.name]
        if lvl == DETAIL_LEVEL:
            return [""] * len(row)
        bg = _total_shade(lvl)          # deeper sub-total -> lighter; grand -> darkest
        return [f"background-color: {bg}; font-weight: bold;"] * len(row)

    styler = (disp.style
                  .apply(_highlight, axis=1)
                  .format("{:,.0f}", subset=measure_labels, na_rep=""))  # blank NA totals
    st.dataframe(styler, use_container_width=True, hide_index=True, key=key)

# ---------------------------------------------------------------------------
# Crosstab (pivot) view
#   Reshapes the long result into a matrix: chosen dimension(s) down the rows,
#   chosen dimension(s) across the columns, ONE measure in the cells. The user
#   picks the layout with multiselects, so any orientation is one click away and
#   multiple dimensions on an axis nest (MultiIndex rows / grouped headers).
#   Empty cells are filled with 0. For an additive measure a single Grand Total
#   row + column is added (margins). A non-additive measure (avg / ratio) uses
#   mean and drops the totals, because summing averages is meaningless. Any
#   dimension left off BOTH axes is summed away by pivot_table.
# ---------------------------------------------------------------------------
_MAX_CROSSTAB_COLS = 40   # guard: too many distinct column values = unreadable

def _blank_repeats(disp: pd.DataFrame, col_positions: list):
    """Blank a group-column cell (in place) when it repeats the value directly
    above it within the SAME parent group — the classic report look where each
    label is printed once and left blank on its repeat rows. Positional (iat) so
    it works whether the columns are plain or a MultiIndex. When a higher level
    changes, deeper levels reprint even if their value is unchanged."""
    prev = [object()] * len(col_positions)     # sentinels: never equal a real value
    for i in range(len(disp)):
        broke = False
        for k, pos in enumerate(col_positions):
            cur = disp.iat[i, pos]
            if not broke and cur == prev[k]:
                disp.iat[i, pos] = ""          # same value, parent unchanged -> blank
            else:
                broke = True                   # this level changed -> deeper reprint
                prev[k] = cur

def _show_crosstab(df: pd.DataFrame, dims: list, measures: list, key: str = None):
    # Layout pickers. Defaults track the SQL column order: first dim -> Rows,
    # second dim -> Columns. Both boxes offer the FULL dimension list (constant
    # options — Streamlit errors if a selected value vanishes from the options
    # on a rerun). Overlap is resolved below: a dimension chosen for both axes
    # stays in Rows and is dropped from Columns.
    # The measure is the cell value. A Values picker is only shown when a result
    # carries more than one measure (e.g. "quantity and amount by ..."); with a
    # single measure there's nothing to choose, so we use it directly and just
    # label the cells.
    multi_measure = len(measures) > 1
    if multi_measure:
        c1, c2, c3 = st.columns(3)
        val = c3.selectbox(
            "Values", measures, index=0,
            format_func=_pretty, key=f"{key}_ct_val",
        )
    else:
        c1, c2 = st.columns(2)
        val = measures[0]
    row_dims = c1.multiselect(
        "Rows", dims, default=[dims[0]],
        format_func=_pretty, key=f"{key}_ct_rows",
    )
    col_default = [d for d in dims if d not in row_dims][:1]
    col_dims = c2.multiselect(
        "Columns", dims, default=col_default,
        format_func=_pretty, key=f"{key}_ct_cols",
    )

    # A dimension can only live on one axis: if the user put it in both, keep it
    # in Rows and drop it from Columns (report it so the layout isn't a mystery).
    dupes = [d for d in col_dims if d in row_dims]
    col_dims = [d for d in col_dims if d not in row_dims]
    if dupes:
        st.caption("In both Rows and Columns — kept in Rows: "
                   + ", ".join(_pretty(d) for d in dupes))

    if not row_dims or not col_dims:
        st.info("Pick at least one Rows dimension and one Columns dimension "
                "to build the crosstab.")
        return

    additive = _is_additive(val)
    pivot = pd.pivot_table(
        df, index=row_dims, columns=col_dims, values=val,
        aggfunc="sum" if additive else "mean",
        margins=additive, margins_name="Total",
        fill_value=0,
    )

    # Wide-column guard: if the crosstab would be too wide to read, fall back to
    # the plain table. Exclude the margin column from the count when present.
    n_value_cols = pivot.shape[1] - (1 if additive else 0)
    if n_value_cols > _MAX_CROSSTAB_COLS:
        st.warning(
            f"The chosen Columns dimension(s) produce {n_value_cols} columns — "
            f"too wide to show as a crosstab (limit {_MAX_CROSSTAB_COLS}). "
            "Showing the plain table instead; pick a narrower Columns dimension."
        )
        _show_table(df)
        return

    # Whole numbers (nullable int keeps things clean).
    pivot = pivot.round(0).astype("Int64")

    # "Repeating value blank in a group": move the Row dimension(s) out of the
    # index into real left-hand columns, then blank any label that repeats the
    # one directly above it within the same parent group (each Item shown once,
    # its Shades beneath). Streamlit's grid doesn't sparsify a MultiIndex on its
    # own, so we do it explicitly and hide the index. The value columns keep
    # their (possibly multi-level) headers.
    n_row = len(row_dims)
    disp = pivot.reset_index()
    _blank_repeats(disp, list(range(n_row)))          # blank repeats in row-dim cols

    # Proper-case the Row-dimension headers, keeping the tuple shape when the
    # value columns are multi-level so the header stays consistent.
    multilevel = isinstance(disp.columns, pd.MultiIndex)
    val_cols = list(disp.columns[n_row:])             # numeric cells (labels unchanged)
    rename = {}
    for i in range(n_row):
        pretty = _pretty(row_dims[i])
        rename[disp.columns[i]] = (pretty, "") if multilevel else pretty
    disp = disp.rename(columns=rename)

    # The column-axis and measure names aren't shown in the grid header once the
    # index is hidden, so surface them in a caption instead.
    cols_lbl = " / ".join(_pretty(d) for d in col_dims)
    st.caption(f"Columns: {cols_lbl}  \u00b7  Cell values: {_pretty(val)}")
    st.dataframe(
        disp.style.format("{:,.0f}", subset=val_cols),
        use_container_width=True, hide_index=True, key=key,
    )

def show_result(table: pd.DataFrame, sql: str, mode: str, key: str = None):
    # SQL is hidden from the UI — results only. (sql is still stored in
    # session_state for history/debugging; re-add an st.expander here to show it.)
    with st.expander("SQL"):
        st.code(sql, language="sql")
    can = chartable(table)
    num, dim = _split_cols(table)       # measures, dimensions
    if len(table) <= 1:                 # single result -> table only, no graph
        _show_table(table)
        return
    if mode == "Table only":
        # 1 dim -> grand total; 2+ dims -> sub-totals + grand total.
        if len(dim) >= 1 and len(num) >= 1:
            _show_table_with_totals(table, dim, num, key=key)
        else:                            # no groupable dimension/measure -> plain
            _show_table(table)
    elif mode == "Graph only":
        if can:
            render_chart(table, key=key)
        else:
            st.info("This result can't be charted — showing the table instead.")
            _show_table(table)
    elif mode == "Crosstab":
        # Needs 2+ dimensions to form a matrix and 1+ measure for the cells.
        if len(dim) >= 2 and len(num) >= 1:
            _show_crosstab(table, dim, num, key=key)
        elif len(dim) >= 1 and len(num) >= 1:
            st.info("A crosstab needs at least two dimensions and one measure — "
                    "showing the table with totals instead.")
            _show_table_with_totals(table, dim, num, key=key)
        else:
            st.info("A crosstab needs at least two dimensions and one measure — "
                    "showing the plain table instead.")
            _show_table(table)
    else:  # Table + Graph
        _show_table(table)
        if can:
            render_chart(table, key=key)

# ---------------------------------------------------------------------------
# Sticky header CSS — pins the title + display selector to the top of the page.
# Uses the :has() trick to make the header container sticky. Adjust `top` if
# it overlaps the Streamlit toolbar on your version.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sales Order Analytics Chatbot", layout="wide")
st.markdown("""
<style>
  /* tighten page whitespace */
  div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
  /* smaller vertical gaps between elements */
  div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
  /* nominal chat input height */
  div[data-testid="stChatInput"] textarea { min-height: 2.2rem; }
  /* tighter chat message bubbles */
  div[data-testid="stChatMessage"] { padding: 0.35rem 0.6rem; }
  /* sticky header, no border */
  div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
      position: sticky; top: 0; z-index: 999;
      background-color: var(--background-color, white);
  }
</style>
""", unsafe_allow_html=True)

header = st.container()
header.markdown(
    "<h1 style='font-size:1.6rem; margin:0 0 0.3rem 0;'>Sales Order Analytics Chatbot</h1>",
    unsafe_allow_html=True,
)
c1, c2 = header.columns([1, 10], vertical_alignment="center")
c1.markdown("**Display**")
view_mode = c2.radio(
    "Display", ["Table + Graph", "Table only", "Graph only", "Crosstab"],
    index=1, horizontal=True, label_visibility="collapsed", key="view_mode",
)
header.markdown('<div class="fixed-header"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — searchable prompt history; each item links to its anchor in the page.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_seq" not in st.session_state:
    st.session_state.msg_seq = 0

def _next_msg_id() -> str:
    """Stable, unique id per message. Per-result widgets (the crosstab Row/Column
    pickers) key off this so they keep the SAME identity whether the result is
    drawn live by handle_prompt or re-drawn by the history loop on later reruns.
    Without it the key flips live_* -> hist_* on the first interaction and the
    widget resets to its default — which looked like the multiselect refusing
    more than one value."""
    st.session_state.msg_seq += 1
    return f"m{st.session_state.msg_seq}"

with st.sidebar:
    st.subheader("Prompt history")
    search = st.text_input("Search", placeholder="filter prompts...")
    user_msgs = [(i, m["content"]) for i, m in enumerate(st.session_state.messages)
                 if m["role"] == "user"]
    if not user_msgs:
        st.caption("No prompts yet.")
    for i, text in reversed(user_msgs):          # newest first
        if search and search.lower() not in text.lower():
            continue
        label = text if len(text) <= 45 else text[:42] + "..."
        safe = text.replace('"', "&quot;")
        st.markdown(f'<a href="#msg{i}" title="{safe}">{label}</a>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Conversation — anchor before each user turn so the sidebar can scroll to it.
# ---------------------------------------------------------------------------
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        st.markdown(f"<div id='msg{i}'></div>", unsafe_allow_html=True)
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("table") is not None:
            show_result(m["table"], m["sql"], view_mode,
                        key=m.get("id") or f"hist_{i}")

def handle_prompt(prompt: str):
    """Run one turn: record the user prompt, generate SQL, query, render.
    Shared by both typed (chat_input) and spoken (voice) prompts."""
    idx = len(st.session_state.messages)
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "id": _next_msg_id()})
    st.markdown(f"<div id='msg{idx}'></div>", unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                sql = generate_sql(prompt)
                result = run_query(sql)
            st.markdown(f"Returned {len(result)} row(s).")
            assistant_id = _next_msg_id()
            # SAME key the history loop will use for this message, so any
            # per-result widgets survive the live -> history rerun intact.
            show_result(result, sql, view_mode, key=assistant_id)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Returned {len(result)} row(s).",
                 "sql": sql, "table": result, "id": assistant_id}
            )
        except Exception as e:
            msg = f"Couldn't run that: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

# ── Voice input ───────────────────────────────────────────────────────────────
# st.audio_input records from the browser mic (requires Streamlit >= 1.36 and a
# secure context: https:// or localhost). It returns WAV bytes which we send to
# Groq Whisper. The transcript is dropped INTO the chat box (via its session_state
# key) so the user can read / edit it and press Enter to send — nothing is
# submitted automatically. We remember the last recording so each one is
# transcribed once, not on every rerun.
mic_col, tip_col = st.columns([1, 3], vertical_alignment="center")
with mic_col:
    audio = st.audio_input("Speak your question", key="voice_input",
                           label_visibility="collapsed")
with tip_col:
    st.caption("🎤 Record a question — the text lands in the box below to review "
               "and edit, then press Enter to send. Or just type.")

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = hash(audio_bytes)                      # identify this recording
    if st.session_state.get("last_audio_id") != audio_id:   # new recording only
        st.session_state["last_audio_id"] = audio_id  # mark as handled
        try:
            with st.spinner("Transcribing..."):
                text = transcribe_audio(audio_bytes)
            if text:
                # Prefill the chat box. This MUST run before st.chat_input below
                # is instantiated; the box then shows the text for the user to
                # verify/edit. chat_input only returns it once the user submits.
                st.session_state["chat_box"] = text
            else:
                st.warning("Didn't catch anything — please try recording again.")
        except Exception as e:
            st.error(f"Couldn't transcribe audio: {e}")

# Typed or voice-prefilled: the user reviews/edits here, then submits with Enter.
# chat_input must stay at the top level (not nested in columns/containers).
if prompt := st.chat_input(
    "e.g. Total amount per month for the period 01/04/2024 to 30/06/2024",
    key="chat_box",
):
    handle_prompt(prompt)
