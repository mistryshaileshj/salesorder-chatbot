"""
Sales-order analytics chatbot — natural language -> T-SQL -> result.
Data source: Microsoft SQL Server view `Vw_SalesOrder_dash` (live query).

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

def show_result(table: pd.DataFrame, sql: str, mode: str, key: str = None):
    with st.expander("SQL"):
        st.code(sql, language="sql")
    can = chartable(table)
    if len(table) <= 1:                 # single result -> table only, no graph
        _show_table(table)
        return
    if mode == "Table only":
        _show_table(table)
    elif mode == "Graph only":
        if can:
            render_chart(table, key=key)
        else:
            st.info("This result can't be charted — showing the table instead.")
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
    "Display", ["Table + Graph", "Table only", "Graph only"],
    index=1, horizontal=True, label_visibility="collapsed", key="view_mode",
)
header.markdown('<div class="fixed-header"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — searchable prompt history; each item links to its anchor in the page.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            show_result(m["table"], m["sql"], view_mode, key=f"hist_{i}")

if prompt := st.chat_input("e.g. Total amount per month for the period 01/04/2024 to 30/06/2024"):
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div id='msg{idx}'></div>", unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                sql = generate_sql(prompt)
                result = run_query(sql)
            st.markdown(f"Returned {len(result)} row(s).")
            show_result(result, sql, view_mode, key=f"live_{idx}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Returned {len(result)} row(s).",
                 "sql": sql, "table": result}
            )
        except Exception as e:
            msg = f"Couldn't run that: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
