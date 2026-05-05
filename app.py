"""
app.py - Streamlit chat app for the microbiome modifier database.

Run locally:
    streamlit run app.py

Reads SUPABASE_DB_URL and ANTHROPIC_API_KEY from .env (local) or
st.secrets (Streamlit Cloud).
"""

import json
import os
import re
from pathlib import Path

import psycopg2
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 10            # safety cap on tool-use loop
ROW_LIMIT = 500           # max rows returned to the LLM per query

load_dotenv(Path(__file__).parent / ".env")


def get_secret(key):
    val = os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return None


def parse_db_url(url):
    if not url.startswith(("postgres://", "postgresql://")):
        raise ValueError("DB URL must start with postgres:// or postgresql://")
    _, rest = url.split("://", 1)
    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" in userinfo:
        username, password = userinfo.split(":", 1)
    else:
        username, password = userinfo, None
    if "/" in hostpart:
        hostport, db = hostpart.split("/", 1)
        if "?" in db:
            db = db.split("?", 1)[0]
    else:
        hostport, db = hostpart, "postgres"
    if ":" in hostport:
        host, port_str = hostport.rsplit(":", 1)
        port = int(port_str)
    else:
        host, port = hostport, 5432
    return {"host": host, "port": port, "dbname": db, "user": username, "password": password}


# --------------------------------------------------------------------
# Database
# --------------------------------------------------------------------

@st.cache_resource
def get_db_kwargs():
    db_url = get_secret("SUPABASE_DB_URL")
    if not db_url:
        st.error("SUPABASE_DB_URL is not set. Add it to .env (local) or Streamlit secrets (cloud).")
        st.stop()
    return parse_db_url(db_url)


def db_connect():
    return psycopg2.connect(**get_db_kwargs())


def run_select(query, params=None):
    """Run a SELECT and return a list of dicts. Caller is responsible for safety."""
    with db_connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            return [dict(r) for r in cur.fetchmany(ROW_LIMIT)]


# --------------------------------------------------------------------
# Schema context (built dynamically from the DB so the LLM sees real data)
# --------------------------------------------------------------------

STATIC_SCHEMA = """\
TABLES (Postgres). Every table except `modifiers` has a `modifier` column
that names the compound (e.g. 'curcumin', 'cranberry'). Use it in WHERE
or GROUP BY for cross-compound queries.

modifiers (one row per loaded compound)
    modifier text PK, display_name text, source_file text, loaded_at timestamptz

studies (one row per included study, per compound)
    modifier text, study_id text, cohort_cluster_id text, first_author text,
    year int, journal text, doi text, study_type text, quality_tier text,
    population text, sample_size int, microbiome_method text, sample_type text,
    intervention text, duration text, is_disease_state_model bool,
    disease_model text, include_for_scoring text, n_findings_rows int,
    alpha_diversity_change text, beta_diversity_change text
    PK (modifier, study_id)

excluded_studies (studies considered but rejected)
    modifier, study_id, first_author, year, exclusion_reason, evidence_basis

rollup (per-taxon aggregated score for each compound)
    modifier text, taxon_listed_name text, taxon_id text,
    direction text,                      -- 'increase' / 'decrease' / 'mixed' / 'none' / 'insufficient_detail'
    increase_effect smallint,            -- 0..3
    increase_confidence smallint,        -- 0..3
    decrease_effect smallint,            -- 0..3
    decrease_confidence smallint,        -- 0..3
    evidence_tier text,                  -- 'Direct' / 'None'
    n_studies_contributing int, n_human_direct int, n_animal_direct int,
    n_disease_state_model int,
    discordance_warning text, cohort_cluster_summary text,
    contributing_study_ids text,         -- comma-separated study_ids
    evidence_basis text                  -- prose summary of evidence
    PK (modifier, taxon_listed_name)

composite_breakdown (which member taxa rolled up into each composite group)
    modifier, composite_taxon, member_taxon, member_taxon_id,
    n_studies_total, n_studies_increase, n_studies_decrease,
    n_studies_mixed, n_studies_insufficient,
    increase_effect, increase_confidence, decrease_effect, decrease_confidence,
    contributing_study_ids, sample_quotes

per_study_findings (one row per study x taxon x intervention arm)
    id bigserial PK, modifier, study_id, study_first_author, study_year,
    study_type, sample_size, is_disease_state_model, disease_model,
    taxon_listed_name, taxon_id, intervention_arm, direction,
    effect_magnitude, fold_change, p_value, statistical_significance,
    multiple_comparison_correction, comparison_type, low_signal, captured_via,
    computed_increase_effect_size, computed_increase_confidence,
    computed_decrease_effect_size, computed_decrease_confidence,
    computed_evidence_tier, computed_quality_tier,
    baseline_value, post_intervention_value, timepoint, subgroup_context,
    mechanism_notes, paper_interpretation, additional_quotes,
    direction_quote, quote_source, evidence_basis, notes

per_study_narrative (study-level prose, may have multiple rows per study)
    id bigserial PK, modifier, study_id, first_author, year, study_type,
    paper_discussion_summary, off_canonical_taxa_observed, raw_extraction_notes,
    study_design_details, statistical_methods, co_interventions, dose_details,
    limitations, additional_diversity_notes, notes

per_finding_narrative (taxon-level prose for each finding)
    id bigserial PK, modifier, study_id, taxon_listed_name, intervention_arm,
    direction, mechanism_notes, paper_interpretation, baseline_value,
    post_intervention_value, timepoint, subgroup_context, evidence_basis,
    additional_quotes, notes

QUERY GUIDANCE
- For a specific compound, always filter `WHERE modifier = '<name>'`.
- For comparisons across compounds, use GROUP BY modifier or a JOIN
  with the `modifiers` table for the display_name.
- The `rollup` table is the right starting point for "what does X do
  to taxon Y?" questions. Use `per_study_findings` for evidence detail.
- Free-text search the prose columns (paper_interpretation, limitations,
  mechanism_notes, evidence_basis, additional_quotes) with ILIKE '%word%'.
- `taxon_listed_name` is case-sensitive. Use ILIKE for fuzzy matching.
- Wrap user-facing strings (like compound names) in single quotes.
- Always limit large result sets; the runtime caps at 500 rows.

WHEN ANSWERING
- After running queries, write a clear answer in plain prose.
- Cite specific studies inline with `(first_author year)` and include
  the DOI when available so the user can follow up.
- If a question can't be answered from the data, say so.
- Never invent taxa, study IDs, or numbers. If the rollup says
  insufficient_detail or none, report that honestly.
"""


@st.cache_data(ttl=600)
def fetch_dynamic_schema():
    """Pull a few small lookups from the DB so the LLM knows what's loaded."""
    out = {}
    out["compounds"] = run_select(
        "SELECT modifier, display_name FROM modifiers ORDER BY display_name"
    )
    out["taxa"] = run_select(
        "SELECT DISTINCT taxon_listed_name FROM rollup ORDER BY taxon_listed_name"
    )
    out["study_types"] = run_select(
        "SELECT DISTINCT study_type FROM studies WHERE study_type IS NOT NULL ORDER BY study_type"
    )
    out["evidence_tiers"] = run_select(
        "SELECT DISTINCT evidence_tier FROM rollup WHERE evidence_tier IS NOT NULL ORDER BY evidence_tier"
    )
    return out


def build_system_prompt():
    dyn = fetch_dynamic_schema()
    compounds = "\n".join(f"  - {c['display_name']} (modifier='{c['modifier']}')" for c in dyn["compounds"])
    taxa = ", ".join(t["taxon_listed_name"] for t in dyn["taxa"])
    study_types = ", ".join(t["study_type"] for t in dyn["study_types"]) or "(none)"
    tiers = ", ".join(t["evidence_tier"] for t in dyn["evidence_tiers"]) or "(none)"

    return f"""You are a research assistant for a microbiome modifier evidence database.
Each "modifier" is a compound (food, supplement, herb, drug) that has been
scored against a fixed list of gut bacterial taxa using published studies.

Use the `run_sql` tool to query Postgres. Run multiple queries if needed.
Then write a clear prose answer with inline citations to specific studies.

{STATIC_SCHEMA}

LOADED COMPOUNDS:
{compounds}

CANONICAL TAXA (in `rollup.taxon_listed_name`, exact spelling):
{taxa}

STUDY TYPES present in `studies.study_type`: {study_types}
EVIDENCE TIERS present in `rollup.evidence_tier`: {tiers}
"""


# --------------------------------------------------------------------
# Tool definition + execution
# --------------------------------------------------------------------

TOOLS = [
    {
        "name": "run_sql",
        "description": (
            "Execute a SELECT query against the microbiome research Postgres database "
            "and return up to 500 rows as JSON. Only SELECT/WITH queries are allowed; "
            "any data-modifying statement is rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single Postgres SELECT or WITH (CTE) query.",
                },
            },
            "required": ["query"],
        },
    },
]


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|copy|vacuum|reindex)\b",
    re.IGNORECASE,
)


def execute_tool(name, args):
    if name != "run_sql":
        return {"error": f"unknown tool: {name}"}
    query = (args.get("query") or "").strip().rstrip(";")
    if not query:
        return {"error": "empty query"}
    if _FORBIDDEN.search(query):
        return {"error": "only read-only queries are allowed"}
    head = query.lstrip().split()[0].lower() if query.lstrip() else ""
    if head not in ("select", "with"):
        return {"error": "query must start with SELECT or WITH"}
    try:
        rows = run_select(query)
        return {"row_count": len(rows), "rows": rows}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------
# Chat agent loop
# --------------------------------------------------------------------

def _serialize_for_llm(obj):
    """JSON-friendly serializer that handles dates, decimals, etc."""
    return json.dumps(obj, default=str)


def run_agent(client, system_prompt, history, user_message, tool_log):
    """
    Run a tool-use loop until Claude produces a final text answer.
    Mutates `tool_log` with each (query, result) pair so the UI can show them.
    Returns the final assistant text.
    """
    messages = list(history) + [{"role": "user", "content": user_message}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = "\n\n".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            return text or "(no response)"

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result = execute_tool(block.name, block.input)
                tool_log.append({
                    "tool": block.name,
                    "input": block.input,
                    "result": result,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _serialize_for_llm(result),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        return f"(unexpected stop_reason: {response.stop_reason})"

    return "(stopped: hit MAX_TURNS without a final answer)"


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

st.set_page_config(page_title="Microbiome modifier chat", layout="wide")
st.title("Microbiome modifier chat")


def check_password():
    """Tiny shared-password gate. If APP_PASSWORD isn't set, the gate is bypassed."""
    expected = get_secret("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Enter access password", type="password")
    if pw:
        if pw == expected:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


api_key = get_secret("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it to .env (local) or Streamlit secrets (cloud).")
    st.stop()

client = Anthropic(api_key=api_key)

# Sidebar: loaded compounds + example questions
with st.sidebar:
    st.header("Loaded compounds")
    try:
        dyn = fetch_dynamic_schema()
        for c in dyn["compounds"]:
            st.write(f"- {c['display_name']}")
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        st.stop()

    st.divider()
    st.header("Example questions")
    st.caption(
        "- Which compounds increase Akkermansia, and how strong is the evidence?\n"
        "- Compare cranberry and curcumin on Bifidobacterium\n"
        "- What does the literature say about saccharomyces boulardii and Lactobacillus, with citations?\n"
        "- Which studies on berberine had concerns about technique resolution?\n"
        "- Show me the strongest human-direct findings across all compounds"
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, tool_log?}]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            with st.expander(f"queries ({len(msg['tool_log'])})", expanded=False):
                for i, t in enumerate(msg["tool_log"], 1):
                    st.markdown(f"**Query {i}**")
                    st.code(t["input"].get("query", ""), language="sql")
                    if "error" in t["result"]:
                        st.error(t["result"]["error"])
                    else:
                        st.caption(f"{t['result']['row_count']} row(s)")
                        if t["result"]["rows"]:
                            st.dataframe(t["result"]["rows"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about the microbiome data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build prior history for the API (strip the just-added user msg, the agent adds it back)
    history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] in ("user", "assistant"):
            history.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tool_log = []
            try:
                system_prompt = build_system_prompt()
                answer = run_agent(client, system_prompt, history, prompt, tool_log)
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)
        if tool_log:
            with st.expander(f"queries ({len(tool_log)})", expanded=False):
                for i, t in enumerate(tool_log, 1):
                    st.markdown(f"**Query {i}**")
                    st.code(t["input"].get("query", ""), language="sql")
                    if "error" in t["result"]:
                        st.error(t["result"]["error"])
                    else:
                        st.caption(f"{t['result']['row_count']} row(s)")
                        if t["result"]["rows"]:
                            st.dataframe(t["result"]["rows"], use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_log": tool_log,
    })
