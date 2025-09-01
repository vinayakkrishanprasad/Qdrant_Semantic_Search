import os, time, re
import streamlit as st
from typing import List, Tuple, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, Range
from fastembed import TextEmbedding

# -----------------------------
# Config (use env vars!)
# -----------------------------
QDRANT_URL = "https://fa3846b5-7346-4ecc-83e1-574627eabcd2.us-east-1-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9HAtIhes8BmSQP_Z4EkOBxkwS10jVPb-fYx2cZwTClU"
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_demo")

if not QDRANT_URL or not QDRANT_API_KEY:
    st.warning("Set QDRANT_URL and QDRANT_API_KEY env vars before running.")
st.set_page_config(page_title="Qdrant: Hybrid News Search Demo", layout="wide")

# -----------------------------
# Heavy resources cached
# -----------------------------
@st.cache_resource
def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def get_embedder():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

client = get_client()
embedder = get_embedder()

st.title("Hybrid Search on News: Semantic + Structured + Custom Scoring (Qdrant)")
st.caption("Step-by-step: ① Keyword baseline → ② Semantic → ③ + Filters → ④ + Custom scoring")

# -----------------------------
# Utils: scroll payloads
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_payloads() -> List[Dict[str, Any]]:
    """Load all payloads (200 docs) to support keyword baseline and facets."""
    all_points = []
    offset = None
    while True:
        ret = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if isinstance(ret, tuple) and len(ret) == 2:
            points, next_off = ret
        else:
            points, next_off = ret, None
        all_points.extend(points)
        if next_off is None:
            break
        offset = next_off
    # Convert to list of plain dicts (payload + id)
    docs = []
    for p in all_points:
        pl = p.payload or {}
        pl["_id"] = p.id
        docs.append(pl)
    return docs

@st.cache_data(ttl=600, show_spinner=False)
def load_facets() -> Tuple[List[str], List[str], List[str], Tuple[int,int]]:
    docs = fetch_all_payloads()
    cats = sorted({d.get("category") for d in docs if d.get("category")})
    authors = sorted({d.get("author") for d in docs if d.get("author")})
    sources = sorted({d.get("source") for d in docs if d.get("source")})
    years = [d.get("year") for d in docs if isinstance(d.get("year"), int)]
    yr_min = min(years) if years else 2018
    yr_max = max(years) if years else 2025
    return cats, authors, sources, (yr_min, yr_max)

cats_all, authors_all, sources_all, (yr_min_all, yr_max_all) = load_facets()

# -----------------------------
# Sidebar: filters + scoring
# -----------------------------
with st.sidebar:
    st.header("Filters (used in steps ③ & ④)")
    cats = st.multiselect("Category", cats_all)
    yr_min, yr_max = st.slider("Year range", min_value=yr_min_all, max_value=yr_max_all,
                               value=(max(yr_min_all, 2019), yr_max_all))
    auths = st.multiselect("Author", authors_all[:50])
    srcs = st.multiselect("Source", sources_all)

    st.header("Custom scoring weights (used in ④)")
    w_rec = st.slider("Recency", 0.0, 1.0, 0.30, 0.05)
    w_cat = st.slider("Category", 0.0, 1.0, 0.25, 0.05)
    w_aut = st.slider("Author", 0.0, 1.0, 0.15, 0.05)
    w_src = st.slider("Source", 0.0, 1.0, 0.10, 0.05)

    st.header("Preferences (boost) (used in ④)")
    pref_cats = st.multiselect("Prefer categories", cats_all)
    pref_author = st.text_input("Prefer author (exact)", "")
    pref_sources = st.multiselect("Prefer sources", sources_all)

    st.header("Results")
    topk = st.number_input("Top-K", 5, 50, 10)

# -----------------------------
# Query + presets
# -----------------------------
c1, c2 = st.columns([3, 1])
with c1:
    query = st.text_input("Query", value="climate change policies")

st.markdown("— Use the tabs below to show the evolution from keyword → semantic → hybrid → boosted.")

# -----------------------------
# Helpers
# -----------------------------
def make_filter() -> Filter | None:
    must = []
    if cats:  must.append(FieldCondition(key="category", match=MatchAny(any=cats)))
    if auths: must.append(FieldCondition(key="author",   match=MatchAny(any=auths)))
    if srcs:  must.append(FieldCondition(key="source",   match=MatchAny(any=srcs)))
    rng = {}
    if yr_min is not None: rng["gte"] = int(yr_min)
    if yr_max is not None: rng["lte"] = int(yr_max)
    must.append(FieldCondition(key="year", range=Range(**rng)))
    return Filter(must=must)

def normalize_recency(year: int | None, now=2025, floor=2019):
    if not isinstance(year, int):
        return 0.0
    year = max(floor, min(now, year))
    return (year - floor) / max(1, (now - floor))

def custom_score(base: float, pl: dict):
    s = float(base)
    s += w_rec * normalize_recency(pl.get("year"))
    if pl.get("category") in (pref_cats or []):         s += w_cat
    if pref_author and pl.get("author") == pref_author:  s += w_aut
    if pl.get("source") in (pref_sources or []):         s += w_src
    return s

def qdrant_semantic(qtext: str, qfilter: Filter | None, overfetch: int):
    qvec = list(embedder.embed([qtext]))[0]
    t0 = time.time()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=overfetch,
        with_payload=True,
        with_vectors=False,
        query_filter=qfilter,
    )
    ms = (time.time() - t0) * 1000
    return hits.points, ms

def keyword_baseline(qtext: str, topk: int) -> List[Dict[str, Any]]:
    """Very simple keyword search over local payloads; demonstrates brittleness."""
    docs = fetch_all_payloads()
    tokens = [t for t in re.findall(r"\w+", qtext.lower()) if len(t) > 2]
    scored = []
    for d in docs:
        hay = f"{d.get('title','')} {d.get('description','')} {d.get('body','')}".lower()
        score = sum(hay.count(tok) for tok in tokens)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:topk]]

def render_doc(p: Dict[str, Any], meta_line: str, extra_line: str | None = None):
    st.markdown(f"### {p.get('title','(untitled)')}")
    st.write(meta_line)
    if extra_line:
        st.caption(extra_line)
    # Preview + expander for full text
    preview = (p.get("description") or p.get("body") or "")[:280]
    st.write(preview + ("…" if len(preview) == 280 else ""))
    with st.expander("Show full text"):
        if p.get("description"):
            st.markdown("**Description**")
            st.write(p.get("description"))
        if p.get("body"):
            st.markdown("**Body**")
            st.write(p.get("body"))
    st.write("---")

# -----------------------------
# Tabs for the 4-step story
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "① Keyword baseline",
    "② Semantic (no filters)",
    "③ Semantic + Filters",
    "④ + Custom scoring"
])

with tab1:
    st.subheader("① Traditional keyword search (brittle)")
    st.caption("Matches exact words in title/description/body; easily misses meaning.")
    if st.button("Run keyword search"):
        docs = keyword_baseline(query, topk)
        if not docs:
            st.warning("No keyword matches. Try simpler words or remove stopwords.")
        for d in docs:
            meta = f"**Category:** {d.get('category')} | **Year:** {d.get('year')} | **Author:** {d.get('author')} | **Source:** {d.get('source')}"
            render_doc(d, meta_line=meta, extra_line="(keyword match)")

with tab2:
    st.subheader("② Pure semantic search (embeddings; no filters)")
    st.caption("Understands meaning, but can be fuzzy (e.g., 'election' vs 'protest').")
    if st.button("Run semantic (no filters)"):
        points, ms = qdrant_semantic(query.strip(), qfilter=None, overfetch=max(50, topk))
        st.caption(f"Qdrant semantic in {ms:.1f} ms")
        for h in points[:topk]:
            p = h.payload or {}
            meta = f"**Category:** {p.get('category')} | **Year:** {p.get('year')} | **Author:** {p.get('author')} | **Source:** {p.get('source')}"
            extra = f"Base similarity: {h.score:.4f}"
            render_doc(p, meta_line=meta, extra_line=extra)

with tab3:
    st.subheader("③ Semantic + structured filters (hybrid)")
    st.caption("Narrow by category/year/author/source using Qdrant payload indexes.")
    if st.button("Run semantic + filters"):
        qfilter = make_filter()
        points, ms = qdrant_semantic(query.strip(), qfilter=qfilter, overfetch=max(50, topk))
        st.caption(f"Qdrant semantic+filters in {ms:.1f} ms")
        for h in points[:topk]:
            p = h.payload or {}
            meta = f"**Category:** {p.get('category')} | **Year:** {p.get('year')} | **Author:** {p.get('author')} | **Source:** {p.get('source')}"
            extra = f"Base similarity: {h.score:.4f}"
            render_doc(p, meta_line=meta, extra_line=extra)

with tab4:
    st.subheader("④ Hybrid + custom re-scoring (business rules)")
    st.caption("Blend similarity with recency/category/author/source boosts.")
    if st.button("Run + custom scoring"):
        qfilter = make_filter()
        points, ms = qdrant_semantic(query.strip(), qfilter=qfilter, overfetch=max(50, topk))
        rescored = sorted(
            ((custom_score(h.score, h.payload or {}), h) for h in points),
            key=lambda x: x[0], reverse=True
        )[:topk]
        st.caption(f"Qdrant semantic+filters in {ms:.1f} ms • custom re-score applied")
        for final, h in rescored:
            p = h.payload or {}
            meta = f"**Category:** {p.get('category')} | **Year:** {p.get('year')} | **Author:** {p.get('author')} | **Source:** {p.get('source')}"
            extra = f"Base: {h.score:.4f} → Final (custom): {final:.4f}"
            render_doc(p, meta_line=meta, extra_line=extra)

# Helpful sidebar note
with st.sidebar:
    st.markdown("---")
    st.info("Demo flow: ① Keyword → ② Semantic → ③ Add filters → ④ Add business boosts")
