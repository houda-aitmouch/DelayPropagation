"""
AIR-ROBUST
Simulateur de Propagation de Retards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import sys, os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_uploaded_data, build_ref_dicts, load_turnaround_table
from src.simulation  import run_monte_carlo
from src.optimizer   import run_swap_optimizer, load_aircraft_authorizations

# ─────────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIR-ROBUST",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

:root {
    --ram-red:      #C8102E;
    --ram-red-dark: #9b0b22;
    --navy:         #0d1b2a;
    --navy-mid:     #fa9366;
    --blue-soft:    #0d1b2a;
    --bg-main:      #f0f2f6;
    --card-bg:      #ffffff;
    --text-primary: #1a1a2e;
    --text-muted:   #6b7280;
    --border:       #e5e7eb;
    --font:         'Barlow', sans-serif;
    --font-cond:    'Barlow Condensed', sans-serif;
    --hub-orange:   #e67e22;
    --hub-amber:    #f39c12;
    --slack-teal:   #0e7490;
    --slack-light:  #ecfeff;
}

html, body, [class*="st-"] { font-family: var(--font) !important; }
.stApp { background: var(--bg-main); }

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 3px solid var(--ram-red) !important;
}
section[data-testid="stSidebar"] * {
    color: #333333 !important;
    font-family: var(--font) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #555555 !important;
    font-size: 0.77rem;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background: #fafafa !important;
    border: 1px dashed #d0d5dd !important;
    border-radius: 6px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
    color: #888888 !important;
    font-size: 0.75rem !important;
}
section[data-testid="stSidebar"] hr { border-color: #e5e7eb !important; }
section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--ram-red) !important;
}
section[data-testid="stSidebar"] .stCaption { color: #888888 !important; }
section[data-testid="stSidebar"] [data-testid="stImage"] img {
    display: block;
    margin: 0 auto;
}

.stButton > button[kind="primary"],
.stButton > button[kind="primary"]:disabled,
.stButton > button[disabled] {
    background: var(--blue-soft) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: var(--font-cond) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 0.8rem;
    height: 3rem;
    border-radius: 4px;
    box-shadow: 0 4px 10px rgba(13, 27, 42, 0.18);
    transition: background 0.2s, box-shadow 0.2s;
    opacity: 1 !important;
}
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"]:disabled *,
.stButton > button[disabled] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--ram-red) !important;
    box-shadow: 0 6px 14px rgba(13, 27, 42, 0.24) !important;
}
.stButton > button[kind="primary"]:hover * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="primary"]:active {
    background: var(--ram-red-dark) !important;
    box-shadow: 0 2px 6px rgba(13, 27, 42, 0.30) !important;
}

/* ── Section Headers ── */
.section-header, .section-header-hub, .section-header-slack, .section-header-opt {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-cond) !important;
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 22px 0 14px 0;
}
.section-header::before, .section-header-hub::before,
.section-header-slack::before, .section-header-opt::before {
    content: '';
    display: block;
    width: 4px;
    height: 18px;
    border-radius: 2px;
    flex-shrink: 0;
}
.section-header::after, .section-header-hub::after,
.section-header-slack::after, .section-header-opt::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-header::before      { background: var(--ram-red); }
.section-header-hub::before  { background: var(--hub-orange); }
.section-header-slack::before{ background: var(--slack-teal); }
.section-header-opt::before  { background: #1d4ed8; }

/* ── KPI Cards ── */
.kpi-card, .kpi-card-hub, .kpi-card-slack, .kpi-card-opt {
    background: var(--card-bg);
    border-radius: 8px;
    padding: 16px 18px 14px;
    transition: box-shadow 0.2s;
}
.kpi-card {
    border: 1px solid var(--border);
    border-top: 3px solid var(--ram-red);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.kpi-card-hub {
    border: 1px solid #fde8c8;
    border-top: 3px solid var(--hub-orange);
    box-shadow: 0 2px 8px rgba(230,126,34,0.08);
}
.kpi-card-hub:hover { box-shadow: 0 4px 16px rgba(230,126,34,0.14); }
.kpi-card-slack {
    border: 1px solid #a5f3fc;
    border-top: 3px solid var(--slack-teal);
    box-shadow: 0 2px 8px rgba(14,116,144,0.08);
}
.kpi-card-slack:hover { box-shadow: 0 4px 16px rgba(14,116,144,0.14); }
.kpi-card-opt {
    border: 1px solid #bfdbfe;
    border-top: 3px solid #1d4ed8;
    box-shadow: 0 2px 8px rgba(29,78,216,0.08);
}
.kpi-card-opt:hover { box-shadow: 0 4px 16px rgba(29,78,216,0.14); }

.kpi-value {
    font-family: var(--font-cond) !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.67rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 6px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.kpi-delta { font-size: 0.72rem; color: #9aa3b2; margin-top: 5px; }

hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

[data-testid="stExpander"] {
    background: var(--card-bg);
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
[data-testid="stDataFrame"] thead th {
    background: var(--navy) !important;
    color: white !important;
    font-size: 0.74rem;
    font-family: var(--font-cond) !important;
}
[data-baseweb="select"] { border-radius: 6px !important; }
[data-testid="stInfo"] {
    background: #eef4ff !important;
    border-left: 3px solid #6b9cff !important;
    border-radius: 6px;
}

/* ── Badges ── */
.scenario-badge {
    display: inline-block;
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 6px 12px;
    font-size: 0.78rem; font-weight: 600; color: #856404;
    margin-bottom: 12px;
}
.hub-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fff4ec; border: 1.5px solid var(--hub-orange);
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.78rem; font-weight: 600; color: #7a3800;
    margin-bottom: 14px; box-shadow: 0 2px 8px rgba(230,126,34,0.10);
}
.slack-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--slack-light); border: 1.5px solid var(--slack-teal);
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.78rem; font-weight: 600; color: #164e63;
    margin-bottom: 14px; box-shadow: 0 2px 8px rgba(14,116,144,0.10);
}

/* ── Swap Cards ── */
.swap-card {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-left: 4px solid #16a34a; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; font-size: 0.82rem;
}
.swap-card-warn {
    background: #fffbeb; border: 1.5px solid #fcd34d;
    border-left: 4px solid #d97706; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; font-size: 0.82rem;
}
.swap-card-none {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-left: 4px solid #dc2626; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────
def mhm(m):
    m = int(m) % 1440
    return f"{m//60:02d}:{m%60:02d}"

def color_delay(val):
    try:
        v = float(val)
        if v <= 0:  return "color:#27ae60; font-weight:600"
        if v <= 15: return "color:#e67e22; font-weight:600"
        return "color:#e74c3c; font-weight:600"
    except Exception:
        return ""

def color_markov(val):
    return {
        "Normal":  "color:#27ae60; font-weight:700",
        "Alerte":  "color:#e67e22; font-weight:700",
        "Bloque":  "color:#e74c3c; font-weight:700",
    }.get(str(val), "")

def color_gain(val):
    try:
        v = float(val)
        if v > 2:  return "color:#27ae60; font-weight:700"
        if v > 0:  return "color:#2dd4bf; font-weight:600"
        if v < -2: return "color:#e74c3c; font-weight:700"
        return "color:#6b7280"
    except Exception:
        return ""

def color_otp(val):
    if val == "A l heure": return "color:#27ae60; font-weight:700"
    if val == "En retard":  return "color:#e74c3c; font-weight:700"
    return ""

def color_tampon(val):
    if val != "Non": return "color:#0e7490; font-weight:700"
    return "color:#9aa3b2"

def color_tampon_reco(val):
    try:
        v = float(val)
        if v <= 0:  return "color:#27ae60; font-weight:600"
        if v <= 15: return "color:#e67e22; font-weight:600"
        return "color:#e74c3c; font-weight:600"
    except Exception:
        return ""

def color_role(val):
    """Coloration du rôle dans df_prop."""
    s = str(val)
    if "Injecte"       in s: return "color:#C8102E; font-weight:700"
    if "Propage"       in s: return "color:#e67e22; font-weight:700"
    if "Absorbe"       in s: return "color:#0e7490; font-weight:700"
    if "Protege"       in s: return "color:#27ae60; font-weight:600"
    if "Avant"         in s: return "color:#9aa3b2"
    return "color:#6b7280"

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def section_header_hub(title):
    st.markdown(f'<div class="section-header-hub">{title}</div>', unsafe_allow_html=True)

def section_header_slack(title):
    st.markdown(f'<div class="section-header-slack">{title}</div>', unsafe_allow_html=True)

def _kpi_card_html(css_class, label, value, delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="{css_class}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def kpi_card(label, value, delta=""):
    return _kpi_card_html("kpi-card", label, value, delta)

def kpi_card_hub(label, value, delta=""):
    return _kpi_card_html("kpi-card-hub", label, value, delta)

def kpi_card_slack(label, value, delta=""):
    return _kpi_card_html("kpi-card-slack", label, value, delta)


# ── Date helpers ──────────────────────────────────────────
def _norm_date_key_ui(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    dt = (
        pd.to_datetime(s, errors="coerce", dayfirst=False)
        if (len(s) >= 10 and s[4] == "-" and s[7] == "-")
        else pd.to_datetime(s, errors="coerce", dayfirst=True)
    )
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")

def _date_series_ui(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    out = pd.Series(index=values.index, dtype="datetime64[ns]")
    out.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce", dayfirst=False)
    out.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], errors="coerce", dayfirst=True)
    return out

def _date_scalar_ui(value):
    s = str(value).strip()
    if not s:
        return pd.NaT
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def _filter_by_injected_triplet(df_in, fid, fdate="", msn=""):
    out = df_in[df_in["flight_id"].astype(str).str.strip() == str(fid).strip()]
    date_key = _norm_date_key_ui(fdate)
    if date_key and "flight_date" in out.columns:
        out_key = _date_series_ui(out["flight_date"]).dt.strftime("%Y-%m-%d")
        out = out[out_key == date_key]
    if str(msn).strip() and "aircraft_msn" in out.columns:
        out = out[out["aircraft_msn"].astype(str).str.strip() == str(msn).strip()]
    return out


# ── Optimizer helpers ─────────────────────────────────────
def _next_flights(df_sched, msn, after_dep, flight_date, limit=5):
    sel = df_sched[(df_sched["aircraft_msn"] == msn) & (df_sched["dep_min"] > after_dep)].copy()
    if "flight_date" in sel.columns and flight_date:
        d_sel = _date_series_ui(sel["flight_date"])
        d_ref = _date_scalar_ui(flight_date)
        if pd.notna(d_ref):
            sel = sel[d_sel == d_ref]
    return sel.sort_values("dep_min").head(limit)

def _last_flight_before(df_sched, msn, at_dep, flight_date):
    prev = df_sched[(df_sched["aircraft_msn"] == msn) & (df_sched["dep_min"] < at_dep)].copy()
    if "flight_date" in prev.columns and flight_date:
        d_prev = _date_series_ui(prev["flight_date"])
        d_ref = _date_scalar_ui(flight_date)
        if pd.notna(d_ref):
            prev = prev[d_prev == d_ref]
    prev = prev.sort_values("dep_min")
    if prev.empty:
        return "-"
    return str(prev.iloc[-1].get("flight_id", "-"))

def _crew_for_flight(df_crew, fid, flight_date):
    """Recherche robuste de l'équipage d'un vol dans df_crew."""
    import re as _re
    if not fid or fid == "-" or "flight_sequence" not in df_crew.columns:
        return "-"

    fid_raw = str(fid).strip()

    def _strip_zero(s):
        return s[:-2] if s.endswith(".0") else s

    fid_norm = _strip_zero(fid_raw)
    digits = _re.sub(r"[^\d]", "", fid_norm)

    seen, candidates = set(), []
    for c in [fid_raw, fid_norm, "AT" + digits, digits]:
        if c and c not in seen:
            seen.add(c)
            candidates.append(c)

    rows = pd.DataFrame()
    for candidate in candidates:
        pat = r"(?:^|;)" + _re.escape(candidate) + r"(?:;|$)"
        found = df_crew[df_crew["flight_sequence"].str.contains(pat, na=False, regex=True)].copy()
        if not found.empty:
            rows = found
            break

    if rows.empty:
        return "-"

    if flight_date and "flight_date" in rows.columns:
        d_ref = _date_scalar_ui(flight_date)
        if pd.notna(d_ref):
            d_rows = _date_series_ui(rows["flight_date"])
            by_date = rows[d_rows == d_ref]
            if not by_date.empty:
                rows = by_date

    id_col = "CREW_ID_OPS" if "CREW_ID_OPS" in rows.columns else "crew_id"
    vals = rows[id_col].astype(str).dropna().unique().tolist()
    return ", ".join(vals[:3]) if vals else "-"


def _format_flight_with_times(df_sched, fid, flight_date=""):
    """Retourne un libellé vol avec date/heure départ-arrivée."""
    if not fid or str(fid).strip() in {"", "-"}:
        return "-"

    fid_txt = str(fid).strip()
    rows = df_sched[df_sched["flight_id"].astype(str).str.strip() == fid_txt].copy()

    if "flight_date" in rows.columns and flight_date:
        d_ref = _date_scalar_ui(flight_date)
        if pd.notna(d_ref):
            d_rows = _date_series_ui(rows["flight_date"])
            by_date = rows[d_rows == d_ref]
            if not by_date.empty:
                rows = by_date

    if rows.empty:
        return fid_txt

    row = rows.sort_values("dep_min").iloc[0]
    dep_dt = pd.NaT
    arr_dt = pd.NaT

    if "scheduled_departure" in row.index:
        dep_dt = pd.to_datetime(row.get("scheduled_departure"), errors="coerce", dayfirst=True)
    if "scheduled_arrival" in row.index:
        arr_dt = pd.to_datetime(row.get("scheduled_arrival"), errors="coerce", dayfirst=True)

    d_ref = _date_scalar_ui(flight_date)
    if pd.isna(dep_dt) and pd.notna(d_ref) and "dep_min" in row.index and pd.notna(row.get("dep_min")):
        dep_dt = d_ref.normalize() + pd.to_timedelta(float(row.get("dep_min", 0.0)), unit="m")

    if pd.isna(arr_dt):
        if pd.notna(d_ref) and "arr_min" in row.index and pd.notna(row.get("arr_min")):
            arr_dt = d_ref.normalize() + pd.to_timedelta(float(row.get("arr_min", 0.0)), unit="m")
            if pd.notna(dep_dt) and arr_dt < dep_dt:
                arr_dt = arr_dt + pd.Timedelta(days=1)
        elif pd.notna(dep_dt) and "flight_duration_min" in row.index and pd.notna(row.get("flight_duration_min")):
            arr_dt = dep_dt + pd.to_timedelta(float(row.get("flight_duration_min", 0.0)), unit="m")

    dep_txt = dep_dt.strftime("%d/%m %H:%M") if pd.notna(dep_dt) else "?"
    arr_txt = arr_dt.strftime("%d/%m %H:%M") if pd.notna(arr_dt) else "?"
    return f"{fid_txt} ({dep_txt} -> {arr_txt})"


def _format_sequence_with_times(df_sched, sequence_fids, flight_date=""):
    if not sequence_fids:
        return "Fin de rotation"
    return " ; ".join(_format_flight_with_times(df_sched, fid, flight_date) for fid in sequence_fids)


# ═══════════════════════════════════════════════════════════
# BUG 1 CORRIGÉ — _load_sidebar_schedule charge réellement
# les données dès le premier upload, avant toute simulation.
#
# AVANT : la fonction faisait seek() puis rien — le code de
#         chargement était du dead code dans _save_uploaded_file_temp
#         après un `return`, donc jamais exécuté. Résultat : la liste
#         "Vol(s) à perturber" restait vide au premier chargement.
# ═══════════════════════════════════════════════════════════
def _load_sidebar_schedule(schedule_file, crew_file):
    """Charge df_sched dans session_state si pas encore fait."""
    if "df_sched_sidebar" not in st.session_state:
        schedule_file.seek(0)
        crew_file.seek(0)
        try:
            _ds, _, _ = load_uploaded_data(schedule_file, crew_file, None)
            if _ds is not None:
                st.session_state["df_sched_sidebar"] = _ds
        except Exception:
            pass
        finally:
            schedule_file.seek(0)
            crew_file.seek(0)


def _save_uploaded_file_temp(uploaded_file):
    """Sauvegarde un fichier uploadé dans un fichier temporaire et retourne le chemin."""
    if uploaded_file is None:
        return None
    suffix = ".csv" if uploaded_file.name.lower().endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:inline-flex; align-items:center; gap:9px; margin-top:9px;">
            <svg width="13" height="19" viewBox="0 0 24 24" fill="#C8102E"
                 xmlns="http://www.w3.org/2000/svg">
                <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67
                         10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5
                         1v-1.5L13 19v-5.5l8 2.5z"/>
            </svg>
            <span style="font-family:'Barlow Condensed',sans-serif;
                         font-size:1rem; font-weight:700;
                         color:#888888 !important; letter-spacing:2px;
                         text-transform:uppercase;">
                Configuration
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "<p style='color:#888888 !important; font-size:0.72rem; font-weight:600;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;'>"
        "Données d'entrée</p>",
        unsafe_allow_html=True,
    )

    schedule_file = st.file_uploader("1 — Programme de vols",   type=["csv", "xlsx"])
    crew_file     = st.file_uploader("2 — Rotations équipage",  type=["csv", "xlsx"])
    ref_file      = st.file_uploader("3 — Paramètres de référence", type=["csv", "xlsx"])
    turnaround_file = st.file_uploader(
        "4 — Turnaround par aéroport",
        type=["csv", "xlsx"],
        help="CSV: airport, aircraft_type, min_turnaround_min",
    )

    files_ok = bool(schedule_file and crew_file and ref_file and turnaround_file)

    st.divider()

    st.markdown(
        "<p style='color:#888888 !important; font-size:0.72rem; font-weight:600;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;'>"
        "Paramètres de simulation</p>",
        unsafe_allow_html=True,
    )
    n_sims     = st.slider("Nombre de simulations",    50, 500, 200, 50)
    otp_thresh = st.slider("Seuil OTP (min)",            5,  30,  15,   5,
                           help="Un vol est à l'heure si son retard est < ce seuil.")
    buffer     = st.slider("Temps tampon global (min)",  0,  60,   0,   5)

    st.divider()

    st.markdown(
        "<p style='color:#888888 !important; font-size:0.72rem; font-weight:600;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;'>"
        "Mode de simulation</p>",
        unsafe_allow_html=True,
    )

    mode_sim = st.radio(
        "Choisir le mode",
        ["Auto — Loi Gamma", "Manuel — Injection de retard"],
        help=(
            "Auto : Monte Carlo avec loi Gamma sur tous les vols.\n"
            "Manuel : un seul vol est perturbé, les autres partent à l'heure "
            "→ propagation pure et lisible."
        ),
    )
    mode = "manuel" if mode_sim.startswith("Manuel") else "auto"

    injected = {}
    injected_targets = []
    use_markov = True

    if mode == "manuel":
        if files_ok:
            _load_sidebar_schedule(schedule_file, crew_file)

        markov_mode = st.radio(
            "Turnaround",
            ["Avec Markov — turnaround aléatoire", "Sans Markov — turnaround minimum fixe"],
            help=(
                "Avec Markov : le turnaround est tiré aléatoirement via la chaîne de Markov.\n"
                "Sans Markov : le turnaround = min_turnaround fixe par type d'avion."
            ),
        )
        use_markov = markov_mode.startswith("Avec")

        vols_disponibles = []
        target_map = {}
        if "df_sched_sidebar" in st.session_state:
            df_side = st.session_state["df_sched_sidebar"].copy()
            if "flight_date" not in df_side.columns:
                df_side["flight_date"] = ""

            for _, rr in df_side.iterrows():
                fid = str(rr.get("flight_id", "")).strip()
                if not fid:
                    continue
                fdate = str(rr.get("flight_date", "")).strip()
                msn = str(rr.get("aircraft_msn", "")).strip()
                base_label = f"{fid} | {msn or 'MSN?'}"

                label = base_label
                suffix = 2
                while label in target_map:
                    label = f"{base_label} #{suffix}"
                    suffix += 1

                target_map[label] = {
                    "flight_id": fid,
                    "flight_date": fdate,
                    "aircraft_msn": msn,
                }

            vols_disponibles = sorted(target_map.keys())

        vols_injectes = st.multiselect(
            "Vol(s) à perturber",
            vols_disponibles,
            help="Sélectionnez la rotation précise.",
        )
        per_flight_delays = {}
        for lbl in vols_injectes:
            per_flight_delays[lbl] = float(st.number_input(
                f"Retard {lbl} (min)",
                min_value=0.0,
                max_value=300.0,
                value=60.0,
                step=5.0,
                key=f"manual_flight_delay_{lbl}",
            ))

        dedup_targets = {}

        for lbl in vols_injectes:
            if lbl not in target_map:
                continue
            tgt = dict(target_map[lbl])
            minutes = float(per_flight_delays.get(lbl, 0.0))
            k = (tgt.get("flight_id", ""), tgt.get("flight_date", ""), tgt.get("aircraft_msn", ""))
            prev = dedup_targets.get(k)
            if prev is None or minutes > float(prev.get("minutes", 0.0)):
                tgt["minutes"] = minutes
                dedup_targets[k] = tgt

        injected_targets = list(dedup_targets.values())

        injected = {}
        for t in injected_targets:
            fid = str(t.get("flight_id", "")).strip()
            if not fid:
                continue
            injected[fid] = max(float(t.get("minutes", 0.0)), float(injected.get(fid, 0.0)))

        if not injected_targets:
            st.caption("Sélectionnez au moins un vol pour activer le scénario.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SCÉNARIO HUB CONGESTION
    # ══════════════════════════════════════════════════════════
    st.markdown(
        "<p style='color:#e67e22 !important; font-size:0.72rem; font-weight:700;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;'>"
        "Scénario Hub Congestion</p>",
        unsafe_allow_html=True,
    )

    hub_enabled = st.toggle("Activer la congestion hub", value=False,
        help="Simule une journée où un aéroport hub est fortement perturbé.")

    hub_airport = None
    hub_factor  = 1.2

    if hub_enabled:
        if files_ok:
            _load_sidebar_schedule(schedule_file, crew_file)

        airports_disponibles = []
        if "df_sched_sidebar" in st.session_state:
            airports_disponibles = sorted(
                st.session_state["df_sched_sidebar"]["origin"].dropna().unique().tolist()
            )

        hub_selected = st.multiselect(
            "Aéroport(s) congestionné(s)",
            airports_disponibles,
            help="Tous les vols au départ de ces aéroports seront amplifiés.",
        )

        if hub_selected:
            hub_airport = hub_selected if len(hub_selected) > 1 else hub_selected[0]
            if mode == "auto":
                hub_factor = st.slider(
                    "Facteur d'amplification Gamma (×)",
                    min_value=1.0, max_value=3.0, value=1.2, step=0.1,
                    help="Shape k × hub_factor pour les vols des hubs.",
                )
                st.caption(f"⚙ Shape Gamma effectif sur {', '.join(hub_selected)} : {hub_factor:.1f}× la valeur normale.")
            else:
                hub_factor = st.slider(
                    "Intensité de la perturbation (×)",
                    min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                    help="Tout retard accumulé au départ du hub est amplifié par ce facteur.",
                )
                st.caption(
                    f"⚙Tout retard au départ de {', '.join(hub_selected)} "
                    f"sera amplifié par {hub_factor:.1f}×."
                )
        else:
            st.caption("Sélectionnez un aéroport pour activer le scénario.")
            hub_airport = None

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SCÉNARIO TAMPON VARIABLE (SLACK TIME)
    # ══════════════════════════════════════════════════════════
    st.markdown(
        "<p style='color:#0e7490 !important; font-size:0.72rem; font-weight:700;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;'>"
        "Scénario Tampon Variable</p>",
        unsafe_allow_html=True,
    )

    slack_enabled = st.toggle("Activer le tampon variable", value=False,
        help="Réduit le turnaround effectif des vols concernés pour libérer l'avion plus tôt.")

    slack_config  = None
    slack_minutes = 10

    if slack_enabled:
        slack_minutes = st.select_slider(
            "Minutes de tampon",
            options=[5, 10, 15, 20, 30, 40, 50, 60],
            value=10,
            help="Minutes ajoutées au turnaround Markov des vols concernés.",
        )

        slack_scope = st.radio(
            "Portée du tampon",
            ["Plage horaire", "Aéroport(s)",
             "Avion(s)", "Avion(s) + plage horaire", "Vol(s) spécifique(s)"],
        )

        avions_disponibles = []
        airports_slack_disponibles = []
        if "df_sched_sidebar" in st.session_state:
            avions_disponibles = sorted(
                st.session_state["df_sched_sidebar"]["aircraft_msn"].dropna().unique().tolist()
            )
            airports_slack_disponibles = sorted(
                st.session_state["df_sched_sidebar"]["origin"].dropna().unique().tolist()
            )

        def _window_inputs(key_prefix):
            sl1, sl2 = st.columns(2)
            with sl1: wsh = st.number_input("Début (heure)", 0, 23, 6, 1, key=f"{key_prefix}_s")
            with sl2: weh = st.number_input("Fin (heure)",   0, 23, 9, 1, key=f"{key_prefix}_e")
            return int(wsh) * 60, int(weh) * 60, int(wsh), int(weh)

        if slack_scope == "Plage horaire":
            window_start, window_end, wsh, weh = _window_inputs("win")
            n_win = 0
            if "df_sched_sidebar" in st.session_state:
                ds = st.session_state["df_sched_sidebar"]
                n_win = int(((ds["dep_min"] >= window_start) & (ds["dep_min"] <= window_end)).sum())
            st.caption(f"⏱ {n_win} vols entre {wsh:02d}:00 et {weh:02d}:00 (+{slack_minutes} min).")
            slack_config = {
                "minutes": slack_minutes, "scope": "window",
                "window_start": window_start, "window_end": window_end,
            }

        elif slack_scope == "Aéroport(s)":
            apts_slack = st.multiselect("Aéroport(s) à tamponner", airports_slack_disponibles, key="slack_apts")
            if apts_slack:
                n_apts = 0
                if "df_sched_sidebar" in st.session_state:
                    n_apts = int(st.session_state["df_sched_sidebar"]["origin"].isin(apts_slack).sum())
                st.caption(f"⏱ {n_apts} vols au départ de {', '.join(apts_slack)} (+{slack_minutes} min).")
                slack_config = {"minutes": slack_minutes, "scope": "airports", "airports": apts_slack}
            else:
                st.caption("Sélectionnez au moins un aéroport.")

        elif slack_scope == "Avion(s)":
            msn_slack = st.multiselect("Avion(s) à tamponner", avions_disponibles, key="slack_msn")
            if msn_slack:
                n_msn = 0
                if "df_sched_sidebar" in st.session_state:
                    n_msn = int(st.session_state["df_sched_sidebar"]["aircraft_msn"].isin(msn_slack).sum())
                st.caption(f"⏱ {n_msn} vols de {', '.join(msn_slack)} (+{slack_minutes} min).")
                slack_config = {"minutes": slack_minutes, "scope": "aircraft", "aircraft_msn": msn_slack}
            else:
                st.caption("Sélectionnez au moins un avion.")

        elif slack_scope == "Avion(s) + plage horaire":
            msn_slack2 = st.multiselect("Avion(s) à tamponner", avions_disponibles, key="slack_msn2")
            window_start2, window_end2, wsh2, weh2 = _window_inputs("win2")
            if msn_slack2:
                n_combo = 0
                if "df_sched_sidebar" in st.session_state:
                    ds = st.session_state["df_sched_sidebar"]
                    n_combo = int((ds["aircraft_msn"].isin(msn_slack2) &
                                   (ds["dep_min"] >= window_start2) &
                                   (ds["dep_min"] <= window_end2)).sum())
                st.caption(f"⏱ {n_combo} vols entre {wsh2:02d}:00 et {weh2:02d}:00 (+{slack_minutes} min).")
                slack_config = {
                    "minutes": slack_minutes, "scope": "window_aircraft",
                    "aircraft_msn": msn_slack2,
                    "window_start": window_start2, "window_end": window_end2,
                }
            else:
                st.caption("Sélectionnez au moins un avion.")

        elif slack_scope == "Vol(s) spécifique(s)":
            vols_slack = []
            if "df_sched_sidebar" in st.session_state:
                vols_slack = sorted(st.session_state["df_sched_sidebar"]["flight_id"].tolist())
            vols_slack_sel = st.multiselect("Vol(s) à tamponner", vols_slack, key="slack_flight")
            if vols_slack_sel:
                st.caption(f"⏱ +{slack_minutes} min de turnaround sur {len(vols_slack_sel)} vol(s).")
                slack_config = {"minutes": slack_minutes, "scope": "flight", "flight_id": vols_slack_sel}
            else:
                st.caption("Sélectionnez au moins un vol.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SCÉNARIO RÉAFFECTATION AVION
    # ══════════════════════════════════════════════════════════
    st.markdown(
        "<p style='color:#1d4ed8 !important; font-size:0.72rem; font-weight:700;"
        " letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;'>"
        "Scénario Réaffectation Avion</p>",
        unsafe_allow_html=True,
    )

    optimizer_enabled = st.toggle(
        "Activer la réaffectation avion", value=False,
        key="sidebar_optimizer_enabled",
        help="Prépare l'analyse de substitution d'avion (A1 -> A2).",
    )

    optimizer_config = {
        "enabled": optimizer_enabled,
        "delay_threshold": 30,
        "show_infeasible": False,
        "authorizations_file": None,
    }
    if optimizer_enabled:
        optimizer_config["delay_threshold"] = st.slider(
            "Seuil de retard moyen à analyser (min)",
            min_value=10, max_value=120, value=30, step=5,
            key="sidebar_optimizer_delay_threshold",
        )
        optimizer_config["show_infeasible"] = st.checkbox(
            "Afficher aussi les substitutions infaisables",
            value=False, key="sidebar_optimizer_show_infeasible",
        )

        st.markdown(
            "<p style='color:#6b7280; font-size:0.70rem; margin-top:12px; margin-bottom:6px;'>"
            "Autorisations aeroports (optionnel)</p>",
            unsafe_allow_html=True,
        )
        authorizations_file = st.file_uploader(
            "Fichier autorisations",
            type=["csv", "xlsx"],
            key="sidebar_authorizations_file",
            help=(
                "CSV avec colonnes :\n"
                "- Par avion : aircraft_msn, aircraft_type, authorized_airports\n"
                "- Par type : aircraft_type, authorized_airports\n"
                "Aeroports separes par ; ou ,"
            ),
        )
        if authorizations_file:
            optimizer_config["authorizations_file"] = authorizations_file
            st.caption(f"Fichier charge : {authorizations_file.name}")
        else:
            st.caption("Sans fichier -> autorisations deduites du planning.")

    st.divider()

    run_btn = st.button(
        "Lancer la simulation",
        type="primary",
        use_container_width=True,
        disabled=not files_ok,
    )
    if not files_ok:
        st.caption("Chargez les 4 fichiers pour activer la simulation.")
    elif mode == "manuel" and not injected:
        st.error("Sélectionnez un vol à perturber avant de lancer.")
    elif hub_enabled and not hub_airport:
        st.warning("Sélectionnez un aéroport hub ou désactivez le scénario.")
    elif slack_enabled and slack_config is None:
        st.warning("Configurez le tampon variable ou désactivez le scénario.")


# ─────────────────────────────────────────────────────────
# ECRAN D'ATTENTE — FICHIERS MANQUANTS
# ─────────────────────────────────────────────────────────
if not files_ok:
    st.markdown(
        '<div style="display:flex; flex-direction:column; align-items:center;'
        'justify-content:center; min-height:70vh; padding:20px;">'
        '<p style="font-size:0.85rem; color:#6b7280; margin:0 0 20px 0; text-align:center;">'
        'Chargez vos quatre fichiers dans le panneau de gauche pour démarrer.'
        '</p>'
        '<div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px;'
        'overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); width:100%; max-width:560px;">'
        '<div style="background:linear-gradient(135deg,#0d1b2a,#1a2e45);'
        'padding:12px 22px; border-bottom:2px solid #C8102E; text-align:center;">'
        '<span style="color:#8fa3bc; font-size:0.68rem; letter-spacing:2px;'
        'text-transform:uppercase; font-family:\'Barlow Condensed\',sans-serif;'
        'font-weight:700;">Fichiers requis</span>'
        '</div>'
        '<div style="padding:0 22px;">'
        '<div style="padding:16px 0; border-bottom:1px solid #f0f1f3; text-align:center;">'
        '<div style="font-size:0.82rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;">1 — Programme de vols</div>'
        '<div style="font-size:0.76rem;color:#9aa3b2;">Un vol par ligne : identifiant, aéroports, horaires, immatriculation et type.</div>'
        '</div>'
        '<div style="padding:16px 0; border-bottom:1px solid #f0f1f3; text-align:center;">'
        '<div style="font-size:0.82rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;">2 — Rotations équipage</div>'
        '<div style="font-size:0.76rem;color:#9aa3b2;">Identifiant équipage, séquence des vols et temps de repos minimum.</div>'
        '</div>'
        '<div style="padding:16px 0; border-bottom:1px solid #f0f1f3; text-align:center;">'
        '<div style="font-size:0.82rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;">3 — Paramètres de référence</div>'
        '<div style="font-size:0.76rem;color:#9aa3b2;">Temps de rotation minimum, paramètres Gamma et Markov.</div>'
        '</div>'
        '<div style="padding:16px 0; text-align:center;">'
        '<div style="font-size:0.82rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;">4 — Turnaround par aéroport</div>'
        '<div style="font-size:0.76rem;color:#9aa3b2;">CSV : airport, aircraft_type, min_turnaround_min.</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────
# CHARGEMENT ET VALIDATION DES DONNEES
# ─────────────────────────────────────────────────────────
df_sched, df_crew, df_ref = load_uploaded_data(schedule_file, crew_file, ref_file)
if df_sched is None:
    st.stop()

st.session_state["df_sched_sidebar"] = df_sched
turnaround_table = load_turnaround_table(turnaround_file)
mt, gs, gsc, mm, mp, turnaround_table = build_ref_dicts(
    df_ref,
    turnaround_table=turnaround_table,
)


# ─────────────────────────────────────────────────────────
# APERCU DES DONNEES
# ─────────────────────────────────────────────────────────
st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

with st.expander("Aperçu des données chargées", expanded=False):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vols",               len(df_sched))
    m2.metric("Avions",             df_sched["aircraft_msn"].nunique())
    m3.metric("Pairings équipage",  len(df_crew))
    m4.metric("Retard moyen Gamma", f"{gs * gsc:.0f} min")

    t1, t2, t3, t4 = st.tabs([
        "Programme de vols",
        "Rotations équipage",
        "Paramètres Markov",
        "Turnaround",
    ])

    cols_sched = [c for c in [
        "flight_id", "FN_CARRIER", "FN_NUMBER", "DAY_OF_ORIGIN", "flight_date",
        "AC_OWNER", "AC_SUBTYPE", "aircraft_type", "AC_REGISTRATION", "aircraft_msn",
        "DEP_AP_SCHED", "ARR_AP_SCHED", "origin", "destination",
        "DEP_TIME_SCHED", "ARR_TIME_SCHED", "scheduled_departure", "scheduled_arrival",
        "flight_duration_min", "LEG_TYPE", "leg_number", "legs_total",
    ] if c in df_sched.columns]

    with t1:
        st.dataframe(df_sched[cols_sched], use_container_width=True, height=260)
    with t2:
        df_crew_view = df_crew.copy()

        def _route_detaillee(row):
            oseq = str(row.get("ORIGINE_SEQUENCE", "")).strip()
            dseq = str(row.get("DESTINATION_SEQUENCE", "")).strip()
            orig_nodes = [x.strip() for x in oseq.split(";") if x.strip()] if oseq else []
            dest_nodes = [x.strip() for x in dseq.split(";") if x.strip()] if dseq else []
            if orig_nodes and dest_nodes:
                return " -> ".join(orig_nodes + [dest_nodes[-1]])
            o = str(row.get("ORIGINE", "")).strip()
            d = str(row.get("DESTINATION", "")).strip()
            if o or d:
                return f"{o} -> {d}".strip()
            return ""

        df_crew_view["Route detaillee"] = df_crew_view.apply(_route_detaillee, axis=1)

        cols_crew = [c for c in [
            "CC_ID", "CA_ID",
            "flight_date", "aircraft_msn", "aircraft_type",
            "flight_sequence", "Route detaillee", "n_flights",
            "ORIGINE", "DESTINATION",
            "route_sequence_type",
        ] if c in df_crew_view.columns]

        if "crew_label" in df_crew_view.columns:
            df_crew_view = df_crew_view.rename(columns={"crew_label": "Équipage (CC | CA)"})
            cols_crew = ["Équipage (CC | CA)" if c == "crew_label" else c for c in cols_crew]

        st.dataframe(
            df_crew_view[cols_crew] if cols_crew else df_crew_view,
            use_container_width=True, height=260,
        )
    with t3:
        df_mk = pd.DataFrame(
            mm,
            index=["Normal", "Alerte", "Bloqué"],
            columns=["vers Normal", "vers Alerte", "vers Bloqué"],
        )
        st.markdown(
            f"**Loi Gamma :** alpha = {gs} | theta = {gsc} min"
            f" | Retard moyen = **{gs * gsc:.0f} min**"
        )
        st.dataframe(
            df_mk.style.format("{:.0%}").background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

    with t4:
        if turnaround_table:
            df_turn = pd.DataFrame(
                [
                    {
                        "airport": ap,
                        "aircraft_type": ac,
                        "min_turnaround_min": val,
                    }
                    for (ap, ac), val in turnaround_table.items()
                ]
            ).sort_values(["airport", "aircraft_type"]).reset_index(drop=True)
            st.caption(f"{len(df_turn)} combinaison(s) chargée(s).")
            st.dataframe(
                df_turn.style.format({"min_turnaround_min": "{:.1f}"}),
                use_container_width=True,
                height=260,
                hide_index=True,
            )
        else:
            st.info("Aucune table turnaround chargée.")


# ─────────────────────────────────────────────────────────
# ATTENTE AVANT LANCEMENT
# ─────────────────────────────────────────────────────────
if not run_btn and "sim_results" not in st.session_state:
    st.info(
        "Données chargées avec succès. "
        "Configurez les paramètres dans la barre latérale, "
        "puis cliquez sur **Lancer la simulation**."
    )
    st.stop()


# ─────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────
if run_btn:
    if mode == "manuel" and not injected_targets:
        st.error("Veuillez sélectionner un vol à perturber dans la barre latérale.")
        st.stop()

    with st.spinner("Simulation Monte Carlo en cours…"):
        pb = st.progress(0, text="Initialisation…")

        np.random.seed(42)
        rng_state = np.random.get_state()

        df_agg, otp_arr, prop_arr, all_results = run_monte_carlo(
            df_sched, df_crew, mt, gs, gsc, mm, mp,
            n_simulations=n_sims,
            otp_threshold=otp_thresh,
            progress_bar=pb,
            mode=mode,
            use_markov=use_markov,
            injected_delays=injected,
            injected_targets=injected_targets,
            hub_airport=hub_airport if hub_enabled else None,
            hub_factor=hub_factor,
            slack_config=slack_config,
            turnaround_table=turnaround_table,
        )
        pb.empty()

        # ── Simulation de référence (sans tampon) si slack actif ──
        df_agg_base = otp_arr_base = prop_arr_base = None
        if slack_enabled and slack_config:
            pb2 = st.progress(0, text="Simulation de référence (sans tampon)…")
            np.random.set_state(rng_state)

            df_agg_base, otp_arr_base, prop_arr_base, _ = run_monte_carlo(
                df_sched, df_crew, mt, gs, gsc, mm, mp,
                n_simulations=n_sims,
                otp_threshold=otp_thresh,
                progress_bar=pb2,
                mode=mode,
                use_markov=use_markov,
                injected_delays=injected,
                injected_targets=injected_targets,
                hub_airport=hub_airport if hub_enabled else None,
                hub_factor=hub_factor,
                slack_config=None,
                turnaround_table=turnaround_table,
            )
            pb2.empty()

        # ── Optimiseur de réaffectation ──
        optimizer_results = None
        if optimizer_enabled and optimizer_config.get("enabled", False):
            opt_threshold = float(optimizer_config.get("delay_threshold", 30))

            auth_file_path = None
            auth_file_uploaded = optimizer_config.get("authorizations_file")
            if auth_file_uploaded is not None:
                auth_file_path = _save_uploaded_file_temp(auth_file_uploaded)

            with st.spinner("Analyse réaffectation avion en cours…"):
                df_opt, swap_list_opt = run_swap_optimizer(
                    df_agg=df_agg,
                    all_results=all_results,
                    df_sched=df_sched,
                    df_crew=df_crew,
                    min_turnaround=mt,
                    authorizations_file=auth_file_path,
                    otp_threshold=otp_thresh,
                    delay_threshold=opt_threshold,
                )

            if auth_file_path and os.path.exists(auth_file_path):
                try:
                    os.unlink(auth_file_path)
                except Exception:
                    pass

            optimizer_results = {
                "df_opt": df_opt,
                "swap_list_opt": swap_list_opt,
                "delay_threshold": opt_threshold,
                "show_infeasible": bool(optimizer_config.get("show_infeasible", False)),
                "authorizations_loaded": auth_file_uploaded is not None,
            }

    st.session_state["sim_results"] = dict(
        df_agg=df_agg, otp_arr=otp_arr, prop_arr=prop_arr,
        all_results=all_results,
        sim_example=all_results[int(np.argsort(otp_arr)[len(otp_arr) // 2])],
        n_sims=n_sims, otp_thresh=otp_thresh,
        injected=injected, injected_targets=injected_targets,
        mode=mode, use_markov=use_markov,
        hub_airport=hub_airport if hub_enabled else None,
        hub_factor=hub_factor, hub_enabled=hub_enabled,
        slack_enabled=slack_enabled, slack_config=slack_config,
        optimizer_enabled=optimizer_enabled,
        optimizer_config=optimizer_config,
        optimizer_results=optimizer_results,
        df_agg_base=df_agg_base,
        otp_arr_base=otp_arr_base,
        prop_arr_base=prop_arr_base,
    )

# ─────────────────────────────────────────────────────────
# RECUPERATION DES RESULTATS
# ─────────────────────────────────────────────────────────
sr              = st.session_state["sim_results"]
df_agg          = sr["df_agg"]
otp_arr         = sr["otp_arr"]
prop_arr        = sr["prop_arr"]
all_results     = sr["all_results"]
sim_example     = sr["sim_example"]
n_sims          = sr["n_sims"]
otp_thresh      = sr["otp_thresh"]
injected        = sr.get("injected", {})
injected_targets = sr.get("injected_targets", [])
mode            = sr.get("mode", "auto")
hub_airport     = sr.get("hub_airport", None)
hub_factor      = sr.get("hub_factor", 1.2)
hub_enabled     = sr.get("hub_enabled", False)
slack_enabled   = sr.get("slack_enabled", False)
slack_config    = sr.get("slack_config", None)
optimizer_enabled  = sr.get("optimizer_enabled", False)
optimizer_config   = sr.get("optimizer_config", {"enabled": False})
optimizer_results  = sr.get("optimizer_results", None)
df_agg_base     = sr.get("df_agg_base", None)
otp_arr_base    = sr.get("otp_arr_base", None)
prop_arr_base   = sr.get("prop_arr_base", None)
delay_map       = df_agg.set_index("flight_id")["mean_dep_delay"].to_dict()
otp_map         = df_agg.set_index("flight_id")["otp_rate"].to_dict()

# ── Références du vol injecté (premier vol pour compatibilité) ──
inj_ref     = injected_targets[0] if injected_targets else {}
inj_fid     = str(inj_ref.get("flight_id",    "")).strip() if inj_ref else (list(injected.keys())[0] if injected else "")
inj_date    = str(inj_ref.get("flight_date",  "")).strip() if inj_ref else ""
inj_msn     = str(inj_ref.get("aircraft_msn", "")).strip() if inj_ref else ""
inj_minutes = float(inj_ref.get("minutes", 0.0)) if inj_ref else (float(list(injected.values())[0]) if injected else 0.0)

# ── Ensemble de tous les flight_id injectés ──
inj_fids_set = {str(t.get("flight_id", "")).strip() for t in injected_targets if t.get("flight_id")}

# ── scope_label initialisé ici pour éviter NameError ──
scope_label = ""


# ─────────────────────────────────────────────────────────
# BADGES MODE ACTIF
# ─────────────────────────────────────────────────────────
badge_row = []

if mode == "manuel" and injected_targets:
    n_inj = len(injected_targets)
    markov_label = "Markov actif" if sr.get("use_markov", True) else "Sans Markov (turnaround fixe)"
    if n_inj == 1:
        row_inj = _filter_by_injected_triplet(df_sched, inj_fid, inj_date, inj_msn)
        route_inj = (f"{row_inj['origin'].iloc[0]} → {row_inj['destination'].iloc[0]}"
                     if not row_inj.empty else "")
        badge_row.append(
            f'<div class="scenario-badge">'
            f'Mode Manuel — <strong>{inj_minutes} min</strong> sur '
            f'<strong>{inj_fid}</strong> ({route_inj}) — {markov_label}'
            f'</div>'
        )
    else:
        fids_label = " + ".join(sorted(inj_fids_set))
        inj_vals = [float(t.get("minutes", 0.0)) for t in injected_targets]
        inj_min_multi = min(inj_vals) if inj_vals else 0.0
        inj_max_multi = max(inj_vals) if inj_vals else 0.0
        inj_label = f"{inj_min_multi:.0f} min" if inj_min_multi == inj_max_multi else f"{inj_min_multi:.0f}-{inj_max_multi:.0f} min"
        badge_row.append(
            f'<div class="scenario-badge">'
            f'Mode Manuel — <strong>{inj_label}</strong> sur '
            f'<strong>{n_inj} vols</strong> ({fids_label}) — {markov_label}'
            f'</div>'
        )
elif mode == "auto":
    badge_row.append(
        '<div style="display:inline-block; background:#e8f5e9; border:1px solid #81c784;'
        ' border-radius:6px; padding:6px 12px; font-size:0.78rem; font-weight:600;'
        ' color:#2e7d32; margin-bottom:12px;">'
        'Mode Auto — Loi Gamma sur tous les vols</div>'
    )

if optimizer_enabled:
    badge_row.append(
        '<div style="display:inline-block; background:#e8f5e9; border:1px solid #81c784;'
        ' border-radius:6px; padding:6px 12px; font-size:0.78rem; font-weight:600;'
        ' color:#2e7d32; margin-bottom:12px;">'
        'SCÉNARIO RÉAFFECTATION AVION</div>'
    )

if optimizer_enabled and hub_airport:
    hub_list = hub_airport if isinstance(hub_airport, list) else [hub_airport]
    hub_label = ", ".join(hub_list)
    n_hub_vols = int(df_sched["origin"].isin(hub_list).sum())
    badge_row.append(
        f'<div class="hub-badge">'
        f'Hub Congestion — {hub_label} ({n_hub_vols} vols) — facteur {hub_factor:.1f}×'
        f'</div>'
    )

if slack_enabled and slack_config:
    s_min   = slack_config["minutes"]
    s_scope = slack_config.get("scope", "global")
    if s_scope == "global":
        scope_label = "tous les vols"
    elif s_scope == "window":
        ws, we = slack_config.get("window_start", 0), slack_config.get("window_end", 0)
        scope_label = f"vols {ws//60:02d}:00–{we//60:02d}:00"
    elif s_scope == "aircraft":
        msn_val = slack_config.get("aircraft_msn", "")
        scope_label = f"avion(s) {', '.join(msn_val) if isinstance(msn_val, list) else msn_val}"
    elif s_scope == "window_aircraft":
        ws, we = slack_config.get("window_start", 0), slack_config.get("window_end", 0)
        msn_val = slack_config.get("aircraft_msn", "")
        scope_label = f"avion(s) {', '.join(msn_val) if isinstance(msn_val, list) else msn_val} · {ws//60:02d}:00–{we//60:02d}:00"
    elif s_scope == "airports":
        apt_val = slack_config.get("airports", [])
        scope_label = f"aéroport(s) {', '.join(apt_val) if isinstance(apt_val, list) else apt_val}"
    elif s_scope == "flight":
        fid_val = slack_config.get("flight_id", "")
        scope_label = f"vol(s) {', '.join(fid_val) if isinstance(fid_val, list) else fid_val}"
    badge_row.append(
        f'<div class="slack-badge">'
        f'Tampon Variable — <strong>+{s_min} min</strong> sur {scope_label}'
        f'</div>'
    )

if badge_row:
    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:4px;">'
        + "".join(badge_row) + "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════
# SECTION RÉAFFECTATION AVION — RÉSULTAT FINAL
# ═════════════════════════════════════════════════════════════
if optimizer_enabled and optimizer_results is not None:
    section_header("Réaffectation Avion — Résultat Final")

    df_opt = optimizer_results.get("df_opt", pd.DataFrame())
    swap_list_opt = optimizer_results.get("swap_list_opt", [])
    show_infeasible_opt = optimizer_results.get("show_infeasible", False)
    opt_threshold = optimizer_results.get("delay_threshold", 30)
    auth_loaded = optimizer_results.get("authorizations_loaded", False)

    if auth_loaded:
        st.markdown(
            '<div style="display:inline-block; background:#dbeafe; border:1px solid #3b82f6;'
            ' border-radius:6px; padding:5px 10px; font-size:0.75rem; font-weight:600;'
            ' color:#1e40af; margin-bottom:10px;">',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Autorisations deduites du planning (aucun fichier fourni).")

    if df_opt is not None and not df_opt.empty:
        n_total = len(swap_list_opt)
        n_ok = sum(1 for s in swap_list_opt if s.has_solution)
        n_otp = sum(1 for s in swap_list_opt if s.has_solution and s.best.new_delay <= otp_thresh)

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Vols analysés", n_total, f"Seuil >= {int(opt_threshold)} min")
        oc2.metric("Switch OK", f"{n_ok}/{n_total}")
        oc3.metric("OTP restauré", n_otp, f"Seuil OTP {otp_thresh} min")

        cols = [c for c in [
            "Vol", "Date vol", "Route", "Avion A1", "Meilleur A2", "Statut",
            "Retard moyen (min)", "Gain (min)", "Nouveau retard (min)",
            "OTP restauré", "Raison infaisabilité",
        ] if c in df_opt.columns]

        df_view = df_opt.copy()
        if not show_infeasible_opt and "Statut" in df_view.columns:
            df_view = df_view[df_view["Statut"].str.contains("Faisable|Partiel", na=False)]

        switch_rows = []
        for sw in swap_list_opt:
            if sw.best is None:
                continue
            status = "Faisable" if sw.best.feasible else "Partiel"
            if status not in {"Faisable", "Partiel"}:
                continue

            a1, a2 = sw.a1_msn, sw.best.a2_msn
            dep_ref = sw.dep_scheduled
            fdate = str(getattr(sw, "flight_date", ""))

            vol_a2_utilise = _last_flight_before(df_sched, a2, dep_ref, fdate)
            a1_next = _next_flights(df_sched, a1, dep_ref, fdate)
            a2_next = _next_flights(df_sched, a2, dep_ref, fdate)

            seq_executee_par_a2 = [sw.flight_id] + a1_next["flight_id"].astype(str).tolist()
            seq_executee_par_a1 = a2_next["flight_id"].astype(str).tolist()

            next_fid_after_switch = seq_executee_par_a2[1] if len(seq_executee_par_a2) > 1 else None
            crew_a2_operera = _crew_for_flight(df_crew, vol_a2_utilise, fdate)
            crew_next_vol_prevu = (
                _crew_for_flight(df_crew, next_fid_after_switch, fdate)
                if next_fid_after_switch else "-"
            )

            switch_rows.append({
                "Vol switche":                       sw.flight_id,
                "Date vol":                          fdate,
                "Statut":                            status,
                "Vol switche (dep-arr)":             _format_flight_with_times(df_sched, sw.flight_id, fdate),
                "Avion A1 (retarde)":                a1,
                "Avion A2 (substitut)":              a2,
                "Dernier vol A2 avant switch":       vol_a2_utilise,
                "Dernier vol A2 (dep-arr)":          _format_flight_with_times(df_sched, vol_a2_utilise, fdate),
                "Equipage A1 prevu (vol switche)":   _crew_for_flight(df_crew, sw.flight_id, fdate),
                "Equipage A2 (operera vol switche)": crew_a2_operera,
                "Vol suivant le switch":             next_fid_after_switch or "Fin de rotation",
                "Vol suivant (dep-arr)":             _format_flight_with_times(df_sched, next_fid_after_switch, fdate),
                "Equipage A2 sur vol suivant":       crew_a2_operera,
                "Equipage A1 libere (vol suivant)":  crew_next_vol_prevu,
                "Sequence A2 apres switch":          " ; ".join(seq_executee_par_a2) or "Fin de rotation",
                "Sequence A2 (dep-arr)":             _format_sequence_with_times(df_sched, seq_executee_par_a2, fdate),
                "Sequence A1 apres switch":          " ; ".join(seq_executee_par_a1) or "Fin de rotation",
                "Sequence A1 (dep-arr)":             _format_sequence_with_times(df_sched, seq_executee_par_a1, fdate),
            })

        base_cols = [c for c in cols if c in df_view.columns]
        df_consolide = df_view[base_cols].copy()

        if switch_rows:
            df_switch = pd.DataFrame(switch_rows).rename(columns={"Vol switche": "Vol"})

            if "Statut" in df_switch.columns and "Statut" in df_consolide.columns:
                df_switch = df_switch.drop(columns=["Statut"])

            if {"Vol", "Date vol"}.issubset(df_consolide.columns) and {"Vol", "Date vol"}.issubset(df_switch.columns):
                df_consolide = df_consolide.merge(df_switch, on=["Vol", "Date vol"], how="left")
            elif "Vol" in df_consolide.columns and "Vol" in df_switch.columns:
                df_consolide = df_consolide.merge(df_switch, on=["Vol"], how="left")

            st.markdown("**Tableau consolidé des switches (faisable / partiel)**")
            st.caption(
                "**Equipage A2** = équipage de A2 qui prend le vol retardé.  "
                "**Vol suivant** = 1er vol aval de A1 exécuté par A2.  "
                "**Equipage A1 libéré** = à réaffecter sur la suite de A2."
            )
        st.dataframe(
            df_consolide, use_container_width=True, hide_index=True,
            height=min(560, 80 + len(df_consolide) * 38),
        )
    else:
        st.info("Aucun vol ne dépasse le seuil configuré pour la réaffectation.")

    st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION 1 — KPI
# ═════════════════════════════════════════════════════════════
st.markdown("")
section_header("Indicateurs Clés de Performance")

if mode == "manuel" and injected_targets:
    _row_kpi = _filter_by_injected_triplet(df_sched, inj_fid, inj_date, inj_msn)
    if not _row_kpi.empty:
        _msn_kpi  = inj_msn if inj_msn else _row_kpi["aircraft_msn"].iloc[0]
        _fids_kpi = df_sched[df_sched["aircraft_msn"] == _msn_kpi]["flight_id"].tolist()
        df_agg_kpi = df_agg[df_agg["flight_id"].isin(_fids_kpi)]
        _kpi_scope = f"Rotation {_msn_kpi}"
    else:
        df_agg_kpi = df_agg
        _kpi_scope = "Réseau complet"
else:
    df_agg_kpi = df_agg
    _kpi_scope = "Réseau complet"

if df_agg_kpi.empty:
    df_agg_kpi = df_agg

otp_mean  = float(otp_arr.mean())
otp_p5    = float(np.percentile(otp_arr, 5))
prop_mean = float(prop_arr.mean())

if df_agg_kpi.empty:
    worst = pd.Series({"flight_id": "N/A", "mean_arr_delay": 0.0})
    avg_delay = p95_delay = 0.0
else:
    worst     = df_agg_kpi.nlargest(1, "mean_arr_delay").iloc[0]
    avg_delay = float(df_agg_kpi["mean_arr_delay"].mean())
    p95_delay = float(df_agg_kpi["p95_arr_delay"].mean())

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("OTP moyen estimé", f"{otp_mean:.1f} %",
                     f"P5 : {otp_p5:.1f} %"), unsafe_allow_html=True)
k2.markdown(kpi_card("Coefficient de propagation", f"{prop_mean:.2f} ×",
                     "1.0 = aucun effet cascade"), unsafe_allow_html=True)
k3.markdown(kpi_card("Vol le plus impacté", worst["flight_id"],
                     f"Retard moyen : {worst['mean_arr_delay']:.0f} min"), unsafe_allow_html=True)
k4.markdown(kpi_card("Retard moyen global", f"{avg_delay:.1f} min",
                     f"P95 : {p95_delay:.1f} min  |  {_kpi_scope}"), unsafe_allow_html=True)

st.markdown("")
st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION TAMPON VARIABLE — ANALYSE COÛT / BÉNÉFICE
# ═════════════════════════════════════════════════════════════
if slack_enabled and slack_config and df_agg_base is not None:
    merged = df_agg[["flight_id", "mean_arr_delay", "slack_affected"]].merge(
        df_agg_base[["flight_id", "mean_arr_delay"]].rename(
            columns={"mean_arr_delay": "mean_arr_delay_base"}),
        on="flight_id"
    )

    st.caption("**Impact vol par vol** — gain en minutes et en OTP.")
    df_delta = merged.copy()
    df_delta["delta"] = df_delta["mean_arr_delay_base"] - df_delta["mean_arr_delay"]
    df_delta = df_delta.merge(
        df_agg[["flight_id", "origin", "destination", "otp_rate"]], on="flight_id"
    ).merge(
        df_sched[["flight_id", "dep_min"]], on="flight_id"
    ).merge(
        df_agg_base[["flight_id", "otp_rate"]].rename(columns={"otp_rate": "otp_rate_base"}),
        on="flight_id"
    ).sort_values("dep_min")
    df_delta["delta_otp"] = df_delta["otp_rate"] - df_delta["otp_rate_base"]
    df_delta = df_delta.rename(columns={
        "flight_id": "Vol", "origin": "Depart", "destination": "Arrivee",
        "mean_arr_delay_base": "Retard sans tampon (min)",
        "mean_arr_delay": "Retard avec tampon (min)",
        "delta": "Gain (min)", "otp_rate_base": "OTP base (%)",
        "otp_rate": "OTP tampon (%)", "delta_otp": "Delta OTP (pts)",
        "slack_affected": "Tamponne",
    })

    st.dataframe(
        df_delta[["Vol", "Depart", "Arrivee",
                  "Retard sans tampon (min)", "Retard avec tampon (min)", "Gain (min)",
                  "OTP base (%)", "OTP tampon (%)", "Delta OTP (pts)", "Tamponne"]]
        .style
        .map(color_delay, subset=["Retard sans tampon (min)", "Retard avec tampon (min)"])
        .map(color_gain,  subset=["Gain (min)", "Delta OTP (pts)"])
        .format({
            "Retard sans tampon (min)": "{:.1f}", "Retard avec tampon (min)": "{:.1f}",
            "Gain (min)": "{:.1f}", "OTP base (%)": "{:.1f}",
            "OTP tampon (%)": "{:.1f}", "Delta OTP (pts)": "{:+.1f}",
        }),
        use_container_width=True, hide_index=True,
        height=min(420, 80 + len(df_delta) * 38),
    )

    st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION HUB CONGESTION
# ═════════════════════════════════════════════════════════════
if hub_airport:
    section_header_hub(f"Impact de la Congestion Hub — {hub_airport}")

    df_hub_vols   = df_agg[df_agg["hub_affected"] == True]
    df_other_vols = df_agg[df_agg["hub_affected"] == False]

    n_hub       = len(df_hub_vols)
    otp_hub     = float(df_hub_vols["otp_rate"].mean())         if n_hub > 0 else 0.0
    otp_other   = float(df_other_vols["otp_rate"].mean())       if len(df_other_vols) > 0 else 0.0
    delay_hub   = float(df_hub_vols["mean_arr_delay"].mean())   if n_hub > 0 else 0.0
    delay_other = float(df_other_vols["mean_arr_delay"].mean()) if len(df_other_vols) > 0 else 0.0
    p95_hub     = float(df_hub_vols["p95_arr_delay"].mean())    if n_hub > 0 else 0.0
    otp_delta   = otp_hub - otp_other

    hk1, hk2, hk3, hk4 = st.columns(4)
    hk1.markdown(kpi_card_hub(f"Vols depuis {hub_airport}", str(n_hub),
        f"Sur {len(df_agg)} vols"), unsafe_allow_html=True)
    hk2.markdown(kpi_card_hub("OTP hub vs réseau", f"{otp_hub:.1f} % / {otp_other:.1f} %",
        f"Écart : {otp_delta:+.1f} pts"), unsafe_allow_html=True)
    hk3.markdown(kpi_card_hub("Retard moy. hub", f"{delay_hub:.1f} min",
        f"Hors hub : {delay_other:.1f} min"), unsafe_allow_html=True)
    hk4.markdown(kpi_card_hub("Retard P95 hub", f"{p95_hub:.1f} min",
        f"Facteur : {hub_factor:.1f}×"), unsafe_allow_html=True)

    st.markdown("")

    if n_hub > 0:
        dest_cols = st.columns([3, 2])

        with dest_cols[0]:
            st.caption(f"**Retard moyen par destination** depuis **{hub_airport}**")
            df_by_dest = (
                df_hub_vols.groupby("destination")
                .agg(mean_delay=("mean_arr_delay","mean"), p95_delay=("p95_arr_delay","mean"),
                     otp_dest=("otp_rate","mean"), n_vols=("flight_id","count"))
                .reset_index().sort_values("mean_delay", ascending=False)
            )
            def dest_bar_color(d):
                if d <= 10: return "#27ae60"
                if d <= 25: return "#f39c12"
                if d <= 60: return "#e67e22"
                return "#e74c3c"
            fig_dest = go.Figure()
            fig_dest.add_trace(go.Bar(
                x=df_by_dest["destination"], y=df_by_dest["mean_delay"],
                marker_color=[dest_bar_color(d) for d in df_by_dest["mean_delay"]],
                text=df_by_dest["mean_delay"].apply(lambda x: f"{x:.0f} min"),
                textposition="outside",
                customdata=np.column_stack([df_by_dest["otp_dest"].round(1), df_by_dest["n_vols"], df_by_dest["p95_delay"].round(1)]),
                hovertemplate="<b>%{x}</b><br>Retard : <b>%{y:.1f} min</b><br>P95 : %{customdata[2]} min<br>OTP : %{customdata[0]} %<br>Vols : %{customdata[1]}<extra></extra>",
            ))
            fig_dest.update_layout(
                template="plotly_white", height=320,
                xaxis_title=f"Destination depuis {hub_airport}", yaxis_title="Retard moy. (min)",
                margin=dict(l=10,r=10,t=20,b=60), paper_bgcolor="white", plot_bgcolor="#fafafa",
                font=dict(family="Barlow, sans-serif"), showlegend=False,
            )
            fig_dest.update_xaxes(showgrid=False, tickangle=-35)
            st.plotly_chart(fig_dest, use_container_width=True)

        with dest_cols[1]:
            st.caption(f"**OTP** : {hub_airport} vs reste du réseau")
            df_compare = pd.DataFrame({
                "Segment": [f"Depuis {hub_airport}", "Reste du réseau"],
                "OTP (%)": [otp_hub, otp_other],
                "Retard moy (min)": [delay_hub, delay_other],
            })
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=df_compare["Segment"], y=df_compare["OTP (%)"],
                marker_color=["#e67e22","#1a6fa0"],
                text=df_compare["OTP (%)"].apply(lambda x: f"{x:.1f} %"),
                textposition="outside",
            ))
            fig_cmp.update_layout(
                template="plotly_white", height=240, yaxis_title="OTP (%)",
                yaxis_range=[0,115], margin=dict(l=10,r=10,t=10,b=20),
                paper_bgcolor="white", plot_bgcolor="#fafafa",
                font=dict(family="Barlow, sans-serif"), showlegend=False,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            fig_cmp2 = go.Figure()
            fig_cmp2.add_trace(go.Bar(
                x=df_compare["Segment"], y=df_compare["Retard moy (min)"],
                marker_color=["#e67e22","#1a6fa0"],
                text=df_compare["Retard moy (min)"].apply(lambda x: f"{x:.1f} min"),
                textposition="outside",
            ))
            fig_cmp2.update_layout(
                template="plotly_white", height=200, yaxis_title="Retard moy. (min)",
                margin=dict(l=10,r=10,t=10,b=20), paper_bgcolor="white", plot_bgcolor="#fafafa",
                font=dict(family="Barlow, sans-serif"), showlegend=False,
            )
            st.plotly_chart(fig_cmp2, use_container_width=True)

        st.caption(f"**Détail vol par vol** depuis **{hub_airport}**")
        df_hub_detail = (
            df_hub_vols[["flight_id","destination","scheduled_departure","aircraft_msn","aircraft_type",
                         "mean_arr_delay","p95_arr_delay","otp_rate"]]
            .rename(columns={"flight_id":"Vol","destination":"Dest.","scheduled_departure":"Départ",
                             "aircraft_msn":"Avion","aircraft_type":"Type",
                             "mean_arr_delay":"Ret. moy. (min)","p95_arr_delay":"P95 (min)","otp_rate":"OTP (%)"})
            .sort_values("Ret. moy. (min)", ascending=False)
        )
        st.dataframe(
            df_hub_detail.style
                .map(color_delay, subset=["Ret. moy. (min)"])
                .format({"Ret. moy. (min)":"{:.1f}","P95 (min)":"{:.1f}","OTP (%)":"{:.1f}"}),
            use_container_width=True, hide_index=True,
            height=min(400, 80+len(df_hub_detail)*38),
        )
    else:
        st.info(f"Aucun vol au départ de **{hub_airport}** trouvé.")

    st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION PROPAGATION — MODE MANUEL
# ═════════════════════════════════════════════════════════════
# Variable pour stocker df_prop (export Bug 3)
df_prop_export = None

if mode == "manuel" and injected_targets:
    fid_inj = inj_fid
    min_inj = inj_minutes

    fids_label = " + ".join(sorted(inj_fids_set)) if len(inj_fids_set) > 1 else fid_inj
    section_header(f"Propagation des Retards Injectés — {fids_label}")

    row_inj = _filter_by_injected_triplet(df_sched, fid_inj, inj_date, inj_msn)
    if not row_inj.empty:
        msn_inj = inj_msn if inj_msn else row_inj["aircraft_msn"].iloc[0]
        rot_sort_cols = [c for c in ["flight_date", "dep_min", "aircraft_msn", "flight_id"]
                         if c in df_sched.columns] or ["dep_min"]
        df_rotation = df_sched[df_sched["aircraft_msn"] == msn_inj].sort_values(rot_sort_cols).reset_index(drop=True)
        fids_rotation = df_rotation["flight_id"].tolist()
        sim_result = all_results[0]

        if "flight_date" not in df_rotation.columns:
            df_rotation["flight_date"] = ""
        df_rotation["_date_key"] = _date_series_ui(df_rotation["flight_date"]).dt.strftime("%Y-%m-%d")

        inj_pos_map = {}
        for tgt in injected_targets:
            tgt_fid  = str(tgt.get("flight_id",    "")).strip()
            tgt_msn  = str(tgt.get("aircraft_msn", "")).strip()
            tgt_date = _norm_date_key_ui(str(tgt.get("flight_date", "")).strip())
            _tm = (df_rotation["flight_id"].astype(str).str.strip() == tgt_fid)
            if tgt_msn:
                _tm = _tm & (df_rotation["aircraft_msn"].astype(str).str.strip() == tgt_msn)
            if tgt_date:
                _tm = _tm & (df_rotation["_date_key"] == tgt_date)
            _pl = df_rotation.index[_tm].tolist()
            if _pl:
                inj_pos_map[tgt_fid] = int(_pl[0])

        earliest_inj_pos = min(inj_pos_map.values()) if inj_pos_map else -1
        inj_dep_min = (float(df_rotation.iloc[earliest_inj_pos]["dep_min"])
                       if earliest_inj_pos >= 0 else None)

        rows_prop = []
        for i, row_r in df_rotation.iterrows():
            fid = row_r["flight_id"]
            sim_row = sim_result[sim_result["flight_id"] == fid].copy()

            if "aircraft_msn" in sim_row.columns:
                msn_filter = sim_row[sim_row["aircraft_msn"].astype(str).str.strip() == str(row_r.get("aircraft_msn", "")).strip()]
                if not msn_filter.empty:
                    sim_row = msn_filter

            if "flight_date" in sim_result.columns and not sim_row.empty:
                try:
                    sim_dates = pd.to_datetime(sim_row["flight_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
                    ref_date = pd.to_datetime(str(row_r.get("flight_date", "")), errors="coerce")
                    ref_str = ref_date.strftime("%Y-%m-%d") if pd.notna(ref_date) else ""
                    if ref_str:
                        by_date = sim_row[sim_dates == ref_str]
                        if not by_date.empty:
                            sim_row = by_date
                except Exception:
                    pass

            if sim_row.empty:
                sim_row = pd.Series({
                    "dep_actual": float(row_r["dep_min"]),
                    "arr_actual": float(row_r.get("arr_min", row_r["dep_min"])),
                    "dep_delay": 0.0, "arr_delay": 0.0,
                    "initial_delay": 0.0, "slack_applied": 0.0,
                    "turnaround_effectif": 0.0, "turnaround_actual": 0.0,
                })
            else:
                sim_row = sim_row.iloc[0]

            ac_reg     = row_r.get("ac_registration", row_r.get("AC_REGISTRATION", row_r.get("aircraft_msn", "")))
            dep_prevu  = row_r["dep_min"]
            arr_prevue = row_r["arr_min"]
            dep_reel   = float(sim_row["dep_actual"])
            arr_reelle = float(sim_row["arr_actual"])
            dep_delay  = float(sim_row["dep_delay"])
            arr_delay  = float(sim_row["arr_delay"])
            inj_delay  = float(sim_row.get("initial_delay", 0.0))
            prop_recv  = max(0.0, dep_delay - inj_delay)
            slack_val  = float(sim_row.get("slack_applied", 0.0))

            is_before_injection = (inj_dep_min is not None and float(row_r["dep_min"]) < inj_dep_min)

            if is_before_injection:
                dep_reel   = float(dep_prevu)
                arr_reelle = float(row_r.get("arr_min", dep_prevu))
                dep_delay  = arr_delay = inj_delay = prop_recv = 0.0
                role = "Avant injection"
            elif fid in inj_fids_set:
                role = "Injecte"
            elif dep_delay > 0:
                role = "Propage"
            elif earliest_inj_pos == -1:
                role = "A l heure"
            else:
                # ═══════════════════════════════════════════════════════
                # BUG 2 CORRIGÉ — Distinction "Absorbe" vs "Protege"
                #
                # AVANT : tous les vols avec dep_delay==0 après l'injection
                #         recevaient le rôle "Absorbe", même ceux non atteints
                #         par la propagation (ex: AT1411, AT270 dans le test).
                #         → Label trompeur : "Absorbe" suggère que le vol
                #           a reçu du retard et l'a absorbé via le tampon,
                #           alors qu'il n'a simplement rien reçu.
                #
                # APRÈS : si slack_applied > 0, le tampon a été utilisé
                #         → "Absorbe" (le tampon suffix sera ajouté ci-dessous).
                #         Sinon, la propagation s'est arrêtée avant ce vol
                #         → "Protege" (chaîne stoppée en amont).
                # ═══════════════════════════════════════════════════════
                if slack_val > 0:
                    role = "Absorbe"   # tampon suffix ajouté ci-dessous
                else:
                    role = "Protege"   # chaîne arrêtée avant ce vol

            if (not is_before_injection) and hub_airport and (row_r["origin"] in hub_airport if isinstance(hub_airport, list) else row_r["origin"] == hub_airport) and dep_delay > 0:
                role += f" hub x{hub_factor:.1f}"
            if (not is_before_injection) and slack_config and slack_val > 0:
                role += f" tampon+{int(slack_val)}m"

            rows_prop.append({
                "Seq": i + 1, "Vol": fid, "Immatriculation": ac_reg,
                "Date vol": str(row_r.get("flight_date", "")),
                "Route": f"{row_r['origin']} -> {row_r['destination']}",
                "Dep. prevu": mhm(dep_prevu), "Depart reel": mhm(dep_reel),
                "Retard injecte (min)": round(inj_delay, 1),
                "Retard propage recu (min)": round(prop_recv, 1),
                "Ret. depart (min)": round(dep_delay, 1),
                "Arr. prevue": mhm(arr_prevue), "Arr. reelle": mhm(arr_reelle),
                "Ret. arrivee (min)": round(arr_delay, 1),
                "Role": role,
                "_sort_date": _date_scalar_ui(row_r.get("flight_date", "")),
                "_sort_dep": float(dep_prevu),
            })

        df_prop = pd.DataFrame(rows_prop)
        if not df_prop.empty:
            df_prop = df_prop.sort_values(
                ["Immatriculation", "_sort_date", "_sort_dep", "Seq"], kind="mergesort"
            ).drop(columns=["_sort_date", "_sort_dep"])

        # Stocker pour l'export (Bug 3)
        df_prop_export = df_prop.copy() if not df_prop.empty else None

        colors_bar = []
        for _, r in df_prop.iterrows():
            if "Injecte"  in r["Role"]: colors_bar.append("#C8102E")
            elif "Propage" in r["Role"]: colors_bar.append("#e67e22")
            elif "Absorbe" in r["Role"]: colors_bar.append("#0e7490")
            elif "Protege" in r["Role"]: colors_bar.append("#27ae60")
            else:                        colors_bar.append("#b0b8c1")

        fig_prop = go.Figure()
        fig_prop.add_trace(go.Bar(
            x=df_prop["Vol"], y=df_prop["Ret. arrivee (min)"],
            marker_color=colors_bar,
            text=df_prop["Ret. arrivee (min)"].apply(lambda x: f"{x:.0f} min"),
            textposition="outside",
            customdata=df_prop[["Route","Dep. prevu","Depart reel","Role"]].values,
            hovertemplate="<b>%{x}</b>  %{customdata[0]}<br>Prévu : %{customdata[1]}<br>Réel : %{customdata[2]}<br>Rôle : %{customdata[3]}<br>Retard : <b>%{y:.0f} min</b><extra></extra>",
        ))
        fig_prop.update_layout(
            template="plotly_white", height=300,
            xaxis_title="Vols de la rotation", yaxis_title="Retard arrivée (min)",
            margin=dict(l=10,r=10,t=20,b=40), paper_bgcolor="white", plot_bgcolor="#fafafa",
            font=dict(family="Barlow, sans-serif"), showlegend=False,
        )

        hub_note   = f" + congestion {hub_airport} (x{hub_factor:.1f})" if hub_airport else ""
        slack_note = f" + tampon +{slack_config['minutes']} min ({scope_label})" if slack_config else ""
        st.caption(
            f"Rotation {msn_inj} — rouge = injecté, orange = propagé, "
            f"bleu = absorbé (tampon actif), vert = protégé (chaîne stoppée).{hub_note}{slack_note}"
        )
        st.plotly_chart(fig_prop, use_container_width=True)

        st.dataframe(
            df_prop.style
                .map(color_delay, subset=["Retard injecte (min)", "Retard propage recu (min)",
                                          "Ret. depart (min)", "Ret. arrivee (min)"])
                .map(color_role, subset=["Role"])
                .format({
                    "Retard injecte (min)": "{:.1f}", "Retard propage recu (min)": "{:.1f}",
                    "Ret. depart (min)": "{:.1f}", "Ret. arrivee (min)": "{:.1f}",
                }),
            use_container_width=True, hide_index=True, height=260,
        )

    st.divider()



# ═════════════════════════════════════════════════════════════
# TABLEAU — RETARDS ET RECOMMANDATIONS
# ═════════════════════════════════════════════════════════════
st.markdown("")
section_header("Vols — Retards et Recommandations")

if mode == "manuel" and injected_targets:
    _row_inj_tbl = _filter_by_injected_triplet(df_sched, inj_fid, inj_date, inj_msn)
    _msn_tbl = inj_msn if inj_msn else (_row_inj_tbl["aircraft_msn"].iloc[0] if not _row_inj_tbl.empty else "")
    df_g = df_sched[df_sched["aircraft_msn"] == _msn_tbl].copy() if _msn_tbl else df_sched.copy()
else:
    df_g = df_sched.copy()

slack_flight_ids = set()
if slack_config and all_results:
    sim0_tbl = all_results[0]
    if "slack_applied" in sim0_tbl.columns:
        slack_flight_ids = set(sim0_tbl[sim0_tbl["slack_applied"] > 0]["flight_id"].tolist())

G_imp = nx.DiGraph()
for _, r in df_sched.iterrows():
    G_imp.add_node(r["flight_id"], delay=float(delay_map.get(r["flight_id"],0)))
for msn_x, grp in df_sched.sort_values("dep_min").groupby("aircraft_msn"):
    fids_x = grp["flight_id"].tolist()
    for i in range(len(fids_x)-1):
        G_imp.add_edge(fids_x[i], fids_x[i+1])
impacts_all = {}
for node in G_imp.nodes():
    succ = list(nx.descendants(G_imp, node))
    casc = float(np.mean([G_imp.nodes[s]["delay"] for s in succ])) if succ else 0.0
    impacts_all[node] = G_imp.nodes[node]["delay"] + 0.4*casc
p90_all = float(np.percentile(list(impacts_all.values()), 90))

p80_map = df_agg.set_index("flight_id")["p80_dep_delay"].to_dict()
p90_map = df_agg.set_index("flight_id")["p90_dep_delay"].to_dict()

prop_delay_map = {}
if mode == "manuel" and all_results:
    sim0 = all_results[0]
    for _, row_s in sim0.iterrows():
        prop_delay_map[row_s["flight_id"]] = max(0.0, float(row_s["dep_delay"]) - float(row_s["initial_delay"]))

table_rows = []
for _, r in df_g.sort_values("dep_min").iterrows():
    fid    = r["flight_id"]
    delay  = float(delay_map.get(fid, 0))
    otp_v  = float(otp_map.get(fid, 0))
    impact = impacts_all.get(fid, delay)
    ac_reg = r.get("ac_registration", r.get("AC_REGISTRATION", r.get("aircraft_msn", "")))
    is_hub = bool(hub_airport and (r["origin"] in hub_airport if isinstance(hub_airport, list) else r["origin"] == hub_airport))
    is_slack = fid in slack_flight_ids

    if mode == "manuel":
        prop_recv = prop_delay_map.get(fid, 0.0)
        dep_delay_vol = float(delay_map.get(fid, 0))

        if prop_recv <= 0:
            reco, reco_min, reco_abs = "Aucun retard propagé reçu", 0, 0
        else:
            tampon_otp = max(0, int(dep_delay_vol - otp_thresh) + 1)
            tampon_abs = int(np.ceil(prop_recv))
            reco_min, reco_abs = tampon_otp, tampon_abs
            reco = "Déjà sous seuil OTP" if tampon_otp <= 0 else f"OTP : +{tampon_otp} min  |  Absorb. : +{tampon_abs} min"

        table_rows.append({
            "Vol": fid, "ac_registration": ac_reg,
            "Route": f"{r['origin']} -> {r['destination']}",
            "Avion": r["aircraft_msn"],
            "Depart": r["scheduled_departure"], "Arrivee": r["scheduled_arrival"],
            "Ret. propagé (min)": round(prop_recv, 1),
            "Ret. total (min)": round(dep_delay_vol, 1),
            "OTP (%)": round(otp_v, 1),
            "Tampon OTP (min)": reco_min, "Tampon absorb. (min)": reco_abs,
            "Recommandation": reco,
            "Priorite": "Critique" if impact>=p90_all else ("Haute" if delay>=45 else "Normale"),
            "Hub": "Hub" if is_hub else "", "Tampon": "T" if is_slack else "",
        })
    else:
        p80 = max(5, int(p80_map.get(fid, 0)))
        p90 = max(5, int(p90_map.get(fid, 0)))
        table_rows.append({
            "Vol": fid, "ac_registration": ac_reg,
            "Route": f"{r['origin']} -> {r['destination']}",
            "Avion": r["aircraft_msn"],
            "Depart": r["scheduled_departure"], "Arrivee": r["scheduled_arrival"],
            "Retard moy. (min)": round(delay, 1), "Impact cascade": round(impact, 1),
            "OTP (%)": round(otp_v, 1), "P80 (min)": p80, "P90 (min)": p90,
            "Recommandation": f"Courant : +{p80} min  |  Sécurisé : +{p90} min",
            "Priorite": "Critique" if impact>=p90_all else ("Haute" if delay>=45 else "Normale"),
            "Hub": "Hub" if is_hub else "", "Tampon": "T" if is_slack else "",
        })

df_table = pd.DataFrame(table_rows)

diag_rows_for_merge = []
for _, row in sim_example.sort_values(["aircraft_msn", "dep_min"]).iterrows():
    prop_sim = float(max(0, row["dep_delay"] - row["initial_delay"]))
    diag_row = {
        "Vol": row["flight_id"],
        "Dep. reel sim": mhm(row["dep_actual"]),
        "Arr. reelle sim": mhm(row["arr_actual"]),
        "Ret. propage sim (min)": round(prop_sim, 1),
        "Ret. depart sim (min)": round(float(row["dep_delay"]), 1),
        "Ret. arrivee sim (min)": round(float(row["arr_delay"]), 1),
        "Turnaround sim (min)": round(float(row.get("turnaround_effectif", row["turnaround_actual"])), 1),
        "Etat Markov": {0: "Normal", 1: "Alerte", 2: "Bloque"}.get(int(row["markov_state"]), "?"),
        "OTP sim": "A l heure" if row["on_time"] else "En retard",
        "Tampon applique": (f"+{int(row['slack_applied'])} min" if float(row.get("slack_applied", 0)) > 0 else "Non"),
        "Hub sim": ("Oui" if hub_airport and (row["origin"] in hub_airport if isinstance(hub_airport, list)
                    else row["origin"] == hub_airport) else "Non"),
        "Avion diag": row["aircraft_msn"],
    }
    if mode == "auto":
        diag_row["Ret. Gamma sim (min)"] = round(float(row["initial_delay"]), 1)
    diag_rows_for_merge.append(diag_row)

df_diag_merge = pd.DataFrame(diag_rows_for_merge)
if not df_diag_merge.empty:
    df_diag_merge = df_diag_merge.drop_duplicates(subset=["Vol"], keep="first")

    if mode == "auto":
        cd1, cd2 = st.columns([2, 1])
        with cd1:
            avion_f = st.selectbox(
                "Filtrer par avion :",
                ["Tous"] + sorted(df_diag_merge["Avion diag"].dropna().astype(str).unique().tolist()),
                key="diag_filter",
            )
        with cd2:
            st.caption("**Ret. total sim = Gamma + Propagé** | Vert ≤ 15 min | Rouge > 15 min")
        if avion_f != "Tous":
            vols_keep = set(df_diag_merge.loc[df_diag_merge["Avion diag"] == avion_f, "Vol"].astype(str).tolist())
            df_table = df_table[df_table["Vol"].astype(str).isin(vols_keep)].copy()
            df_diag_merge = df_diag_merge[df_diag_merge["Vol"].astype(str).isin(vols_keep)].copy()
    else:
        row_diag = _filter_by_injected_triplet(df_sched, inj_fid, inj_date, inj_msn)
        if not row_diag.empty:
            msn_diag = inj_msn if inj_msn else row_diag["aircraft_msn"].iloc[0]
            st.caption(f"Rotation {msn_diag} | Vert ≤ 15 min | Rouge > 15 min")

    df_diag_merge = df_diag_merge.drop(columns=["Avion diag"], errors="ignore")
    df_table = df_table.merge(df_diag_merge, on="Vol", how="left")

if mode == "manuel":
    st.caption(
        f"**Tampon OTP** : slack pour passer sous {otp_thresh} min.  "
        "**Tampon absorb.** : slack pour absorber tout le retard propagé."
    )
    cols_table = ["Vol","ac_registration","Route","Avion","Depart","Arrivee",
                  "Ret. propagé (min)","Ret. total (min)","OTP (%)",
                  "Dep. reel sim","Arr. reelle sim",
                  "Ret. propage sim (min)","Ret. depart sim (min)","Ret. arrivee sim (min)",
                  "Turnaround sim (min)","Etat Markov","OTP sim",
                  "Tampon OTP (min)","Tampon absorb. (min)",
                  "Recommandation","Priorite","Hub","Hub sim","Tampon applique","Tampon"]
    fmt_table  = {
        "Ret. propagé (min)": "{:.1f}", "Ret. total (min)": "{:.1f}",
        "OTP (%)": "{:.1f}", "Ret. propage sim (min)": "{:.1f}",
        "Ret. depart sim (min)": "{:.1f}", "Ret. arrivee sim (min)": "{:.1f}",
        "Turnaround sim (min)": "{:.1f}",
        "Tampon OTP (min)": "{:.0f}", "Tampon absorb. (min)": "{:.0f}",
    }
    delay_subset = ["Ret. propagé (min)", "Ret. total (min)",
                    "Ret. propage sim (min)", "Ret. depart sim (min)", "Ret. arrivee sim (min)"]
else:
    st.caption(
        "**P80** : tampon absorbant 8 retards sur 10.  "
        "**P90** : tampon absorbant 9 retards sur 10."
    )
    cols_table = ["Vol","ac_registration","Route","Avion","Depart","Arrivee",
                  "Retard moy. (min)","Impact cascade","OTP (%)","P80 (min)","P90 (min)",
                  "Dep. reel sim","Arr. reelle sim",
                  "Ret. Gamma sim (min)","Ret. propage sim (min)","Ret. depart sim (min)","Ret. arrivee sim (min)",
                  "Turnaround sim (min)","Etat Markov","OTP sim",
                  "Recommandation","Priorite","Hub","Hub sim","Tampon applique","Tampon"]
    fmt_table  = {
        "Retard moy. (min)": "{:.1f}", "Impact cascade": "{:.1f}",
        "OTP (%)": "{:.1f}", "P80 (min)": "{:.0f}", "P90 (min)": "{:.0f}",
        "Ret. Gamma sim (min)": "{:.1f}", "Ret. propage sim (min)": "{:.1f}",
        "Ret. depart sim (min)": "{:.1f}", "Ret. arrivee sim (min)": "{:.1f}",
        "Turnaround sim (min)": "{:.1f}",
    }
    delay_subset = ["Retard moy. (min)", "P80 (min)", "P90 (min)",
                    "Ret. Gamma sim (min)", "Ret. propage sim (min)", "Ret. depart sim (min)", "Ret. arrivee sim (min)"]

cols_table   = [c for c in cols_table   if c in df_table.columns]
delay_subset = [c for c in delay_subset if c in cols_table]

style_table = df_table[cols_table].style.map(color_delay, subset=delay_subset).format(fmt_table)
if mode == "manuel":
    tampon_cols = [c for c in ["Tampon OTP (min)", "Tampon absorb. (min)"] if c in cols_table]
    if tampon_cols:
        style_table = style_table.map(color_tampon_reco, subset=tampon_cols)

if "Etat Markov" in cols_table:
    style_table = style_table.map(color_markov, subset=["Etat Markov"])
if "OTP sim" in cols_table:
    style_table = style_table.map(color_otp, subset=["OTP sim"])
if "Tampon applique" in cols_table:
    style_table = style_table.map(color_tampon, subset=["Tampon applique"])

st.dataframe(style_table, use_container_width=True, hide_index=True, height=320)
st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION 5 — RETARDS PAR AVION (mode auto)
# ═════════════════════════════════════════════════════════════
if mode == "auto":
    section_header("Analyse des Retards par Avion")

    _g = df_agg.groupby("aircraft_msn")
    df_by_ac = pd.DataFrame({
        "aircraft_msn": list(_g.groups.keys()),
        "mean_delay":   [float(_g.get_group(k)["mean_arr_delay"].mean()) for k in _g.groups],
        "n_flights":    [int(len(_g.get_group(k)))                       for k in _g.groups],
        "otp_moy":      [float(_g.get_group(k)["otp_rate"].mean())       for k in _g.groups],
    }).sort_values("mean_delay", ascending=False).reset_index(drop=True)

    fig_ac = go.Figure()
    fig_ac.add_trace(go.Bar(
        x=df_by_ac["aircraft_msn"], y=df_by_ac["mean_delay"],
        text=(df_by_ac["mean_delay"].round(0).astype(int).astype(str)+" min"),
        textposition="outside",
        marker=dict(color=df_by_ac["mean_delay"],
                    colorscale=[[0,"#27ae60"],[0.4,"#f1c40f"],[0.7,"#e67e22"],[1,"#e74c3c"]],
                    showscale=True, colorbar=dict(title="Retard<br>(min)", thickness=12)),
        customdata=np.column_stack([df_by_ac["n_flights"], df_by_ac["otp_moy"]]),
        hovertemplate="<b>%{x}</b><br>Retard : %{y:.1f} min<br>Vols : %{customdata[0]}<br>OTP : %{customdata[1]:.1f} %<extra></extra>",
    ))
    fig_ac.update_layout(
        template="plotly_white", height=380,
        xaxis_title="Avion", yaxis_title="Retard moy. arrivée (min)",
        margin=dict(l=10, r=60, t=20, b=80), xaxis_tickangle=-45,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Barlow, sans-serif"),
    )
    st.plotly_chart(fig_ac, use_container_width=True)
    st.divider()


# ═════════════════════════════════════════════════════════════
# SECTION 6 — RÉCAPITULATIF
# ═════════════════════════════════════════════════════════════
if mode == "auto":
    section_header("Récapitulatif Global — Tous les Vols")
    df_recap_source = df_agg.copy()
else:
    row_recap = _filter_by_injected_triplet(df_sched, inj_fid, inj_date, inj_msn)
    if not row_recap.empty:
        msn_recap = inj_msn if inj_msn else row_recap["aircraft_msn"].iloc[0]
        fids_recap = df_sched[df_sched["aircraft_msn"] == msn_recap]["flight_id"].tolist()
        df_recap_source = df_agg[df_agg["flight_id"].isin(fids_recap)].copy()
        section_header(f"Récapitulatif — Rotation {msn_recap}")
    else:
        df_recap_source = df_agg.copy()
        section_header("Récapitulatif Global — Tous les Vols")

if mode == "auto":
    cols_recap   = ["flight_id","origin","destination","scheduled_departure",
                    "aircraft_type","mean_dep_delay","mean_arr_delay","p95_arr_delay","otp_rate"]
    rename_recap = {
        "flight_id":"Vol","origin":"Départ","destination":"Arrivée",
        "scheduled_departure":"Heure prévue","aircraft_type":"Type",
        "mean_dep_delay":"Ret. dép. (min)","mean_arr_delay":"Ret. arr. (min)",
        "p95_arr_delay":"P95 (min)","otp_rate":"OTP (%)",
    }
    fmt_recap    = {"Ret. dép. (min)":"{:.1f}","Ret. arr. (min)":"{:.1f}",
                    "P95 (min)":"{:.1f}","OTP (%)":"{:.1f}"}
    col_gradient = "Ret. arr. (min)"
else:
    cols_recap   = ["flight_id","origin","destination","scheduled_departure",
                    "aircraft_type","mean_arr_delay","otp_rate"]
    rename_recap = {
        "flight_id":"Vol","origin":"Départ","destination":"Arrivée",
        "scheduled_departure":"Heure prévue","aircraft_type":"Type",
        "mean_arr_delay":"Ret. arrivée (min)","otp_rate":"OTP (%)",
    }
    fmt_recap    = {"Ret. arrivée (min)":"{:.1f}","OTP (%)":"{:.1f}"}
    col_gradient = "Ret. arrivée (min)"

df_display = df_recap_source.sort_values("mean_arr_delay", ascending=False)[
    [c for c in cols_recap if c in df_recap_source.columns]
].rename(columns=rename_recap)
max_ret = float(df_display[col_gradient].max())

st.dataframe(
    df_display.style
        .format(fmt_recap)
        .background_gradient(subset=[col_gradient], cmap="RdYlGn_r",
                             vmin=0, vmax=max_ret if max_ret > 0 else 1),
    use_container_width=True,
    height=min(460, 80 + len(df_display) * 38),
    hide_index=True,
)
st.divider()


# ═════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# BUG 3 CORRIGÉ — Export du détail de propagation (df_prop)
#
# AVANT : le bouton exportait uniquement df_agg (résumé OTP par vol),
#         qui ne contient pas les colonnes de propagation (Rôle,
#         Retard injecté, Retard propagé reçu…) issues du mode manuel.
#         Les deux CSV que tu avais exportés correspondaient à deux
#         appels différents à st.download_button sur df_agg et df_delta.
#
# APRÈS : en mode manuel, deux boutons d'export sont proposés :
#   1. Résumé OTP (df_agg) — comme avant
#   2. Détail propagation rotation (df_prop) — nouveau
# ═══════════════════════════════════════════════════════════
dcol1, dcol2, _ = st.columns([1, 1, 2])

with dcol1:
    export_df = df_agg.copy()
    if hub_airport:
        export_df["hub_congestion_airport"] = hub_airport
        export_df["hub_congestion_factor"]  = hub_factor
    if slack_config:
        export_df["slack_minutes"] = slack_config.get("minutes", 0)
        export_df["slack_scope"]   = slack_config.get("scope", "")
    st.download_button(
        "Exporter résumé OTP (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="AIR_ROBUST_RAM_resultats.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dcol2:
    if df_prop_export is not None and not df_prop_export.empty:
        st.download_button(
            "Exporter propagation rotation (CSV)",
            data=df_prop_export.to_csv(index=False).encode("utf-8"),
            file_name="AIR_ROBUST_RAM_propagation.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Export propagation disponible en mode Manuel.")