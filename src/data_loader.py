"""
src/data_loader.py — Chargement et validation des données
"""
import pandas as pd
import numpy as np
import streamlit as st
import re

# ─── Colonnes obligatoires (format interne simulateur) ───────
REQUIRED_SCHEDULE = {
    "flight_id", "origin", "destination",
    "scheduled_departure", "scheduled_arrival",
    "aircraft_msn", "aircraft_type",
}
REQUIRED_CREW = {
    "crew_id", "flight_sequence",
}
REQUIRED_REF = {
    "param_type", "value",
}

# ─── Colonnes clés pour détecter le format opérationnel ──────
OPS_LEG_COLS = {"FN_CARRIER", "FN_NUMBER", "DEP_AP_SCHED", "ARR_AP_SCHED"}
OPS_DOA_COLS = {"CREW_ID", "ACTIVITY", "ACTIVITY_GROUP", "ORIGINE"}

# ─── Référentiel route domestique ────────────────────────────
DOM = {
    "CMN", "RAK", "AGA", "TNG", "FEZ", "RBA", "OUD", "NDR",
    "EUN", "VIL", "TTU", "OZZ", "ERH", "ESU", "AHU", "GLN",
    "MEK", "TTA", "OZG",
}

# ─── Utilitaires horaires ────────────────────────────────────

def hhmm_to_minutes(t: str) -> int:
    try:
        h, m = map(int, str(t).strip().split(":"))
        return h * 60 + m
    except Exception:
        return 0


def minutes_to_hhmm(minutes: int) -> str:
    h = (int(minutes) // 60) % 24
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """Normalise en YYYY-MM-DD sans inverser les formats ISO (YYYY-MM-DD)."""
    s = series.astype(str).str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)

    out = pd.Series(index=series.index, dtype="object")
    out.loc[iso_mask] = pd.to_datetime(
        s.loc[iso_mask], errors="coerce", dayfirst=False
    ).dt.strftime("%Y-%m-%d")
    out.loc[~iso_mask] = pd.to_datetime(
        s.loc[~iso_mask], errors="coerce", dayfirst=True
    ).dt.strftime("%Y-%m-%d")
    return out


def _to_date_str(series: pd.Series) -> pd.Series:
    """
    Convertit n'importe quelle série (Timestamp, str, mixed) en YYYY-MM-DD.
    Méthode unique et robuste utilisée par _recompute_crew_constraints.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.strftime("%Y-%m-%d")
    if isinstance(dt, pd.DatetimeIndex):
        idx = series.index if isinstance(series, pd.Series) else None
        return pd.Series(dt.strftime("%Y-%m-%d"), index=idx)
    return pd.Series(pd.to_datetime([series], errors="coerce")).dt.strftime("%Y-%m-%d")


def _normalize_flight_token(value) -> str:
    """Normalise un identifiant vol unitaire ; retourne "" si invalide."""
    if pd.isna(value):
        return ""

    raw = str(value).strip().upper()
    if not raw:
        return ""
    if raw in {"NAN", "NONE", "NULL", "<NA>", "NAT"}:
        return ""
    try:
        f = float(raw)
        if np.isfinite(f) and f.is_integer():
            raw = str(int(f))
    except (ValueError, TypeError):
        pass

    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    if not raw or raw in {"NAN", "NONE", "NULL", "NA"}:
        return ""
    return raw


def _normalize_flight_sequence(value) -> str:
    """Nettoie une séquence 'A;B;C' et supprime les tokens invalides."""
    tokens = []
    for tok in str(value).split(";"):
        t = _normalize_flight_token(tok)
        if t:
            tokens.append(t)
    return ";".join(tokens)


# ─── Lecture brute d'un fichier uploadé ──────────────────────

def _read_file(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        uploaded_file.seek(0)
        if name.endswith((".xlsx", ".xls")):
            def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
                if df is None:
                    return df
                out = df.copy()
                cols = []
                for c in out.columns:
                    s = "" if pd.isna(c) else str(c).strip()
                    cols.append(s)
                out.columns = cols
                drop_cols = [c for c in out.columns if not c or c.lower().startswith("unnamed")]
                if drop_cols:
                    out = out.drop(columns=drop_cols, errors="ignore")
                return out

            def _sheet_score(df: pd.DataFrame) -> tuple[int, int, int, int]:
                if df is None:
                    return (-1, -1, -1, -1)
                cols = [str(c).strip() for c in df.columns if str(c).strip()]
                colset = set(cols)

                known_hits = len(colset & (
                    REQUIRED_SCHEDULE
                    | REQUIRED_CREW
                    | REQUIRED_REF
                    | OPS_LEG_COLS
                    | OPS_DOA_COLS
                ))
                non_empty_cols = len(cols)
                n_rows = int(len(df))
                data_size = n_rows * max(non_empty_cols, 1)
                return (known_hits, non_empty_cols, n_rows, data_size)

            best_df = None
            best_score = (-1, -1, -1, -1)
            uploaded_file.seek(0)
            xls = pd.ExcelFile(uploaded_file)

            for sh in xls.sheet_names:
                for header_row in range(0, 8):
                    try:
                        sdf = pd.read_excel(xls, sheet_name=sh, header=header_row)
                    except Exception:
                        continue
                    sdf = _sanitize_columns(sdf)
                    score = _sheet_score(sdf)
                    if score > best_score:
                        best_df = sdf
                        best_score = score

            if best_df is not None:
                return best_df

            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";")
    except Exception as e:
        st.error(f"Impossible de lire **{uploaded_file.name}** : {e}")
        return None


# ─── Validation des colonnes ─────────────────────────────────

def _check_columns(df: pd.DataFrame, required: set, label: str) -> bool:
    missing = required - set(df.columns)
    if missing:
        st.error(
            f"Le fichier **{label}** ne contient pas les colonnes requises :\n\n"
            f"`{', '.join(sorted(missing))}`\n\n"
            f"Colonnes trouvées : `{', '.join(df.columns.tolist())}`"
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════
# Détection et conversion du format opérationnel
# ══════════════════════════════════════════════════════════════

def _is_ops_leg(df: pd.DataFrame) -> bool:
    return OPS_LEG_COLS.issubset(set(df.columns))


def _is_ops_doa(df: pd.DataFrame) -> bool:
    return OPS_DOA_COLS.issubset(set(df.columns))


def _convert_leg_futur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit LEG_FUTUR -> format schedule interne.
    Toutes les colonnes originales sont conservees dans le DataFrame.
    """
    if "LEG_STATE" in df.columns:
        df = df[df["LEG_STATE"] == "NEW"].copy()
    else:
        df = df.copy()

    if "DAY_OF_ORIGIN" in df.columns:
        df["DAY_OF_ORIGIN"] = (
            df["DAY_OF_ORIGIN"]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(r"\.0+$", "", regex=True)
            .str.extract(r"(\d{8})", expand=False)
            .fillna("")
        )

    carriers = df["FN_CARRIER"].apply(_normalize_flight_token)
    numbers  = df["FN_NUMBER"].apply(_normalize_flight_token)
    df["flight_id"] = [
        (f"{c}{n}" if c and n and any(ch.isdigit() for ch in f"{c}{n}")
         else (n if (not c and n and any(ch.isdigit() for ch in n)) else ""))
        for c, n in zip(carriers, numbers)
    ]
    df["origin"]        = df["DEP_AP_SCHED"]
    df["destination"]   = df["ARR_AP_SCHED"]
    df["aircraft_msn"]  = df["AC_REGISTRATION"]
    df["aircraft_type"] = df["AC_SUBTYPE"].astype(str)

    def _parse_hhmm(value) -> str:
        if pd.isna(value):
            return "0000"
        raw = str(value).strip()
        if not raw:
            return "0000"
        try:
            hhmm_int = int(float(raw))
            hhmm = f"{hhmm_int:04d}"[-4:]
        except (ValueError, TypeError):
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                return "0000"
            hhmm = digits[-4:].zfill(4)
        hh = int(hhmm[:2])
        mm = int(hhmm[2:4])
        if hh > 23 or mm > 59:
            return "0000"
        return hhmm

    def _parse_day(value) -> int:
        if pd.isna(value):
            return 0
        raw = str(value).strip()
        if not raw:
            return 0
        try:
            day_val = int(float(raw))
            return day_val % 100
        except (ValueError, TypeError):
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                return 0
            return int(digits[-2:])

    dep_t = df["DEP_TIME_SCHED"].apply(_parse_hhmm)
    arr_t = df["ARR_TIME_SCHED"].apply(_parse_hhmm)
    df["scheduled_departure"] = dep_t.str[:2] + ":" + dep_t.str[2:4]
    df["scheduled_arrival"]   = arr_t.str[:2] + arr_t.str[2:4]

    if "DEP_DAY_SCHED" in df.columns:
        dep_day = df["DEP_DAY_SCHED"].apply(_parse_day)
    elif "DAY_OF_ORIGIN" in df.columns:
        dep_day = df["DAY_OF_ORIGIN"].apply(_parse_day)
    else:
        dep_day = pd.Series([0] * len(df), index=df.index)

    if "ARR_DAY_SCHED" in df.columns:
        arr_day = df["ARR_DAY_SCHED"].apply(_parse_day)
    elif "DAY_OF_ORIGIN" in df.columns:
        arr_day = df["DAY_OF_ORIGIN"].apply(_parse_day)
    else:
        arr_day = pd.Series([0] * len(df), index=df.index)

    dep_min = dep_t.apply(lambda x: int(x[:2]) * 60 + int(x[2:4]))
    arr_min = arr_t.apply(lambda x: int(x[:2]) * 60 + int(x[2:4]))
    dur = (arr_day - dep_day) * 1440 + (arr_min - dep_min)
    dur.loc[dur < 0] += 30 * 1440
    df["flight_duration_min"] = dur

    if "DAY_OF_ORIGIN" in df.columns:
        d = df["DAY_OF_ORIGIN"].astype(str)
        df["flight_date"] = (
            d.str[:4] + "-" + d.str[4:6] + "-" + d.str[6:8]
        )
        bad_mask = df["flight_date"].str.contains(r"[^0-9\-]", na=True) | (df["flight_date"].str.len() != 10)
        df.loc[bad_mask, "flight_date"] = ""


    df["origin_city"] = df["origin"]
    df["dest_city"]   = df["destination"]
    df["leg_number"]  = 1
    df["legs_total"]  = 1

    df = df[df["flight_id"].astype(str).str.strip() != ""].copy()

    for _col in ["FN_NUMBER", "DEP_TIME_SCHED", "ARR_TIME_SCHED",
                 "DEP_DAY_SCHED", "ARR_DAY_SCHED", "LEG_SEQUENCE_NUMBER"]:
        if _col in df.columns:
            df[_col] = (
                df[_col]
                .astype(str)
                .str.replace(r"\.0+$", "", regex=True)
                .str.replace(",", "", regex=False)
            )

    return df.reset_index(drop=True)


_EXCEL_EPOCH = pd.Timestamp("1899-12-30")

def _clean_ac_msn(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, pd.Timestamp):
        serial = (val.normalize() - _EXCEL_EPOCH).days
        return str(serial)
    s = str(val).strip()
    if not s or s in {"nan", "NaT", "None", ""}:
        return ""
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            t = pd.Timestamp(s)
            if t.year < 2000:
                serial = (t.normalize() - _EXCEL_EPOCH).days
                return str(serial)
            return s[:10]
        except Exception:
            pass
    if s.endswith(".0"):
        return s[:-2]
    return s


def _format_date_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d") if pd.notna(val) else ""
    s = str(val).strip()
    if not s or s in {"nan", "NaT", "None"}:
        return ""
    if "T" in s:
        s = s.split("T")[0]
    return s[:10]


def _convert_doa_programe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit DOA_PROGRAME → format crew_pairing interne.

    Groupement par CREW_ID + DATE_LT (date locale opérationnelle).
    Tri des vols par DATE_LT + START_LT
    Clé de jointure avec le schedule : numéro de vol + date + ORIGINE + DESTINATION.
    """
    df = df.copy()

    # ── Remappage colonnes si nécessaire ──────────────────────
    if {"ACTIVITY_GROUP", "POS"}.issubset(df.columns):
        actg = df["ACTIVITY_GROUP"].astype(str).str.strip().str.upper()
        posv = df["POS"].astype(str).str.strip().str.upper()

        # Détection : POS contient FLT au lieu de ACTIVITY_GROUP
        colonnes_decalees = (not (actg == "FLT").any()) and (posv == "FLT").any()

        if colonnes_decalees:
            src = df.copy()  # snapshot sécurisé

            # Correction principale
            df["ACTIVITY_GROUP"] = src["POS"]

            # Ne pas casser les colonnes métier
            for col in [
                "ACTIVITY", "ORIGINE", "DESTINATION",
                "START_LT", "END_LT", "A_C"
            ]:
                if col in src.columns:
                    df[col] = src[col]

            # Gestion des dates/temps (corriger les manquants)
            if "START_UTC" in src.columns:
                df["START_UTC"] = src["START_UTC"].fillna(src.get("START_LT"))

            if "END_UTC" in src.columns:
                df["END_UTC"] = src["END_UTC"].fillna(src.get("END_LT"))

            # Colonnes optionnelles (sécurisées)
            for col in [
                "LAYOVER", "HOTEL", "BLOCK_HOURS",
                "BLC", "AUGMENTATION",
                "flight_sequence"
            ]:
                if col in src.columns:
                    df[col] = src[col]

    # Référence date
    _REF_DATE = pd.Timestamp("2000-01-01")

    def _epoch_minutes_lt(row) -> float:
        """
        Ordonne les vols par DATE_LT + START_LT
        """
        try:
            dt = pd.to_datetime(row["DATE_LT"], errors="coerce")
            if pd.isna(dt):
                return 0.0
            day_offset = (dt.normalize() - _REF_DATE).days * 1440

            # ── Heure locale : START_LT prioritaire ──────────
            t = row.get("START_LT")
            if t is None or (isinstance(t, float) and np.isnan(t)):
                return float(day_offset)
            if hasattr(t, "hour"):
                return day_offset + t.hour * 60 + t.minute
            parts = str(t).split(":")
            if len(parts) >= 2:
                try:
                    return day_offset + int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    pass
            return float(day_offset)
        except Exception:
            return 0.0

    # ── Filtrer les lignes de type FLT ────────────────────────
    df_flt = df[
        df["ACTIVITY_GROUP"].astype(str).str.strip().str.upper() == "FLT"
    ].copy()

    df_flt["_epoch"] = df_flt.apply(_epoch_minutes_lt, axis=1)
    df_flt = df_flt.sort_values(["CREW_ID", "DATE_LT", "_epoch"])

    # ── Groupement par CREW_ID + DATE_LT ─────────────────────
    crew_rows = []

    for (cid, duty_date), grp in df_flt.groupby(
        ["CREW_ID", "DATE_LT"], observed=True, sort=False
    ):
        grp = grp.sort_values("_epoch")

        # Collecter les vols avec leur origine et destination
        fids, origins, dests = [], [], []
        for _, r in grp.iterrows():
            tok = _normalize_flight_token(r["ACTIVITY"])
            if not tok:
                continue
            # Normaliser le numéro de vol (ex: "570" → "AT570")
            if tok.isdigit():
                tok = f"AT{tok}"
            elif not any(ch.isdigit() for ch in tok):
                continue  # Pas un vol (activité sol, etc.)

            fids.append(tok)
            origins.append(
                str(r["ORIGINE"]).strip().upper()
                if "ORIGINE" in grp.columns and pd.notna(r.get("ORIGINE"))
                else ""
            )
            dests.append(
                str(r["DESTINATION"]).strip().upper()
                if "DESTINATION" in grp.columns and pd.notna(r.get("DESTINATION"))
                else ""
            )

        if not fids:
            continue

        # ── Données complémentaires ───────────────────────────
        quals = (
            grp["AC_QUALS"].iloc[0]
            if "AC_QUALS" in grp.columns and not grp["AC_QUALS"].isna().all()
            else ""
        )
        rank     = grp["RANK_"].iloc[0]    if "RANK_" in grp.columns    else ""
        ac_codes = grp["A_C"].dropna().unique() if "A_C" in grp.columns else []
        msn      = _clean_ac_msn(ac_codes[0]) if len(ac_codes) > 0 else ""
        block    = grp["BLOCK_HOURS"].iloc[0] if "BLOCK_HOURS" in grp.columns else ""

        quals_str = str(quals).strip()
        ac_type   = quals_str if quals_str else str(msn)

        duty_start_str = _format_date_str(duty_date)

        # Normaliser l'identifiant équipage
        if pd.isna(cid):
            cid_str = "UNKNOWN"
        elif str(cid).replace(".", "", 1).isdigit():
            cid_str = str(int(float(cid)))
        else:
            cid_str = str(cid).strip()

        date_token = "".join(ch for ch in duty_start_str if ch.isdigit())[:8]
        if not date_token:
            date_token = str(duty_date).strip().replace(" ", "_")

        crew_rows.append({
            "crew_id":              f"CREW-{cid_str}-{date_token}",
            "CREW_ID_OPS":          cid_str,
            "flight_date":          duty_start_str,
            "aircraft_msn":         msn,
            "aircraft_type":        ac_type,
            "flight_sequence":      ";".join(fids),
            "n_flights":            len(fids),
            "RANK_":                rank,
            "AC_QUALS":             quals,
            "BLOCK_HOURS":          block,
            "ORIGINE":              origins[0] if origins else "",
            "DESTINATION":          dests[-1]  if dests   else "",
            # Séquences origine/destination par vol — utilisées pour la jointure 4-clés
            "ORIGINE_SEQUENCE":     ";".join(origins),
            "DESTINATION_SEQUENCE": ";".join(dests),
        })

    return pd.DataFrame(crew_rows)


# ─── Traitement interne du schedule ──────────────────────────

def _process_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "flight_id" in df.columns:
        df["flight_id"] = df["flight_id"].apply(_normalize_flight_token)
        df = df[df["flight_id"] != ""].copy()

    df["dep_min"] = df["scheduled_departure"].apply(hhmm_to_minutes)
    df["arr_min"] = df["scheduled_arrival"].apply(hhmm_to_minutes)

    mask_night = df["arr_min"] < df["dep_min"]
    df.loc[mask_night, "arr_min"] += 1440

    if "flight_duration_min" not in df.columns:
        df["flight_duration_min"] = df["arr_min"] - df["dep_min"]

    if "flight_date" in df.columns:
        dates = pd.to_datetime(df["flight_date"], errors="coerce")
        valid = dates.notna()
        if valid.any():
            j0 = dates.loc[valid].min()
            day_offset = (dates - j0).dt.days.fillna(0).astype(int)
            df["dep_min"] += day_offset * 1440
            df["arr_min"]  = df["dep_min"] + df["flight_duration_min"]

    if "flight_date" not in df.columns: df["flight_date"] = ""
    if "leg_number"  not in df.columns: df["leg_number"]  = 1
    if "legs_total"  not in df.columns: df["legs_total"]  = 1
    return df


def _recompute_crew_constraints(
    df_sched: pd.DataFrame, df_crew: pd.DataFrame
) -> pd.DataFrame:
    """
    Recalcule les séquences crew.

    Cle de jointure : numero de vol + date + origine + destination (4 cles).
    Date utilisee cote equipage : flight_date (issue de DATE_LT, heure locale).
    """
    if "flight_id" not in df_sched.columns:
        return df_crew
    if "crew_id" not in df_crew.columns or "flight_sequence" not in df_crew.columns:
        return df_crew

    # ── 1. Référentiel schedule normalisé ─────────────────────────────────────
    sched = df_sched[["flight_id"]].copy()
    sched["flight_id"]        = sched["flight_id"].astype(str).str.strip().str.upper()
    sched["flight_date_norm"] = (
        _to_date_str(df_sched["flight_date"])
        if "flight_date" in df_sched.columns
        else pd.Series([""] * len(df_sched), index=df_sched.index)
    )
    sched["origin"] = (
        df_sched["origin"].astype(str).str.strip().str.upper()
        if "origin" in df_sched.columns
        else ""
    )
    sched["destination"] = (
        df_sched["destination"].astype(str).str.strip().str.upper()
        if "destination" in df_sched.columns
        else ""
    )

    # ── 2. Exploser les séquences crew (1 ligne par vol) ──────────────────────
    rows = []
    for _, cr in df_crew.iterrows():
        seq_fids = [f.strip() for f in str(cr.get("flight_sequence", "")).split(";")]
        seq_orgs = [x.strip().upper() for x in str(cr.get("ORIGINE_SEQUENCE", "")).split(";")]
        seq_dsts = [x.strip().upper() for x in str(cr.get("DESTINATION_SEQUENCE", "")).split(";")]
        date_norm = _to_date_str(pd.Series([cr.get("flight_date", "")])).iloc[0]

        for i, fid_raw in enumerate(seq_fids):
            fid = str(fid_raw).strip().upper()
            if not fid:
                continue
            org = seq_orgs[i] if i < len(seq_orgs) else ""
            dst = seq_dsts[i] if i < len(seq_dsts) else ""
            rows.append({
                "crew_id":          cr["crew_id"],
                "flight_id":        fid,
                "flight_date_norm": date_norm,
                "origin":           org,
                "destination":      dst,
                "seq_pos":          i,
            })

    if not rows:
        return df_crew

    seq = pd.DataFrame(rows)

    # ── 3. Jointure 4-clés : vol + date + origine + destination ───────────────
    merged = seq.merge(
        sched[["flight_id", "flight_date_norm", "origin", "destination"]],
        on=["flight_id", "flight_date_norm", "origin", "destination"],
        how="left",
    )

    # ── 4. Agreger par crew_id ────────────────────────────────────────────────
    # Pas d'agrégation pour repos, simplement fusionner
    
    # ── 5. Fusionner avec df_crew ─────────────────────────────────────────────
    out = df_crew.copy()


    return out


def _group_crew_pairs(df_crew: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les 2 lignes équipage (CC/CA) en 1 seule ligne par vol.

    Regroupement: flight_sequence + flight_date + ORIGINE + DESTINATION
    (avec fallbacks vers flight_id / origin / destination si disponibles).
    """
    if df_crew is None or df_crew.empty:
        return df_crew

    df = df_crew.copy()

    # Colonnes de regroupement avec fallbacks pour le format DOA
    _fallbacks = {
        "flight_sequence": "flight_id",
        "flight_date":     None,
        "ORIGINE":         "origin",
        "DESTINATION":     "destination",
    }
    group_keys = []
    for primary, fallback in _fallbacks.items():
        if primary in df.columns:
            group_keys.append(primary)
        elif fallback and fallback in df.columns:
            group_keys.append(fallback)

    if not group_keys:
        df["CC_ID"] = df.get("crew_id", "")
        df["CA_ID"] = ""
        df["crew_label"] = df["CC_ID"].astype(str).apply(lambda x: f"CC: {x}")
        return df

    cc_ranks = {"CMD", "CDB", "CC", "CPT", "CAPT", "PIC", "P1"}
    if "RANK_" in df.columns:
        df["_is_cc"] = df["RANK_"].apply(lambda r: str(r).strip().upper() in cc_ranks)
    else:
        df["_sort_num"] = pd.to_numeric(df["crew_id"], errors="coerce").fillna(0)
        df = df.sort_values(group_keys + ["_sort_num"])
        df["_is_cc"] = df.groupby(group_keys, sort=False, observed=True).cumcount() == 0

    rows = []
    for _, grp in df.groupby(group_keys, sort=False, observed=True):
        cc_grp = grp[grp["_is_cc"] == True]
        ca_grp = grp[grp["_is_cc"] == False]

        base  = grp.iloc[0].to_dict()
        cc_id = str(cc_grp["crew_id"].iloc[0]) if not cc_grp.empty else ""
        ca_id = str(ca_grp["crew_id"].iloc[0]) if not ca_grp.empty else ""

        base["CC_ID"]      = cc_id
        base["CA_ID"]      = ca_id
        base["crew_label"] = f"CC: {cc_id} | CA: {ca_id}" if ca_id else f"CC: {cc_id}"
        base["crew_id"]    = f"PAIR-{cc_id}-{ca_id or 'X'}"
        rows.append(base)

    result = pd.DataFrame(rows)
    drop_cols = [c for c in ["_is_cc", "_sort_num"] if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)
    return result.reset_index(drop=True)
# ══════════════════════════════════════════════════════════════
# Chargement avec détection automatique du format
# ══════════════════════════════════════════════════════════════

def load_uploaded_data(schedule_file, crew_file, ref_file=None):
    """
    Charge les fichiers uploadés dans Streamlit.
    Détecte automatiquement le format et traduit si nécessaire.
    """
    df_sched_raw = _read_file(schedule_file)
    df_crew_raw  = _read_file(crew_file)
    df_ref       = _read_file(ref_file) if ref_file else None

    if df_sched_raw is None or df_crew_raw is None:
        return None, None, None

    converted_sched = _is_ops_leg(df_sched_raw)
    converted_crew  = _is_ops_doa(df_crew_raw)

    df_sched = _convert_leg_futur(df_sched_raw) if converted_sched else df_sched_raw
    df_crew  = _convert_doa_programe(df_crew_raw) if converted_crew else df_crew_raw

    ok_s = _check_columns(df_sched, REQUIRED_SCHEDULE, schedule_file.name)
    ok_c = _check_columns(df_crew,  REQUIRED_CREW,     crew_file.name)
    if not ok_s or not ok_c:
        return None, None, None

    if df_ref is not None:
        ref_cols_norm = {str(c).strip().lower() for c in df_ref.columns}
        # Evite le message d'erreur "param_type, value" quand un fichier turnaround est chargé ici.
        if REQUIRED_REF.issubset(ref_cols_norm):
            df_ref = df_ref.copy()
            df_ref.columns = [str(c).strip().lower() for c in df_ref.columns]
        elif {"airport", "aircraft_type", "min_turnaround_min"}.issubset(ref_cols_norm):
            st.info(
                f"Le fichier **{ref_file.name}** ressemble à un fichier turnaround. "
                "Les paramètres de référence par défaut seront utilisés."
            )
            df_ref = None
        else:
            _check_columns(df_ref, REQUIRED_REF, ref_file.name)

    df_sched = _process_schedule(df_sched)
    if "flight_sequence" in df_crew.columns:
        df_crew = df_crew.copy()
        df_crew["flight_sequence"] = df_crew["flight_sequence"].fillna("").apply(_normalize_flight_sequence)
        df_crew = df_crew[df_crew["flight_sequence"] != ""].copy()

    if "flight_sequence" in df_crew.columns:
        df_crew = _recompute_crew_constraints(df_sched, df_crew)
    elif converted_crew:
        df_crew = _recompute_crew_constraints(df_sched, df_crew)

    df_crew = _group_crew_pairs(df_crew)

    return df_sched, df_crew, df_ref


# ─── Chargement du fichier turnaround CSV ────────────────────

def load_turnaround_table(uploaded_file) -> dict:
    """
    Charge le fichier CSV/Excel turnaround uploadé via Streamlit.

    Format attendu : airport, aircraft_type, min_turnaround_min
    Retourne un dict avec clés (airport, aircraft_type) → minutes.
    La ligne (DEFAULT, DEFAULT) sert de valeur par défaut globale.

    Hiérarchie de lookup (dans _resolve_min_turnaround) :
      1. (airport, aircraft_type)   — correspondance exacte

    """
    if uploaded_file is None:
        return {}

    df = _read_file(uploaded_file)
    if df is None or df.empty:
        st.warning("Fichier turnaround vide ou illisible — valeurs par défaut utilisées.")
        return {}

    required = {"airport", "aircraft_type", "min_turnaround_min"}
    raw_cols = [str(c).strip() for c in df.columns]
    raw_to_norm = {c: c.lower() for c in raw_cols}

    alias_map = {
        "airport": {
            "airport", "aeroport", "apt", "station", "origin", "dep_ap_sched",
        },
        "aircraft_type": {
            "aircraft_type", "ac_type", "aircraft", "type_avion", "fleet", "ac_subtype",
        },
        "min_turnaround_min": {
            "min_turnaround_min", "min_turnaround", "turnaround_min", "tat_min",
        },
    }

    rename_dict = {}
    taken_targets = set()
    for raw, norm in raw_to_norm.items():
        for target, aliases in alias_map.items():
            if norm in aliases and target not in taken_targets:
                rename_dict[raw] = target
                taken_targets.add(target)
                break

    df = df.rename(columns=rename_dict)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = required - set(df.columns)
    if missing:
        st.error(
            "Le fichier turnaround doit contenir les colonnes : "
            "`airport, aircraft_type, min_turnaround_min`\n\n"
            f"Colonnes manquantes : `{', '.join(sorted(missing))}`\n\n"
            f"Colonnes trouvées : `{', '.join([str(c) for c in raw_cols])}`"
        )
        return {}

    table = {}
    for _, row in df.iterrows():
        airport = str(row["airport"]).strip().upper()
        ac_type = str(row["aircraft_type"]).strip().upper()
        try:
            minutes = float(str(row["min_turnaround_min"]).replace(",", "."))
        except (ValueError, TypeError):
            continue
        table[(airport, ac_type)] = minutes

    return table


# ─── Extraction paramètres de référence ──────────────────────

def build_ref_dicts(df_ref, turnaround_table=None):
    """
    Retourne les paramètres de simulation.

    Si turnaround_table est fourni (dict issu du CSV uploadé),
    il est retourné tel quel en 6ème position.
    Sinon, un fallback DEFAULT est utilisé.

    Retourne: (min_turnaround, gamma_shape, gamma_scale,
               markov_matrix, markov_multipliers, turnaround_table)
    """
    min_turnaround = {
        "DEFAULT": 45.0,
    }
    gamma_shape        = 2.5
    gamma_scale        = 15.0
    markov_matrix      = np.array([
        [0.65, 0.28, 0.07],
        [0.25, 0.52, 0.23],
        [0.08, 0.28, 0.64],
    ])
    markov_multipliers = [1.0, 1.45, 2.30]

    if turnaround_table is None:
        turnaround_table = {}

    if df_ref is None or "param_type" not in df_ref.columns:
        return min_turnaround, gamma_shape, gamma_scale, \
               markov_matrix, markov_multipliers, turnaround_table

    states = {"Normal": 0, "Alerte": 1, "Bloqué": 2}

    for _, row in df_ref.iterrows():
        pt  = str(row.get("param_type", "")).strip()
        at  = str(row.get("aircraft_type", "")).strip()
        try:
            val = float(row.get("value", 0))
        except (ValueError, TypeError):
            continue

        if pt == "min_turnaround_min":
            pass
        elif pt == "gamma_shape":
            gamma_shape = val
        elif pt == "gamma_scale":
            gamma_scale = val
        elif pt == "markov_transition":
            try:
                src, dst = at.split("→")
                i = states[src.strip()]
                j = states[dst.strip()]
                markov_matrix[i, j] = val
            except Exception:
                pass
        elif pt == "markov_turnaround_multiplier_normal":
            markov_multipliers[0] = val
        elif pt == "markov_turnaround_multiplier_alerte":
            markov_multipliers[1] = val
        elif pt == "markov_turnaround_multiplier_bloque":
            markov_multipliers[2] = val

    return min_turnaround, gamma_shape, gamma_scale, \
           markov_matrix, markov_multipliers, turnaround_table


if __name__ == "__main__":
    mt, gs, gsc, mm, mp, tt = build_ref_dicts(None)
    print("data_loader: defaults OK")
    print(f"Turnaround DEFAULT={mt.get('DEFAULT', 45.0)}")
