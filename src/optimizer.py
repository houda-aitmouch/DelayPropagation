"""
src/optimizer.py — Moteur de substitution d'avion
Algorithme de récupération de retard par échange d'appareil.

Pipeline par vol candidat :
  1. Trouver avions A2  → même type, même aéroport AU MOMENT du vol retardé
  2. Vérifier dispo A2  → arr_actual_A2 + turnaround ≤ dep_actual_A1
  3. Simuler le switch  → new_dep = max(dep_scheduled, arr_A2 + turn)
  4. Swapper les rotations → les vols suivants de A1/A2 (et équipages) sont échangés
  5. Vérifier chaîne    → vérification GÉOGRAPHIQUE + TEMPORELLE des vols avals
  6. Vérifier autorisations → A2 autorisé sur TOUS aéroports séquence A1 et vice-versa
  7. Vérifier équipage  → équipage affecté au vol disponible à new_dep
  8. Calculer gain      → delay_saved, new_delay, score composite
  9. Choisir meilleur   → max(score) parmi les candidats A2 valides

CORRECTIONS v5 :
  - load_aircraft_authorizations gère les colonnes RAM réelles
    (immatriculation / type_avion / aeroports_autorises)
  - Nouvelle fonction is_aircraft_allowed_for_route() : vérifie une LISTE
    d'aéroports en un seul appel (plus propre, évite les boucles redondantes)
  - _evaluate_candidate step 0 : vérifie ORIGIN + DESTINATION dès le départ
    (pas seulement la destination)
  - step 3bis simplifié : utilise is_aircraft_allowed_for_route()
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════

def _parse_date_scalar(value) -> pd.Timestamp:
    s = str(value).strip()
    if not s:
        return pd.NaT
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _parse_date_series(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    out = pd.Series(index=values.index, dtype="datetime64[ns]")
    out.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce", dayfirst=False)
    out.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], errors="coerce", dayfirst=True)
    return out


def _flight_match_key(value) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits if digits else raw


def _schedule_match_key(row_like) -> str:
    fn_key = _flight_match_key(row_like.get("FN_NUMBER", ""))
    if fn_key:
        return fn_key
    return _flight_match_key(row_like.get("flight_id", ""))


def _norm_airport(value) -> str:
    return str(value).strip().upper() if value is not None else ""


def _format_time(minutes: float) -> str:
    return f"{int(minutes)} min"



# ═══════════════════════════════════════════════════════════════════════
# CORRECTION 1 — Chargement des autorisations
# Supporte les colonnes RAM réelles :
#   immatriculation / type_avion / aeroports_autorises
# ET les colonnes standard :
#   aircraft_msn / aircraft_type / authorized_airports
# ═══════════════════════════════════════════════════════════════════════

# Mapping des noms de colonnes RAM → noms internes
_COL_ALIASES = {
    # colonne MSN / immatriculation
    "immatriculation":    "aircraft_msn",
    "aircraft_msn":       "aircraft_msn",
    "msn":                "aircraft_msn",
    "registration":       "aircraft_msn",
    "ac_registration":    "aircraft_msn",
    # colonne type avion
    "type_avion":         "aircraft_type",
    "aircraft_type":      "aircraft_type",
    "ac_subtype":         "aircraft_type",
    "type":               "aircraft_type",
    # colonne aéroports
    "aeroports_autorises":  "authorized_airports",
    "authorized_airports":  "authorized_airports",
    "airports":             "authorized_airports",
    "aeroports":            "authorized_airports",
}


def _normalize_auth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes du CSV vers les noms internes standard."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_").replace("-", "_")
        if key in _COL_ALIASES:
            rename_map[col] = _COL_ALIASES[key]
    return df.rename(columns=rename_map)


def load_aircraft_authorizations(filepath: str) -> dict:
    """
    Charge les autorisations d'aéroports depuis un fichier CSV.

    Formats supportés (colonnes détectées automatiquement) :

    Format RAM (votre cas) :
        immatriculation | type_avion | aeroports_autorises
        CNRGK           | 73H        | CDG;ORY;CMN;...
        CNRGM           | 73H        | CDG;ORY;CMN;...

    Format standard :
        aircraft_msn | aircraft_type | authorized_airports
        aircraft_type | authorized_airports   (sans MSN)

    Retourne :
        {
          "by_msn":  { "CNRGK": {"CDG", "ORY", ...}, "CNRGM": {"CDG", ...} },
          "by_type": { "73H":   {"CDG", "ORY", ...} }
        }

    Note : by_type est l'UNION des aéroports de tous les avions du même type.
    La vérification granulaire (par immatriculation) utilise by_msn.
    """
    result = {"by_msn": {}, "by_type": {}}

    if not filepath:
        print("Aucun fichier d'autorisations spécifié — mode permissif activé")
        return result

    if not os.path.exists(filepath):
        print(f"[WARN] Fichier d'autorisations non trouvé : {filepath}")
        return result

    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
    except Exception as e:
        print(f"[ERROR] Lecture fichier autorisations : {e}")
        return result

    # Normaliser les noms de colonnes (gère RAM et standard)
    df = _normalize_auth_columns(df)

    if "authorized_airports" not in df.columns:
        print(f"[ERROR] Colonne aéroports introuvable. Colonnes détectées : {list(df.columns)}")
        return result

    has_msn  = "aircraft_msn"  in df.columns
    has_type = "aircraft_type" in df.columns

    if not has_msn and not has_type:
        print(f"[ERROR] Ni colonne MSN/immatriculation ni colonne type trouvée.")
        return result

    def _parse_airports(raw: str) -> set:
        raw = str(raw).replace(",", ";").replace(" ", ";")
        return {a.strip().upper() for a in raw.split(";") if a.strip()}

    for _, row in df.iterrows():
        airports = _parse_airports(row["authorized_airports"])
        if not airports:
            continue

        # Indexation par MSN (immatriculation) — granulaire
        if has_msn:
            msn = str(row["aircraft_msn"]).strip().upper()
            if msn and msn not in ("NAN", ""):
                result["by_msn"][msn] = airports

        # Indexation par type — union (fallback)
        if has_type:
            ac_type = str(row["aircraft_type"]).strip().upper()
            if ac_type and ac_type not in ("NAN", ""):
                result["by_type"].setdefault(ac_type, set()).update(airports)

    n_msn  = len(result["by_msn"])
    n_type = len(result["by_type"])
    print(f"[OK] Autorisations chargées : {n_msn} avions (immat.), {n_type} types")
    if n_msn > 0:
        print(f"     Immatriculations : {', '.join(sorted(result['by_msn']))}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# CORRECTION 2 — Vérification par avion (immatriculation)
# ═══════════════════════════════════════════════════════════════════════

def is_airport_authorized(
        msn: str,
        aircraft_type: str,
        airport: str,
        authorizations: dict,
        fallback_allowed: Optional[set] = None,
) -> tuple[bool, str]:
    """
    Vérifie si un avion (par immatriculation, puis par type) est autorisé
    sur un aéroport donné.

    Priorité :
      1. by_msn  (immatriculation — plus précis)
      2. by_type (type avion — moins précis)
      3. fallback (planning opérationnel)
      4. Mode permissif si aucune info
    """
    msn_norm     = str(msn).strip().upper()
    type_norm    = str(aircraft_type).strip().upper()
    airport_norm = _norm_airport(airport)

    # 1. Par immatriculation
    if msn_norm in authorizations.get("by_msn", {}):
        allowed = authorizations["by_msn"][msn_norm]
        if airport_norm in allowed:
            return True, f"Autorisé par immatriculation ({msn_norm})"
        return False, (
            f"Aéroport {airport_norm} NON autorisé pour {msn_norm} "
            f"[autorisés: {', '.join(sorted(allowed))}]"
        )

    # 2. Par type
    if type_norm in authorizations.get("by_type", {}):
        allowed = authorizations["by_type"][type_norm]
        if airport_norm in allowed:
            return True, f"Autorisé par type ({type_norm})"
        return False, (
            f"Aéroport {airport_norm} NON autorisé pour type {type_norm} "
            f"[autorisés: {', '.join(sorted(allowed))}]"
        )

    # 3. Fallback planning
    if fallback_allowed is not None:
        if airport_norm in fallback_allowed:
            return True, "Autorisé par planning (fallback)"
        return False, f"Aéroport {airport_norm} absent du planning pour type {type_norm}"

    # 4. Pas d'info → permissif
    return True, "Pas de restriction définie"


def is_aircraft_allowed_for_route(
        msn: str,
        aircraft_type: str,
        airports: set,
        authorizations: dict,
        fallback_allowed: Optional[set] = None,
) -> tuple[bool, str]:
    """
    NOUVEAU — Vérifie qu'un avion est autorisé sur TOUS les aéroports
    d'une liste (séquence de vols complète).

    Remplace les boucles `for airport in airports: is_airport_authorized()`
    dispersées dans le code.

    Paramètres :
        msn             : immatriculation de l'avion (ex: "CNRGK")
        aircraft_type   : type avion (ex: "73H")
        airports        : ensemble d'aéroports à vérifier
        authorizations  : dict chargé par load_aircraft_authorizations()
        fallback_allowed: aéroports du planning pour ce type (fallback)

    Retourne :
        (True,  "")           si tous les aéroports sont autorisés
        (False, raison)       dès le premier aéroport non autorisé
    """
    for apt in sorted(airports):               # tri pour logs déterministes
        ok, reason = is_airport_authorized(
            msn=msn,
            aircraft_type=aircraft_type,
            airport=apt,
            authorizations=authorizations,
            fallback_allowed=fallback_allowed,
        )
        if not ok:
            return False, reason
    return True, ""


def _build_allowed_airports_by_type(df_sched: pd.DataFrame) -> dict[str, set]:
    """Construit la liste des aéroports autorisés par type depuis le planning (fallback)."""
    allowed: dict[str, set] = {}
    if df_sched is None or df_sched.empty:
        return allowed
    required = {"aircraft_type", "origin", "destination"}
    if not required.issubset(set(df_sched.columns)):
        return allowed
    for _, row in df_sched.iterrows():
        ac_type = str(row.get("aircraft_type", "")).strip().upper()
        if not ac_type:
            continue
        allowed.setdefault(ac_type, set())
        org = _norm_airport(row.get("origin", ""))
        dst = _norm_airport(row.get("destination", ""))
        if org:
            allowed[ac_type].add(org)
        if dst:
            allowed[ac_type].add(dst)
    return allowed


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURES DE RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SwapCandidate:
    flight_id: str
    a1_msn: str
    a2_msn: str
    a2_type: str
    dep_scheduled: float
    dep_original: float
    arr_original: float
    a2_last_arr: float
    new_dep: float
    new_arr: float
    flight_duration: float
    turnaround_used: float
    a2_next_fid: Optional[str]
    a2_next_dep: Optional[float]
    a2_next_origin: Optional[str]
    a2_next_conflict: bool
    a2_conflict_margin: float
    a1_downstream: int
    a1_chain_delay_saved: float
    crew_ok: bool
    crew_note: str
    delay_saved: float
    new_delay: float
    score: float
    feasible: bool
    infeasibility_reason: str = ""

    @property
    def gain_pct(self) -> float:
        orig_delay = self.dep_original - self.dep_scheduled
        if orig_delay <= 0:
            return 0.0
        return max(0.0, self.delay_saved / orig_delay * 100)


@dataclass
class SwapResult:
    flight_id: str
    flight_date: str
    origin: str
    destination: str
    dep_scheduled: float
    dep_original: float
    a1_msn: str
    a1_type: str
    mean_arr_delay: float
    best: Optional[SwapCandidate] = None
    all_candidates: list[SwapCandidate] = field(default_factory=list)
    n_candidates_found: int = 0
    n_candidates_feasible: int = 0

    @property
    def has_solution(self) -> bool:
        return self.best is not None and self.best.feasible

    @property
    def original_delay(self) -> float:
        return max(0.0, self.dep_original - self.dep_scheduled)


# ═══════════════════════════════════════════════════════════════════════
# ÉTAT FLOTTE
# ═══════════════════════════════════════════════════════════════════════

def _build_fleet_state(
        df_sched: pd.DataFrame,
        sim_result: pd.DataFrame,
        before_dep_min: Optional[float] = None,
        before_flight_date: Optional[str] = None,
) -> dict[str, dict]:
    fleet = {}
    merged = df_sched.copy()

    if len(sim_result) == len(df_sched):
        for col in ["dep_actual", "arr_actual", "dep_delay", "arr_delay"]:
            if col in sim_result.columns:
                merged[col] = sim_result[col].values
    else:
        merged = df_sched.merge(
            sim_result[["flight_id", "dep_actual", "arr_actual", "dep_delay", "arr_delay"]],
            on="flight_id", how="left",
        )

    ref_abs = None
    if before_dep_min is not None and before_flight_date:
        ref_date = _parse_date_scalar(before_flight_date)
        if pd.notna(ref_date):
            ref_abs = ref_date + pd.Timedelta(minutes=before_dep_min)

    for msn, grp in merged.groupby("aircraft_msn"):
        grp_sorted = grp.sort_values("dep_actual")

        if before_dep_min is not None:
            if ref_abs is not None and "flight_date" in grp_sorted.columns:
                abs_dep = (
                    _parse_date_series(grp_sorted["flight_date"])
                    + pd.to_timedelta(grp_sorted["dep_actual"], unit="m")
                )
                grp_before = grp_sorted[abs_dep < ref_abs]
            else:
                grp_before = grp_sorted[grp_sorted["dep_actual"] < before_dep_min]

            if grp_before.empty:
                continue
            last = grp_before.iloc[-1]
        else:
            last = grp_sorted.iloc[-1]

        fleet[msn] = {
            "last_flight_id":    last["flight_id"],
            "last_arr_actual":   float(last["arr_actual"]),
            "last_destination":  str(last["destination"]).strip().upper(),
            "all_flights":       grp_sorted.to_dict("records"),
            "type":              last["aircraft_type"],
        }
    return fleet


def _get_next_flight(
        msn: str,
        after_dep_min: float,
        after_flight_date: Optional[str],
        df_sched: pd.DataFrame,
) -> Optional[dict]:
    df_msn = df_sched[df_sched["aircraft_msn"] == msn].copy()
    if df_msn.empty:
        return None

    if "flight_date" in df_msn.columns and after_flight_date:
        ref_date = _parse_date_scalar(after_flight_date)
        if pd.notna(ref_date):
            dep_keys = (
                _parse_date_series(df_msn["flight_date"])
                + pd.to_timedelta(df_msn["dep_min"], unit="m")
            )
            ref_key = ref_date + pd.to_timedelta(after_dep_min, unit="m")
            df_msn = df_msn.assign(_dep_key=dep_keys)
            df_after = df_msn[df_msn["_dep_key"] > ref_key].sort_values("_dep_key")
        else:
            df_after = df_msn[df_msn["dep_min"] > after_dep_min].sort_values("dep_min")
    else:
        df_after = df_msn[df_msn["dep_min"] > after_dep_min].sort_values("dep_min")

    if df_after.empty:
        return None
    return df_after.iloc[0].to_dict()


# ═══════════════════════════════════════════════════════════════════════
# VÉRIFICATION ÉQUIPAGE
# ═══════════════════════════════════════════════════════════════════════

def _check_crew(
        flight_id: str,
        flight_date: Optional[str],
        new_dep: float,
        df_sched: pd.DataFrame,
        df_crew: pd.DataFrame,
        swap_crews_with_aircraft: bool = True,
        otp_threshold: float = 15,
) -> tuple[bool, str]:
    if swap_crews_with_aircraft:
        return True, "Équipage swappé avec l'avion (mode rotation complète)"

    if df_crew is None or df_crew.empty:
        return True, "Pas de données équipage"

    row_sched = df_sched[df_sched["flight_id"] == flight_id].copy()
    if flight_date is not None and "flight_date" in df_sched.columns:
        target_date = _parse_date_scalar(flight_date)
        if pd.notna(target_date):
            by_date = row_sched[_parse_date_series(row_sched["flight_date"]) == target_date]
            if not by_date.empty:
                row_sched = by_date
    if row_sched.empty:
        return True, "Vol non trouvé dans le schedule"

    row_ref  = row_sched.iloc[0]
    row_key  = _schedule_match_key(row_ref)
    row_org  = _norm_airport(row_ref.get("origin", ""))
    row_dst  = _norm_airport(row_ref.get("destination", ""))
    row_msn  = str(row_ref.get("aircraft_msn", "")).strip()
    target_date = _parse_date_scalar(flight_date)

    crew_rows = df_crew[df_crew["flight_sequence"].notna()].copy()
    if flight_date is not None and "flight_date" in df_crew.columns:
        if pd.notna(target_date):
            by_date = crew_rows[_parse_date_series(crew_rows["flight_date"]) == target_date]
            if not by_date.empty:
                crew_rows = by_date
    if row_msn and "aircraft_msn" in crew_rows.columns:
        by_msn = crew_rows[crew_rows["aircraft_msn"].astype(str).str.strip() == row_msn]
        if not by_msn.empty:
            crew_rows = by_msn

    if crew_rows.empty:
        return True, "Aucun équipage affecté — libre"

    for _, cr in crew_rows.iterrows():
        seq_raw = [f.strip() for f in str(cr["flight_sequence"]).split(";")]
        seq_key = [_flight_match_key(x) for x in seq_raw]
        if row_key not in seq_key:
            continue

        org_seq = [_norm_airport(x) for x in str(cr.get("ORIGINE_SEQUENCE", "")).split(";") if str(x).strip()]
        dst_seq = [_norm_airport(x) for x in str(cr.get("DESTINATION_SEQUENCE", "")).split(";") if str(x).strip()]

        pos = -1
        for i, k in enumerate(seq_key):
            if k != row_key:
                continue
            if i < len(org_seq) and i < len(dst_seq):
                if org_seq[i] != row_org or dst_seq[i] != row_dst:
                    continue
            pos = i
            break

        if pos < 0:
            continue
        if pos == 0:
            return True, f"1er vol de la rotation équipage {cr.get('crew_id', '')}"

        prev_key = seq_key[pos - 1]
        prev_org = org_seq[pos - 1] if pos - 1 < len(org_seq) else ""
        prev_dst = dst_seq[pos - 1] if pos - 1 < len(dst_seq) else ""

        if "FN_NUMBER" in df_sched.columns:
            prev_match = df_sched[df_sched["FN_NUMBER"].astype(str).map(_flight_match_key) == prev_key]
        else:
            prev_match = df_sched[df_sched["flight_id"].astype(str).map(_flight_match_key) == prev_key]

        prev_row = prev_match.copy()
        if prev_org and prev_dst and {"origin", "destination"}.issubset(set(prev_row.columns)):
            by_apt = prev_row[
                (prev_row["origin"].astype(str).str.strip().str.upper() == prev_org)
                & (prev_row["destination"].astype(str).str.strip().str.upper() == prev_dst)
            ]
            if not by_apt.empty:
                prev_row = by_apt
        if flight_date is not None and "flight_date" in df_sched.columns:
            if pd.notna(target_date):
                by_date = prev_row[_parse_date_series(prev_row["flight_date"]) == target_date]
                if not by_date.empty:
                    prev_row = by_date
        if prev_row.empty:
            continue

        arr_prev = float(prev_row["arr_min"].iloc[0])
        # Pas de repos d'équipage requis
        repos = 0.0
        crew_ready = arr_prev + repos

        if new_dep >= crew_ready:
            return True, (
                f"Équipage {cr.get('crew_id', '')} disponible à {_format_time(crew_ready)} "
                f"(repos {repos:.0f} min après vol précédent #{prev_key})"
            )
        else:
            return False, (
                f"Équipage {cr.get('crew_id', '')} disponible à {_format_time(crew_ready)}, "
                f"soit {crew_ready - new_dep:.0f} min après le nouveau départ prévu"
            )

    return True, "Équipage compatible"


# ═══════════════════════════════════════════════════════════════════════
# CORRECTION 3 — Évaluation d'un candidat A2
# ═══════════════════════════════════════════════════════════════════════

def _evaluate_candidate(
        flight_id: str,
        flight_date: Optional[str],
        a1_msn: str,
        a2_msn: str,
        dep_scheduled: float,
        dep_original: float,
        arr_original: float,
        flight_duration: float,
        origin: str,
        destination: str,
        a2_last_arr: float,
        a1_last_arr: float,
        fleet_state: dict,
        df_sched: pd.DataFrame,
        df_crew: pd.DataFrame,
        min_turnaround: dict,
        authorizations: Optional[dict] = None,
        fallback_allowed_by_type: Optional[dict[str, set]] = None,
        swap_crews_with_aircraft: bool = True,
        otp_threshold: float = 15,
) -> SwapCandidate:
    """
    Évalue la faisabilité et le gain d'utiliser A2 sur le vol retardé.

    Ordre des vérifications :
      0. Autorisations A2 sur ORIGIN + DESTINATION du vol switché
         (vérification rapide avant calcul)
      1. Disponibilité temporelle de A2
      2. Prochain vol de A2
      3. Collecte des vols avals A1 et A2
      3bis. Autorisations complètes sur TOUTES les séquences post-switch
             → A2 sur tous les aéroports de la séquence A1
            → A1 sur tous les aéroports de la séquence A2
      4. Vérification géographique + temporelle
      5. Vérification équipage
      6. Score composite
    """
    turnaround   = 45.0
    a2_type      = fleet_state[a2_msn]["type"]
    a1_type      = str(fleet_state.get(a1_msn, {}).get("type", "")).strip().upper()
    origin_norm  = _norm_airport(origin)
    dest_norm    = _norm_airport(destination)
    a2_type_norm = str(a2_type).strip().upper()

    # Helper pour créer un candidat infaisable
    def _infeasible(reason: str) -> SwapCandidate:
        return SwapCandidate(
            flight_id=flight_id, a1_msn=a1_msn, a2_msn=a2_msn, a2_type=a2_type,
            dep_scheduled=dep_scheduled, dep_original=dep_original,
            arr_original=arr_original, a2_last_arr=a2_last_arr,
            new_dep=dep_original, new_arr=arr_original,
            flight_duration=flight_duration, turnaround_used=turnaround,
            a2_next_fid=None, a2_next_dep=None, a2_next_origin=None,
            a2_next_conflict=False, a2_conflict_margin=0.0,
            a1_downstream=0, a1_chain_delay_saved=0.0,
            crew_ok=False, crew_note="",
            delay_saved=0.0, new_delay=max(0.0, dep_original - dep_scheduled),
            score=-9999, feasible=False, infeasibility_reason=reason,
        )

    # Fallback aéroports par type
    fallback_a2 = fallback_allowed_by_type.get(a2_type_norm) if fallback_allowed_by_type else None
    fallback_a1 = fallback_allowed_by_type.get(a1_type)      if fallback_allowed_by_type else None

    # ── 0. CORRECTION : Vérification autorisation ORIGIN + DESTINATION ──────
    # On vérifie les deux aéroports du vol switché dès le départ,
    # avant tout calcul coûteux. Cela rejette rapidement les candidats invalides.
    if authorizations is not None and (authorizations.get("by_msn") or authorizations.get("by_type")):
        airports_vol = {origin_norm, dest_norm}
        ok, reason = is_aircraft_allowed_for_route(
            msn=a2_msn, aircraft_type=a2_type,
            airports=airports_vol,
            authorizations=authorizations,
            fallback_allowed=fallback_a2,
        )
        if not ok:
            return _infeasible(f"Vol principal — {reason}")

    elif fallback_allowed_by_type is not None:
        # Fallback : vérification par type via planning
        allowed = fallback_allowed_by_type.get(a2_type_norm, set())
        for apt in (origin_norm, dest_norm):
            if apt not in allowed:
                return _infeasible(
                    f"Destination/origine {apt} non autorisée "
                    f"pour type {a2_type_norm} (planning fallback)"
                )

    # ── 1. Calcul disponibilité A2 et nouveau départ ─────────────────────
    a2_ready  = a2_last_arr + turnaround
    new_dep   = max(dep_scheduled, a2_ready)
    new_arr   = new_dep + flight_duration
    new_delay = max(0.0, new_dep - dep_scheduled)

    if a2_ready > dep_original + 1:
        return _infeasible(
            f"A2 disponible à {_format_time(a2_ready)}, "
            f"trop tard vs départ retardé A1 ({_format_time(dep_original)})"
        )

    # ── 2. Prochain vol de A2 ─────────────────────────────────────────────
    next_a2 = _get_next_flight(a2_msn, dep_scheduled, flight_date, df_sched)
    a2_next_fid      = None
    a2_next_dep      = None
    a2_next_origin   = None
    a2_next_conflict = False
    a2_conflict_margin = 9999.0

    if next_a2 is not None:
        a2_next_fid    = next_a2["flight_id"]
        a2_next_dep    = float(next_a2["dep_min"])
        a2_next_origin = _norm_airport(next_a2.get("origin", ""))
        turn_next      = 45.0
        a2_conflict_margin = a2_next_dep - (new_arr + turn_next)

    # ── 3. Collecte des vols avals A1 et A2 ──────────────────────────────
    def _collect_downstream(msn: str, start_dep: float) -> list:
        chain = []
        cur = _get_next_flight(msn, start_dep, flight_date, df_sched)
        while cur is not None and len(chain) <= 10:
            chain.append(cur)
            cur = _get_next_flight(
                msn, float(cur["dep_min"]),
                str(cur.get("flight_date", flight_date or "")), df_sched,
            )
        return chain

    a1_downstream_flights = _collect_downstream(a1_msn, dep_scheduled)
    a2_downstream_flights = _collect_downstream(a2_msn, dep_scheduled)

    # ── 3bis. CORRECTION : Autorisations complètes sur TOUTES les séquences ──
    # Après le switch :
    #   • A2 prendra les vols de A1 → doit être autorisé sur tous les aéroports A1
    #   • A1 prendra les vols de A2 → doit être autorisé sur tous les aéroports A2
    feasible      = True
    infeas_reason = ""

    if feasible and authorizations is not None and (authorizations.get("by_msn") or authorizations.get("by_type")):

        # Séquence A1 que A2 devra opérer (destination + avals)
        airports_a1_seq = {dest_norm}
        for f in a1_downstream_flights:
            if _norm_airport(f.get("origin", "")):
                airports_a1_seq.add(_norm_airport(f["origin"]))
            if _norm_airport(f.get("destination", "")):
                airports_a1_seq.add(_norm_airport(f["destination"]))

        ok, reason = is_aircraft_allowed_for_route(
            msn=a2_msn, aircraft_type=a2_type,
            airports=airports_a1_seq,
            authorizations=authorizations,
            fallback_allowed=fallback_a2,
        )
        if not ok:
            feasible      = False
            infeas_reason = f"Séquence A1 → A2 : {reason}"

        # Séquence A2 que A1 devra opérer (avals A2 seulement)
        if feasible and a2_downstream_flights:
            airports_a2_seq = set()
            for f in a2_downstream_flights:
                if _norm_airport(f.get("origin", "")):
                    airports_a2_seq.add(_norm_airport(f["origin"]))
                if _norm_airport(f.get("destination", "")):
                    airports_a2_seq.add(_norm_airport(f["destination"]))

            ok, reason = is_aircraft_allowed_for_route(
                msn=a1_msn, aircraft_type=a1_type,
                airports=airports_a2_seq,
                authorizations=authorizations,
                fallback_allowed=fallback_a1,
            )
            if not ok:
                feasible      = False
                infeas_reason = f"Séquence A2 → A1 : {reason}"

    # ── 4. Vérification géographique + temporelle de la chaîne ───────────
    if feasible and a1_downstream_flights:
        first_origin = _norm_airport(a1_downstream_flights[0].get("origin", ""))
        if first_origin and first_origin != dest_norm:
            feasible      = False
            infeas_reason = (
                f"Chaîne A1 impossible : 1er vol aval part de {first_origin} "
                f"mais A2 arrivera à {dest_norm}"
            )

        if feasible:
            first_dep = float(a1_downstream_flights[0].get("dep_min", 9999))
            turn      = 45.0
            if new_arr + turn > first_dep + 1:
                feasible      = False
                infeas_reason = (
                    f"Chaîne A1 : A2 prêt à {_format_time(new_arr + turn)} "
                    f"mais 1er vol aval part à {_format_time(first_dep)}"
                )

    if feasible and a2_downstream_flights:
        first_origin = _norm_airport(a2_downstream_flights[0].get("origin", ""))
        if first_origin and first_origin != origin_norm:
            feasible      = False
            infeas_reason = (
                f"Chaîne A2 impossible : 1er vol aval part de {first_origin} "
                f"mais A1 restera à {origin_norm}"
            )

        if feasible:
            first_dep = float(a2_downstream_flights[0].get("dep_min", 9999))
            turn      = 45.0
            if a1_last_arr + turn > first_dep + 1:
                feasible      = False
                infeas_reason = (
                    f"Chaîne A2 : A1 prêt à {_format_time(a1_last_arr + turn)} "
                    f"mais 1er vol aval A2 part à {_format_time(first_dep)}"
                )

    # ── 5. Vérification équipage ──────────────────────────────────────────
    crew_ok, crew_note = _check_crew(
        flight_id, flight_date, new_dep, df_sched, df_crew,
        swap_crews_with_aircraft=swap_crews_with_aircraft,
        otp_threshold=otp_threshold,
    )
    if not crew_ok and feasible:
        feasible      = False
        infeas_reason = crew_note

    # ── 6. Score composite ────────────────────────────────────────────────
    delay_saved       = max(0.0, dep_original - new_dep)
    n_downstream      = len(a1_downstream_flights) + len(a2_downstream_flights)
    a1_chain_delay_saved = delay_saved * n_downstream * 0.6

    otp_bonus    = 50.0 if new_delay <= otp_threshold else 0.0
    chain_bonus  = min(a1_chain_delay_saved * 0.3, 100.0)
    margin_bonus = min(a2_conflict_margin * 0.1, 30.0) if not a2_next_conflict else 0.0

    score = (
        delay_saved + otp_bonus + chain_bonus + margin_bonus
    ) if feasible else -9999.0 + delay_saved

    return SwapCandidate(
        flight_id=flight_id, a1_msn=a1_msn, a2_msn=a2_msn, a2_type=a2_type,
        dep_scheduled=dep_scheduled, dep_original=dep_original,
        arr_original=arr_original, a2_last_arr=a2_last_arr,
        new_dep=new_dep, new_arr=new_arr, flight_duration=flight_duration,
        turnaround_used=turnaround,
        a2_next_fid=a2_next_fid, a2_next_dep=a2_next_dep, a2_next_origin=a2_next_origin,
        a2_next_conflict=a2_next_conflict, a2_conflict_margin=a2_conflict_margin,
        a1_downstream=n_downstream, a1_chain_delay_saved=a1_chain_delay_saved,
        crew_ok=crew_ok, crew_note=crew_note,
        delay_saved=delay_saved, new_delay=new_delay, score=score,
        feasible=feasible, infeasibility_reason=infeas_reason,
    )


# ═══════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════

def find_swap_for_flight(
        flight_id: str,
        df_sched: pd.DataFrame,
        sim_result: pd.DataFrame,
        df_crew: pd.DataFrame,
        min_turnaround: dict,
        authorizations: Optional[dict] = None,
        fallback_allowed_by_type: Optional[dict[str, set]] = None,
        swap_crews_with_aircraft: bool = True,
        row_idx: Optional[int] = None,
        otp_threshold: float = 15,
        mean_arr_delay: float = 0.0,
) -> SwapResult:
    """Trouve le meilleur avion de substitution pour un vol retardé."""
    if row_idx is not None and row_idx in df_sched.index:
        row = df_sched.loc[[row_idx]]
    else:
        row = df_sched[df_sched["flight_id"] == flight_id]

    if row.empty:
        return SwapResult(
            flight_id=flight_id, flight_date="", origin="?", destination="?",
            dep_scheduled=0, dep_original=0, a1_msn="?", a1_type="?",
            mean_arr_delay=mean_arr_delay,
        )

    row          = row.iloc[0]
    a1_msn       = row["aircraft_msn"]
    a1_type      = row["aircraft_type"]
    origin       = _norm_airport(row["origin"])
    destination  = _norm_airport(row["destination"])
    dep_scheduled   = float(row["dep_min"])
    flight_duration = float(row["flight_duration_min"])
    arr_scheduled   = float(row["arr_min"])
    flight_date     = str(row.get("flight_date", ""))

    if row_idx is not None and row_idx in sim_result.index:
        dep_original = float(sim_result.loc[row_idx, "dep_actual"])
        arr_original = float(sim_result.loc[row_idx, "arr_actual"])
    else:
        sim_row = sim_result[sim_result["flight_id"] == flight_id]
        if sim_row.empty:
            dep_original = dep_scheduled
            arr_original = arr_scheduled
        else:
            dep_original = float(sim_row.iloc[0]["dep_actual"])
            arr_original = float(sim_row.iloc[0]["arr_actual"])

    result = SwapResult(
        flight_id=flight_id, flight_date=flight_date,
        origin=origin, destination=destination,
        dep_scheduled=dep_scheduled, dep_original=dep_original,
        a1_msn=a1_msn, a1_type=a1_type,
        mean_arr_delay=mean_arr_delay,
    )

    fleet_state = _build_fleet_state(
        df_sched, sim_result,
        before_dep_min=dep_scheduled,
        before_flight_date=flight_date,
    )

    a1_state    = fleet_state.get(a1_msn, {})
    a1_last_arr = float(a1_state.get("last_arr_actual", dep_scheduled))

    candidates = []
    for a2_msn, fstate in fleet_state.items():
        if a2_msn == a1_msn:
            continue
        if fstate["type"] != a1_type:      # même type avion requis
            continue
        if fstate["last_destination"] != origin:   # même aéroport requis
            continue

        cand = _evaluate_candidate(
            flight_id=flight_id, flight_date=flight_date,
            a1_msn=a1_msn, a2_msn=a2_msn,
            dep_scheduled=dep_scheduled, dep_original=dep_original,
            arr_original=arr_original, flight_duration=flight_duration,
            origin=origin, destination=destination,
            a2_last_arr=fstate["last_arr_actual"],
            a1_last_arr=a1_last_arr,
            fleet_state=fleet_state,
            df_sched=df_sched, df_crew=df_crew,
            min_turnaround=min_turnaround,
            authorizations=authorizations,
            fallback_allowed_by_type=fallback_allowed_by_type,
            swap_crews_with_aircraft=swap_crews_with_aircraft,
            otp_threshold=otp_threshold,
        )
        candidates.append(cand)

    result.n_candidates_found    = len(candidates)
    result.all_candidates        = sorted(candidates, key=lambda c: -c.score)
    result.n_candidates_feasible = sum(1 for c in candidates if c.feasible)

    feasible_cands = [c for c in candidates if c.feasible]
    if feasible_cands:
        result.best = max(feasible_cands, key=lambda c: c.score)
    elif candidates:
        result.best = max(candidates, key=lambda c: c.score)

    return result


# ═══════════════════════════════════════════════════════════════════════
# ANALYSE GLOBALE
# ═══════════════════════════════════════════════════════════════════════

def run_swap_optimizer(
        df_agg: pd.DataFrame,
        all_results: list,
        df_sched: pd.DataFrame,
        df_crew: pd.DataFrame,
        min_turnaround: dict,
        authorizations_file: Optional[str] = None,
        swap_crews_with_aircraft: bool = True,
        otp_threshold: float = 15,
        delay_threshold: float = 30,
        use_median_sim: bool = True,
) -> tuple[pd.DataFrame, list[SwapResult]]:
    """
    Lance l'optimiseur de substitution sur tous les vols en retard.

    Le fichier CSV des autorisations peut utiliser les colonnes RAM :
        immatriculation | type_avion | aeroports_autorises
    """
    if not all_results:
        return pd.DataFrame(), []

    otp_per_sim = [r["on_time"].mean() for r in all_results]
    median_idx  = int(np.argsort(otp_per_sim)[len(otp_per_sim) // 2])
    sim_rep     = all_results[median_idx]

    delayed = df_agg[df_agg["mean_arr_delay"] >= delay_threshold].copy()

    # Charger les autorisations (supporte colonnes RAM)
    authorizations = None
    if authorizations_file:
        authorizations = load_aircraft_authorizations(authorizations_file)

    # Fallback depuis le planning
    fallback_allowed_by_type = _build_allowed_airports_by_type(df_sched)

    swap_list = []
    rows_out  = []

    for idx_delayed, r in delayed.iterrows():
        row_idx_val = idx_delayed if isinstance(idx_delayed, (int, np.integer)) else None
        sw = find_swap_for_flight(
            flight_id=r["flight_id"],
            df_sched=df_sched, sim_result=sim_rep, df_crew=df_crew,
            min_turnaround=min_turnaround,
            authorizations=authorizations,
            fallback_allowed_by_type=fallback_allowed_by_type,
            swap_crews_with_aircraft=swap_crews_with_aircraft,
            row_idx=row_idx_val,
            otp_threshold=otp_threshold,
            mean_arr_delay=float(r["mean_arr_delay"]),
        )
        swap_list.append(sw)

        row_out = {
            "Vol":                  sw.flight_id,
            "Date vol":             sw.flight_date,
            "Route":                f"{sw.origin} → {sw.destination}",
            "Avion A1":             sw.a1_msn,
            "Type":                 sw.a1_type,
            "Retard moyen (min)":   round(sw.mean_arr_delay, 1),
            "Retard prévu (min)":   round(sw.original_delay, 1),
            "Candidats trouvés":    sw.n_candidates_found,
            "Candidats valides":    sw.n_candidates_feasible,
        }

        if sw.best is not None:
            b = sw.best
            row_out.update({
                "Meilleur A2":          b.a2_msn,
                "Statut":               "Faisable" if b.feasible else "Partiel",
                "Nouveau départ":       f"{int(b.new_dep // 60):02d}:{int(b.new_dep % 60):02d}",
                "Gain (min)":           round(b.delay_saved, 1),
                "Nouveau retard (min)": round(b.new_delay, 1),
                "OTP restauré":         "Oui" if b.new_delay <= otp_threshold else "Non",
                "Conflit A2 suivant":   "Oui (Conflit)" if b.a2_next_conflict else "Non",
                "Équipage":             "OK" if b.crew_ok else "À vérifier",
                "Vols libérés":         b.a1_downstream,
                "Raison infaisabilité": b.infeasibility_reason,
            })
        else:
            row_out.update({
                "Meilleur A2":          "-",
                "Statut":               "Aucun candidat",
                "Nouveau départ":       "-",
                "Gain (min)":           0.0,
                "Nouveau retard (min)": round(sw.original_delay, 1),
                "OTP restauré":         "Non",
                "Conflit A2 suivant":   "-",
                "Équipage":             "-",
                "Vols libérés":         0,
                "Raison infaisabilité": "Aucun avion du même type disponible sur cet aéroport",
            })

        rows_out.append(row_out)

    df_results = pd.DataFrame(rows_out) if rows_out else pd.DataFrame()
    return df_results, swap_list


if __name__ == "__main__":
    print("optimizer.py v5 — OK")
    print("\nChangements v5 :")
    print("  [1] load_aircraft_authorizations : colonnes RAM (immatriculation/type_avion/aeroports_autorises) supportées")
    print("  [2] is_aircraft_allowed_for_route() : nouvelle fonction (vérifie une liste d'aéroports en un appel)")
    print("  [3] _evaluate_candidate step 0 : vérifie ORIGIN + DESTINATION (pas seulement destination)")
    print("  [3bis] Autorisations séquences : utilise is_aircraft_allowed_for_route() (code simplifié)")