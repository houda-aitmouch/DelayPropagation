"""
src/simulation.py — Moteur de simulation Monte Carlo
"""
import numpy as np


def _norm_date_key(value):
    """Normalise une date en clé stable YYYY-MM-DD (ou "" si absente/invalide)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return str(np.datetime64(s, "D"))
    except Exception:
        return ""


def _flight_match_key(value):
    """Clé de matching vol tolérante: 'AT570' et '570' -> '570'."""
    if value is None:
        return ""
    raw = str(value).strip().upper()
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits if digits else raw


def _norm_airport(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def _schedule_match_key(row_like):
    """Clé de matching côté schedule: FN_NUMBER prioritaire, sinon flight_id."""
    fn_key = _flight_match_key(row_like.get("FN_NUMBER", ""))
    if fn_key:
        return fn_key
    return _flight_match_key(row_like.get("flight_id", ""))


def _resolve_min_turnaround(min_turnaround, aircraft_type,
                            turnaround_table=None, airport=None):
    """
    Résout le turnaround minimum.

    Utilise uniquement la correspondance exacte turnaround_table[(airport, aircraft_type)].
    Fallback : 45 min.
    """
    if turnaround_table and airport:
        ap  = str(airport).strip().upper()
        act = str(aircraft_type).strip().upper()

        # Correspondance exacte (airport, aircraft_type)
        val = turnaround_table.get((ap, act))
        if val is not None:
            return float(val)

    return 45.0




def markov_turnaround(
    aircraft_type, min_turnaround,
    markov_matrix, markov_multipliers,
    n_steps=3, initial_state=0,
    turnaround_table=None, airport=None,
):
    """
    Tire un turnaround stochastique via la chaîne de Markov à 3 états.
    États : 0=Normal, 1=Alerte, 2=Bloqué
    Retourne : (turnaround_minutes, état_final)
    """
    base  = _resolve_min_turnaround(
        min_turnaround, aircraft_type,
        turnaround_table=turnaround_table, airport=airport,
    )
    state = initial_state
    for _ in range(n_steps):
        state = np.random.choice([0, 1, 2], p=markov_matrix[state])
    return base * markov_multipliers[state], state


def simulate_once(
    df_sched, df_crew, min_turnaround,
    gamma_shape, gamma_scale,
    markov_matrix, markov_multipliers,
    otp_threshold=15,
    mode="auto",
    use_markov=True,
    injected_delays=None,
    injected_targets=None,
    hub_airport=None,
    hub_factor=1.2,
    slack_config=None,
    turnaround_table=None,
):
    """
    Simule une journée complète de vols.

    Clé de jointure équipage/vol : numéro de vol + date locale + origine + destination.

    Formule de propagation :
      turn_effectif = max(min_turn, turn_Markov − slack_minutes)
      avion_prêt    = arr_prev + turn_effectif
      dep_réel      = max(dep_prévu + retard_initial, avion_prêt, équipage_prêt)

    Colonnes résultat :
      turnaround_actual   : valeur brute tirée par Markov
      turnaround_effectif : valeur après réduction par slack + plancher min_turn
      slack_applied       : absorption réelle = actual − effectif
    """
    n = len(df_sched)

    # ── Masque hub ────────────────────────────────────────────
    hub_mask = np.zeros(n, dtype=bool)
    if hub_airport:
        if isinstance(hub_airport, list):
            hub_mask = df_sched["origin"].isin(hub_airport).values
        else:
            hub_mask = (df_sched["origin"].values == hub_airport)

    # ── Retards initiaux ──────────────────────────────────────
    if mode == "manuel" and (injected_targets or injected_delays):
        initial_delays = np.zeros(n)
        if injected_targets:
            for tgt in injected_targets:
                fid = str(tgt.get("flight_id", "")).strip()
                if not fid:
                    continue
                minutes = float(tgt.get("minutes", 0.0))

                mask = (df_sched["flight_id"].astype(str).str.strip() == fid)

                if "flight_date" in df_sched.columns:
                    tgt_date = _norm_date_key(tgt.get("flight_date", ""))
                    if tgt_date:
                        d_sched = df_sched["flight_date"].apply(_norm_date_key)
                        mask = mask & (d_sched == tgt_date)

                tgt_msn = str(tgt.get("aircraft_msn", "")).strip()
                if tgt_msn and "aircraft_msn" in df_sched.columns:
                    mask = mask & (df_sched["aircraft_msn"].astype(str).str.strip() == tgt_msn)

                initial_delays[mask.values] = minutes
        else:
            for fid, minutes in injected_delays.items():
                mask = df_sched["flight_id"] == fid
                initial_delays[mask.values] = float(minutes)
    else:
        shapes = np.where(hub_mask, gamma_shape * hub_factor, gamma_shape)
        initial_delays = np.array([
            np.random.gamma(shape=float(shapes[i]), scale=gamma_scale)
            for i in range(n)
        ])

    # ── Initialisation du DataFrame résultat ─────────────────
    df = df_sched.copy()
    df["initial_delay"]       = initial_delays
    df["dep_actual"]          = df["dep_min"] + initial_delays
    df["arr_actual"]          = df["dep_actual"] + df["flight_duration_min"]
    df["turnaround_actual"]   = 0.0
    df["turnaround_effectif"] = 0.0
    df["markov_state"]        = 0
    df["hub_affected"]        = hub_mask
    df["slack_applied"]       = 0.0

    # ── Index schedule : (fkey, date, org, dst) → label ──────
    # Utilisé pour retrouver un vol précédent depuis la contrainte équipage.
    idx_by_fkey_date_apt = {}   # clé primaire 4-clés
    idx_by_fkey_date     = {}   # fallback 2-clés (overnight sans orig/dest)
    idx_by_fkey          = {}   # fallback ultime (sans date)

    for label, row in df.iterrows():
        fkey = _schedule_match_key(row)
        dkey = _norm_date_key(row.get("flight_date", ""))
        org  = _norm_airport(row.get("origin", ""))
        dst  = _norm_airport(row.get("destination", ""))

        if fkey:
            idx_by_fkey_date_apt[(fkey, dkey, org, dst)] = label
            idx_by_fkey_date[(fkey, dkey)]                = label
            idx_by_fkey[fkey]                             = label

    # ── Index équipage : (fkey, date, org, dst) → liste de records ───────────
    # Chaque record stocke la position dans la séquence (pos_seq) et le vol
    # précédent, pour calculer la contrainte de repos.
    #
    # Clé de jointure : numéro de vol + date locale (flight_date issue de
    # DATE_LT) + ORIGINE + DESTINATION — cohérent avec data_loader.
    crew_idx = {}   # (fkey, date_key, org, dst) → [rec, ...]

    for _, cr in df_crew[df_crew["flight_sequence"].notna()].iterrows():
        seq_raw  = [f.strip() for f in str(cr["flight_sequence"]).split(";")]
        seq_key  = [_flight_match_key(f) for f in seq_raw]

        org_seq  = [
            _norm_airport(x)
            for x in str(cr.get("ORIGINE_SEQUENCE", "")).split(";")
            if str(x).strip()
        ]
        dst_seq  = [
            _norm_airport(x)
            for x in str(cr.get("DESTINATION_SEQUENCE", "")).split(";")
            if str(x).strip()
        ]
        date_key = _norm_date_key(cr.get("flight_date", ""))
        msn      = str(cr.get("aircraft_msn", "")).strip()

        for pos_seq, fkey in enumerate(seq_key):
            if not fkey:
                continue
            org = org_seq[pos_seq] if pos_seq < len(org_seq) else ""
            dst = dst_seq[pos_seq] if pos_seq < len(dst_seq) else ""

            lookup_key = (fkey, date_key, org, dst)
            crew_idx.setdefault(lookup_key, []).append({
                "row":       cr,
                "seq_key":   seq_key,
                "org_seq":   org_seq,
                "dst_seq":   dst_seq,
                "date_key":  date_key,
                "msn":       msn,
                "pos_seq":   pos_seq,   # position dans la séquence (0-based)
            })

    # ── Tri des vols pour la propagation ──────────────────────
    if "flight_date" in df.columns:
        df["_flight_date_key"] = df["flight_date"].apply(_norm_date_key)
        df_sorted = df.sort_values(["_flight_date_key", "dep_min"])
    else:
        df["_flight_date_key"] = ""
        df_sorted = df.sort_values("dep_min")

    ac_last_flight = {}

    for pos, row in df_sorted.iterrows():
        msn        = row["aircraft_msn"]
        ac_type    = row["aircraft_type"]
        flight_date_key = str(row.get("_flight_date_key", ""))

        # ── Turnaround Markov (ou valeur fixe) ────────────────
        # L'aéroport de turnaround = origin du vol courant
        turn_airport = row.get("origin", "")
        if use_markov:
            turn_dur, m_state = markov_turnaround(
                ac_type, min_turnaround,
                markov_matrix, markov_multipliers,
                turnaround_table=turnaround_table,
                airport=turn_airport,
            )
        else:
            turn_dur = _resolve_min_turnaround(
                min_turnaround, ac_type,
                turnaround_table=turnaround_table,
                airport=turn_airport,
            )
            m_state  = 0

        df.at[pos, "turnaround_actual"] = turn_dur
        df.at[pos, "markov_state"]      = m_state

        # ── Minutes de slack applicables à ce vol ─────────────
        slack_minutes = 0.0
        if slack_config and slack_config.get("minutes", 0) > 0:
            s_min   = float(slack_config["minutes"])
            s_scope = slack_config.get("scope", "global")

            if s_scope == "global":
                slack_minutes = s_min

            elif s_scope == "window":
                w_start = float(slack_config.get("window_start", 0))
                w_end   = float(slack_config.get("window_end", 1440))
                if w_start <= row["dep_min"] <= w_end:
                    slack_minutes = s_min

            elif s_scope == "aircraft":
                msn_val = slack_config.get("aircraft_msn", "")
                if isinstance(msn_val, list):
                    if row["aircraft_msn"] in msn_val:
                        slack_minutes = s_min
                elif row["aircraft_msn"] == msn_val:
                    slack_minutes = s_min

            elif s_scope == "window_aircraft":
                w_start = float(slack_config.get("window_start", 0))
                w_end   = float(slack_config.get("window_end", 1440))
                msn_val = slack_config.get("aircraft_msn", "")
                msn_match = (
                    row["aircraft_msn"] in msn_val
                    if isinstance(msn_val, list)
                    else row["aircraft_msn"] == msn_val
                )
                if msn_match and w_start <= row["dep_min"] <= w_end:
                    slack_minutes = s_min

            elif s_scope == "flight":
                fid_val = slack_config.get("flight_id", "")
                if isinstance(fid_val, list):
                    if row["flight_id"] in fid_val:
                        slack_minutes = s_min
                elif row["flight_id"] == fid_val:
                    slack_minutes = s_min

            elif s_scope == "airports":
                apt_val = slack_config.get("airports", [])
                if isinstance(apt_val, list):
                    if row["origin"] in apt_val:
                        slack_minutes = s_min
                elif row["origin"] == apt_val:
                    slack_minutes = s_min

        # ── Turnaround minimum (valeur plancher) ─────────────────
        min_turn = _resolve_min_turnaround(
            min_turnaround, ac_type,
            turnaround_table=turnaround_table, airport=turn_airport,
        )
        # On garde turn_dur comme turnaround réel (tiré par Markov)
        turn_effectif = max(min_turn, turn_dur)
        df.at[pos, "turnaround_effectif"] = turn_effectif

        # ── Contrainte AVION avec absorption du retard AMONT ──────
        #
        # NOUVELLE LOGIQUE:
        # 1. Calculer le retard AMONT = retard du vol précédent
        # 2. Le slack ABSORBE ce retard (pas le turnaround!)
        # 3. Seul le retard non-absorbé se propage
        #
        earliest_avion = row["dep_min"]
        slack_utilise = 0.0

        if msn in ac_last_flight:
            prev_pos = ac_last_flight[msn]

            # Arrivée réelle vs prévue du vol précédent
            arr_prev_actual = df.at[prev_pos, "arr_actual"]
            arr_prev_scheduled = df.at[prev_pos, "arr_min"]

            # Retard AMONT = combien le vol précédent était en retard
            retard_amont = max(0.0, arr_prev_actual - arr_prev_scheduled)

            # Le slack ABSORBE le retard amont (règle métier aviation)
            slack_utilise = min(slack_minutes, retard_amont)
            retard_propague = retard_amont - slack_utilise

            # L'avion est prêt après:
            # - Arrivée prévue du vol précédent
            # - + turnaround minimum
            # - + retard propagé (non absorbé par le slack)
            avion_pret = arr_prev_scheduled + min_turn + retard_propague

            # On ne peut pas partir avant l'heure prévue
            earliest_avion = max(row["dep_min"], avion_pret)

            # Enregistrer le slack utilisé
            df.at[pos, "slack_applied"] = slack_utilise

        ac_last_flight[msn] = pos

        # ── Contrainte ÉQUIPAGE ───────────────────────────────
        # Jointure 4-clés : numéro + date + origine + destination
        earliest_crew = row["dep_min"]

        row_fkey = _schedule_match_key(row)
        row_org  = _norm_airport(row.get("origin", ""))
        row_dst  = _norm_airport(row.get("destination", ""))

        lookup_key = (row_fkey, flight_date_key, row_org, row_dst)
        candidates = crew_idx.get(lookup_key, [])

        for rec in candidates:
            pos_seq = rec["pos_seq"]
            if pos_seq <= 0:
                # Premier vol de la séquence : pas de contrainte équipage amont
                continue

            seq_key  = rec["seq_key"]
            org_seq  = rec["org_seq"]
            dst_seq  = rec["dst_seq"]
            prev_date_key = rec["date_key"]

            prev_fkey = seq_key[pos_seq - 1]
            prev_org  = org_seq[pos_seq - 1] if pos_seq - 1 < len(org_seq) else ""
            prev_dst  = dst_seq[pos_seq - 1] if pos_seq - 1 < len(dst_seq) else ""

            # Chercher le vol précédent dans le schedule (4-clés prioritaire)
            prev_label = idx_by_fkey_date_apt.get(
                (prev_fkey, flight_date_key, prev_org, prev_dst)
            )
            # Fallback overnight : le vol précédent peut être J-1 (date crew)
            if prev_label is None and prev_date_key != flight_date_key:
                prev_label = idx_by_fkey_date_apt.get(
                    (prev_fkey, prev_date_key, prev_org, prev_dst)
                )
            # Fallback 2-clés sans orig/dest (cas données incomplètes)
            if prev_label is None:
                prev_label = idx_by_fkey_date.get((prev_fkey, flight_date_key))
            if prev_label is None and prev_date_key != flight_date_key:
                prev_label = idx_by_fkey_date.get((prev_fkey, prev_date_key))
            # Dernier recours : numéro de vol seul
            if prev_label is None:
                prev_label = idx_by_fkey.get(prev_fkey)

            if prev_label is None:
                continue

            arr_crew = df.at[prev_label, "arr_actual"]
            # Contrainte demandee: l'equipage est disponible des l'arrivee du vol precedent.
            earliest_crew = max(earliest_crew, arr_crew)

        # ── Départ réel — formule MAX ─────────────────────────
        dep_real = max(
            row["dep_min"] + row["initial_delay"],
            earliest_avion,
            earliest_crew,
        )

        hub_check = False
        if hub_airport:
            hub_check = (
                row["origin"] in hub_airport
                if isinstance(hub_airport, list)
                else row["origin"] == hub_airport
            )

        if mode == "manuel" and hub_check:
            retard_accumule = dep_real - row["dep_min"]
            if retard_accumule > 0:
                dep_real = row["dep_min"] + (retard_accumule * hub_factor)

        df.at[pos, "dep_actual"] = dep_real
        df.at[pos, "arr_actual"] = dep_real + row["flight_duration_min"]

    # ── Indicateurs finaux ────────────────────────────────────
    if "_flight_date_key" in df.columns:
        df = df.drop(columns=["_flight_date_key"])

    df["dep_delay"] = (df["dep_actual"] - df["dep_min"]).clip(lower=0)
    df["arr_delay"] = (df["arr_actual"] - df["arr_min"]).clip(lower=0)
    df["on_time"]   = df["dep_delay"] <= otp_threshold

    return df


def run_monte_carlo(
    df_sched, df_crew, min_turnaround,
    gamma_shape, gamma_scale,
    markov_matrix, markov_multipliers,
    n_simulations=200,
    otp_threshold=15,
    progress_bar=None,
    mode="auto",
    use_markov=True,
    injected_delays=None,
    injected_targets=None,
    hub_airport=None,
    hub_factor=1.2,
    slack_config=None,
    turnaround_table=None,
):
    """
    Lance N simulations et agrège les résultats par vol.
    """
    if mode == "manuel":
        n_simulations = 1

    all_results  = []
    otp_per_sim  = []
    prop_per_sim = []

    for i in range(n_simulations):
        if progress_bar:
            progress_bar.progress(
                (i + 1) / n_simulations,
                text=f"Simulation {i+1}/{n_simulations}…",
            )

        result = simulate_once(
            df_sched, df_crew, min_turnaround,
            gamma_shape, gamma_scale,
            markov_matrix, markov_multipliers,
            otp_threshold=otp_threshold,
            mode=mode,
            use_markov=use_markov,
            injected_delays=injected_delays,
            injected_targets=injected_targets,
            hub_airport=hub_airport,
            hub_factor=hub_factor,
            slack_config=slack_config,
            turnaround_table=turnaround_table,
        )
        all_results.append(result)
        otp_per_sim.append(result["on_time"].mean() * 100)

        total_initial = result["initial_delay"].sum()
        total_final   = result["arr_delay"].sum()
        prop_per_sim.append(
            total_final / total_initial if total_initial > 0 else 1.0
        )

    # ── Construction des matrices (N simulations × M vols) ────
    arr_matrix   = np.array([r["arr_delay"].values           for r in all_results])
    dep_matrix   = np.array([r["dep_delay"].values           for r in all_results])
    turn_matrix  = np.array([r["turnaround_actual"].values   for r in all_results])
    teff_matrix  = np.array([r["turnaround_effectif"].values for r in all_results])
    slack_matrix = np.array([r["slack_applied"].values       for r in all_results])

    # ── Agrégation par vol ────────────────────────────────────
    agg = df_sched[[
        "flight_id", "origin", "destination",
        "scheduled_departure", "aircraft_msn", "aircraft_type",
    ]].copy()

    agg["mean_arr_delay"] = arr_matrix.mean(axis=0)
    agg["p95_arr_delay"]  = np.percentile(arr_matrix, 95, axis=0)
    agg["mean_dep_delay"] = dep_matrix.mean(axis=0)
    agg["otp_rate"]       = (dep_matrix <= otp_threshold).mean(axis=0) * 100

    agg["p80_dep_delay"] = np.percentile(dep_matrix, 80, axis=0).clip(min=0)
    agg["p90_dep_delay"] = np.percentile(dep_matrix, 90, axis=0).clip(min=0)

    agg["mean_turnaround_actual"]   = turn_matrix.mean(axis=0)
    agg["mean_turnaround_effectif"] = teff_matrix.mean(axis=0)
    agg["mean_slack_absorbed"]      = slack_matrix.mean(axis=0)

    # ── Marquage hub ──────────────────────────────────────────
    if hub_airport:
        if isinstance(hub_airport, list):
            agg["hub_affected"] = df_sched["origin"].isin(hub_airport).values
        else:
            agg["hub_affected"] = (df_sched["origin"].values == hub_airport)
    else:
        agg["hub_affected"] = False

    # ── Marquage slack ────────────────────────────────────────
    if slack_config and slack_config.get("minutes", 0) > 0:
        s_scope = slack_config.get("scope", "global")

        if s_scope == "global":
            agg["slack_affected"] = True

        elif s_scope == "window":
            w_start = float(slack_config.get("window_start", 0))
            w_end   = float(slack_config.get("window_end", 1440))
            agg["slack_affected"] = (
                (df_sched["dep_min"].values >= w_start) &
                (df_sched["dep_min"].values <= w_end)
            )

        elif s_scope == "aircraft":
            msn_val = slack_config.get("aircraft_msn", "")
            if isinstance(msn_val, list):
                agg["slack_affected"] = df_sched["aircraft_msn"].isin(msn_val).values
            else:
                agg["slack_affected"] = (df_sched["aircraft_msn"].values == msn_val)

        elif s_scope == "window_aircraft":
            w_start = float(slack_config.get("window_start", 0))
            w_end   = float(slack_config.get("window_end", 1440))
            msn_val = slack_config.get("aircraft_msn", "")
            if isinstance(msn_val, list):
                msn_mask = df_sched["aircraft_msn"].isin(msn_val).values
            else:
                msn_mask = (df_sched["aircraft_msn"].values == msn_val)
            agg["slack_affected"] = (
                msn_mask &
                (df_sched["dep_min"].values >= w_start) &
                (df_sched["dep_min"].values <= w_end)
            )

        elif s_scope == "flight":
            fid_val = slack_config.get("flight_id", "")
            if isinstance(fid_val, list):
                agg["slack_affected"] = df_sched["flight_id"].isin(fid_val).values
            else:
                agg["slack_affected"] = (df_sched["flight_id"].values == fid_val)

        elif s_scope == "airports":
            apt_val = slack_config.get("airports", [])
            if isinstance(apt_val, list):
                agg["slack_affected"] = df_sched["origin"].isin(apt_val).values
            else:
                agg["slack_affected"] = (df_sched["origin"].values == apt_val)

        else:
            agg["slack_affected"] = False
    else:
        agg["slack_affected"] = False


    return agg, np.array(otp_per_sim), np.array(prop_per_sim), all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import build_ref_dicts

    mt, gs, gsc, mm, mp, tt = build_ref_dicts(None)
    print("simulation.py: OK")
    print(
        f"Turnaround defaults: "
        f"DOM={mt['Domestique']} MC={mt['Moyen-courrier']} LC={mt['Long-courrier']}"
    )
    print(f"Turnaround table: {len(tt)} entrées")