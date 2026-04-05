# RAPPORT TECHNIQUE — Simulateur AIR-ROBUST de Propagation de Retards

---

## Architecture générale

Le projet est une application **Streamlit** composée de 4 fichiers Python dans `/src/` :

```
src/
├── app.py          → Interface utilisateur (UI Streamlit)
├── data_loader.py  → Chargement, validation, conversion des données
├── simulation.py   → Moteur Monte Carlo de propagation de retards
└── optimizer.py    → Optimiseur de substitution d'avion
```

---

## 1. `data_loader.py` — Chargement et préparation des données

### 1.1 Constantes et référentiels (lignes 10–31)

```python
REQUIRED_SCHEDULE = {"flight_id", "origin", "destination", ...}
REQUIRED_CREW     = {"crew_id", "flight_sequence", "rest_min_required"}
OPS_LEG_COLS      = {"FN_CARRIER", "FN_NUMBER", "DEP_AP_SCHED", "ARR_AP_SCHED"}
OPS_DOA_COLS      = {"CREW_ID", "ACTIVITY", "ACTIVITY_GROUP", "ORIGINE"}
DOM               = {"CMN", "RAK", "AGA", ...}  # Aéroports marocains domestiques
```

- `REQUIRED_SCHEDULE` / `REQUIRED_CREW` : colonnes minimales obligatoires dans les fichiers uploadés.
- `OPS_LEG_COLS` / `OPS_DOA_COLS` : colonnes clés pour **détecter automatiquement** le format opérationnel (LEG_FUTUR et DOA_PROGRAME). Si ces colonnes sont présentes, le système fait une conversion automatique.
- `DOM` : liste des codes IATA marocains. Sert à classifier un vol comme **Domestique** (si départ ET arrivée dans DOM).

---

### 1.2 Utilitaires horaires (lignes 35–46)

```python
def hhmm_to_minutes(t: str) -> int:
    h, m = map(int, str(t).strip().split(":"))
    return h * 60 + m
```

**Formule :** `minutes = heures × 60 + minutes`
Convertit `"06:45"` → `405` minutes depuis minuit.
Utilisé pour tous les calculs de propagation (qui travaillent en minutes entières).

```python
def minutes_to_hhmm(minutes: int) -> str:
    h = (int(minutes) // 60) % 24
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"
```

**Formule inverse :** Modulo 24 pour rester dans la plage horaire d'une journée.

---

### 1.3 Normalisation des dates (lignes 49–75)

```python
def _normalize_date_series(series):
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    # Format ISO (YYYY-MM-DD) → dayfirst=False
    # Autres formats (DD/MM/YYYY) → dayfirst=True
```

Implémente un choix conditionnel : si la date ressemble déjà au format ISO, on ne l'inverse pas (évite de confondre `"2024-03-01"` avec un format jour-premier).

---

### 1.4 Normalisation des numéros de vol (lignes 78–109)

```python
def _normalize_flight_token(value) -> str:
    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"[^A-Z0-9]", "", raw)
```

Nettoie les numéros de vol : supprime espaces et caractères spéciaux, garde uniquement `[A-Z0-9]`.
Gère les cas de valeurs numériques Excel (ex : `570.0` → `"570"`).

---

### 1.5 Lecture intelligente des fichiers Excel (lignes 114–186)

La fonction `_read_file` implémente une **sélection automatique de la meilleure feuille** :

```python
def _sheet_score(df) -> tuple[int, int, int, int]:
    known_hits = len(colset & (REQUIRED_SCHEDULE | REQUIRED_CREW | ...))
    return (known_hits, non_empty_cols, n_rows, data_size)
```

**Score à 4 niveaux :**
1. Nombre de colonnes reconnues parmi les sets obligatoires
2. Nombre total de colonnes non vides
3. Nombre de lignes
4. Taille totale `n_rows × n_cols`

La feuille avec le meilleur score lexicographique est choisie.
Pour chaque feuille, les lignes d'en-tête 0 à 7 sont testées.

---

### 1.6 Conversion LEG_FUTUR → schedule interne (lignes 215–348)

Fonction de transformation principale pour le format opérationnel Royal Air Maroc.

**Classification `route_type` (lignes 322–329) :**

```python
dom_mask  = df["origin"].isin(DOM) & df["destination"].isin(DOM)
long_mask = df["flight_duration_min"] >= 330   # >= 5h30

df["route_type"] = np.select(
    [dom_mask, long_mask & ~dom_mask],
    ["Domestique", "Long-courrier"],
    default="Moyen-courrier",
)
```

**Règle (priorité décroissante) :**
1. **Domestique** : départ ET arrivée dans le référentiel marocain `DOM`
2. **Long-courrier** : durée ≥ 330 min ET non-domestique
3. **Moyen-courrier** : tout le reste

**Calcul de la durée de vol (lignes 308–312) :**

```python
dep_min = hhmm → minutes
arr_min = hhmm → minutes
dur = (arr_day - dep_day) * 1440 + (arr_min - dep_min)
dur.loc[dur < 0] += 30 * 1440   # correction vol de nuit "invraisemblable"
```

**Formule :** `durée = (écart de jours × 1440) + (arrivée_min − départ_min)`
Le cas `dur < 0` est corrigé en ajoutant `30 × 1440` minutes pour couvrir les vols passant minuit.

---

### 1.7 Conversion DOA_PROGRAME → crew pairing interne (lignes 389–556)

La fonction `_convert_doa_programe` reconstitue les rotations équipage.

**Tri chronologique absolu :**

```python
def _epoch_minutes_lt(row) -> float:
    day_offset = (dt.normalize() - REF_DATE).days * 1440
    return day_offset + heure_lt.hour * 60 + heure_lt.minute
```

Convertit chaque ligne en une valeur de minutes depuis le `01/01/2000` pour un tri absolu multi-jours.

**Normalisation numéro de vol :**

```python
if tok.isdigit():
    tok = f"AT{tok}"   # "570" → "AT570"
```

Préfixe automatiquement le code IATA carrier `AT` (Air Maroc) si le token est purement numérique.

**Groupement par `(CREW_ID, DATE_LT)` :** Chaque groupe devient une ligne dans le DataFrame équipage avec :

| Colonne | Exemple |
|---------|---------|
| `flight_sequence` | `"AT570;AT571;AT572"` |
| `ORIGINE_SEQUENCE` | `"CMN;CDG;LHR"` |
| `DESTINATION_SEQUENCE` | `"CDG;LHR;CMN"` |
| `rest_sequence_min` | calculé par `_recompute_crew_constraints` |

---

### 1.8 Traitement interne du schedule (lignes 561–589)

```python
def _process_schedule(df):
    df["dep_min"] = df["scheduled_departure"].apply(hhmm_to_minutes)
    df["arr_min"] = df["scheduled_arrival"].apply(hhmm_to_minutes)

    mask_night = df["arr_min"] < df["dep_min"]
    df.loc[mask_night, "arr_min"] += 1440    # vol de nuit : +1440 min

    # Si flight_date présent : décalage absolu multi-jours
    j0 = dates.loc[valid].min()
    day_offset = (dates - j0).dt.days
    df["dep_min"] += day_offset * 1440
```

**Principe :** Tous les temps sont exprimés en **minutes depuis minuit du premier jour** de la simulation.
Un vol du 2ème jour aura `dep_min = 1440 + hhmm_to_minutes(dep)`.
Cela permet des comparaisons directes entre vols de jours différents.

---

### 1.9 Jointure 4-clés crew ↔ schedule (lignes 592–755)

La fonction `_recompute_crew_constraints` effectue la jointure la plus critique du système.

**Étape 1 — Explosion des séquences crew (1 ligne par vol) :**

```python
for i, fid_raw in enumerate(seq_fids):
    rows.append({
        "crew_id": cr["crew_id"],
        "flight_id": fid,
        "flight_date_norm": date_norm,
        "origin": org,
        "destination": dst,
        "seq_pos": i,
    })
```

**Étape 2 — Jointure LEFT sur 4 clés :**

```python
merged = seq.merge(
    sched[["flight_id", "flight_date_norm", "origin", "destination", "route_type"]],
    on=["flight_id", "flight_date_norm", "origin", "destination"],
    how="left",
)
```

Un vol équipage qui ne trouve pas son correspondant → `route_type = ""`.

**Étape 3 — Fallback ±1 jour pour vols overnight (lignes 670–701) :**

```python
for delta in [-1, 1]:
    tmp["flight_date_norm"] = (tmp["_dt"] + pd.Timedelta(days=delta)).dt.strftime(...)
```

Résout le cas d'un vol décollant à 23h50 (date = J) mais enregistré J+1 dans le schedule.

**Étape 4 — Calcul des repos requis par vol (lignes 714–724) :**

```python
merged["rest_leg_min"] = np.select(
    [route_norm == "domestique",
     route_norm == "moyen-courrier",
     route_norm == "long-courrier"],
    [30.0, 45.0, 60.0],
    default=45.0,
)
```

**Règle métier :** Domestique = 30 min | Moyen-courrier = 45 min | Long-courrier = 60 min.

**Étape 5 — Agrégation par `crew_id` :**

```python
crew_agg = (
    merged.sort_values(["crew_id", "seq_pos"])
    .groupby("crew_id")
    .agg(
        route_sequence_type=("route_type", lambda s: ";".join(s)),
        rest_sequence_min=("rest_leg_min", lambda s: ";".join(str(int(v)) for v in s)),
    )
)
```

Produit par exemple : `rest_sequence_min = "30;45;60"` pour une rotation DOM → MC → LC.

---

### 1.10 Paramètres de référence — `build_ref_dicts` (lignes 953–1020)

Valeurs par défaut calibrées :

```python
min_turnaround = {
    "Domestique":     30.0,
    "Moyen-courrier": 45.0,
    "Long-courrier":  60.0,
    "DEFAULT":        45.0,
}
gamma_shape        = 2.5
gamma_scale        = 15.0
markov_matrix      = [[0.65, 0.28, 0.07],
                      [0.25, 0.52, 0.23],
                      [0.08, 0.28, 0.64]]
markov_multipliers = [1.0, 1.45, 2.30]
```

Ces valeurs peuvent être surchargées par un fichier CSV de référence uploadé par l'utilisateur via les colonnes `param_type` / `value`.

---

## 2. `simulation.py` — Moteur Monte Carlo

### 2.1 Résolution du turnaround minimum (lignes 45–78)

```python
def _resolve_min_turnaround(min_turnaround, route_type, aircraft_type,
                            turnaround_table=None, airport=None):
```

**Hiérarchie de priorité (5 niveaux) :**

| Priorité | Source | Clé |
|----------|--------|-----|
| 1 | CSV turnaround | `(airport, aircraft_type)` — correspondance exacte |
| 2 | CSV turnaround | `(airport, "DEFAULT")` — aéroport connu, type inconnu |
| 3 | CSV turnaround | `("DEFAULT", "DEFAULT")` — fallback global CSV |
| 4 | Paramètre interne | `min_turnaround[route_type]` — par type de route |
| 5 | Valeur ultime | `min_turnaround["DEFAULT"]` → 45 min |

---

### 2.2 Chaîne de Markov — turnaround stochastique (lignes 97–115)

```python
def markov_turnaround(..., n_steps=3, initial_state=0):
    base  = _resolve_min_turnaround(...)    # turnaround minimum réglementaire
    state = initial_state                   # état initial = 0 (Normal)
    for _ in range(n_steps):               # 3 pas de transition
        state = np.random.choice([0, 1, 2], p=markov_matrix[state])
    return base * markov_multipliers[state], state
```

**Modèle à 3 états :**

| État | Nom | Multiplicateur | Signification |
|------|-----|---------------|---------------|
| 0 | Normal | × 1.00 | Turnaround nominal |
| 1 | Alerte | × 1.45 | Retard modéré (opérations lentes) |
| 2 | Bloqué | × 2.30 | Blocage sévère (incident, maintenance) |

**Matrice de transition :**

```
                  → Normal   Alerte   Bloqué
De Normal    :      65%       28%       7%
De Alerte    :      25%       52%      23%
De Bloqué    :       8%       28%      64%
```

**Résultat :** `turnaround_réel = turnaround_minimum × multiplicateur[état_final_après_3_pas]`

---

### 2.3 Simulation d'une journée — `simulate_once` (lignes 118–452)

#### Retards initiaux — loi Gamma (lignes 185–189)

```python
shapes = np.where(hub_mask, gamma_shape * hub_factor, gamma_shape)
initial_delays = np.array([
    np.random.gamma(shape=float(shapes[i]), scale=gamma_scale)
    for i in range(n)
])
```

**Loi Gamma(`shape=2.5, scale=15`) :**
- Moyenne = `shape × scale` = `2.5 × 15` = **37.5 min**
- Distribution asymétrique : majorité des vols < 40 min, mais queue longue (quelques vols très retardés)
- Les vols depuis un hub reçoivent une forme amplifiée : `shape × hub_factor` (hub_factor = 1.2)

#### Indexation multi-niveaux pour la jointure équipage (lignes 202–261)

Trois dictionnaires d'index sont construits avec une priorité décroissante :

```python
idx_by_fkey_date_apt[(fkey, dkey, org, dst)] = label  # 4 clés (prioritaire)
idx_by_fkey_date[(fkey, dkey)]               = label  # 2 clés (fallback)
idx_by_fkey[fkey]                            = label  # 1 clé  (dernier recours)
```

#### Boucle de propagation (lignes 273–442)

Pour chaque vol trié par `(date, dep_min)` :

**1. Turnaround effectif avec absorption de slack :**

```python
min_turn      = _resolve_min_turnaround(...)
turn_effectif = max(min_turn, turn_dur - slack_minutes)
```

La marge slack réduit le turnaround aléatoire, avec plancher au minimum réglementaire.

**2. Contrainte avion :**

```python
arr_prev       = df.at[prev_pos, "arr_actual"]    # arrivée réelle du vol précédent
avion_pret     = arr_prev + turn_effectif           # disponibilité de l'avion
earliest_avion = max(row["dep_min"], avion_pret)   # au plus tôt : départ prévu
```

**3. Contrainte équipage — jointure 4-clés :**

```python
lookup_key = (row_fkey, flight_date_key, row_org, row_dst)
candidates = crew_idx.get(lookup_key, [])

for rec in candidates:
    if pos_seq <= 0: continue          # 1er vol → pas de contrainte amont
    arr_crew      = df.at[prev_label, "arr_actual"]
    earliest_crew = max(earliest_crew, arr_crew)
```

Fallback overnight : si la clé 4-clés échoue avec la date courante, on essaie la date de l'équipage (qui peut être J-1).

**4. Départ réel — formule centrale de propagation :**

```
dep_real = max(dep_prévu + retard_initial,   ← retard opérationnel initial
               arr_prev + turn_effectif,      ← contrainte avion
               arr_crew)                      ← contrainte équipage
```

**C'est la formule fondamentale.** Un retard se propage si l'avion ou l'équipage n'est pas prêt à temps.

**5. Facteur hub (amplification) :**

```python
retard_accumule = dep_real - row["dep_min"]
dep_real = row["dep_min"] + (retard_accumule * hub_factor)   # hub_factor = 1.2
```

Amplifie le retard accumulé d'un facteur 1.2 pour les vols opérés depuis un hub.

**6. Indicateurs finaux :**

```python
df["dep_delay"] = (df["dep_actual"] - df["dep_min"]).clip(lower=0)
df["arr_delay"] = (df["arr_actual"] - df["arr_min"]).clip(lower=0)
df["on_time"]   = df["dep_delay"] <= otp_threshold    # seuil OTP = 15 min
```

---

### 2.4 Monte Carlo — `run_monte_carlo` (lignes 455–603)

Lance N simulations (N = 200 par défaut, N = 1 en mode manuel).

**Construction des matrices résultats :**

```python
arr_matrix  = np.array([r["arr_delay"].values for r in all_results])   # shape (N, M)
dep_matrix  = np.array([r["dep_delay"].values for r in all_results])
```

Chaque ligne = une simulation, chaque colonne = un vol.

**Agrégation statistique par vol :**

```python
agg["mean_arr_delay"] = arr_matrix.mean(axis=0)               # retard moyen sur N simulations
agg["p95_arr_delay"]  = np.percentile(arr_matrix, 95, axis=0) # percentile 95 (pire cas)
agg["otp_rate"]       = (dep_matrix <= otp_threshold).mean(axis=0) * 100  # % ponctuel
agg["p80_dep_delay"]  = np.percentile(dep_matrix, 80, axis=0).clip(min=0)
agg["p90_dep_delay"]  = np.percentile(dep_matrix, 90, axis=0).clip(min=0)
```

**Taux de propagation :**

```python
total_initial = result["initial_delay"].sum()
total_final   = result["arr_delay"].sum()
prop = total_final / total_initial   # ratio > 1 = amplification
```

**Interprétation :** Un ratio > 1 signifie que les retards se sont amplifiés en se propageant sur la journée. Un ratio < 1 indique une absorption (slack, turnaround absorbé).

---

## 3. `optimizer.py` — Moteur de substitution d'avion

### 3.1 Chargement des autorisations (lignes 85–165)

```python
def load_aircraft_authorizations(filepath: str) -> dict:
    # Format 1 — par MSN (prioritaire) :
    #   aircraft_msn, aircraft_type, authorized_airports
    #   MSN001,A320,CDG;LHR;AMS;FCO
    #
    # Format 2 — par type :
    #   aircraft_type, authorized_airports
    #   A320,CDG;LHR;AMS;FCO;MAD;BCN
    return {"by_msn": {...}, "by_type": {...}}
```

Auto-détecte le séparateur (virgule, point-virgule, espace).
Construit deux index : un par MSN (individuel) et un par type (générique).

---

### 3.2 Vérification d'autorisation aéroport (lignes 168–213)

```python
def is_airport_authorized(msn, aircraft_type, airport, authorizations, fallback_allowed):
    # Priorité 1 : vérification par MSN
    if msn in authorizations["by_msn"]:
        return airport in authorizations["by_msn"][msn], raison
    # Priorité 2 : vérification par type d'avion
    if aircraft_type in authorizations["by_type"]:
        return airport in authorizations["by_type"][aircraft_type], raison
    # Priorité 3 : fallback depuis le planning (aéroports déjà desservis)
    if fallback_allowed:
        return airport in fallback_allowed, raison
    # Mode permissif : aucune restriction définie
    return True, "Pas de restriction"
```

---

### 3.3 Structures de données (lignes 243–303)

**`SwapCandidate`** : résultat de l'évaluation d'un avion A2 candidat.

Champs clés :
- `delay_saved` : minutes gagnées grâce au swap
- `new_delay` : retard résiduel après swap
- `score` : score composite (voir §3.5)
- `feasible` : True si toutes les vérifications passent
- `infeasibility_reason` : raison d'un rejet

**Propriété calculée `gain_pct` :**

```python
@property
def gain_pct(self) -> float:
    orig_delay = self.dep_original - self.dep_scheduled
    return max(0.0, self.delay_saved / orig_delay * 100)
```

---

### 3.4 État de la flotte (lignes 308–362)

```python
def _build_fleet_state(df_sched, sim_result, before_dep_min, before_flight_date):
    for msn, grp in merged.groupby("aircraft_msn"):
        grp_before = grp[abs_dep < ref_abs]   # vols AVANT le vol retardé
        last = grp_before.iloc[-1]             # dernier vol connu de cet avion
        fleet[msn] = {
            "last_arr_actual":  last["arr_actual"],     # arrivée réelle simulée
            "last_destination": last["destination"],    # localisation courante
            "type":             last["aircraft_type"],
        }
```

Construit une **photo instantanée de la flotte** au moment du vol retardé.
Pour chaque avion : où il se trouve et quand il arrive.

---

### 3.5 Évaluation d'un candidat A2 (lignes 523–742)

Pipeline d'évaluation en 6 étapes séquentielles :

#### Étape 0 — Autorisation aéroport destination

```python
is_authorized, auth_reason = is_airport_authorized(a2_msn, a2_type, destination, ...)
if not is_authorized:
    return SwapCandidate(feasible=False, infeasibility_reason=auth_reason)
```

#### Étape 1 — Disponibilité A2 et calcul du nouveau départ

```python
a2_ready  = a2_last_arr + turnaround       # quand A2 est opérationnellement disponible
new_dep   = max(dep_scheduled, a2_ready)   # nouveau départ : au plus tôt l'heure prévue
new_arr   = new_dep + flight_duration
new_delay = max(0.0, new_dep - dep_scheduled)

if a2_ready > dep_original + 1:            # A2 arrive plus tard que A1 retardé → inutile
    return SwapCandidate(feasible=False)
```

#### Étape 2 — Vérification géographique et temporelle de la chaîne

Après le swap, A2 prend les vols de A1 et A1 prend les vols de A2.
Il faut vérifier que les connexions géographiques et temporelles sont cohérentes.

```python
# Chaîne A1 (reprise par A2) :
first_origin = a1_downstream_flights[0]["origin"]
if first_origin != dest_norm:              # A2 doit arriver là où commence la chaîne A1
    feasible = False

a2_ready_for_chain = new_arr + turn
if a2_ready_for_chain > first_dep + 1:    # A2 doit être prêt avant le premier vol de la chaîne
    feasible = False

# Chaîne A2 (reprise par A1) :
if first_origin_a2 != origin_norm:        # A1 doit être là où commence la chaîne A2
    feasible = False
```

#### Étape 3 — Vérification équipage

```python
crew_ok, crew_note = _check_crew(flight_id, flight_date, new_dep, ...)
if not crew_ok:
    feasible = False
```

Vérifie que l'équipage affecté au vol est disponible à `new_dep` après son repos réglementaire.

#### Étape 4 — Score composite

```python
delay_saved          = max(0.0, dep_original - new_dep)
n_downstream         = len(a1_downstream) + len(a2_downstream)
a1_chain_delay_saved = delay_saved * n_downstream * 0.6    # 60% du gain propagé sur la chaîne

otp_bonus    = 50.0  if new_delay <= otp_threshold else 0.0  # bonus si vol redevient ponctuel
chain_bonus  = min(a1_chain_delay_saved * 0.3, 100.0)         # bonus chaîne (plafonné à 100)
margin_bonus = min(a2_conflict_margin * 0.1, 30.0)             # bonus marge libre A2 (plafonné à 30)

score = delay_saved + otp_bonus + chain_bonus + margin_bonus   # si faisable
score = -9999.0 + delay_saved                                   # si infaisable (trié en dernier)
```

**Interprétation :**
- `delay_saved` : gain principal en minutes
- `otp_bonus` : prime de 50 pts si le vol redevient OTP (< 15 min de retard)
- `chain_bonus` : valorise les swaps qui libèrent plusieurs vols aval
- `margin_bonus` : favorise les swaps laissant A2 avec de la marge pour son prochain vol

---

### 3.6 Fonction principale `find_swap_for_flight` (lignes 747–850)

**Filtre initial des candidats A2 :**

```python
for a2_msn, fstate in fleet_state.items():
    if a2_msn == a1_msn:                        continue  # pas l'avion lui-même
    if fstate["type"] != a1_type:               continue  # même type obligatoire
    if fstate["last_destination"] != origin:    continue  # doit être sur l'aéroport
```

Seuls les avions du **même type** et **déjà présents sur l'aéroport de départ** sont candidats.

**Sélection du meilleur candidat :**

```python
result.all_candidates = sorted(candidates, key=lambda c: -c.score)
feasible_cands = [c for c in candidates if c.feasible]
result.best = max(feasible_cands, key=lambda c: c.score)  # meilleur faisable
# Si aucun faisable : on retourne quand même le moins mauvais
```

---

### 3.7 Optimiseur global `run_swap_optimizer` (lignes 855–952)

**Sélection de la simulation représentative :**

```python
otp_per_sim = [r["on_time"].mean() for r in all_results]
median_idx  = int(np.argsort(otp_per_sim)[len(otp_per_sim) // 2])
sim_rep     = all_results[median_idx]   # simulation médiane = la plus représentative
```

On ne prend ni la meilleure ni la pire simulation, mais la médiane pour une évaluation réaliste.

**Sélection des vols à optimiser :**

```python
delayed = df_agg[df_agg["mean_arr_delay"] >= delay_threshold]   # seuil = 30 min
```

Seuls les vols avec un **retard moyen ≥ 30 min** sur toutes les simulations sont traités.

---

## 4. `app.py` — Interface utilisateur Streamlit

### 4.1 Configuration visuelle

- **Police** : Barlow + Barlow Condensed (Google Fonts)
- **Couleurs** : rouge Royal Air Maroc `#C8102E`, bleu marine `#0d1b2a`
- **Sidebar** : fond blanc avec bordure rouge, labels compacts `0.77rem`

### 4.2 Fichiers uploadés (sidebar)

| Fichier | Format | Rôle |
|---------|--------|------|
| Schedule | LEG_FUTUR ou CSV/Excel interne | Planning des vols |
| Équipage | DOA_PROGRAME ou CSV/Excel interne | Rotations équipages |
| Référence | CSV `param_type,value` | Paramètres de simulation |
| Turnaround | CSV `airport,aircraft_type,min_turnaround_min` | Temps de rotation par aéroport |
| Autorisations | CSV `aircraft_msn,authorized_airports` | Restrictions opérationnelles |

### 4.3 Pipeline d'exécution

```
load_uploaded_data()          → df_sched, df_crew, df_ref
    ↓
build_ref_dicts(df_ref)       → min_turnaround, gamma_shape, gamma_scale,
                                 markov_matrix, markov_multipliers, turnaround_table
    ↓
run_monte_carlo(...)          → df_agg, otp_per_sim, prop_per_sim, all_results
    ↓
run_swap_optimizer(...)       → df_swap_results, swap_list
```

### 4.4 Visualisations

- **Histogrammes Plotly** : distribution des retards par vol, OTP par simulation
- **Graphe NetworkX** : réseau de propagation (nœuds = vols, arêtes = dépendances avion/équipage)
- **Tableaux interactifs** : résultats détaillés par vol avec filtres

---

## 5. Synthèse des formules clés

| Formule | Expression | Fichier / Ligne |
|---------|-----------|----------------|
| **Départ réel** | `dep_real = max(dep_prévu + retard_initial, arr_prev + turn_effectif, arr_crew)` | simulation.py:422 |
| **Turnaround effectif** | `turn_effectif = max(min_turn, turn_markov − slack)` | simulation.py:356 |
| **Turnaround Markov** | `turnaround = base × multiplicateur[état_final]` | simulation.py:115 |
| **Retard initial** | `Gamma(shape=2.5, scale=15)` ou injection manuelle | simulation.py:187 |
| **Taux de propagation** | `taux = Σ arr_delay / Σ initial_delay` | simulation.py:508 |
| **Score swap** | `delay_saved + 50·[OTP] + min(0.3·chain, 100) + min(0.1·margin, 30)` | optimizer.py:726 |
| **Durée vol** | `(arr_day − dep_day) × 1440 + (arr_min − dep_min)` | data_loader.py:310 |
| **Repos équipage** | `DOM=30 min, MC=45 min, LC=60 min` | data_loader.py:716 |
| **Heure → minutes** | `h × 60 + m` | data_loader.py:38 |
| **Hub factor** | `dep_real = dep_prévu + retard_accumulé × hub_factor` | simulation.py:439 |

---

## 6. Jointures et clés de matching

| Jointure | Clés | Fallback | Fichier |
|----------|------|---------|---------|
| Crew ↔ Schedule (contrainte propagation) | `(flight_num, date, origin, destination)` | → `(num, date)` → `(num)` | simulation.py:202–261 |
| Crew ↔ Schedule (repos requis) | `(flight_id, flight_date, origin, destination)` | ±1 jour | data_loader.py:662–701 |
| Crew ↔ Équipage (vérification swap) | `(flight_num, date, origin, destination)` | par MSN, par date | optimizer.py:447–493 |
| Numéro vol (tolérance préfixe) | extrait digits : `"AT570"` → `"570"` | retourne raw si non-numérique | simulation.py:20–28 |
| Date (tolérance format) | ISO `YYYY-MM-DD` ou `dayfirst=True` | NaT si invalide | data_loader.py:49–61 |

---

*Rapport généré le 2026-04-04 — Projet AIR-ROBUST / DelayPropagation*