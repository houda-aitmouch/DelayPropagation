# RAPPORT D'EXPLICATION DU CODE — AIR-ROBUST / DelayPropagation
**Préparé pour la présentation à l'encadrant**
Date : 05 Avril 2026

---

## TABLE DES MATIÈRES

1. [Architecture générale du projet](#1-architecture-générale-du-projet)
2. [main.py — Application de démonstration](#2-mainpy--application-de-démonstration)
3. [generate_data.py — Générateur de données fictives](#3-generate_datapy--générateur-de-données-fictives)
4. [src/simulation.py — Moteur de simulation Monte Carlo](#4-srcsimulationpy--moteur-de-simulation-monte-carlo)
5. [src/data_loader.py — Chargement et validation des données](#5-srcdata_loaderpy--chargement-et-validation-des-données)
6. [src/optimizer.py — Optimiseur d'échanges d'avions](#6-srcoptimizerpy--optimiseur-déchanges-davions)
7. [src/app.py — Interface web Streamlit principale](#7-srcapppy--interface-web-streamlit-principale)
8. [Flux de données global](#8-flux-de-données-global)

---

## 1. ARCHITECTURE GÉNÉRALE DU PROJET

```
DelayPropagation/
├── main.py               ← Application démo simple (graphe réseau)
├── generate_data.py      ← Génère les données fictives RAM
├── src/
│   ├── app.py            ← Interface web principale (2 259 lignes)
│   ├── data_loader.py    ← Lecture et nettoyage des fichiers
│   ├── simulation.py     ← Moteur Monte Carlo + Markov
│   └── optimizer.py      ← Algorithme d'échange d'avions
├── data/                 ← Fichiers de données générés
└── requirements.txt      ← Dépendances Python
```

**Rôle de chaque module :**
- `main.py` : démonstrateur simplifié avec graphe NetworkX (utilisé pour les tests initiaux)
- `generate_data.py` : crée des fichiers Excel/CSV réalistes au format Royal Air Maroc
- `src/app.py` : interface utilisateur Streamlit complète (tableau de bord)
- `src/data_loader.py` : lit, détecte le format, et nettoie les données uploadées
- `src/simulation.py` : cœur du simulateur (propagation des retards, Markov, Monte Carlo)
- `src/optimizer.py` : cherche des avions de remplacement pour réduire les retards

**Technologies utilisées :**
| Bibliothèque | Rôle |
|---|---|
| `streamlit` | Interface web interactive |
| `pandas` | Manipulation de tableaux de données |
| `numpy` | Calculs numériques (distributions, matrices) |
| `plotly` | Graphiques interactifs |
| `networkx` | Graphe de réseau aéroportuaire |
| `scipy` | Distributions statistiques |

---

## 2. `main.py` — Application de démonstration

> **But :** Application Streamlit simplifiée qui modélise la propagation d'un retard initial à travers un réseau d'aéroports, en utilisant un graphe orienté.

---

### Lignes 1–8 : Imports

```python
from __future__ import annotations   # Ligne 1 : permet les annotations de type modernes
from pathlib import Path              # Ligne 3 : manipulation de chemins de fichiers
import networkx as nx                 # Ligne 5 : bibliothèque de graphes (réseau d'aéroports)
import pandas as pd                   # Ligne 6 : manipulation de données tabulaires
import plotly.express as px           # Ligne 7 : création de graphiques interactifs
import streamlit as st                # Ligne 8 : framework d'interface web
```

---

### Lignes 10–20 : Configuration et colonnes requises

```python
DATA_FILE = Path("data/flights.csv")   # Chemin vers le fichier de données CSV
REQUIRED_COLUMNS = {                    # Ensemble des colonnes que le CSV DOIT contenir
    "flight_id",          # Identifiant unique du vol
    "origin",             # Code IATA de l'aéroport de départ
    "destination",        # Code IATA de l'aéroport d'arrivée
    "departure_time",     # Heure de départ programmée
    "weather_index",      # Indice météo (facteur externe)
    "congestion_index",   # Indice de congestion aéroportuaire
    "turnaround_slack_min", # Marge de rotation en minutes
    "delay_minutes",      # Retard en minutes
}
```

**Pourquoi un ensemble (`set`) ?** Pour vérifier rapidement si une colonne manque : `REQUIRED_COLUMNS - set(df.columns)` donne les colonnes absentes en O(1).

---

### Lignes 23–32 : Fonction `load_dataset()`

```python
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)                          # Ligne 24 : lit le fichier CSV en DataFrame
    missing = REQUIRED_COLUMNS - set(df.columns)    # Ligne 25 : calcule les colonnes manquantes
    if missing:                                      # Ligne 26 : s'il en manque...
        raise ValueError(f"Missing: {sorted(missing)}")  # Ligne 27 : erreur explicite

    df["departure_time"] = pd.to_datetime(          # Ligne 29 : convertit la colonne en DateTime
        df["departure_time"], errors="coerce"       # "coerce" = remplace les invalides par NaT
    )
    df = df.dropna(subset=["departure_time"]).copy()  # Ligne 30 : supprime les lignes sans heure
    df["delay_minutes"] = pd.to_numeric(            # Ligne 31 : force les retards en nombre
        df["delay_minutes"], errors="coerce"
    ).fillna(0.0)                                   # Remplace les NaN par 0
    return df                                       # Retourne le DataFrame nettoyé
```

**Concept clé :** `errors="coerce"` est une stratégie défensive : au lieu de planter sur une valeur non convertible, on la transforme en valeur nulle puis on la gère.

---

### Lignes 35–50 : Fonction `build_airport_graph()`

```python
def build_airport_graph(df: pd.DataFrame) -> nx.DiGraph:
    grouped = (
        df.groupby(["origin", "destination"], as_index=False)  # Groupe par paire (départ, arrivée)
        .agg(
            avg_delay=("delay_minutes", "mean"),   # Calcule le retard moyen pour chaque route
            flights=("flight_id", "count")         # Compte le nombre de vols sur cette route
        )
        .sort_values(["origin", "destination"])    # Trie pour la reproductibilité
    )

    graph = nx.DiGraph()                           # Crée un graphe ORIENTÉ (A→B ≠ B→A)
    for row in grouped.itertuples(index=False):    # Parcourt chaque route
        graph.add_edge(
            row.origin,                            # Nœud source = aéroport de départ
            row.destination,                       # Nœud cible = aéroport d'arrivée
            avg_delay=float(row.avg_delay),        # Attribut sur l'arête : retard moyen
            flights=int(row.flights),              # Attribut : nombre de vols
        )
    return graph
```

**Concept clé :** Un `DiGraph` (graphe dirigé) est parfait pour modéliser un réseau aérien car CMN→CDG ≠ CDG→CMN (les retards peuvent être asymétriques).

---

### Lignes 53–77 : Fonction `propagate_delay()`

C'est la **fonction centrale de l'algorithme** : elle simule comment un retard initial à un aéroport se propage à travers le réseau.

```python
def propagate_delay(graph, source, initial_delay, decay, max_hops):
    if source not in graph.nodes:          # Vérifie que l'aéroport existe dans le graphe
        raise ValueError(...)

    queue = [(source, max(initial_delay, 0.0), 0)]  # File d'attente BFS : (aéroport, retard, profondeur)
    best_delay = {source: max(initial_delay, 0.0)}  # Dictionnaire : meilleur retard connu par aéroport
```

**Algorithme BFS (Breadth-First Search) :**

```python
    while queue:                                       # Tant qu'il reste des nœuds à traiter
        airport, delay_value, depth = queue.pop(0)    # Prend le premier élément (FIFO)
        if depth >= max_hops or delay_value <= 0:      # Arrêt : trop loin ou retard nul
            continue

        for neighbor in graph.successors(airport):    # Pour chaque aéroport atteignable
            edge = graph.get_edge_data(airport, neighbor)

            # FORMULE DE PROPAGATION :
            # - delay_value * decay    → le retard se dissipe à chaque saut (facteur < 1)
            # - edge["avg_delay"]*0.15 → la route elle-même contribue 15% de son retard moyen
            next_delay = delay_value * decay + float(edge.get("avg_delay", 0.0)) * 0.15

            if next_delay <= best_delay.get(neighbor, -1.0):  # On garde seulement le pire cas
                continue
            best_delay[neighbor] = next_delay          # Mise à jour du retard maximum
            queue.append((neighbor, next_delay, depth + 1))  # Ajoute à la file
```

**Exemple concret :**
- Retard initial à CMN = 60 min, decay = 0.65
- Propagation à CDG : `60 × 0.65 + retard_moyen_route × 0.15 ≈ 39 + qqs min`
- Propagation ensuite de CDG vers ses voisins avec le retard réduit

---

### Lignes 80–88 : Fonction `render_metrics()`

```python
def render_metrics(df: pd.DataFrame) -> None:
    mean_delay = df["delay_minutes"].mean()          # Moyenne de tous les retards
    p90_delay  = df["delay_minutes"].quantile(0.90)  # Percentile 90 : 90% des vols ont moins que ça
    impacted   = int((df["delay_minutes"] >= 15).sum())  # Vols avec >= 15 min de retard

    c1, c2, c3 = st.columns(3)    # Crée 3 colonnes côte à côte dans l'interface
    c1.metric("Delay moyen", f"{mean_delay:.1f} min")
    c2.metric("P90 delay",   f"{p90_delay:.1f} min")
    c3.metric("Vols >= 15 min", f"{impacted}")
```

**Pourquoi le P90 ?** En aviation, on utilise les percentiles pour mesurer la fiabilité : "90% des vols partent avec moins de X minutes de retard" est un indicateur plus robuste que la moyenne seule (car quelques vols très retardés ne font pas exploser la statistique).

---

### Lignes 119–156 : Fonction `run_app()`

C'est le **point d'entrée** de l'application Streamlit.

```python
def run_app() -> None:
    st.set_page_config(page_title="DelayPropagation", layout="wide")  # Configuration page web
    st.title("DelayPropagation - Analyse de propagation des retards")

    st.sidebar.header("Paramètres")
    data_path = Path(st.sidebar.text_input("Fichier CSV", str(DATA_FILE)))  # Champ texte sidebar

    if not data_path.exists():      # Si le fichier n'existe pas, on affiche une erreur
        st.error("Dataset introuvable...")
        return

    df    = load_dataset(data_path)      # Chargement des données
    graph = build_airport_graph(df)      # Construction du graphe
    airports = sorted(graph.nodes)       # Liste triée des aéroports

    # Contrôles interactifs dans la barre latérale :
    source        = st.sidebar.selectbox("Aéroport source", airports)  # Sélecteur d'aéroport
    initial_delay = st.sidebar.slider("Retard initial (min)", 5, 180, 45, 5)     # 5 à 180 min
    decay         = st.sidebar.slider("Facteur de dissipation", 0.30, 0.95, 0.65, 0.05)  # 0.3 à 0.95
    max_hops      = st.sidebar.slider("Nombre max de correspondances", 1, 8, 4, 1)

    propagated = propagate_delay(graph, source, float(initial_delay), float(decay), int(max_hops))

    render_metrics(df)          # Affiche les KPIs
    render_charts(df, propagated)  # Affiche les graphiques

    st.subheader("Données de propagation")
    st.dataframe(propagated, use_container_width=True, hide_index=True)  # Tableau résultat

if __name__ == "__main__":
    run_app()   # Lance l'app si exécuté directement
```

---

## 3. `generate_data.py` — Générateur de données fictives

> **But :** Générer des données réalistes au format opérationnel de Royal Air Maroc (RAM) pour pouvoir tester le simulateur sans données confidentielles réelles.

---

### Lignes 1–10 : Docstring et imports

```python
"""
AIR-ROBUST — Génération de données fictives
Produit 3 fichiers dans data/:
  1. LEG_FUTUR_011125_301125.xlsx   (vols - format opérationnel RAM)
  2. DOA_PROGRAME_011125_301125.xlsx (équipages - format opérationnel RAM)
  3. reference_params_RAM.csv        (paramètres simulation)
"""
import random       # Génération aléatoire (choix d'avions, états de vol)
import pandas as pd # Création de DataFrames et export Excel
import numpy as np  # Graine aléatoire reproductible
from datetime import datetime, timedelta  # Manipulation des dates
import os           # Création du répertoire data/
```

---

### Lignes 17–18 : Graines aléatoires

```python
random.seed(42)    # Graine fixe pour reproductibilité : les mêmes données à chaque exécution
np.random.seed(42) # Même principe pour NumPy
```

**Pourquoi c'est important ?** Un simulateur doit être reproductible pour être testable. Avec la même graine, on obtient toujours les mêmes données fictives.

---

### Lignes 23–36 : Définition de la flotte RAM

```python
FLEET = {
    "CNROH": ("73H", "AT", "000", "JY159"),  # immatriculation: (sous-type, opérateur, version, code)
    "CNROI": ("73H", "AT", "000", "JY160"),
    # ... 21 avions au total
    "CNRGB": ("788", "AT", "000", "JY180"),  # Boeing 787-8
    "CNRGE": ("789", "AT", "000", "JY185"),  # Boeing 787-9
    "CNCOB": ("AT7", "RXP", "001", "JY70"),  # ATR 72 (vols domestiques)
    "CNRGT": ("E90", "AT", "000", "JY190"),  # Embraer 190
}
```

**Format réel RAM :** Ces immatriculations correspondent au format utilisé dans les systèmes opérationnels de Royal Air Maroc (CN = Maroc, puis code alphanumérique).

---

### Lignes 38–39 : Correspondances de types

```python
SUBTYPE_TO_AC   = {"73H":"738", "7M8":"7M8", "788":"788", ...}  # Sous-type → code A/C
SUBTYPE_TO_QUAL = {"73H":"B737", "788":"B737,B787", "E90":"B737,B787,E90"}  # Qualifications pilote
```

**Logique métier :** Un pilote qualifié sur 787 l'est aussi sur 737 (qualification croisée), c'est pourquoi la 789 a `"B737,B787"`.

---

### Lignes 41–67 : Table des routes

```python
ROUTES = [
    # (origine, destination, num_aller, num_retour, heure_dep_h, heure_dep_min, durée_min)
    ("CMN", "RAK",  409,  410,  7, 30,  55),   # Casablanca → Marrakech, 7h30, 55 min
    ("CMN", "AGA",  417,  418, 10,  0,  65),   # Casablanca → Agadir, 10h00, 65 min
    ("CMN", "CDG",  700,  701,  6,  0, 175),   # Casablanca → Paris CDG, 6h00, 2h55
    ("CMN", "JFK",  200,  201,  1,  0, 480),   # Casablanca → New York, 1h00, 8h00
    # ... 47 routes en tout
]
```

---

### Lignes 69–75 : Catégorisation des avions et aéroports

```python
DOM_REGS  = ["CNCOB", "CNCOC", "CNCOD", "CNCOE", "CNRGT", "CNRGU"]  # ATR et E90 pour domestique
EURO_REGS = ["CNROH", ...]   # Boeing 737 pour court/moyen-courrier Europe
MAX_REGS  = ["CNMAX1", ...]  # Boeing 737 MAX pour moyen-courrier
WIDE_REGS = ["CNRGB", ...]   # Boeing 787 pour long-courrier
DOM_AP    = {"CMN","RAK","AGA","TNG","FEZ","OUD","NDR","EUN","VIL"}  # Aéroports domestiques
LONG_DEST = {"JFK","MTL","IAD","DXB","JED","DOH"}  # Destinations long-courrier
LEG_STATES = ["NEW"]*9 + ["CNL"]  # 90% de vols opérés, 10% annulés
```

---

### Lignes 82–86 : Fonction `pick_reg()`

```python
def pick_reg(dur, orig, dest):
    """Sélectionne l'immatriculation d'avion appropriée selon la durée et le type de route."""
    if dur >= 300:                                    # Long-courrier (> 5h) : Boeing 787
        return random.choice(WIDE_REGS)
    if orig in DOM_AP and dest in DOM_AP and dur <= 90:  # Domestique court : ATR ou Embraer
        return random.choice(DOM_REGS)
    if dur >= 200:                                    # Moyen-long : 787 MAX ou 737 récents
        return random.choice(MAX_REGS + EURO_REGS[-2:])
    return random.choice(EURO_REGS + MAX_REGS[:2])   # Standard : 737
```

---

### Lignes 88–96 : Fonctions utilitaires de formatage

```python
def fday(dt):   return dt.strftime("%Y%m%d")    # Format YYYYMMDD  ex: "20251101"
def fslash(dt): return dt.strftime("%d/%m/%Y")  # Format JJ/MM/AAAA ex: "01/11/2025"
def fhm(h,m):   return f"{h:02d}{m:02d}"        # HHMM sans séparateur ex: "0730"
def fhmc(h,m):  return f"{h:02d}:{m:02d}"       # HH:MM avec séparateur ex: "07:30"
def blk(d):     return f"{d//60:02d}:{d%60:02d}" # Durée en HH:MM ex: blk(175) = "02:55"

def addm(h, m, mins):
    """Ajoute 'mins' minutes à l'heure (h,m), retourne (nouvelle_h, nouveau_m, jours_supplémentaires)."""
    t = h*60 + m + mins   # Total en minutes
    d = t // 1440          # Nombre de jours dépassés (1440 = 24h × 60min)
    t = t % 1440           # Minutes restantes dans la journée
    return t//60, t%60, d  # Heure, minutes, jours de dépassement
```

---

### Lignes 98–208 : Fonction principale `generate_all()`

Cette fonction génère les 3 fichiers de données en simulant 30 jours d'exploitation.

```python
def generate_all():
    os.makedirs("data", exist_ok=True)   # Crée le dossier data/ s'il n'existe pas
    leg_rows, doa_rows = [], []          # Listes qui accumuleront les lignes de chaque fichier
    start = datetime(2025,11,1)          # Début de la période : 1er novembre 2025
    end   = datetime(2025,11,30)         # Fin : 30 novembre 2025
```

**Boucle principale (lignes 109–208) :**

```python
    for d in range((end-start).days+1):  # Boucle sur les 30 jours
        dt = start + timedelta(days=d)
        day_routes = [r for r in ROUTES if random.random() < 0.88]  # 88% des routes actives chaque jour
```

**Génération d'un vol aller (lignes 114–130) :**

```python
        for orig, dest, fna, fnr, dh, dm, dur in day_routes:
            # Variation aléatoire de l'heure de départ : ±10 à +15 minutes
            dm2 = dm + random.randint(-10, 15)
            # ... ajustement si dépassement 60 minutes ...

            reg = pick_reg(dur, orig, dest)     # Sélectionne l'avion
            state = random.choice(LEG_STATES)   # NEW (opéré) ou CNL (annulé), 90/10%

            # Calcul de l'heure d'arrivée
            ah, am, dex = addm(dh2, dm2, dur)   # dex = jours supplémentaires si vol de nuit

            # Ajout de la ligne dans leg_rows avec toutes les colonnes RAM
            leg_rows.append({
                "FN_CARRIER": "AT",          # Code IATA de RAM
                "FN_NUMBER": str(fna),        # Numéro du vol aller
                "AC_REGISTRATION": reg,       # Immatriculation de l'avion
                "DEP_AP_SCHED": orig,         # Aéroport de départ
                "ARR_AP_SCHED": dest,         # Aéroport d'arrivée
                "LEG_STATE": state,           # NEW ou CNL
                "DEP_TIME_SCHED": fhm(dh2, dm2),  # Heure de départ HHMM
                "ARR_TIME_SCHED": fhm(ah, am),    # Heure d'arrivée HHMM
                # ... autres colonnes ...
            })
```

**Génération du vol retour (lignes 132–144) :**

```python
            # Turnaround (temps d'escale à destination)
            turn = random.randint(120, 300) if dest in LONG_DEST else ...

            rdh, rdm, rd = addm(ah, am, turn)   # Heure de départ retour = arrivée + turnaround
            leg_rows.append({...})               # Même structure, origine/destination inversées
```

**Génération des équipages (lignes 146–184) :**

```python
            if state != "CNL":   # Seulement pour les vols non-annulés
                caid = CA_IDS[ca_i % len(CA_IDS)]  # Identifiant Commandant de Bord
                ccid = CC_IDS[cc_i % len(CC_IDS)]  # Identifiant Copilote

                roff = 60 if dur <= 90 else 120    # Temps de présence avant départ

                for cid, rank, pos in [(caid,"CA","CA"), (ccid,"CC","CC")]:
                    doa_rows.append({
                        "CREW_ID": cid,
                        "RANK_": rank,              # CA = Commandant, CC = Copilote
                        "ACTIVITY": str(fna),       # Numéro du vol effectué
                        "ACTIVITY_GROUP": "FLT",    # Type : vol (FLT) vs repos (SBA) etc.
                        "ORIGINE": orig,
                        "DESTINATION": dest,
                        "BLOCK_HOURS": blk(dur),    # Heures de vol
                        # ... autres colonnes ...
                    })
```

**Activités non-vol (lignes 186–207) :**

```python
        # Chaque jour, 10 à 20 activités non-vol aléatoires
        for _ in range(random.randint(10, 20)):
            act, grp = random.choice(NON_FLT)  # SB4/SBA = standby, CRE/LVE = congé, etc.
            # Ces données représentent les jours de repos, simulateurs, etc.
```

**Export des fichiers (lignes 220–235) :**

```python
    df_leg.to_excel("data/LEG_FUTUR_011125_301125.xlsx", ...)    # Fichier vols
    df_doa.to_excel("data/DOA_PROGRAME_011125_301125.xlsx", ...) # Fichier équipages
    pd.DataFrame(params).to_csv("data/reference_params_RAM.csv", ...)  # Paramètres
```

**Paramètres de simulation exportés :**

```python
    # Turnaround minimum par type d'avion
    {"param_type":"min_turnaround_min", "aircraft_type":"B737-800", "value":45}

    # Distribution Gamma pour les retards initiaux
    {"param_type":"gamma_shape", "value":2.0}   # Forme alpha de la distribution
    {"param_type":"gamma_scale", "value":15.0}  # Échelle thêta (moyenne = alpha × theta = 30 min)

    # Matrice de transition Markov 3×3
    {"param_type":"markov_transition", "aircraft_type":"Normal->Normal", "value":0.65}  # 65% rester normal
    {"param_type":"markov_transition", "aircraft_type":"Normal->Alerte", "value":0.28}  # 28% passer en alerte
    {"param_type":"markov_transition", "aircraft_type":"Normal->Bloque", "value":0.07}  # 7% se bloquer
```

---

## 4. `src/simulation.py` — Moteur de simulation Monte Carlo

> **But :** Simuler la propagation des retards à travers tout le programme de vols d'une journée. C'est le cœur mathématique du projet.

---

### Lignes 7–17 : `_norm_date_key()` — Normalisation des dates

```python
def _norm_date_key(value):
    """Convertit n'importe quelle représentation de date en YYYY-MM-DD stable."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return str(np.datetime64(s, "D"))  # np.datetime64 accepte ISO 8601 et formats courants
    except Exception:
        return ""   # Si conversion impossible : clé vide (pas d'erreur fatale)
```

**Pourquoi ?** Les données peuvent contenir des dates sous forme de string "2025-11-01", de Timestamp pandas, ou même de float Excel (44932 = 2025-11-01). Cette fonction harmonise tout.

---

### Lignes 20–28 : `_flight_match_key()` — Clé de matching des vols

```python
def _flight_match_key(value):
    """Rend 'AT570' et '570' identiques pour la jointure vol/équipage."""
    raw = str(value).strip().upper()
    digits = "".join(ch for ch in raw if ch.isdigit())  # Extrait seulement les chiffres
    return digits if digits else raw   # "AT570" → "570", "ZZTEST" → "ZZTEST"
```

**Problème résolu :** Dans le schedule, un vol s'appelle "AT570", mais dans le fichier équipage, il est noté "570". Sans cette normalisation, la jointure échouerait.

---

### Lignes 45–62 : `_resolve_min_turnaround()` — Turnaround minimum

```python
def _resolve_min_turnaround(min_turnaround, aircraft_type, turnaround_table=None, airport=None):
    """
    Détermine le turnaround minimum applicable.
    Priorité : turnaround_table[(airport, aircraft_type)] → fallback 45 min.
    """
    if turnaround_table and airport:
        ap  = str(airport).strip().upper()    # "cmn" → "CMN"
        act = str(aircraft_type).strip().upper()

        val = turnaround_table.get((ap, act))  # Recherche exacte (aéroport, type avion)
        if val is not None:
            return float(val)

    return 45.0  # Valeur par défaut si pas d'entrée dans la table
```

---

### Lignes 67–85 : `markov_turnaround()` — Turnaround stochastique

```python
def markov_turnaround(aircraft_type, min_turnaround, markov_matrix, markov_multipliers,
                      n_steps=3, initial_state=0, turnaround_table=None, airport=None):
    """
    Simule un turnaround réaliste via une chaîne de Markov à 3 états.

    États :
      0 = Normal  → multiplicateur ×1.00 (tout se passe bien)
      1 = Alerte  → multiplicateur ×1.45 (légers problèmes)
      2 = Bloqué  → multiplicateur ×2.30 (retard opérationnel sévère)
    """
    base  = _resolve_min_turnaround(...)  # Turnaround de base selon type et aéroport
    state = initial_state                  # On commence en état Normal

    for _ in range(n_steps):               # On fait 3 transitions Markov
        state = np.random.choice(
            [0, 1, 2],
            p=markov_matrix[state]         # Probabilités selon l'état actuel
        )

    return base * markov_multipliers[state], state  # Durée × multiplicateur, état final
```

**Exemple numérique :**
- Turnaround de base pour B737-800 à CMN = 45 min
- Si état final = Bloqué : `45 × 2.30 = 103.5 min` de turnaround
- Si état final = Normal  : `45 × 1.00 = 45 min`

**Matrice de Markov typique (issue des paramètres) :**
```
État courant    | →Normal | →Alerte | →Bloqué
Normal          |  65%    |  28%    |   7%
Alerte          |  25%    |  52%    |  23%
Bloqué          |   8%    |  28%    |  64%
```

---

### Lignes 88–447 : `simulate_once()` — Une simulation complète

C'est la fonction la plus complexe du projet. Elle simule **une journée entière** de vols en propageant les retards.

**Signature et paramètres :**
```python
def simulate_once(
    df_sched,          # DataFrame : programme de vols (une ligne = un vol)
    df_crew,           # DataFrame : données équipages (séquences de vols)
    min_turnaround,    # Durée minimale de rotation
    gamma_shape,       # Paramètre α de la distribution Gamma (forme)
    gamma_scale,       # Paramètre θ de la distribution Gamma (échelle)
    markov_matrix,     # Matrice 3×3 de transition de la chaîne de Markov
    markov_multipliers,# [1.0, 1.45, 2.30] = multiplicateurs par état
    otp_threshold=15,  # Seuil OTP : vol "à l'heure" si retard < 15 min
    mode="auto",       # "auto" = Monte Carlo, "manuel" = retard injecté manuellement
    injected_delays=None,   # Dict {flight_id: minutes} pour mode manuel
    injected_targets=None,  # Liste de cibles précises {flight_id, date, msn, minutes}
    hub_airport=None,  # Aéroport hub : amplifie les retards (congestion)
    hub_factor=1.2,    # Facteur d'amplification hub (1.2 = +20%)
    slack_config=None, # Configuration des tampons de temps
    turnaround_table=None,  # Table de turnaround par (aéroport, type_avion)
):
```

**ÉTAPE 1 — Masque hub (lignes 120–125) :**
```python
hub_mask = np.zeros(n, dtype=bool)  # Tableau de False par défaut
if hub_airport:
    hub_mask = (df_sched["origin"].values == hub_airport)  # True pour vols depuis le hub
```

**ÉTAPE 2 — Génération des retards initiaux (lignes 128–159) :**

*Mode manuel :*
```python
if mode == "manuel" and injected_targets:
    initial_delays = np.zeros(n)  # Tous les vols à l'heure sauf le vol ciblé
    for tgt in injected_targets:
        fid     = str(tgt["flight_id"]).strip()
        minutes = float(tgt["minutes"])
        mask    = (df_sched["flight_id"].astype(str).str.strip() == fid)
        # Affinage par date et immatriculation si fournis
        initial_delays[mask.values] = minutes  # Injecte le retard sur le vol cible
```

*Mode automatique :*
```python
else:
    # Pour les vols au hub : shape amplifié = gamma_shape × hub_factor
    shapes = np.where(hub_mask, gamma_shape * hub_factor, gamma_shape)
    initial_delays = np.array([
        np.random.gamma(shape=float(shapes[i]), scale=gamma_scale)
        for i in range(n)
    ])
    # Distribution Gamma(α, θ) : moyenne = α×θ, variance = α×θ²
    # Avec α=2.0, θ=15 : moyenne = 30 min, retards concentrés entre 5 et 80 min
```

**ÉTAPE 3 — Construction des index de recherche (lignes 172–229) :**

Ces index permettent de retrouver rapidement un vol ou un équipage pendant la propagation.

```python
# Index schedule : clé (numéro_vol, date, origine, destination) → position
idx_by_fkey_date_apt = {}   # Précision maximale (4 clés)
idx_by_fkey_date     = {}   # Fallback sans origine/destination
idx_by_fkey          = {}   # Dernier recours : numéro seul

for label, row in df.iterrows():
    fkey = _schedule_match_key(row)  # Normalise le numéro de vol
    dkey = _norm_date_key(...)       # Normalise la date
    org  = str(row["origin"]).upper()
    dst  = str(row["destination"]).upper()

    idx_by_fkey_date_apt[(fkey, dkey, org, dst)] = label
    idx_by_fkey_date[(fkey, dkey)]               = label
    idx_by_fkey[fkey]                            = label
```

**Pourquoi 3 niveaux de fallback ?** Les données réelles peuvent avoir des incohérences : un équipage noté "09/11/2025" et le vol noté "2025-11-09". Les 3 niveaux garantissent qu'on trouve toujours la correspondance.

**ÉTAPE 4 — Boucle de propagation principale (lignes 241–437) :**

```python
ac_last_flight = {}   # Mémorise le dernier vol de chaque avion : {immat: position_dans_df}

for pos, row in df_sorted.iterrows():   # Vols triés par date puis heure de départ
    msn     = row["aircraft_msn"]
    ac_type = row["aircraft_type"]
```

*4a. Calcul du turnaround Markov :*
```python
    turn_airport = row["origin"]   # Le turnaround se fait à l'aéroport de départ
    turn_dur, m_state = markov_turnaround(
        ac_type, min_turnaround, markov_matrix, markov_multipliers,
        turnaround_table=turnaround_table, airport=turn_airport,
    )
    df.at[pos, "turnaround_actual"] = turn_dur   # Enregistre la valeur Markov brute
    df.at[pos, "markov_state"]      = m_state    # Enregistre l'état (0, 1 ou 2)
```

*4b. Calcul du slack applicable :*
```python
    slack_minutes = 0.0
    if slack_config and slack_config.get("minutes", 0) > 0:
        s_scope = slack_config.get("scope", "global")

        if s_scope == "global":
            slack_minutes = s_min    # Le tampon s'applique à TOUS les vols

        elif s_scope == "window":    # Seulement dans une fenêtre horaire
            if w_start <= row["dep_min"] <= w_end:
                slack_minutes = s_min

        elif s_scope == "aircraft":  # Seulement pour un avion spécifique
            if row["aircraft_msn"] == msn_val:
                slack_minutes = s_min
        # ... autres portées : flight, airports, window_aircraft
```

*4c. Contrainte avion — FORMULE CENTRALE (lignes 334–362) :*

```python
    earliest_avion = row["dep_min"]   # Par défaut : heure programmée
    slack_utilise  = 0.0

    if msn in ac_last_flight:         # Si cet avion a déjà effectué un vol aujourd'hui
        prev_pos = ac_last_flight[msn]

        arr_prev_actual    = df.at[prev_pos, "arr_actual"]     # Arrivée réelle du vol précédent
        arr_prev_scheduled = df.at[prev_pos, "arr_min"]        # Arrivée prévue du vol précédent

        # Retard accumulé par le vol précédent
        retard_amont = max(0.0, arr_prev_actual - arr_prev_scheduled)

        # Le slack ABSORBE le retard amont
        slack_utilise  = min(slack_minutes, retard_amont)
        retard_propague = retard_amont - slack_utilise   # Ce qui n'a pas été absorbé

        # L'avion est prêt :
        # heure d'arrivée prévue + turnaround minimum + retard non absorbé
        avion_pret = arr_prev_scheduled + min_turn + retard_propague

        earliest_avion = max(row["dep_min"], avion_pret)

    df.at[pos, "slack_applied"] = slack_utilise  # Mémoriser l'absorption réelle
    ac_last_flight[msn] = pos  # Enregistrer ce vol comme dernier vol de l'avion
```

**Interprétation métier :**
> Un avion qui arrive avec 30 min de retard repart avec 30 min de retard sur son prochain vol, SAUF si on a prévu un tampon (slack) de 20 min : dans ce cas, il ne repart qu'avec 10 min de retard.

*4d. Contrainte équipage (lignes 366–414) :*

```python
    row_fkey = _schedule_match_key(row)
    lookup_key = (row_fkey, flight_date_key, row_org, row_dst)
    candidates = crew_idx.get(lookup_key, [])   # Trouvons l'équipage de ce vol

    for rec in candidates:
        pos_seq = rec["pos_seq"]  # Position dans la séquence de l'équipage
        if pos_seq <= 0:
            continue  # Premier vol de l'équipage : pas de contrainte antérieure

        prev_fkey = seq_key[pos_seq - 1]   # Vol précédent de l'équipage
        prev_label = idx_by_fkey_date_apt.get(...)  # Retrouve ce vol dans le schedule

        arr_crew = df.at[prev_label, "arr_actual"]   # L'équipage arrive à cette heure
        earliest_crew = max(earliest_crew, arr_crew)  # L'équipage doit être disponible
```

**Interprétation métier :**
> Si un copilote arrive de Paris avec 45 min de retard, il ne peut pas piloter le prochain vol avant d'être arrivé. C'est la contrainte équipage.

*4e. Calcul du départ réel — Formule MAX (lignes 417–437) :*

```python
    dep_real = max(
        row["dep_min"] + row["initial_delay"],  # Heure programmée + retard initial du vol
        earliest_avion,                          # Contrainte avion (rotation)
        earliest_crew,                           # Contrainte équipage
    )
    # On prend le MAXIMUM car un vol ne peut partir que quand TOUT est prêt

    # Application de l'effet hub en mode manuel
    if mode == "manuel" and hub_check:
        retard_accumule = dep_real - row["dep_min"]
        if retard_accumule > 0:
            dep_real = row["dep_min"] + (retard_accumule * hub_factor)
            # Exemple : +20 min de retard avec hub_factor=1.5 → +30 min

    df.at[pos, "dep_actual"] = dep_real
    df.at[pos, "arr_actual"] = dep_real + row["flight_duration_min"]  # Arrivée = départ + durée
```

**Lignes 443–447 : Indicateurs finaux :**
```python
df["dep_delay"] = (df["dep_actual"] - df["dep_min"]).clip(lower=0)  # Retard départ (jamais négatif)
df["arr_delay"] = (df["arr_actual"] - df["arr_min"]).clip(lower=0)  # Retard arrivée
df["on_time"]   = df["dep_delay"] <= otp_threshold  # True si retard < 15 min
```

---

### Lignes 450–596 : `run_monte_carlo()` — Agrégation des simulations

```python
def run_monte_carlo(df_sched, df_crew, ..., n_simulations=200, ...):
    """Lance N simulations indépendantes et agrège statistiquement les résultats."""

    if mode == "manuel":
        n_simulations = 1   # En mode manuel : une seule simulation déterministe

    all_results  = []   # Liste de N DataFrames résultat
    otp_per_sim  = []   # OTP global par simulation (pour la distribution)
    prop_per_sim = []   # Coefficient de propagation par simulation

    for i in range(n_simulations):
        # Mise à jour de la barre de progression Streamlit
        if progress_bar:
            progress_bar.progress((i+1)/n_simulations, text=f"Simulation {i+1}...")

        result = simulate_once(...)     # Lance une simulation complète
        all_results.append(result)

        otp_per_sim.append(result["on_time"].mean() * 100)  # % vols à l'heure dans cette sim

        total_initial = result["initial_delay"].sum()
        total_final   = result["arr_delay"].sum()
        # Coefficient de propagation : si 1.0 = retards n'ont pas grossi
        #                               si 1.5 = retards ont augmenté de 50%
        prop_per_sim.append(total_final / total_initial if total_initial > 0 else 1.0)
```

**Agrégation des matrices (lignes 506–529) :**

```python
    # Créer des matrices NumPy : N lignes (simulations) × M colonnes (vols)
    arr_matrix   = np.array([r["arr_delay"].values for r in all_results])  # Matrice N×M
    dep_matrix   = np.array([r["dep_delay"].values for r in all_results])

    # Calcul des statistiques par vol (sur l'axe 0 = axe des simulations)
    agg["mean_arr_delay"] = arr_matrix.mean(axis=0)           # Retard moyen d'arrivée
    agg["p95_arr_delay"]  = np.percentile(arr_matrix, 95, axis=0)  # Percentile 95
    agg["mean_dep_delay"] = dep_matrix.mean(axis=0)           # Retard moyen de départ
    agg["otp_rate"]       = (dep_matrix <= 15).mean(axis=0) * 100  # % à l'heure

    agg["p80_dep_delay"]  = np.percentile(dep_matrix, 80, axis=0).clip(min=0)  # P80
    agg["p90_dep_delay"]  = np.percentile(dep_matrix, 90, axis=0).clip(min=0)  # P90

    agg["mean_turnaround_actual"]   = turn_matrix.mean(axis=0)   # Turnaround Markov moyen
    agg["mean_turnaround_effectif"] = teff_matrix.mean(axis=0)   # Turnaround après slack
    agg["mean_slack_absorbed"]      = slack_matrix.mean(axis=0)  # Absorption slack moyenne
```

**Interprétation P80/P90 :**
> P80 = 45 min signifie que dans 80% des 200 simulations, ce vol est parti avec moins de 45 min de retard. C'est un indicateur de robustesse du programme.

**Retour de la fonction :**
```python
    return agg,              # DataFrame agrégé par vol
           np.array(otp_per_sim),  # OTP par simulation → permet d'afficher la distribution
           np.array(prop_per_sim), # Coeff propagation par simulation
           all_results             # Les N DataFrames bruts (pour l'optimiseur)
```

---

## 5. `src/data_loader.py` — Chargement et validation des données

> **But :** Lire les fichiers uploadés par l'utilisateur (CSV, XLSX), détecter leur format, les nettoyer et les convertir vers le format interne du simulateur.

---

### Lignes 9–24 : Définition des colonnes requises

```python
REQUIRED_SCHEDULE = {
    "flight_id",          # Identifiant vol (ex: "AT570")
    "origin",             # IATA départ (ex: "CMN")
    "destination",        # IATA arrivée (ex: "CDG")
    "scheduled_departure","scheduled_arrival",
    "aircraft_msn",       # Immatriculation avion (ex: "CNROH")
    "aircraft_type",      # Type avion (ex: "B737-800")
}
REQUIRED_CREW = {"crew_id", "flight_sequence"}
REQUIRED_REF  = {"param_type", "value"}

# Colonnes spécifiques au format RAM → permet la détection automatique
OPS_LEG_COLS = {"FN_CARRIER", "FN_NUMBER", "DEP_AP_SCHED", "ARR_AP_SCHED"}
OPS_DOA_COLS = {"CREW_ID", "ACTIVITY", "ACTIVITY_GROUP", "ORIGINE"}
```

---

### Lignes 35–46 : Conversions horaires

```python
def hhmm_to_minutes(t: str) -> int:
    """Convertit "07:30" → 450 minutes depuis minuit."""
    h, m = map(int, str(t).strip().split(":"))
    return h * 60 + m   # 7×60 + 30 = 450

def minutes_to_hhmm(minutes: int) -> str:
    """Convertit 450 → "07:30"."""
    h = (int(minutes) // 60) % 24   # % 24 pour gérer les vols de nuit
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"
```

**Pourquoi convertir en minutes ?** La simulation travaille en minutes depuis minuit (entiers), ce qui simplifie tous les calculs d'horaires. "07:30" + 175 min = "10:25" devient juste `450 + 175 = 625`.

---

### Lignes 49–61 : `_normalize_date_series()`

```python
def _normalize_date_series(series: pd.Series) -> pd.Series:
    """Normalise des dates mixtes en YYYY-MM-DD sans inverser les formats ISO."""
    s = series.astype(str).str.strip()

    # Détecte le format ISO (YYYY-MM-DD) pour ne pas l'invertir
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)

    # Dates ISO : pas de dayfirst
    out.loc[iso_mask]  = pd.to_datetime(s[iso_mask],  dayfirst=False)...
    # Autres formats (JJ/MM/AAAA) : dayfirst=True pour RAM
    out.loc[~iso_mask] = pd.to_datetime(s[~iso_mask], dayfirst=True)...
```

**Problème résolu :** "01/11/2025" (format RAM) doit être interprété comme 1er novembre, pas le 11 janvier. Et "2025-11-01" (format ISO) ne doit pas être inversé en 2025-01-11.

---

### Lignes 78–99 : `_normalize_flight_token()`

```python
def _normalize_flight_token(value) -> str:
    """Nettoie un identifiant de vol individuel."""
    raw = str(value).strip().upper()
    if raw in {"NAN", "NONE", "NULL", "<NA>", "NAT"}:
        return ""   # Toutes les formes de "vide" → ""

    try:
        f = float(raw)    # "570.0" → 570
        if np.isfinite(f) and f.is_integer():
            raw = str(int(f))   # → "570"
    except:
        pass

    raw = re.sub(r"\s+", "", raw)         # Supprime espaces internes
    raw = re.sub(r"[^A-Z0-9]", "", raw)  # Supprime tout sauf lettres et chiffres
    return raw  # "AT 570" → "AT570", "570.0" → "570"
```

---

### Lignes 114–186 : `_read_file()` — Lecture intelligente de fichiers

```python
def _read_file(uploaded_file) -> pd.DataFrame | None:
    name = uploaded_file.name.lower()

    if name.endswith((".xlsx", ".xls")):
        # Pour Excel : tester TOUTES les feuilles et TOUS les en-têtes possibles (0 à 7)
        best_df = None
        best_score = (-1, -1, -1, -1)

        for sh in xls.sheet_names:            # Pour chaque feuille du classeur
            for header_row in range(0, 8):    # Pour chaque ligne d'en-tête possible
                sdf = pd.read_excel(xls, sheet_name=sh, header=header_row)
                score = _sheet_score(sdf)     # Score = nombre de colonnes connues trouvées
                if score > best_score:
                    best_df = sdf
                    best_score = score
        return best_df

    # Pour CSV : essayer virgule puis point-virgule comme séparateur
    try:
        df = pd.read_csv(uploaded_file, sep=",")
        if df.shape[1] > 1:   # Si plus d'une colonne → bonne détection
            return df
    except:
        pass
    return pd.read_csv(uploaded_file, sep=";")   # Essai avec point-virgule
```

**Problème résolu :** Les fichiers Excel RAM peuvent avoir des lignes de titre avant l'en-tête réel, ou être dans n'importe quelle feuille. L'algorithme essaie toutes les combinaisons et garde celle qui reconnaît le plus de colonnes connues.

---

## 6. `src/optimizer.py` — Optimiseur d'échanges d'avions

> **But :** Quand un vol est prévu avec un retard important, trouver un autre avion disponible qui pourrait prendre sa place et réduire le retard. C'est ce qu'on appelle un "aircraft swap".

### Principe de l'algorithme

```
Vol A1 est en retard (ex: +90 min)
↓
Recherche d'un avion A2 de même type, déjà sur place, disponible
↓
A2 peut-il partir à temps ? (arrivée A2 + turnaround ≤ départ planifié A1)
↓
Validation équipage (le crew de A1 peut-il voler sur A2 ?)
↓
Calcul du gain : "nouveau retard A2" vs "retard actuel A1"
↓
Recommandation si gain > 0
```

### Structures de données clés

```python
# État de la flotte : position et disponibilité de chaque avion
fleet_state = {
    "CNROH": {
        "last_airport": "CDG",      # Où est l'avion actuellement
        "available_from": 650,      # Disponible à partir de (minutes depuis minuit)
        "aircraft_type": "B737-800"
    },
    ...
}

# Candidat d'échange évalué
SwapCandidate = {
    "flight_id": "AT570",
    "a1_msn": "CNROH",      # Avion original en retard
    "a2_msn": "CNROI",      # Avion de remplacement proposé
    "new_dep": 420,          # Nouvelle heure de départ avec A2 (en minutes)
    "delay_saved": 35,       # Gain en minutes
    "score": 0.85,           # Score de faisabilité (0-1)
    "feasible": True,
}
```

### Fonction principale `find_swap_for_flight()`

```python
def find_swap_for_flight(flight_id, df_sched, df_crew, fleet_state,
                          turnaround_table, auth_table, delayed_departure):
    """
    Cherche le meilleur avion de remplacement pour un vol retardé.

    Étapes :
    1. Récupérer l'avion original A1 et ses caractéristiques
    2. Identifier les candidats A2 (même type, au bon aéroport)
    3. Pour chaque A2 : vérifier la disponibilité et calculer le gain
    4. Valider les contraintes équipage et autorisation d'aéroport
    5. Retourner le meilleur candidat
    """
```

---

### Vérification des autorisations `is_airport_authorized()`

```python
def is_airport_authorized(msn, aircraft_type, airport, auth_table, planning_airports):
    """
    4 niveaux de priorité :
    1. Autorisation spécifique par immatriculation (plus précis)
    2. Autorisation par type d'avion
    3. Fallback : aéroports trouvés dans le planning (l'avion y va déjà)
    4. Mode permissif : autorisé par défaut si pas de données
    """
    # Niveau 1 : par immatriculation
    if msn in auth_table["by_msn"]:
        return airport in auth_table["by_msn"][msn]["authorized_airports"]

    # Niveau 2 : par type
    if aircraft_type in auth_table["by_type"]:
        return airport in auth_table["by_type"][aircraft_type]

    # Niveau 3 : si l'avion dessert déjà cet aéroport dans le planning
    if airport in planning_airports.get(msn, set()):
        return True

    # Niveau 4 : permissif (pas de table de restrictions chargée)
    return True
```

---

## 7. `src/app.py` — Interface web Streamlit principale

> **But :** Orchestrer toute l'application. C'est l'interface que voit l'utilisateur : sidebar, boutons, tableaux, graphiques.

### Structure générale (2 259 lignes)

```
app.py
├── CSS personnalisé (lignes 1-300)
│   ├── Couleurs RAM (#C8102E rouge, bleu marine)
│   ├── Styles cartes KPI
│   └── Formatage tableaux
│
├── Fonctions utilitaires (lignes 300-900)
│   ├── mhm() : minutes → "HH:MM"
│   ├── color_delay() : coloration selon sévérité retard
│   ├── color_otp() : coloration selon taux OTP
│   ├── section_header() : composant titre stylisé
│   └── kpi_card() : carte KPI HTML personnalisée
│
├── Sidebar (lignes 900-1200)
│   ├── Upload fichiers (vols, équipages, référentiel, turnaround, autorisations)
│   ├── Mode simulation (Auto/Manuel)
│   ├── Paramètres (N simulations, shape/scale Gamma)
│   ├── Options hub (aéroport, facteur)
│   ├── Options slack (minutes, portée)
│   └── Options optimiseur (seuil déclenchement)
│
└── Contenu principal (lignes 1200-2259)
    ├── Section 1 : KPIs globaux (OTP, propagation, vols impactés)
    ├── Section 2 : Analyse Hub (si activé)
    ├── Section 3 : Analyse Slack (si activé)
    ├── Section 4 : Tableau de propagation par vol
    ├── Section 5 : Analyse par avion (mode auto)
    ├── Section 6 : Récapitulatif global
    ├── Résultats optimiseur (si activé)
    └── Boutons d'export CSV
```

### Exemples de fonctions clés

**`mhm()` — Formatage horaire :**
```python
def mhm(minutes):
    """Convertit 450 → '07:30', gère les valeurs NaN."""
    if pd.isna(minutes): return "—"
    m = int(minutes)
    return f"{m//60:02d}:{m%60:02d}"
```

**`color_delay()` — Coloration conditionnelle :**
```python
def color_delay(val):
    """Retourne un style CSS selon la valeur du retard."""
    if pd.isna(val): return ""
    if val < 15:  return "background-color: #d4edda"  # Vert : à l'heure
    if val < 30:  return "background-color: #fff3cd"  # Jaune : léger retard
    if val < 60:  return "background-color: #fde8cc"  # Orange : retard modéré
    return "background-color: #f8d7da"                # Rouge : retard sévère
```

**`kpi_card()` — Carte KPI HTML :**
```python
def kpi_card(title, value, subtitle="", color="#C8102E"):
    """Génère une carte HTML styled avec titre, valeur principale et sous-titre."""
    return f"""
    <div style="background:white; border-left:4px solid {color}; padding:16px; ...">
        <div style="font-size:0.85rem; color:#666">{title}</div>
        <div style="font-size:2rem; font-weight:700; color:{color}">{value}</div>
        <div style="font-size:0.75rem; color:#888">{subtitle}</div>
    </div>
    """
```

### Flux d'exécution de l'application

```
1. Utilisateur uploade les fichiers
   ↓
2. data_loader détecte le format et convertit
   ↓
3. Utilisateur clique "Lancer la simulation"
   ↓
4. simulation.run_monte_carlo() tourne N fois
   ↓
5. Agrégation des résultats (mean, P80, P90, OTP)
   ↓
6. Affichage :
   - KPI globaux (st.metric)
   - Tableau détaillé (st.dataframe)
   - Graphiques (st.plotly_chart)
   ↓
7. Si optimiseur activé :
   optimizer.run_swap_optimizer() → recommandations
   ↓
8. Export CSV si demandé
```

---

## 8. FLUX DE DONNÉES GLOBAL

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRÉES UTILISATEUR                       │
│                                                               │
│  LEG_FUTUR.xlsx      DOA_PROGRAME.xlsx    reference_params.csv │
│  (programme vols)    (équipages)          (paramètres simul.)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   data_loader.py                             │
│                                                               │
│  1. Lecture fichier (CSV ou XLSX, auto-détection feuille)    │
│  2. Détection format (OPS RAM ou format interne)             │
│  3. Conversion colonnes (FN_NUMBER → flight_id, etc.)        │
│  4. Nettoyage dates, horaires, séquences vols                │
│  5. Validation colonnes requises                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
              df_sched, df_crew, ref_params
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   simulation.py                              │
│                                                               │
│  Pour chaque simulation i in [1..N]:                         │
│    1. Tirage retards initiaux (Gamma ou injection manuelle)  │
│    2. Pour chaque vol (trié par date+heure):                 │
│       a. Turnaround Markov (Normal/Alerte/Bloqué)            │
│       b. Absorption slack                                    │
│       c. Contrainte avion : arr_prev + turnaround            │
│       d. Contrainte équipage : arrivée vol précédent crew    │
│       e. dep_réel = MAX(dep_sched+retard, avion_prêt, crew)  │
│    3. Calcul dep_delay, arr_delay, on_time                   │
│                                                               │
│  Agrégation : mean, P80, P90, OTP, coeff_propagation         │
└──────────────────────┬──────────────────────────────────────┘
                       │
              agg_results, otp_per_sim, all_results
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              optimizer.py (optionnel)                        │
│                                                               │
│  Pour les vols retardés (> seuil):                           │
│    1. Construction état flotte                               │
│    2. Recherche avions A2 compatibles                        │
│    3. Vérification disponibilité + autorisations             │
│    4. Calcul gain + validation équipage                      │
│    5. Recommandation meilleur swap                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      app.py                                  │
│                                                               │
│  Affichage Streamlit :                                       │
│    - OTP global (%), Coeff propagation, Vols impactés        │
│    - Tableau par vol (retard moyen, P80, P90, état Markov)   │
│    - Graphique retard par avion                              │
│    - Recommandations optimiseur                              │
│    - Export CSV                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## GLOSSAIRE DES TERMES CLÉS

| Terme | Définition |
|---|---|
| **OTP** | On-Time Performance : % de vols partant avec moins de 15 min de retard |
| **Gamma(α, θ)** | Distribution statistique avec forme α et échelle θ. Moyenne = α×θ. Modélise bien les retards (asymétrique, toujours positif) |
| **Chaîne de Markov** | Processus où l'état suivant dépend uniquement de l'état actuel (pas de l'historique). Les 3 états Normal/Alerte/Bloqué évoluent selon des probabilités de transition |
| **Monte Carlo** | Méthode qui lance N simulations aléatoires pour obtenir une distribution statistique du résultat au lieu d'un résultat unique |
| **Turnaround** | Temps de rotation : durée entre l'arrivée d'un avion et son prochain départ (inclut nettoyage, ravitaillement, embarquement) |
| **Slack** | Tampon de temps prévu volontairement dans le programme pour absorber les petits retards sans les propager |
| **BFS** | Breadth-First Search : algorithme de parcours en largeur d'un graphe (explore niveau par niveau) |
| **Propagation** | Mécanisme par lequel le retard d'un vol se transfère aux vols suivants utilisant le même avion ou le même équipage |
| **Hub** | Aéroport central (CMN pour RAM) avec forte densité de vols : un retard y est amplifié car il affecte de nombreuses correspondances |
| **P80 / P90** | Percentiles 80 et 90 : dans 80% (resp. 90%) des simulations, le retard était inférieur à cette valeur |
| **Aircraft Swap** | Remplacement d'un avion retardé par un autre avion disponible de même type |
| **MSN** | Manufacturer Serial Number : immatriculation unique d'un avion |
| **IATA** | International Air Transport Association : codes à 3 lettres des aéroports (CMN, CDG, LHR...) |

---

*Rapport généré le 05 Avril 2026 — Projet AIR-ROBUST / DelayPropagation*
*Royal Air Maroc — Simulation de propagation des retards*