# AIR-ROBUST

**Simulateur de Propagation de Retards Aériens — Royal Air Maroc**

Outil d'aide à la décision pour l'analyse et la simulation de la propagation des retards
dans le réseau de Royal Air Maroc, basé sur des méthodes Monte Carlo et des chaînes de Markov.

---

## Objectif du Projet

Modéliser et simuler la propagation des retards dans le réseau RAM pour :
- Estimer le taux de ponctualité (OTP) du réseau sur une journée type
- Identifier les vols les plus vulnérables aux effets cascade
- Évaluer l'impact de scénarios opérationnels (congestion hub, tampon variable)
- Recommander des tampons de turnaround optimaux par vol

---

## Architecture du Projet

```
air-robust/
│
├── generate_data.py                       # Génération des données de test
├── main.py                                # Application simplifiée (propagation)
│
├── src/
│   ├── app.py                             # Application Streamlit principale
│   ├── simulation.py                      # Moteur Monte Carlo + Chaîne de Markov
│   ├── simulation_copy.py                 # Version documentée de la simulation
│   └── data_loader.py                     # Chargement et validation des données
│
├── data/                                  # Données générées
│   ├── LEG_FUTUR_011125_301125.xlsx       # Programme de vols (format opérationnel)
│   ├── DOA_PROGRAME_011125_301125.xlsx    # Rotations équipage (format opérationnel)
│   └── reference_params_RAM.csv           # Paramètres de simulation
│
├── requirements.txt                       # Dépendances Python
├── Dockerfile                             # Conteneurisation Docker
└── README.md
```

---

## Données d'Entrée

Le simulateur accepte trois fichiers :

### 1. Programme de vols — `LEG_FUTUR`

Format opérationnel avec 23 colonnes :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `FN_CARRIER` | Code compagnie | AT |
| `FN_NUMBER` | Numéro de vol | 600 |
| `DAY_OF_ORIGIN` | Date du vol (YYYYMMDD) | 20251101 |
| `AC_SUBTYPE` | Code type avion | 73H, 7M8, 788, AT7 |
| `AC_REGISTRATION` | Immatriculation | CNROH |
| `DEP_AP_SCHED` | Aéroport de départ | CMN |
| `ARR_AP_SCHED` | Aéroport d'arrivée | CDG |
| `DEP_TIME_SCHED` | Heure de départ (HHMM) | 0600 |
| `ARR_TIME_SCHED` | Heure d'arrivée (HHMM) | 0855 |
| `LEG_STATE` | Statut du vol | NEW / CNL |

### 2. Rotations Équipage — `DOA_PROGRAME`

Format opérationnel avec 30 colonnes :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `CREW_ID` | Identifiant équipage | 45196 |
| `RANK_` | Grade | CA / CC |
| `AC_QUALS` | Qualifications | B737, B737,B787 |
| `ACTIVITY` | Numéro de vol ou code activité | 600, SB4, PDO |
| `ACTIVITY_GROUP` | Type d'activité | FLT, SBA, OFF, ACT, LVE |
| `ORIGINE` | Aéroport d'origine | CMN |
| `DESTINATION` | Aéroport de destination | CDG |
| `A_C` | Code avion | 738, 7M8, 788 |
| `BLOCK_HOURS` | Temps de vol bloc | 02:55 |

### 3. Paramètres de Référence — `reference_params_RAM.csv`

| param_type | aircraft_type | value | Description |
|------------|--------------|-------|-------------|
| `min_turnaround_min` | B737-800 | 45 | Turnaround minimum |
| `gamma_shape` | ALL | 2.0 | Paramètre α loi Gamma |
| `gamma_scale` | ALL | 15.0 | Paramètre θ loi Gamma |
| `markov_transition` | Normal->Alerte | 0.28 | Matrice de transition |
| `markov_turnaround_multiplier_alerte` | ALL | 1.45 | Multiplicateur turnaround |

### Proportionnalité des données

Les données de test respectent la proportionnalité suivante :
- Chaque vol actif (`LEG_STATE = NEW`) a un Capitaine (CA) et un Cabin Crew (CC) dans DOA_PROGRAME
- Ratio crew/vol ≈ 2.0
- Les numéros de vol (`FN_NUMBER`) correspondent aux activités (`ACTIVITY`) dans le fichier crew
- ~10% des vols sont marqués annulés (`LEG_STATE = CNL`)

---

## Modèle de Simulation

### Loi Gamma — Retards initiaux

Chaque vol reçoit un retard initial tiré d'une loi Gamma(α, θ) :
- α = 2.0 (shape), θ = 15.0 min (scale)
- Retard moyen = α × θ = 30 min

### Chaîne de Markov — Turnaround

Le temps de rotation au sol est modulé par une chaîne de Markov à 3 états :

| État | Multiplicateur | Signification |
|------|---------------|---------------|
| Normal | ×1.00 | Opérations fluides |
| Alerte | ×1.45 | Congestion modérée |
| Bloqué | ×2.30 | Perturbation forte |

### Propagation

Le retard se propage via deux mécanismes :
- **Contrainte avion** : l'avion ne peut repartir qu'après turnaround effectif
- **Contrainte équipage** : l'équipage ne peut enchaîner qu'après repos minimum

### Scénarios

- **Hub Congestion** : amplifie les retards au départ d'un aéroport choisi
- **Tampon Variable** : réduit le turnaround effectif pour absorber les retards (portée configurable : global, plage horaire, avion, vol)

---

## Installation et Lancement

### Prérequis

- Python 3.11+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Génération des données de test

```bash
python generate_data.py
```

Produit trois fichiers dans `data/` :
- `LEG_FUTUR_011125_301125.xlsx` — ~2500 vols sur novembre 2025
- `DOA_PROGRAME_011125_301125.xlsx` — ~4700 lignes crew proportionnelles
- `reference_params_RAM.csv` — paramètres Gamma, Markov, turnaround

### Lancement de l'application

```bash
streamlit run src/app.py
```

Puis charger les trois fichiers via l'interface de gauche.

### Avec Docker

```bash
docker build -t air-robust .
docker run -p 8501:8501 air-robust
```

Ouvrir http://localhost:8501

---

## Fonctionnalités de l'Application

| Section | Description |
|---------|-------------|
| **KPI globaux** | OTP moyen, coefficient de propagation, vol le plus impacté, retard moyen |
| **Tampon Variable** | Analyse coût/bénéfice avec simulation de référence (même graine aléatoire) |
| **Hub Congestion** | Impact par destination, comparaison hub vs réseau |
| **Propagation manuelle** | Injection d'un retard sur un vol, visualisation de l'effet cascade |
| **Distributions** | Histogrammes OTP et retards à l'arrivée |
| **Graphe orienté** | Visualisation interactive du réseau par rotation avion |
| **Recommandations** | Tampons P80/P90 par vol (auto) ou tampons OTP/absorption (manuel) |
| **Diagnostic détaillé** | Décomposition retard initial / propagé par vol, état Markov, turnaround |
| **Export CSV** | Téléchargement des résultats agrégés |

---

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit 1.35 |
| Simulation | NumPy, SciPy |
| Visualisation | Plotly |
| Graphes réseau | NetworkX |
| Données | Pandas, OpenPyXL |

---

## Licence

Projet académique — Royal Air Maroc / AIR-ROBUST.
