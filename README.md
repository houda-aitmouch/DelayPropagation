# AIR-ROBUST / DelayPropagation

Simulateur Streamlit de propagation des retards aeriens, avec contraintes avion + equipage, simulation Monte Carlo, turnaround Markovien et scenario optionnel de reaffectation d'avion.

## Vue d'ensemble

Ce projet permet de:

- simuler les retards de depart/arrivee sur un programme de vols,
- mesurer l'OTP et la propagation globale,
- comparer des scenarios operationnels (Hub Congestion, Slack),
- tester un mode manuel d'injection de retard,
- evaluer des opportunites de switch avion (optimizer).

L'application principale est `src/app.py` (Streamlit).

## Structure du projet

```text
DelayPropagation/
├── src/
│   ├── app.py
│   ├── simulation.py
│   ├── data_loader.py
│   └── optimizer.py
├── data/
├── generate_data.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Entrees attendues

L'UI accepte des fichiers au format operationnel (Excel/CSV) .

Fichiers principaux:

1. **Programme de vols** (ex: `LEG_FUTUR*.xlsx`)
2. **Planning equipage** (ex: `DOA_PROGRAME*.xlsx`)
3. **Parametres de simulation** (ex: `reference_params_RAM.csv`)

Fichiers optionnels:

- table turnaround par aeroport/type avion,
- autorisations avion/aeroport pour l'optimizer(Switch).

## Logique de simulation

Le moteur (`src/simulation.py`) applique:

1. **Retard initial**
   - mode `auto`: tirage Gamma,
   - mode `manuel`: injection ciblee.

2. **Turnaround stochastique**
   - chaine de Markov a 3 etats (Normal / Alerte / Bloque),
   - multiplicateurs appliques au turnaround de base.

3. **Propagation des contraintes**
   - **constrainte avion**: un avion repart apres arrivee precedente + turnaround effectif,
   - **constrainte equipage**: un equipage ne peut partir qu'apres l'arrivee du vol precedent de sa sequence.

4. **Depart reel**
   - `dep_actual = max(dep_programme + retard_initial, earliest_avion, earliest_crew)`.

Sorties calculees: `dep_delay`, `arr_delay`, `on_time`, OTP, percentiles et indicateurs de propagation.

## Fonctionnalites UI (`src/app.py`)

- KPI globaux (OTP, propagation, vols impactes)
- mode Auto / Manuel
- scenario **Hub Congestion** (aeroport + facteur)
- scenario **Slack/Tampon variable** (portee configurable)
- tableau detaille par vol
- section **Reaffectation avion** (si activee)
- exports CSV des resultats

## Installation locale

Prerequis:

- Python 3.11+
- `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Lancer l'application:

```bash
streamlit run src/app.py
```

Puis ouvrir:

- `http://localhost:8501`

## Lancement avec Docker

```bash
docker build -t delaypropagation:latest .
docker run --rm -p 8501:8501 delaypropagation:latest
```

Puis ouvrir:

- `http://localhost:8501`

## Stack technique

- Streamlit
- Pandas / NumPy
- Plotly
- NetworkX
- OpenPyXL.

