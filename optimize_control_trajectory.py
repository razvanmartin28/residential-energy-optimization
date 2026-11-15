"""
optimize_control_trajectory.py

Etapa 2 din noua abordare (control-trajectory optimization): functia
obiectiv pentru optimizarea traiectoriei de control a bateriei pe cele 24
de ore ale zilei fixate in optimal_control_24h.py (TARGET_DATE), care va
fi folosita mai tarziu de scipy.optimize.differential_evolution.

Vectorul de decizie x are 48 de variabile, exact cum a specificat
coordonatorul:
  - x[0:24]  -> Actiune_i in {-1, 0, 1} (intreg):
                    -1 = incarcare, 0 = asteptare, 1 = descarcare
                (aceeasi conventie de semn ca in BatteryModel din
                solar_pv_integration.py: p_ESS > 0 inseamna descarcare).
  - x[24:48] -> Putere_i in [0, max_power_kW] (continuu): magnitudinea
                puterii bateriei la ora i, in kW.

Puterea efectiva (semnata) a bateriei la ora i este:
    p_ESS_i [kW] = Actiune_i * Putere_i

Acest modul NU ruleaza inca optimizarea - doar defineste functia
obiectiv, care va fi pasata intr-o etapa urmatoare catre:
    differential_evolution(objective_function_24h, bounds=...,
                            args=(df_24h, battery_params))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution

# Sub acest prag (Wh), productia PV a orei e considerata "zero" - nu se
# poate justifica o incarcare "din soare" daca PV e practic 0 (ar insemna,
# de fapt, incarcare din reteaua electrica, interzisa de articol - Eq.
# 24-25 din Simoiu et al. 2021).
PV_ZERO_THRESHOLD_WH = 1.0

# Toleranta numerica pentru compararea SOC cu pragurile soc_min/soc_max -
# evita respingeri false cauzate doar de erori de rotunjire in virgula
# mobila (ex. SOC_next = 0.19999999999999998 in loc de 0.2 exact).
SOC_TOLERANCE = 1e-9


def _unpack_decision_vector(x, max_power_kW):
    """
    Despacheteaza vectorul de decizie x (48 elemente) si calculeaza
    puterea semnata p_ESS (kW) a bateriei pentru fiecare din cele 24 de
    ore.

    Actiunea si puterea sunt clipate defensiv la intervalele lor admise
    (actiune in {-1, 0, 1}, putere in [0, max_power_kW]) - acelasi motiv
    pentru care s-a facut clipping si in pv_battery_optimization.py: pasul
    de "polish" al lui differential_evolution poate evalua puncte usor in
    afara bounds-urilor declarate.

    Parameters
    ----------
    x : array-like, lungime 48
        x[0:24]  - actiunea orara (asteptata in {-1, 0, 1}).
        x[24:48] - magnitudinea puterii orare (asteptata in
                   [0, max_power_kW]).
    max_power_kW : float
        Limita superioara admisa pentru magnitudinea puterii bateriei.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (action, power_kW, p_ess_kWh) - trei array-uri de lungime 24.
        `p_ess_kWh` e puterea semnata a bateriei ora cu ora, in kWh
        (numeric egala cu puterea in kW, pentru dt = 1h); p_ess > 0 =
        descarcare, p_ess < 0 = incarcare (aceeasi conventie ca in
        BatteryModel).
    """
    x = np.asarray(x, dtype=float)
    action = np.clip(np.rint(x[:24]), -1, 1).astype(int)
    power_kW = np.clip(x[24:48], 0.0, max_power_kW)
    p_ess_kWh = action * power_kW
    return action, power_kW, p_ess_kWh


def objective_function_24h(x, df_24h, battery_params):
    """
    Functia obiectiv (NEEG) pentru optimizarea traiectoriei de control a
    bateriei pe cele 24 de ore ale zilei `df_24h`.

    Simuleaza evolutia SOC ora cu ora, pornind de la
    `battery_params["soc_initial"]`, aplicand actiunea/puterea cerute de
    `x` la fiecare ora (i = 0..23, in ordine). Daca la orice ora se
    incalca una din regulile fizice de mai jos, simularea se opreste
    imediat si functia returneaza `np.inf` (punctul e infezabil):

      1. Fara incarcare din retea: actiune de incarcare (-1) intr-o ora cu
         productie PV sub `PV_ZERO_THRESHOLD_WH`.
      2. Protectie descarcare: SOC-ul de dupa ora ar scadea sub
         `battery_params["soc_min"]`.
      3. Protectie supraincarcare: SOC-ul de dupa ora ar creste peste
         `battery_params["soc_max"]`.

    Daca cele 24 de ore trec fara nicio penalizare, se calculeaza si se
    returneaza NEEG-ul zilei, in Wh:
        NEEG = sum_i |P_PV_i + P_ESS_i - P_Load_i|
    unde P_ESS_i e puterea bateriei convertita din kWh in Wh
    (p_ess_kWh_i * 1000), pastrand aceeasi conventie de semn (> 0 =
    descarcare = "sursa" pentru consum, exact ca PV).

    Parameters
    ----------
    x : array-like, lungime 48
        Vectorul de decizie: x[0:24] = actiune orara (-1/0/1),
        x[24:48] = magnitudinea puterii orare (kW), in [0, max_power_kW].
    df_24h : pd.DataFrame
        24 de randuri (cate unul pe ora, in ordine cronologica), cu
        coloanele "Consum_Wh" si "PV_Wh" (ex. rezultatul lui
        `optimal_control_24h.select_day`).
    battery_params : dict
        Cheile asteptate: "capacity_kWh", "soc_initial", "soc_min",
        "soc_max", "max_power_kW".

    Returns
    -------
    float
        NEEG-ul zilei (Wh) daca traiectoria e fezabila, altfel `np.inf`.
    """
    capacity_kWh = battery_params["capacity_kWh"]
    soc_min = battery_params["soc_min"]
    soc_max = battery_params["soc_max"]
    max_power_kW = battery_params["max_power_kW"]

    action, _power_kW, p_ess_kWh = _unpack_decision_vector(x, max_power_kW)

    pv_Wh = df_24h["PV_Wh"].to_numpy()
    load_Wh = df_24h["Consum_Wh"].to_numpy()

    soc = battery_params["soc_initial"]
    neeg_Wh = 0.0

    for i in range(24):
        # Regula 1: fara incarcare din retea (incarcare ceruta, dar fara
        # soare in ora respectiva).
        if action[i] == -1 and pv_Wh[i] < PV_ZERO_THRESHOLD_WH:
            return np.inf

        soc_next = soc - p_ess_kWh[i] / capacity_kWh

        # Regula 2: protectie descarcare (SOC ar scadea sub prag).
        if soc_next < soc_min - SOC_TOLERANCE:
            return np.inf
        # Regula 3: protectie supraincarcare (SOC ar creste peste prag).
        if soc_next > soc_max + SOC_TOLERANCE:
            return np.inf

        soc = soc_next

        p_ess_Wh_i = p_ess_kWh[i] * 1000.0
        neeg_Wh += abs(pv_Wh[i] + p_ess_Wh_i - load_Wh[i])

    return neeg_Wh


def _simulate_trajectory_clipped(x, df_24h, battery_params):
    """
    Varianta "proiectie pe spatiul fezabil" (Metoda 2) a simularii din
    `objective_function_24h`: in loc sa opreasca simularea si sa
    returneze `np.inf` la prima incalcare a unei reguli fizice, FIECARE
    ora e mai intai "taiata" (clipata) la ce e efectiv posibil, inainte
    sa fie aplicata:

      - Actiune de incarcare (-1) intr-o ora fara soare (PV sub
        `PV_ZERO_THRESHOLD_WH`) -> decizia e suprascrisa la 0 (asteptare)
        - nu se incarca niciodata din retea.
      - Incarcare (-1) cu PV disponibil -> magnitudinea puterii e taiata
        la energia care mai incape pana la `soc_max`.
      - Descarcare (1) -> magnitudinea puterii e taiata la energia
        disponibila pana la `soc_min`.

    Astfel, orice vector de decizie `x` (oricat de "nefizic") produce o
    traiectorie 100% fezabila - spatiul de cautare al lui
    `differential_evolution` devine continuu navigabil (fara "zidul" de
    `np.inf`), ceea ce ii permite algoritmului sa foloseasca diferentele
    de NEEG dintre indivizi ca semnal de cautare, in loc sa se blocheze
    intr-un peisaj dominat de infezabilitate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (p_ess_kWh, soc_trajectory) - puterea bateriei efectiv aplicata
        (dupa proiectie, in kWh, aceeasi conventie de semn: > 0 =
        descarcare) si evolutia SOC, ambele de lungime 24.
    """
    capacity_kWh = battery_params["capacity_kWh"]
    soc_min = battery_params["soc_min"]
    soc_max = battery_params["soc_max"]
    max_power_kW = battery_params["max_power_kW"]

    action_req, power_req, _ = _unpack_decision_vector(x, max_power_kW)
    pv_Wh = df_24h["PV_Wh"].to_numpy()

    p_ess_kWh = np.zeros(24)
    soc_trajectory = np.empty(24)
    soc = battery_params["soc_initial"]

    for i in range(24):
        action_i = action_req[i]
        power_i = power_req[i]

        if action_i == 0 or power_i <= 0:
            p_ess_i = 0.0
        elif action_i == -1:
            if pv_Wh[i] < PV_ZERO_THRESHOLD_WH:
                # Fara incarcare din retea: suprascriem decizia la 0.
                p_ess_i = 0.0
            else:
                max_charge_kWh = max((soc_max - soc) * capacity_kWh, 0.0)
                p_ess_i = -min(power_i, max_charge_kWh)
        else:  # action_i == 1 (descarcare)
            max_discharge_kWh = max((soc - soc_min) * capacity_kWh, 0.0)
            p_ess_i = min(power_i, max_discharge_kWh)

        soc = soc - p_ess_i / capacity_kWh
        soc = float(np.clip(soc, soc_min, soc_max))  # siguranta numerica

        p_ess_kWh[i] = p_ess_i
        soc_trajectory[i] = soc

    return p_ess_kWh, soc_trajectory


def objective_function_24h_clipped(x, df_24h, battery_params):
    """
    Functia obiectiv (NEEG) pentru Metoda 2 (proiectie pe spatiul
    fezabil - vezi `_simulate_trajectory_clipped`): fara `np.inf`, orice
    `x` produce o traiectorie fezabila, deci un NEEG finit.

    Aceasta e functia folosita efectiv de `run_optimal_control_24h`
    (in locul lui `objective_function_24h`, cu penalizare tip "zid" -
    pastrata mai sus neschimbata, ca referinta/comparatie), dupa ce
    varianta cu `np.inf` a dus la convergenta prematura a lui
    `differential_evolution` intr-o solutie mai slaba decat baseline-ul
    greedy.

    Parameters / Returns: identice cu `objective_function_24h`.
    """
    p_ess_kWh, _soc_trajectory = _simulate_trajectory_clipped(x, df_24h, battery_params)
    p_ess_Wh = p_ess_kWh * 1000.0

    pv_Wh = df_24h["PV_Wh"].to_numpy()
    load_Wh = df_24h["Consum_Wh"].to_numpy()

    return float(np.sum(np.abs(pv_Wh + p_ess_Wh - load_Wh)))


def _reconstruct_trajectory(x, df_24h, battery_params):
    """
    Reconstruieste, dintr-un vector de decizie `x` (presupus fezabil - de
    ex. `result.x` intors de `differential_evolution`), seria orara a
    puterii bateriei si evolutia SOC-ului pe cele 24 de ore, exact ca in
    bucla din `objective_function_24h`, dar fara sa se opreasca la
    `np.inf` - in loc, SOC e clipat defensiv in [soc_min, soc_max] (doar
    pentru erori de rotunjire in virgula mobila la limita).

    Returns
    -------
    pd.DataFrame
        Copie a `df_24h`, cu coloanele suplimentare "Action" (-1/0/1),
        "Power_kW" (magnitudinea puterii), "PESS_Wh" (puterea semnata a
        bateriei, in Wh - aceeasi conventie ca `simulate_battery_over_period`
        din solar_pv_integration.py) si "SOC" (starea de incarcare dupa
        fiecare ora).
    """
    capacity_kWh = battery_params["capacity_kWh"]
    soc_min = battery_params["soc_min"]
    soc_max = battery_params["soc_max"]
    max_power_kW = battery_params["max_power_kW"]

    action, power_kW, p_ess_kWh = _unpack_decision_vector(x, max_power_kW)
    p_ess_Wh = p_ess_kWh * 1000.0

    soc_trajectory = np.empty(24)
    soc = battery_params["soc_initial"]
    for i in range(24):
        soc = soc - p_ess_kWh[i] / capacity_kWh
        soc = float(np.clip(soc, soc_min, soc_max))
        soc_trajectory[i] = soc

    trajectory_df = df_24h.copy()
    trajectory_df["Action"] = action
    trajectory_df["Power_kW"] = power_kW
    trajectory_df["PESS_Wh"] = p_ess_Wh
    trajectory_df["SOC"] = soc_trajectory
    return trajectory_df


def _reconstruct_trajectory_clipped(x, df_24h, battery_params):
    """
    Echivalentul lui `_reconstruct_trajectory`, dar pentru Metoda 2
    (proiectie pe spatiul fezabil - `_simulate_trajectory_clipped`).
    Foloseste EXACT aceeasi simulare ca `objective_function_24h_clipped`,
    ca traiectoria raportata sa corespunda mereu scorului NEEG calculat.

    Retine si actiunea/puterea CERUTE de `x` (coloanele
    "Action_requested"/"Power_requested_kW"), separat de "PESS_Wh" (ce s-a
    aplicat efectiv, dupa proiectie) - util pentru a vedea unde
    optimizatorul a "cerut" ceva ce a fost taiat de regulile fizice.

    Returns
    -------
    pd.DataFrame
        Copie a `df_24h`, cu coloanele suplimentare "Action_requested",
        "Power_requested_kW", "PESS_Wh" si "SOC".
    """
    action_req, power_req, _ = _unpack_decision_vector(x, battery_params["max_power_kW"])
    p_ess_kWh, soc_trajectory = _simulate_trajectory_clipped(x, df_24h, battery_params)

    trajectory_df = df_24h.copy()
    trajectory_df["Action_requested"] = action_req
    trajectory_df["Power_requested_kW"] = power_req
    trajectory_df["PESS_Wh"] = p_ess_kWh * 1000.0
    trajectory_df["SOC"] = soc_trajectory
    return trajectory_df


def run_optimal_control_24h(
    df_24h,
    battery_params,
    seed=42,
    workers=-1,
    updating="deferred",
    popsize=30,
    strategy="best1bin",
):
    """
    Configureaza si ruleaza `differential_evolution` pe
    `objective_function_24h_clipped` (Metoda 2 - proiectie pe spatiul
    fezabil, vezi `_simulate_trajectory_clipped`), pentru a gasi
    traiectoria orara optima (actiune + putere, pe cele 24 de ore) care
    minimizeaza NEEG-ul zilei `df_24h`, cu sistemul fixat descris de
    `battery_params`.

    S-a preferat aceasta varianta (fata de `objective_function_24h`, cu
    penalizare `np.inf`) dupa ce prima rulare a aratat ca "zidul" de
    infezabilitate face suprafata de cautare prea abrupta pentru
    `differential_evolution` cu setarile implicite - algoritmul converge
    ("success=True") dar la o solutie mai slaba decat baseline-ul greedy.
    Cu proiectia pe spatiul fezabil, orice individ e valid fizic, deci
    intreaga populatie ofera semnal util de cautare. Am marit si
    `popsize` (15 -> 30) si am fixat explicit `strategy='best1bin'`,
    pentru o explorare mai larga a celor 48 de variabile.

    Configuratia celor 48 de variabile de decizie (exacta, ca specificat):
      - bounds: primele 24 -> (-1, 1) (actiune); urmatoarele 24 ->
        (0, max_power_kW) (putere, extras din `battery_params`).
      - integrality: primele 24 -> True (intregi); urmatoarele 24 ->
        False (continue).

    Parameters
    ----------
    df_24h : pd.DataFrame
        24 de randuri (cate unul pe ora), cu coloanele "Consum_Wh" si
        "PV_Wh".
    battery_params : dict
        Cheile asteptate: "capacity_kWh", "soc_initial", "soc_min",
        "soc_max", "max_power_kW".
    seed, workers, updating
        Trecute direct catre `differential_evolution`; parametrii de
        populatie (popsize, maxiter, mutation, recombination etc.) raman
        pe valorile implicite ale scipy.

    Returns
    -------
    tuple[OptimizeResult, pd.DataFrame, dict, float]
        (result, optimized_df, metrics, neeg_Wh) - `result` e obiectul
        intors de `differential_evolution`; `optimized_df` e traiectoria
        orara reconstruita (vezi `_reconstruct_trajectory_clipped`);
        `metrics` e dict-ul intors de
        `compute_self_consumption_and_sufficiency` (SC, SS, energii in
        kWh); `neeg_Wh` e NEEG-ul traiectoriei optime, in Wh.
    """
    n_hours = 24
    max_power_kW = battery_params["max_power_kW"]

    bounds = [(-1, 1)] * n_hours + [(0, max_power_kW)] * n_hours
    integrality = [True] * n_hours + [False] * n_hours

    result = differential_evolution(
        objective_function_24h_clipped,
        bounds=bounds,
        integrality=integrality,
        args=(df_24h, battery_params),
        seed=seed,
        workers=workers,
        updating=updating,
        popsize=popsize,
        strategy=strategy,
    )

    if not np.isfinite(result.fun):
        print(
            "ATENTIE: differential_evolution nu a gasit nicio traiectorie "
            "fezabila (result.fun = inf) - verifica battery_params / bounds."
        )

    optimized_df = _reconstruct_trajectory_clipped(result.x, df_24h, battery_params)

    # Evita importul greu de solar_pv_integration doar pentru doua functii -
    # se face local, in schimb, pentru a pastra restul modulului (functia
    # obiectiv) usor de folosit fara dependinta de BATEM.
    from solar_pv_integration import compute_neeg_Wh, compute_self_consumption_and_sufficiency

    metrics = compute_self_consumption_and_sufficiency(optimized_df, with_battery=True)
    neeg_Wh = compute_neeg_Wh(optimized_df, with_battery=True)

    return result, optimized_df, metrics, neeg_Wh


def print_baseline_vs_optimized(baseline_metrics, baseline_neeg_Wh, opt_metrics, opt_neeg_Wh, result):
    """
    Printeaza in consola o comparatie scurta intre baseline-ul greedy
    (Etapa 1) si traiectoria optima gasita de `differential_evolution`
    (Etapa 3), pentru aceeasi zi si acelasi sistem fixat.
    """
    improvement_Wh = baseline_neeg_Wh - opt_neeg_Wh
    improvement_pct = (improvement_Wh / baseline_neeg_Wh * 100) if baseline_neeg_Wh > 0 else float("nan")

    print("\n" + "=" * 65)
    print("BASELINE (greedy)  vs.  OPTIMIZAT (differential_evolution)")
    print("=" * 65)
    print(f"{'Indicator':<28}{'Baseline':>15}{'Optimizat':>15}")
    print("-" * 65)
    print(f"{'Autoconsum (SC_disc)':<28}{baseline_metrics['SC']:>15.3f}{opt_metrics['SC']:>15.3f}")
    print(f"{'Autonomie (SS_disc)':<28}{baseline_metrics['SS']:>15.3f}{opt_metrics['SS']:>15.3f}")
    print(f"{'NEEG (Wh)':<28}{baseline_neeg_Wh:>15,.1f}{opt_neeg_Wh:>15,.1f}")
    print("-" * 65)
    print(f"Imbunatatire NEEG: {improvement_Wh:,.1f} Wh ({improvement_pct:.1f}%)")
    print("=" * 65)


def plot_and_save_24h_results(
    df_24h,
    baseline_df,
    optimized_df,
    actiune_bruta,
    putere_bruta,
    house_id,
    house_status_label,
    max_power_kW,
    output_dir,
):
    """
    Etapa 4: genereaza si salveaza 3 grafice Plotly interactive (HTML),
    comparand traiectoria optima (Etapa 3, `differential_evolution`) cu
    baseline-ul greedy (Etapa 1), pe cele 24 de ore ale zilei din
    `df_24h`.

    Organizarea fisierelor: rezultatele merg in folderul casei DEJA
    folosit de `visualizations.py` -
    `output_dir / f"House_{house_id}_{house_status_label}"` (ex.
    "House_2000905_VALIDA") - daca folderul exista deja (dintr-o rulare
    anterioara a acelui script), e refolosit; daca nu, e creat acum.
    Graficele specifice acestei zile merg intr-un sub-folder cu data
    zilei (extrasa din indexul `df_24h`, ex. "1998-12-18"):

        output_dir / f"House_{house_id}_{house_status_label}" / <data_zilei> /
            Power_Balance_24h.html
            SOC_Evolution_24h.html
            Raw_Decision_Vector_24h.html

    Parameters
    ----------
    df_24h : pd.DataFrame
        24 de randuri (Consum_Wh, PV_Wh) - folosit doar pentru a extrage
        data zilei din index.
    baseline_df : pd.DataFrame
        Rezultatul `optimal_control_24h.run_baseline` (coloanele
        "Consum_Wh", "PV_Wh", "PESS_Wh", "SOC") - strategia greedy.
    optimized_df : pd.DataFrame
        Rezultatul `run_optimal_control_24h` (coloanele "Consum_Wh",
        "PV_Wh", "PESS_Wh", "SOC") - traiectoria optima, DUPA proiectia
        pe spatiul fezabil (Metoda 2).
    actiune_bruta, putere_bruta : array-like, lungime 24
        x[0:24] si x[24:48] din `result.x` (differential_evolution) - ce
        a "cerut" optimizatorul pentru fiecare ora, INAINTE ca decizia sa
        fie taiata de regulile fizice din `_simulate_trajectory_clipped`.
    house_id : int
    house_status_label : str
        "VALIDA" sau "INVALIDA" (vezi `data_processing.get_house_status_label`)
        - determina numele folderului casei, exact ca in `visualizations.py`.
    max_power_kW : float
        Limita superioara a puterii bateriei - folosita pentru axa Y a
        Graficului 3.
    output_dir : Path
        Directorul de baza pentru output (ex. `optimal_control_24h.OUTPUT_DIR`).

    Returns
    -------
    tuple[Path, Path, Path]
        (power_balance_path, soc_evolution_path, raw_decision_path)
    """
    data_zilei = df_24h.index[0].strftime("%Y-%m-%d")

    # Kaleido (exportul PNG) nu poate serializa un DatetimeIndex cu fus orar
    # (tz-aware) - folosim copii cu index "naiv" (fara fus), doar pentru
    # trasare; datele originale primite de functie raman neatinse.
    optimized_df = optimized_df.copy()
    optimized_df.index = optimized_df.index.tz_localize(None)
    baseline_df = baseline_df.copy()
    baseline_df.index = baseline_df.index.tz_localize(None)

    house_dir = Path(output_dir) / f"House_{house_id}_{house_status_label}"
    day_dir = house_dir / data_zilei
    day_dir.mkdir(parents=True, exist_ok=True)

    # --- Graficul 1: Balanta de putere (OPTIMIZAT vs. GREEDY) --------------
    fig1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        subplot_titles=(
            "Balanta de putere OPTIMIZATA (differential_evolution)",
            "Balanta de putere GREEDY (baseline)",
        ),
    )

    fig1.add_trace(
        go.Scatter(
            x=optimized_df.index, y=optimized_df["Consum_Wh"], mode="lines",
            name="Consum orar (Wh)", line=dict(color="firebrick", width=1), legendgroup="consum",
        ),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=optimized_df.index, y=optimized_df["PV_Wh"], mode="lines",
            name="Productie PV (Wh)", line=dict(color="seagreen", width=1), legendgroup="pv",
        ),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=optimized_df.index, y=optimized_df["PESS_Wh"], mode="lines",
            name="p_ESS OPTIM (Wh) - (+) descarcare, (-) incarcare",
            line=dict(color="darkorange", width=1),
        ),
        row=1, col=1,
    )
    fig1.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=1, col=1)

    fig1.add_trace(
        go.Scatter(
            x=baseline_df.index, y=baseline_df["Consum_Wh"], mode="lines",
            name="Consum orar (Wh)", line=dict(color="firebrick", width=1),
            legendgroup="consum", showlegend=False,
        ),
        row=2, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=baseline_df.index, y=baseline_df["PV_Wh"], mode="lines",
            name="Productie PV (Wh)", line=dict(color="seagreen", width=1),
            legendgroup="pv", showlegend=False,
        ),
        row=2, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=baseline_df.index, y=baseline_df["PESS_Wh"], mode="lines",
            name="p_ESS GREEDY (Wh) - (+) descarcare, (-) incarcare",
            line=dict(color="mediumpurple", width=1),
        ),
        row=2, col=1,
    )
    fig1.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)

    fig1.update_yaxes(title_text="Energie / Putere (Wh)", row=1, col=1)
    fig1.update_yaxes(title_text="Energie / Putere (Wh)", row=2, col=1)
    fig1.update_xaxes(title_text="Ora", row=2, col=1)
    fig1.update_layout(
        title=f"Balanta de putere pe 24h - Casa {house_id} - {data_zilei}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
    )

    power_balance_path = day_dir / "Power_Balance_24h.html"
    fig1.write_html(str(power_balance_path))
    fig1.write_image(str(day_dir / "Power_Balance_24h.png"), width=1400, height=900, scale=2)

    # --- Graficul 2: Evolutia SOC (OPTIMIZAT vs. GREEDY) --------------------
    fig2 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        subplot_titles=("SOC - traiectorie OPTIMIZATA", "SOC - strategie GREEDY (baseline)"),
    )
    fig2.add_trace(
        go.Scatter(
            x=optimized_df.index, y=optimized_df["SOC"] * 100, mode="lines",
            name="SOC optim (%)", line=dict(color="royalblue", width=1),
            fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=baseline_df.index, y=baseline_df["SOC"] * 100, mode="lines",
            name="SOC baseline (%)", line=dict(color="mediumpurple", width=1),
            fill="tozeroy", fillcolor="rgba(147,112,219,0.15)",
        ),
        row=2, col=1,
    )
    fig2.update_yaxes(title_text="SOC (%)", range=[0, 100], row=1, col=1)
    fig2.update_yaxes(title_text="SOC (%)", range=[0, 100], row=2, col=1)
    fig2.update_xaxes(title_text="Ora", row=2, col=1)
    fig2.update_layout(
        title=f"Evolutia SOC pe 24h - Casa {house_id} - {data_zilei}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
    )

    soc_evolution_path = day_dir / "SOC_Evolution_24h.html"
    fig2.write_html(str(soc_evolution_path))
    fig2.write_image(str(day_dir / "SOC_Evolution_24h.png"), width=1400, height=900, scale=2)

    # --- Graficul 3: Vectorul de decizie BRUT (inainte de proiectie) -------
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(
        go.Scatter(
            x=optimized_df.index,
            y=actiune_bruta,
            mode="markers",
            name="Actiune bruta (-1 = incarcare, 0 = asteptare, 1 = descarcare)",
            marker=dict(color="firebrick", size=10, symbol="diamond"),
        ),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Bar(
            x=optimized_df.index,
            y=putere_bruta,
            name="Putere bruta ceruta (kW)",
            marker=dict(color="darkorange"),
            opacity=0.4,
        ),
        secondary_y=True,
    )
    fig3.update_yaxes(
        title_text="Actiune bruta", range=[-1.2, 1.2], tickvals=[-1, 0, 1], secondary_y=False,
    )
    fig3.update_yaxes(
        title_text="Putere bruta (kW)", range=[0, max_power_kW * 1.05], secondary_y=True,
    )
    fig3.update_xaxes(title_text="Ora")
    fig3.update_layout(
        title=(
            f"Vectorul de decizie BRUT al optimizatorului (inainte de proiectia pe "
            f"spatiul fezabil) - Casa {house_id} - {data_zilei}"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
    )

    raw_decision_path = day_dir / "Raw_Decision_Vector_24h.html"
    fig3.write_html(str(raw_decision_path))
    fig3.write_image(str(day_dir / "Raw_Decision_Vector_24h.png"), width=1400, height=900, scale=2)

    print(f"\nGrafic salvat: {power_balance_path}")
    print(f"Grafic salvat: {soc_evolution_path}")
    print(f"Grafic salvat: {raw_decision_path}")

    return power_balance_path, soc_evolution_path, raw_decision_path


if __name__ == "__main__":
    import data_processing as dp
    import optimal_control_24h as oc
    import solar_pv_integration as pvint

    print(
        f"Sistem fixat: {oc.N_PANELS} panouri, baterie {oc.BATTERY_CAPACITY_KWH:.1f} kWh, "
        f"SOC in [{oc.SOC_MIN:.0%}, {oc.SOC_MAX:.0%}] (initial {oc.SOC_INITIAL:.0%}), "
        f"putere max {oc.MAX_POWER_KW:.1f} kW"
    )
    print(f"Casa: {pvint.TARGET_HOUSE_ID}   Ziua selectata: {oc.TARGET_DATE}\n")

    # 1-2. Acelasi pipeline ca in optimal_control_24h.py: consum orar +
    # meteo + centrala PV fixata + selectia zilei. Pastram si
    # `valid_houses_df`, ca sa aflam statusul (VALIDA/INVALIDA) casei si
    # sa refolosim folderul deja creat de visualizations.py pentru ea.
    (
        _house_df,
        _appliance_df,
        valid_houses_df,
        _consumption_df,
        _filtered_consumption_df,
        _cleaned_consumption_df,
        _spike_summary,
        _hourly_appliance_df,
        hourly_house_df,
    ) = dp.main()
    house_status_label = dp.get_house_status_label(valid_houses_df, pvint.TARGET_HOUSE_ID)
    start_date, end_date, house_hourly_df = pvint.get_house_consumption_period(
        hourly_house_df, pvint.TARGET_HOUSE_ID
    )
    solar_model = pvint.build_solar_model(start_date, end_date)
    pv_plant = oc.build_fixed_pv_plant(solar_model)
    combined_df = pvint.align_pv_with_consumption(pv_plant, solar_model, house_hourly_df)
    day_df = oc.select_day(combined_df, oc.TARGET_DATE)

    battery_params = {
        "capacity_kWh": oc.BATTERY_CAPACITY_KWH,
        "soc_initial": oc.SOC_INITIAL,
        "soc_min": oc.SOC_MIN,
        "soc_max": oc.SOC_MAX,
        "max_power_kW": oc.MAX_POWER_KW,
    }

    # 3. Baseline (Etapa 1) - strategia greedy, pentru comparatie.
    baseline_df, _baseline_battery, baseline_metrics, baseline_neeg_Wh = oc.run_baseline(day_df)

    # 4. Etapa 3: optimizarea traiectoriei de control cu differential_evolution.
    print("Rulez differential_evolution pe cele 48 de variabile de decizie...")
    result, optimized_df, opt_metrics, opt_neeg_Wh = run_optimal_control_24h(day_df, battery_params)

    print("\nTraiectoria optima (actiune/putere CERUTE, PESS_Wh aplicat dupa proiectie, SOC) pe cele 24 de ore:")
    print(
        optimized_df[
            ["Consum_Wh", "PV_Wh", "Action_requested", "Power_requested_kW", "PESS_Wh", "SOC"]
        ].round(3).to_string()
    )

    print_baseline_vs_optimized(baseline_metrics, baseline_neeg_Wh, opt_metrics, opt_neeg_Wh, result)

    # 5. Etapa 4: grafice Plotly - traiectoria optima vs. baseline greedy,
    # plus vectorul de decizie brut (inainte de proiectia pe spatiul fezabil).
    actiune_bruta, putere_bruta, _ = _unpack_decision_vector(result.x, oc.MAX_POWER_KW)
    plot_and_save_24h_results(
        df_24h=day_df,
        baseline_df=baseline_df,
        optimized_df=optimized_df,
        actiune_bruta=actiune_bruta,
        putere_bruta=putere_bruta,
        house_id=pvint.TARGET_HOUSE_ID,
        house_status_label=house_status_label,
        max_power_kW=oc.MAX_POWER_KW,
        output_dir=oc.OUTPUT_DIR,
    )
