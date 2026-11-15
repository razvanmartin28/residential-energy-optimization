"""
pv_battery_optimization.py

Optimizare combinata (dimensionare PV+baterie + cativa parametri de
control) cu scipy.optimize.differential_evolution, minimizand NEEG

Varianta SIMPLIFICATA fata de articol: acolo vectorul de decizie contine
traiectoria orara completa p_ESS (una per ora, ~8760 variabile/an) plus
dimensionarea, rezolvata cu MILP. Un optimizator stocastic global ca
differential_evolution nu se scaleaza la atatea variabile, asa ca aici
decizia se reduce la 4 variabile: dimensionare (n panouri, capacitate
baterie) + cativa parametri ai politicii de control existente (BatteryModel
din solar_pv_integration.py), NU traiectoria orara insasi.

Variabile de decizie x = (n_panouri, capacitate_baterie_kWh, soc_max,
putere_maxima_descarcare_kW).

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import LinearConstraint, differential_evolution

from batem.core.solar import PVplant

import data_processing as dp
import solar_pv_integration as pvint
from solar_pv_integration import (
    BatteryModel,
    compute_neeg_Wh,
    compute_self_consumption_and_sufficiency,
    simulate_battery_over_period,
)

# --- Constrangere fizica: suprafata de acoperis disponibila ----------------
#  - o sectiune de acoperis rezidential orientata favorabil.
ROOF_AREA_M2 = 30.0

PANEL_AREA_M2 = pvint.PANEL_WIDTH_M * pvint.PANEL_HEIGHT_M
N_PANELS_MAX = max(1, int(ROOF_AREA_M2 // PANEL_AREA_M2))

# --- Constrangere economica: buget maxim ------------------------------------
# Costuri estimate la nivel de "procurare instalator" (pretul componentei/
# produsului, INAINTE de manopera de instalare completa) 
PANEL_COST_EUR = 200.0             # per panou (hardware, monocristalin ~400W)
BATTERY_COST_EUR_PER_KWH = 500.0   # per kWh capacitate baterie (produs, fara manopera)
MAX_BUDGET_EUR = 10000.0

# --- Parametri ai bateriei FIXATI  ---
# SOC_min: prag de protectie a duratei de viata bateriei -
# din studiul de caz al articolului (Tabelul 1: SOC_min = 0.2)
FIXED_SOC_MIN = 0.2
# SOC initial: articolul arata (Sectiunea 5.3) ca alegerea SOC initial nu
# schimba SC/SS pe un orizont lung, doar momentul (dis)incarcarii - deci nu
# are rost sa fie variabila de decizie.
FIXED_SOC_INITIAL = 0.2

# --- Bounds pentru variabilele de decizie -----------------------------------
# x = (n_panouri, capacitate_baterie_kWh, soc_max, putere_maxima_descarcare_kW)
BOUNDS = [
    (1, N_PANELS_MAX),   # limitat de suprafata acoperisului
    (0.5, 15.0),         # capacitate baterie (kWh) - de la aproape nimic pana la o baterie mare rezidentiala
    (0.6, 0.95),         # soc_max - cat din capacitate e efectiv utilizabil
    (0.5, 10.0),         # putere maxima de descarcare (kW)
]


def build_base_series():
    """
    Pregateste o sg data, in afara functiei obiectiv: modelul 
    solar BATEM, orientarea optima si productia PV pentru un singur panou.

    La montaj PLAN (fara umbrire intre randuri), puterea PV se
    scaleaza LINIAR cu numarul de panouri (fiecare panou contribuie
    independent, iar factorul termic din model depinde doar de iradianta si
    temperatura exterioara, nu de numarul de panouri).

    Returns
    tuple[pd.DataFrame, float, float, str]
        (base_df, best_exposure_deg, best_slope_deg, house_status_label) -
        base_df are coloanele "Consum_Wh" si "PV_1panel_Wh" (Wh/ora,
        productia unui singur panou); house_status_label e "VALIDA" sau
        "INVALIDA" (vezi `data_processing.get_house_status_label`), pentru
        a salva rezultatele in folderul deja existent al casei
        (House_<id>_<VALIDA|INVALIDA>, ca in visualizations.py).
    """
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

    best_exposure_deg, best_slope_deg = solar_model.best_direction(
        initial_slope_deg=150, initial_exposure_deg=0
    )
    print(
        f"Orientare optima: expunere={best_exposure_deg:.1f} deg, panta={best_slope_deg:.1f} deg"
    )

    pv_plant_1panel = PVplant(
        solar_model=solar_model,
        exposure_deg=best_exposure_deg,
        slope_deg=best_slope_deg,
        mount_type=pvint.MOUNT_TYPE,
        number_of_panels=1,
        panel_width_m=pvint.PANEL_WIDTH_M,
        panel_height_m=pvint.PANEL_HEIGHT_M,
        pv_efficiency=pvint.PANEL_EFFICIENCY,
        temperature_coefficient=pvint.PANEL_TEMPERATURE_COEFFICIENT,
    )

    combined_df = pvint.align_pv_with_consumption(pv_plant_1panel, solar_model, house_hourly_df)
    base_df = combined_df.rename(columns={"PV_Wh": "PV_1panel_Wh"})
    return base_df, best_exposure_deg, best_slope_deg, house_status_label


def build_scenario_df(base_df, n_panels, capacity_kWh, soc_max, max_discharge_kW):
    """
    Construieste scenariul (productie PV scalata cu n_panels + simularea
    bateriei) pentru o configuratie data. Functie comuna, folosita atat de
    `neeg_objective` cat si pentru inspectarea rezultatului final.

    Returns
    -------
    tuple[pd.DataFrame, BatteryModel]
        (df_with_battery, battery)
    """
    pv_Wh = base_df["PV_1panel_Wh"] * n_panels
    df = pd.DataFrame({"Consum_Wh": base_df["Consum_Wh"], "PV_Wh": pv_Wh}, index=base_df.index)

    battery = BatteryModel(
        capacity_kWh=capacity_kWh,
        soc_initial=FIXED_SOC_INITIAL,
        soc_min=FIXED_SOC_MIN,
        soc_max=soc_max,
        max_discharge_power_kW=max_discharge_kW,
    )
    df_with_battery = simulate_battery_over_period(df, battery)
    return df_with_battery, battery


def _unpack_and_clip(x):
    """
    Extrage (n_panouri, capacitate_kWh, soc_max, putere_max_descarcare_kW)
    din vectorul de decizie `x` si le limiteaza (clip) strict la `BOUNDS`.

    """
    n_panels = int(round(np.clip(x[0], BOUNDS[0][0], BOUNDS[0][1])))
    capacity_kWh = float(np.clip(x[1], BOUNDS[1][0], BOUNDS[1][1]))
    soc_max = float(np.clip(x[2], BOUNDS[2][0], BOUNDS[2][1]))
    max_discharge_kW = float(np.clip(x[3], BOUNDS[3][0], BOUNDS[3][1]))
    return n_panels, capacity_kWh, soc_max, max_discharge_kW


def neeg_objective(x, base_df):
    """
    Functia obiectiv pentru `differential_evolution`: NEEG 
    pentru o configuratie (n_panouri, capacitate_baterie_kWh,
    soc_max, putere_maxima_descarcare_kW), cu bateria simulata pe tot
    intervalul disponibil (~1 an).

    `differential_evolution` minimizeaza mereu `func`, deci NEEG (energia
    schimbata cu reteaua) e alegerea directa - articolul arata explicit ca
    minimizarea NEEG e echivalenta cu maximizarea simultana a SC si SS
    (spre deosebire de a optimiza SC sau SS separat, care ar da mereu
    solutia trivial minima/maxima).
    """
    n_panels, capacity_kWh, soc_max, max_discharge_kW = _unpack_and_clip(x)

    df_with_battery, _ = build_scenario_df(base_df, n_panels, capacity_kWh, soc_max, max_discharge_kW)
    return compute_neeg_Wh(df_with_battery, with_battery=True)


def plot_optimized_battery_result(df_optimized, house_id):
    """
    Figura Plotly cu doua subgrafice suprapuse pe aceeasi axa de timp
    (rezultatul simularii CU strategia si dimensionarea optimizate de
    `differential_evolution`, din `df_optimized` - trebuie sa contina
    coloanele "Consum_Wh", "PV_Wh", "PESS_Wh", "SOC"):

      1. Balanta de putere: consum, productie PV si puterea bateriei
         p_ESS (Wh) - convetia de semn face ca p_ESS sa oscileze in zona
         NEGATIVA la incarcare si POZITIVA la descarcare.
      2. Evolutia SOC (%): cand bateria se incarca (p_ESS < 0 in graficul
         1), SOC creste vizibil aici; cand se descarca (p_ESS > 0), SOC
         scade - direct din Ec. 10 (SOC_{i+1} = SOC_i - p_ESS*dt/Q_nom).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=(
            "Balanta de putere: consum, productie PV, putere baterie (p_ESS)",
            "Evolutia SOC a bateriei",
        ),
    )

    # --- Subgraficul 1: balanta de putere ---
    fig.add_trace(
        go.Scatter(
            x=df_optimized.index,
            y=df_optimized["Consum_Wh"],
            mode="lines",
            name="Consum orar (Wh)",
            line=dict(color="firebrick", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimized.index,
            y=df_optimized["PV_Wh"],
            mode="lines",
            name="Productie PV orara (Wh)",
            line=dict(color="seagreen", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimized.index,
            y=df_optimized["PESS_Wh"],
            mode="lines",
            name="Putere baterie p_ESS (Wh) - (+) descarcare, (-) incarcare",
            line=dict(color="darkorange", width=1),
        ),
        row=1,
        col=1,
    )
    # Linie de referinta la 0 - separa vizual incarcare (sub 0) de descarcare (peste 0).
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=1, col=1)

    # --- Subgraficul 2: SOC ---
    fig.add_trace(
        go.Scatter(
            x=df_optimized.index,
            y=df_optimized["SOC"] * 100,
            mode="lines",
            name="SOC (%)",
            line=dict(color="royalblue", width=1),
            fill="tozeroy",
            fillcolor="rgba(65,105,225,0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Energie / Putere (Wh)", row=1, col=1)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="Data/Ora", row=2, col=1)

    fig.update_layout(
        title=f"Strategie optimizata de baterie (ESS) - Casa {house_id}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    )
    return fig


def print_optimization_comparison(df_optimized):
    """
    Printeaza o comparatie clara, cu diferente calculate explicit, intre
    scenariul FARA baterie si CU baterie - folosind ACEEASI dimensionare PV
    optimizata (`df_optimized`), diferenta fiind doar prezenta/absenta
    bateriei controlate prin strategia optimizata. Astfel se izoleaza
    exact contributia bateriei, separat de schimbarea numarului de panouri.

    Indicatori (Ec. 1, 2, 3 din articol), fiecare ca o singura valoare
    agregata pe tot anul simulat: Autoconsum (SC_disc), Autonomie (SS_disc)
    si NEEG.
    """
    metrics_no_batt = compute_self_consumption_and_sufficiency(df_optimized, with_battery=False)
    metrics_with_batt = compute_self_consumption_and_sufficiency(df_optimized, with_battery=True)

    neeg_no_batt_kWh = compute_neeg_Wh(df_optimized, with_battery=False) / 1000
    neeg_with_batt_kWh = compute_neeg_Wh(df_optimized, with_battery=True) / 1000

    sc_delta = metrics_with_batt["SC"] - metrics_no_batt["SC"]
    ss_delta = metrics_with_batt["SS"] - metrics_no_batt["SS"]
    neeg_delta_kWh = neeg_with_batt_kWh - neeg_no_batt_kWh
    neeg_delta_pct = (neeg_delta_kWh / neeg_no_batt_kWh * 100) if neeg_no_batt_kWh else float("nan")

    print("\n" + "=" * 70)
    print("COMPARATIE FINALA - inainte de baterie vs. cu baterie (strategie optimizata)")
    print("=" * 70)
    print(f"{'':26s}{'Fara baterie':>16s}{'Cu baterie (opt)':>19s}{'Diferenta':>14s}")
    print(
        f"{'Autoconsum (SC_disc)':26s}{metrics_no_batt['SC']:>16.3f}"
        f"{metrics_with_batt['SC']:>19.3f}{sc_delta:>+14.3f}"
    )
    print(
        f"{'Autonomie (SS_disc)':26s}{metrics_no_batt['SS']:>16.3f}"
        f"{metrics_with_batt['SS']:>19.3f}{ss_delta:>+14.3f}"
    )
    print(
        f"{'NEEG (kWh)':26s}{neeg_no_batt_kWh:>16,.1f}"
        f"{neeg_with_batt_kWh:>19,.1f}{neeg_delta_kWh:>+14,.1f}"
    )
    print("-" * 70)
    if neeg_delta_kWh < 0:
        print(f"Bateria reduce NEEG cu {abs(neeg_delta_pct):.1f}% fata de scenariul fara baterie.")
    else:
        print(f"Bateria creste NEEG cu {neeg_delta_pct:.1f}% fata de scenariul fara baterie.")
    print("=" * 70)


def run_optimization():

    base_df, _, _, house_status_label = build_base_series()

    # Constrangere economica (LinearConstraint): cost_panou*n + cost_kWh*capacitate <= buget.
    # soc_max si putere_maxima_descarcare nu au cost propriu (nu cumperi
    # "SOC mai mare", cumperi capacitate/panouri) - coeficienti 0.
    budget_constraint = LinearConstraint(
        [[PANEL_COST_EUR, BATTERY_COST_EUR_PER_KWH, 0.0, 0.0]],
        lb=-np.inf,
        ub=MAX_BUDGET_EUR,
    )

    print(
        f"\nConstrangeri: n_panouri in [1,{N_PANELS_MAX}] (acoperis {ROOF_AREA_M2:.0f} m2), "
        f"buget <= {MAX_BUDGET_EUR:.0f} EUR ({PANEL_COST_EUR:.0f} EUR/panou, "
        f"{BATTERY_COST_EUR_PER_KWH:.0f} EUR/kWh baterie)"
    )

    result = differential_evolution(
        neeg_objective,
        bounds=BOUNDS,
        args=(base_df,),
        constraints=(budget_constraint,),
        integrality=[True, False, False, False],
        seed=42,  # reproductibilitate
    )

    n_opt, capacity_opt, soc_max_opt, max_discharge_opt = _unpack_and_clip(result.x)
    neeg_opt_kWh = result.fun / 1000
    cost_opt = PANEL_COST_EUR * n_opt + BATTERY_COST_EUR_PER_KWH * capacity_opt

    df_opt, battery_opt = build_scenario_df(base_df, n_opt, capacity_opt, soc_max_opt, max_discharge_opt)
    metrics_opt = compute_self_consumption_and_sufficiency(df_opt, with_battery=True)

    # Configuratia FIXA din solar_pv_integration.py, pentru comparatie directa.
    df_baseline, _ = build_scenario_df(
        base_df,
        pvint.NUMBER_OF_PANELS,
        pvint.BATTERY_CAPACITY_KWH,
        pvint.BATTERY_SOC_MAX,
        pvint.BATTERY_CAPACITY_KWH,  # echivalent cu max_discharge_power_kW=None (1C) din BatteryModel implicit
    )
    neeg_baseline_kWh = compute_neeg_Wh(df_baseline, with_battery=True) / 1000
    metrics_baseline = compute_self_consumption_and_sufficiency(df_baseline, with_battery=True)
    cost_baseline = PANEL_COST_EUR * pvint.NUMBER_OF_PANELS + BATTERY_COST_EUR_PER_KWH * pvint.BATTERY_CAPACITY_KWH

    print("-" * 70)
    print(f"REZULTAT OPTIMIZARE (differential_evolution) - succes: {result.success}")
    print(f"  {result.message}")
    print("-" * 70)
    print(f"{'':30s}{'Baseline (fix)':>18s}{'Optimizat':>18s}")
    print(f"{'n panouri':30s}{pvint.NUMBER_OF_PANELS:>18d}{n_opt:>18d}")
    print(f"{'capacitate baterie (kWh)':30s}{pvint.BATTERY_CAPACITY_KWH:>18.2f}{capacity_opt:>18.2f}")
    print(f"{'soc_max':30s}{pvint.BATTERY_SOC_MAX:>18.2f}{soc_max_opt:>18.2f}")
    print(f"{'putere max. descarcare (kW)':30s}{'(1C, =capacitate)':>18s}{max_discharge_opt:>18.2f}")
    print(f"{'cost estimat (EUR)':30s}{cost_baseline:>18,.0f}{cost_opt:>18,.0f}")
    print(f"{'NEEG (kWh)':30s}{neeg_baseline_kWh:>18,.1f}{neeg_opt_kWh:>18,.1f}")
    print(f"{'Autoconsum (SC_disc)':30s}{metrics_baseline['SC']:>18.3f}{metrics_opt['SC']:>18.3f}")
    print(f"{'Autonomie (SS_disc)':30s}{metrics_baseline['SS']:>18.3f}{metrics_opt['SS']:>18.3f}")
    print("-" * 70)

    # --- Grafic: balanta de putere + SOC, pentru configuratia optimizata ---
    fig = plot_optimized_battery_result(df_opt, pvint.TARGET_HOUSE_ID)
    house_dir = pvint.OUTPUT_DIR / f"House_{pvint.TARGET_HOUSE_ID}_{house_status_label}"
    house_dir.mkdir(parents=True, exist_ok=True)
    output_path = house_dir / f"Optimized_Battery_House_{pvint.TARGET_HOUSE_ID}.html"
    fig.write_html(str(output_path))
    print(f"\nGrafic salvat: {output_path}")
    if pvint.SHOW_IN_BROWSER:
        fig.show()

    # --- Comparatie finala fara baterie vs. cu baterie (strategie optimizata) ---
    print_optimization_comparison(df_opt)

    return result, base_df, df_opt


if __name__ == "__main__":
    run_optimization()
