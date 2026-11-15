"""
optimal_control_24h.py

Etapa 1 din noua abordare, stabilita impreuna cu coordonatorul: fixam
dimensionarea sistemului (PV + baterie) si ne concentram exclusiv pe
optimizarea TRAIECTORIEI DE CONTROL a bateriei pe o singura zi (24 ore) -
exact formularea din articolul de referinta (Simoiu et al. 2021, Sectiunea
5: "the optimisation problem has been resolved ... for a time interval T
of one day").

Acest script (Etapa 1) NU optimizeaza inca nimic - pregateste terenul:
  1. Fixeaza sistemul (4 panouri, baterie 10 kWh, SOC in [0.2, 0.95],
     putere max 5 kW la incarcare/descarcare) - nu mai sunt variabile de
     decizie, ca in pv_battery_optimization.py.
  2. Selecteaza o singura zi (24 ore) din datele casei, cu productia PV
     calculata de BATEM pentru cele 4 panouri fixate.
  3. Ruleaza strategia noastra actuala (BatteryModel - greedy, bazata pe
     reguli: incarca la surplus, descarca la deficit) pe aceste 24 de ore.
     Acesta e BASELINE-ul fata de care se va compara, intr-o etapa
     urmatoare, traiectoria orara optima gasita de
     scipy.optimize.differential_evolution.
  4. Salveaza rezultatele (seriile orare + indicatorii NEEG/SC/SS) pe disc,
     ca sa poata fi reincarcate direct de scriptul de optimizare, fara sa
     mai fie nevoie sa se redescarce datele meteo / sa se ruleze din nou
     intreg pipeline-ul de procesare a consumului.

Ruleaza cu interpretorul din venv (are BATEM):
    ./venv/Scripts/python.exe optimal_control_24h.py
"""

import json
from pathlib import Path

import pandas as pd

from batem.core.solar import PVplant

import data_processing as dp
import solar_pv_integration as pvint
from solar_pv_integration import (
    BatteryModel,
    compute_neeg_Wh,
    compute_self_consumption_and_sufficiency,
    simulate_battery_over_period,
)

# --- Sistem FIXAT (nu mai e variabila de decizie in aceasta etapa) ---------
N_PANELS = 4
BATTERY_CAPACITY_KWH = 5.0
SOC_MAX = 0.95
SOC_MIN = 0.2
MAX_POWER_KW = 3.0  # aceeasi limita pentru incarcare si descarcare

# SOC de start: mijlocul intervalului [SOC_MIN, SOC_MAX] - o alegere
# neutra (bateria nici "goala", nici "plina") pentru o zi tipica.
SOC_INITIAL = SOC_MIN
#SOC_INITIAL = 0.4
# --- Ziua selectata pentru optimizare ---------------------------------------
# O zi de vara (ex. 1998-07-15) a fost testata initial, dar cu sistemul fixat
# de mai sus (baterie 10kWh, mult mai mare decat fluxurile unei singure zile
# insorite) strategia greedy atinge deja NEEG=0 (optim teoretic) - nu ar
# ramane loc de imbunatatit pentru un optimizator.
#
# Am scanat mai multe zile de iarna (productie solara mica) si am ales
# 1998-12-20: consum 5.61 kWh, productie PV doar 0.94 kWh, NEEG_baseline
# (greedy) = 974 Wh, SS_disc = 0.167 - sistemul e vizibil "sub presiune"
# (bateria+PV nu acopera tot consumul), deci exista spatiu real pentru ca o
# traiectorie optimizata sa faca mai bine decat strategia greedy.
TARGET_DATE = "1998-12-18"

OUTPUT_DIR = Path(__file__).resolve().parent / "output_plots"


def build_fixed_pv_plant(solar_model):
    """
    Construieste centrala PV cu dimensionarea FIXATA (N_PANELS panouri), la
    orientarea optima pentru locatie. Independenta de NUMBER_OF_PANELS din
    solar_pv_integration.py - aici dimensionarea e explicit fixata pentru
    acest experiment, indiferent de valoarea globala din acel modul.
    """
    best_exposure_deg, best_slope_deg = solar_model.best_direction(
        initial_slope_deg=150, initial_exposure_deg=0
    )
    pv_plant = PVplant(
        solar_model=solar_model,
        exposure_deg=best_exposure_deg,
        slope_deg=best_slope_deg,
        mount_type=pvint.MOUNT_TYPE,
        number_of_panels=N_PANELS,
        panel_width_m=pvint.PANEL_WIDTH_M,
        panel_height_m=pvint.PANEL_HEIGHT_M,
        pv_efficiency=pvint.PANEL_EFFICIENCY,
        temperature_coefficient=pvint.PANEL_TEMPERATURE_COEFFICIENT,
    )
    print(
        f"Centrala PV (fixata): {pv_plant.number_of_panels} panouri, "
        f"{pv_plant.peak_power_kW:.2f} kWp instalati, "
        f"orientare {best_exposure_deg:.1f} deg / {best_slope_deg:.1f} deg"
    )
    return pv_plant


def select_day(combined_df, date_str):
    """
    Extrage cele 24 de ore ale zilei `date_str` (format "YYYY-MM-DD") din
    `combined_df` (coloanele "Consum_Wh", "PV_Wh"), pe fusul orar al
    indexului (Europe/Paris).

    Raises
    ------
    ValueError
        Daca ziua nu are exact 24 de citiri orare complete (fara goluri) -
        alegerea unei alte date (TARGET_DATE) e responsabilitatea
        apelantului.
    """
    day_start = pd.Timestamp(date_str, tz=combined_df.index.tz)
    day_end = day_start + pd.Timedelta(hours=23)
    day_df = combined_df.loc[day_start:day_end].copy()

    if len(day_df) != 24 or day_df[["Consum_Wh", "PV_Wh"]].isna().any().any():
        raise ValueError(
            f"Ziua {date_str} nu are 24 de citiri orare complete pentru casa "
            f"{pvint.TARGET_HOUSE_ID} (gasite: {len(day_df)}). Alege alta "
            "data (TARGET_DATE) acoperita complet de date."
        )
    return day_df


def run_baseline(day_df):
    """
    Ruleaza strategia greedy (BatteryModel) pe cele 24 de ore ale zilei
    selectate, cu sistemul FIXAT de mai sus.

    Returns
    -------
    tuple[pd.DataFrame, BatteryModel, dict, float]
        (day_with_battery_df, battery, metrics, neeg_Wh) - metrics e
        dict-ul intors de `compute_self_consumption_and_sufficiency`
        (SC, SS, energii in kWh), iar neeg_Wh e NEEG-ul zilei, in Wh.
    """
    battery = BatteryModel(
        capacity_kWh=BATTERY_CAPACITY_KWH,
        soc_initial=SOC_INITIAL,
        soc_min=SOC_MIN,
        soc_max=SOC_MAX,
        max_charge_power_kW=MAX_POWER_KW,
        max_discharge_power_kW=MAX_POWER_KW,
    )
    day_with_battery_df = simulate_battery_over_period(day_df, battery)

    metrics = compute_self_consumption_and_sufficiency(day_with_battery_df, with_battery=True)
    neeg_Wh = compute_neeg_Wh(day_with_battery_df, with_battery=True)

    return day_with_battery_df, battery, metrics, neeg_Wh


def save_baseline_results(day_with_battery_df, metrics, neeg_Wh):
    """
    Salveaza pe disc rezultatele baseline-ului:
      - seriile orare (Consum_Wh, PV_Wh, PESS_Wh, SOC) intr-un CSV;
      - indicatorii agregati (SC, SS, NEEG) + configuratia sistemului
        fixat, intr-un JSON - usor de reincarcat de etapa urmatoare
        (optimizarea traiectoriei), fara sa recalculeze nimic.

    Returns
    -------
    tuple[Path, Path]
        (csv_path, json_path)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{pvint.TARGET_HOUSE_ID}_{TARGET_DATE}"

    csv_path = OUTPUT_DIR / f"baseline_24h_timeseries_{tag}.csv"
    day_with_battery_df.to_csv(csv_path, index_label="Datetime")

    summary = {
        "house_id": pvint.TARGET_HOUSE_ID,
        "date": TARGET_DATE,
        "system": {
            "n_panels": N_PANELS,
            "battery_capacity_kWh": BATTERY_CAPACITY_KWH,
            "soc_min": SOC_MIN,
            "soc_max": SOC_MAX,
            "soc_initial": SOC_INITIAL,
            "max_power_kW": MAX_POWER_KW,
        },
        "baseline_greedy": {
            "SC": metrics["SC"],
            "SS": metrics["SS"],
            "NEEG_Wh": neeg_Wh,
            "consumption_kWh": metrics["consumption_kWh"],
            "pv_kWh": metrics["pv_kWh"],
            "self_consumed_kWh": metrics["self_consumed_kWh"],
        },
    }
    json_path = OUTPUT_DIR / f"baseline_24h_summary_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def main():
    print(
        f"Sistem fixat: {N_PANELS} panouri, baterie {BATTERY_CAPACITY_KWH:.1f} kWh, "
        f"SOC in [{SOC_MIN:.0%}, {SOC_MAX:.0%}] (initial {SOC_INITIAL:.0%}), "
        f"putere max {MAX_POWER_KW:.1f} kW"
    )
    print(f"Casa: {pvint.TARGET_HOUSE_ID}   Ziua selectata: {TARGET_DATE}\n")

    # 1. Consumul orar al casei (din pipeline-ul existent de procesare a datelor).
    *_, hourly_house_df = dp.main()
    start_date, end_date, house_hourly_df = pvint.get_house_consumption_period(
        hourly_house_df, pvint.TARGET_HOUSE_ID
    )

    # 2. Date meteo + model solar + centrala PV fixata (4 panouri).
    solar_model = pvint.build_solar_model(start_date, end_date)
    pv_plant = build_fixed_pv_plant(solar_model)
    combined_df = pvint.align_pv_with_consumption(pv_plant, solar_model, house_hourly_df)

    # 3. Selectam cele 24 de ore ale zilei alese.
    day_df = select_day(combined_df, TARGET_DATE)
    print(f"\nZiua {TARGET_DATE}: {len(day_df)} ore extrase (Consum_Wh, PV_Wh):")
    print(day_df[["Consum_Wh", "PV_Wh"]].round(1).to_string())

    # 4. Rulam strategia greedy actuala (BASELINE).
    day_with_battery_df, battery, metrics, neeg_Wh = run_baseline(day_df)

    print("\n" + "-" * 65)
    print(f"BASELINE (strategie greedy / bazata pe reguli) - ziua {TARGET_DATE}")
    print("-" * 65)
    print(f"Consum total zi:      {metrics['consumption_kWh']:.2f} kWh")
    print(f"Productie PV totala:  {metrics['pv_kWh']:.2f} kWh")
    print(f"Energie autoconsumata:{metrics['self_consumed_kWh']:.2f} kWh")
    print(f"Autoconsum (SC_disc): {metrics['SC']:.3f}")
    print(f"Autonomie (SS_disc):  {metrics['SS']:.3f}")
    print(f"NEEG:                 {neeg_Wh:,.1f} Wh")
    print("-" * 65)

    # 5. Salvam rezultatele (serii orare + indicatori), pentru etapa urmatoare.
    csv_path, json_path = save_baseline_results(day_with_battery_df, metrics, neeg_Wh)
    print(f"\nSerii orare salvate: {csv_path}")
    print(f"Rezumat (SC/SS/NEEG + configuratie sistem) salvat: {json_path}")

    return day_with_battery_df, battery, metrics, neeg_Wh


if __name__ == "__main__":
    main()
