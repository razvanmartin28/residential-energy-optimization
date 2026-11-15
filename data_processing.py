"""
data_processing.py

Preprocesare a datelor din baza de date iRISE (masuratori de consum de
energie la nivel de aparat, pentru mai multe case din Franta), ca prim pas
pentru proiectul de gestionare optima a energiei intr-o locuinta
(minimizare NEEG - Net Energy Exchanged with the Grid).
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# Calea implicita catre baza de date, relativa la acest fisier.
DB_PATH = Path(__file__).resolve().parent / "irise38.sqlite3"

# O casa este considerata "cu gol mare" daca consumul total este 0 pentru
# mai mult de atat timp consecutiv .
MAX_ALLOWED_GAP = pd.Timedelta(days=20)

# Numele aparatului virtual care reprezinta consumul total al casei
SITE_CONSUMPTION_APPLIANCE_NAME = "Site consumption ()"

# Cuvinte-cheie (cautate case-insensitive, ca substring in Appliance.Name)
# care identifica un consumator "esential/major" al unei case - un aparat a
# carui inactivitate prelungita e neplauzibila intr-o casa locuita normal
# (spre deosebire de o lampa individuala sau un incarcator, unde 2 luni de
# inactivitate sunt banale). Verificat manual pe toate cele 64 de nume
# distincte de aparate din baza de date: prinde frigidere/congelatoare,
# plita/cuptor electric, incalzire electrica, pompa/control boiler
# (combustibil/lemne), masina de spalat si agregatul de iluminat al intregii
# case; exclude lampi individuale, uscator de rufe, masina de spalat vase si
# cuptor cu microunde (folosire discretionara, nu esentiala).
# Atentie: "oven" simplu ar fi prins gresit si "Microwave oven" - de aceea
# se folosesc variantele mai specifice "electric oven"/"electric cooker".
#
# "site light consumption": desi individual o lampa poate sta stinsa 2 luni
# fara sa insemne nimic, AGREGATUL de iluminat al intregii case ("Total site
# light consumption ()") ajungand la 0 pentru o perioada lunga e un semnal
# mult mai puternic - inseamna ca nicio lumina din casa n-a fost aprinsa,
# nu doar una anume. Verificat empiric: la casa 2000902, acest agregat are
# exact acelasi gol de 181 zile ca Electric Cooker si Water heater - foarte
# probabil aceeasi cauza reala (pana de curent/date), pe care lista initiala
# (fara acest cuvant-cheie) o rata complet.
#
# TV exclus intentionat: spre deosebire de frigider/boiler/incalzire, unde o
# inactivitate lunga indica aproape sigur o problema (fizica sau de senzor),
# e complet plauzibil ca un TV sa nu fie folosit luni de zile - mai ales
# intr-o casa cu mai multe televizoare.
ESSENTIAL_APPLIANCE_KEYWORDS = [
    "fridge",
    "freezer",
    "water heater",
    "power supply for",  # "... fuel boiler ()" / "... wood boiler ()"
    "electric heating",
    "electric cooker",
    "electric oven",
    "hot plate",
    "washing machine",
    "site light consumption",  # "Total site light consumption ()"
]

# Prag de inactivitate (gol continuu de Value == 0) pentru un consumator
# esential/major - peste acest prag, casa e eliminata din analiza chiar
# daca golul total al casei (Conditia 1) e in regula.
MAX_APPLIANCE_GAP = pd.Timedelta(days=60)

# Multiplicator implicit pentru pragul robust bazat pe IQR (Q3 + k * IQR)
# folosit la detectarea spike-urilor de consum per aparat.
SPIKE_IQR_MULTIPLIER = 3.0


def load_raw_tables(db_path=DB_PATH):
    """
    Se conecteaza la baza de date SQLite si incarca tabelele.
    """
    with sqlite3.connect(db_path) as conn:
        house_df = pd.read_sql_query("SELECT * FROM House", conn)
        appliance_df = pd.read_sql_query("SELECT * FROM Appliance", conn)
        consumption_df = pd.read_sql_query("SELECT * FROM Consumption", conn)

    return house_df, appliance_df, consumption_df


def exclude_site_consumption(consumption_df, appliance_df, appliance_name=SITE_CONSUMPTION_APPLIANCE_NAME):
    """
    Elimina din Consumption randurile aparatului virtual "Site consumption ()".
    """
    site_consumption_keys = appliance_df.loc[
        appliance_df["Name"] == appliance_name, ["HouseIDREF", "ID"]
    ].rename(columns={"ID": "ApplianceIDREF"})

    merged = consumption_df.merge(
        site_consumption_keys.assign(_is_site_consumption=True),
        on=["HouseIDREF", "ApplianceIDREF"],
        how="left",
    )
    merged = merged[merged["_is_site_consumption"].isna()]
    return merged.drop(columns="_is_site_consumption").reset_index(drop=True)


def convert_time_columns(house_df, consumption_df):
    """
    Converteste coloanele de timp Unix in datetime pe fusul orar al Frantei (Europe/Paris).
    """
    house_df = house_df.copy()
    house_df["StartingDatetime"] = pd.to_datetime(house_df["StartingEpochTime"], unit="s", utc=True).dt.tz_convert("Europe/Paris")
    house_df["EndingDatetime"] = pd.to_datetime(house_df["EndingEpochTime"], unit="s", utc=True).dt.tz_convert("Europe/Paris")

    consumption_df = consumption_df.copy()
    consumption_df["Datetime"] = pd.to_datetime(consumption_df["EpochTime"], unit="s", utc=True).dt.tz_convert("Europe/Paris")
    consumption_df = consumption_df.set_index("Datetime").sort_index()

    return house_df, consumption_df


def _longest_zero_run(sorted_group):
    """
    Calculeaza cea mai lunga secventa continua cu Value == 0 dintr-un grup
    deja sortat cronologic (coloane 'Datetime' si 'Value').

    Returns
    -------
    dict
        {'start', 'end', 'duration'} pentru cel mai lung gol; daca nu exista
        niciun 0, duration = Timedelta(0).
    """
    is_zero = sorted_group["Value"] == 0

    if not is_zero.any():
        return {"start": None, "end": None, "duration": pd.Timedelta(0)}

    block_ids = (is_zero != is_zero.shift()).cumsum()
    zero_blocks = sorted_group[is_zero].groupby(block_ids[is_zero])

    max_duration = pd.Timedelta(0)
    best_start = None
    best_end = None

    for _, block in zero_blocks:
        start_time = block["Datetime"].min()
        end_time = block["Datetime"].max()
        duration = end_time - start_time

        if duration >= max_duration:
            max_duration = duration
            best_start = start_time
            best_end = end_time

    return {"start": best_start, "end": best_end, "duration": max_duration}


def analyze_zero_consumption_gaps(consumption_df):
    """
    Calculeaza cea mai lunga perioada consecutiva cu Value == 0
    (pe suma tuturor aparatelor din casa la un moment dat) - Conditia 1 de
    validare (inactivitate totala a casei).
    """
    total_consumption = consumption_df.groupby(['HouseIDREF', consumption_df.index])['Value'].sum().reset_index()

    results = {}
    for house_id, group in total_consumption.groupby('HouseIDREF'):
        results[house_id] = _longest_zero_run(group.sort_values('Datetime'))

    return results


def is_essential_appliance(name, keywords=ESSENTIAL_APPLIANCE_KEYWORDS):
    """Determina daca `name` descrie un aparat esential/major (vezi ESSENTIAL_APPLIANCE_KEYWORDS)."""
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in keywords)


def analyze_appliance_zero_gaps(consumption_df):
    """
    Calculeaza, pentru fiecare (HouseIDREF, ApplianceIDREF) din
    `consumption_df`, cea mai lunga perioada continua cu Value == 0.

    Generic - nu se limiteaza la aparate esentiale. Folosit ca baza pentru
    `analyze_essential_appliance_gaps` (Conditia 2 de validare) si pentru
    etichetarea graficelor individuale ale oricarui aparat, indiferent daca
    e esential (vezi visualizations.py).

    Returns
    -------
    dict[(int, int), dict]
        Cheie: (HouseIDREF, ApplianceIDREF). Valoare: {'start', 'end', 'duration'}.
    """
    df = consumption_df.reset_index()  # Datetime devine coloana
    results = {}
    for (house_id, appliance_id), group in df.groupby(["HouseIDREF", "ApplianceIDREF"]):
        results[(house_id, appliance_id)] = _longest_zero_run(group.sort_values("Datetime"))
    return results


def analyze_essential_appliance_gaps(consumption_df, appliance_df, essential_keywords=ESSENTIAL_APPLIANCE_KEYWORDS):
    """
    Pentru fiecare aparat esential/major (cf. `essential_keywords`) al
    fiecarei case, cea mai lunga perioada continua cu Value == 0 - Conditia
    2 de validare (inactivitate a unui consumator major). Construit peste
    `analyze_appliance_zero_gaps`, filtrat la aparatele esentiale.

    Returns
    -------
    dict[int, list[dict]]
        Pentru fiecare HouseIDREF care are cel putin un aparat esential cu
        date, o LISTA cu TOATE aparatele esentiale ale casei (nu doar cel
        mai problematic), fiecare ca {'appliance_id', 'name', 'duration',
        'start', 'end'}, sortata descrescator dupa durata golului (primul
        element = cel mai mare gol). Casele fara niciun aparat esential in
        Appliance nu apar in dict (tratate ca neavand nicio problema pe
        Conditia 2).
    """
    essential_appliances = appliance_df.loc[
        appliance_df["Name"].apply(is_essential_appliance, keywords=essential_keywords),
        ["HouseIDREF", "ID", "Name"],
    ].rename(columns={"ID": "ApplianceIDREF"})

    if essential_appliances.empty:
        return {}

    essential_names = {
        (row.HouseIDREF, row.ApplianceIDREF): row.Name for row in essential_appliances.itertuples()
    }

    appliance_gaps = analyze_appliance_zero_gaps(consumption_df)

    per_house = {}
    for (house_id, appliance_id), name in essential_names.items():
        gap = appliance_gaps.get((house_id, appliance_id))
        if gap is None:
            continue

        per_house.setdefault(house_id, []).append(
            {
                "appliance_id": appliance_id,
                "name": name,
                "duration": gap["duration"],
                "start": gap["start"],
                "end": gap["end"],
            }
        )

    for house_id, entries in per_house.items():
        entries.sort(key=lambda entry: entry["duration"], reverse=True)

    return per_house


def build_house_validation_report(
    houses_with_data,
    appliance_df,
    consumption_df,
    gap_analysis,
    essential_gap_analysis,
    max_house_gap=MAX_ALLOWED_GAP,
    max_appliance_gap=MAX_APPLIANCE_GAP,
):
    """
    Construieste raportul de validare pentru TOATE casele cu date (nu doar
    cele valide), aplicand validarea dubla:
      Conditia 1: golul de consum total al casei (toate aparatele simultan
                  0) <= max_house_gap.
      Conditia 2: niciun aparat esential/major din casa nu are un gol
                  continuu de Value == 0 mai mare de max_appliance_gap.

    Returns
    -------
    pd.DataFrame
        O linie per casa: HouseIDREF, Status (VALIDA/INVALIDA),
        MotivInvalidare (gol daca e valida; daca sunt mai multe aparate
        esentiale problematice, sunt listate TOATE, separate prin "; "),
        plus informatii descriptive (nr. aparate fizice, interval de date,
        golul maxim la nivel de casa si cel mai mare gol dintre aparatele
        esentiale).
    """
    physical_appliances = appliance_df[appliance_df["Name"] != SITE_CONSUMPTION_APPLIANCE_NAME]

    rows = []
    for house_id in houses_with_data:
        house_consumption = consumption_df[consumption_df["HouseIDREF"] == house_id]
        num_appliances = physical_appliances[physical_appliances["HouseIDREF"] == house_id]["ID"].nunique()

        start_date = house_consumption.index.min().strftime("%Y-%m-%d") if not house_consumption.empty else "N/A"
        end_date = house_consumption.index.max().strftime("%Y-%m-%d") if not house_consumption.empty else "N/A"

        house_gap = gap_analysis[house_id]["duration"]
        cond1_ok = house_gap <= max_house_gap

        # essential_entries e sortata descrescator dupa durata (vezi
        # analyze_essential_appliance_gaps) - primul element e cel mai grav.
        essential_entries = essential_gap_analysis.get(house_id, [])
        worst_essential_gap = essential_entries[0]["duration"] if essential_entries else pd.Timedelta(0)

        problem_appliances = [e for e in essential_entries if e["duration"] > max_appliance_gap]
        cond2_ok = len(problem_appliances) == 0

        is_valid = cond1_ok and cond2_ok

        reasons = []
        if not cond1_ok:
            reasons.append(f"Inactivitate totala casa > {max_house_gap.days} zile (gol de {house_gap})")
        for entry in problem_appliances:
            reasons.append(
                f"Aparatul '{entry['name']}' a fost inactiv "
                f"{entry['duration'].days} zile"
            )
        reason = "; ".join(reasons)

        rows.append(
            {
                "HouseIDREF": house_id,
                "Status": "VALIDA" if is_valid else "INVALIDA",
                "MotivInvalidare": reason,
                "NumAparateFizice": num_appliances,
                "DataInceput": start_date,
                "DataSfarsit": end_date,
                "GolMaximConsum0Casa": str(house_gap),
                "GolMaximAparatEsential": str(worst_essential_gap),
            }
        )

    return pd.DataFrame(rows)


def get_house_status_label(valid_houses_df, house_id):
    """
    Intoarce eticheta de status ("VALIDA"/"INVALIDA") a casei `house_id`,
    din raportul `valid_houses_df` (`build_house_validation_report`) -
    aceeasi conventie de nume de folder ca in `visualizations.py`
    (`House_<house_id>_<VALIDA|INVALIDA>`), pentru ca scripturile din
    etapele urmatoare (optimal_control_24h.py, optimize_control_trajectory.py
    etc.) sa poata salva rezultatele in folderul deja existent al casei,
    in loc sa creeze unul nou, paralel.

    Daca `house_id` nu apare in `valid_houses_df` (casa necunoscuta),
    intoarce "INVALIDA" defensiv.

    Returns
    -------
    str
        "VALIDA" sau "INVALIDA".
    """
    status_row = valid_houses_df.loc[valid_houses_df["HouseIDREF"] == house_id]
    is_valid = not status_row.empty and status_row.iloc[0]["Status"] == "VALIDA"
    return "VALIDA" if is_valid else "INVALIDA"


def detect_consumption_spikes(
    df,
    value_col="Value",
    group_cols=("HouseIDREF", "ApplianceIDREF"),
    iqr_multiplier=SPIKE_IQR_MULTIPLIER,
    elevated_iqr_multiplier=1.0,
    min_neighbor_ratio=0.3,
    ignore_zero_for_quantiles=True,
):
    """
    Detecteaza spike-uri IZOLATE din `value_col`, separat pe fiecare grup.

    Un punct e candidat de spike daca depaseste pragul strict
    Q3 + `iqr_multiplier` * IQR. E marcat spike doar daca, in plus, e
    IZOLAT: niciun vecin temporal (prev/next, acelasi aparat) nu "protejeaza"
    punctul ca fiind parte dintr-o stare sustinuta reala. Un vecin protejeaza
    doar daca indeplineste SIMULTAN:
      1. E statistic ridicat: >= Q3 + `elevated_iqr_multiplier` * IQR.
      2. E comparabil ca magnitudine cu punctul curent:
         >= `min_neighbor_ratio` * valoarea punctului curent.

    De ce ambele conditii (nu doar prima): verificat empiric pe 3 cazuri
    unde un vecin trecea pragul statistic "la limita" dar era de 4-21 ori
    mai mic decat candidatul - ex. Clothes drier (casa 2000904): candidat
    6449, vecin 303 (prag elevated=300, deci 303 "trecea" pragul, dar
    303/6449 = 4.7%). Un asemenea vecin nu face parte din acelasi eveniment
    fizic, dar cu o singura conditie (doar pragul statistic) proteja gresit
    spike-ul de la detectare. Conditia 2 cere ca vecinul sa fie de aceeasi
    anvergura ca punctul curent (ex. centrifugarea masinii de spalat, casa
    2000902: candidat 306, vecin 203, raport 66% - clar comparabile, deci
    ramane protejat corect).

    Returns
    -------
    pd.Series[bool]
        Masca de spike-uri, in aceeasi ordine si cu acelasi index ca `df`.
    """
    group_cols = list(group_cols)

    sort_pos = df.index.argsort(kind="stable")
    unsort_pos = np.argsort(sort_pos, kind="stable")
    df_sorted = df.iloc[sort_pos]

    stats_source = (
        df_sorted[df_sorted[value_col] > 0] if ignore_zero_for_quantiles else df_sorted
    )

    quantiles = stats_source.groupby(group_cols)[value_col].quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["Q1", "Q3"]
    quantiles["IQR"] = quantiles["Q3"] - quantiles["Q1"]
    quantiles["StrictThreshold"] = quantiles["Q3"] + iqr_multiplier * quantiles["IQR"]
    quantiles["ElevatedThreshold"] = quantiles["Q3"] + elevated_iqr_multiplier * quantiles["IQR"]
    quantiles = quantiles.reset_index()

    merged = df_sorted[group_cols].reset_index(drop=True).merge(
        quantiles[group_cols + ["ElevatedThreshold", "StrictThreshold"]],
        on=group_cols,
        how="left",
    )
    strict_threshold = merged["StrictThreshold"].to_numpy()
    elevated_threshold = merged["ElevatedThreshold"].to_numpy()

    values_sorted = df_sorted[value_col].to_numpy()
    is_candidate = values_sorted > strict_threshold
    is_elevated = values_sorted >= elevated_threshold

    neighbor_frame = pd.DataFrame(
        {
            **{c: df_sorted[c].to_numpy() for c in group_cols},
            "Value": values_sorted,
            "IsElevated": is_elevated,
        }
    )
    grouped = neighbor_frame.groupby(group_cols)

    # Valorile brute ale vecinilor (NaN la limitele grupului - comparatiile
    # cu NaN dau False, deci se comporta corect si acolo fara fillna).
    prev_value = grouped["Value"].shift(1).to_numpy()
    next_value = grouped["Value"].shift(-1).to_numpy()
    # shift() pe o coloana booleana introduce NaN la limitele grupului, ceea
    # ce forteaza pandas sa treaca coloana pe dtype=object; astype(bool)
    # readuce un array boolean curat dupa fillna.
    prev_elevated = grouped["IsElevated"].shift(1).fillna(False).astype(bool).to_numpy()
    next_elevated = grouped["IsElevated"].shift(-1).fillna(False).astype(bool).to_numpy()

    prev_comparable = prev_value >= min_neighbor_ratio * values_sorted
    next_comparable = next_value >= min_neighbor_ratio * values_sorted

    prev_protects = prev_elevated & prev_comparable
    next_protects = next_elevated & next_comparable

    is_isolated = ~(prev_protects | next_protects)
    is_spike_sorted = is_candidate & is_isolated

    is_spike = is_spike_sorted[unsort_pos]
    return pd.Series(is_spike, index=df.index)


def clean_outliers(
    consumption_df,
    appliance_df=None,
    value_col="Value",
    iqr_multiplier=SPIKE_IQR_MULTIPLIER,
    site_consumption_name=SITE_CONSUMPTION_APPLIANCE_NAME,
    site_consumption_iqr_multiplier=None,
):
    """
    Curata spike-urile de consum: le marcheaza ca NaN si le interpoleaza liniar.
    """
    df = consumption_df.copy()
    group_cols = ["HouseIDREF", "ApplianceIDREF"]

    is_site_consumption = np.zeros(len(df), dtype=bool)
    if appliance_df is not None:
        site_rows = appliance_df.loc[
            appliance_df["Name"] == site_consumption_name, ["HouseIDREF", "ID"]
        ].rename(columns={"ID": "ApplianceIDREF"})
        if not site_rows.empty:
            df_keys = pd.MultiIndex.from_frame(df[group_cols].reset_index(drop=True))
            site_keys = pd.MultiIndex.from_frame(site_rows)
            is_site_consumption = df_keys.isin(site_keys)

    spike_mask = np.zeros(len(df), dtype=bool)

    physical_mask = ~is_site_consumption
    if physical_mask.any():
        physical_spikes = detect_consumption_spikes(
            df.loc[physical_mask],
            value_col=value_col,
            group_cols=group_cols,
            iqr_multiplier=iqr_multiplier,
        )
        spike_mask[physical_mask] = physical_spikes.to_numpy()

    if is_site_consumption.any() and site_consumption_iqr_multiplier is not None:
        site_spikes = detect_consumption_spikes(
            df.loc[is_site_consumption],
            value_col=value_col,
            group_cols=group_cols,
            iqr_multiplier=site_consumption_iqr_multiplier,
        )
        spike_mask[is_site_consumption] = site_spikes.to_numpy()

    df.loc[spike_mask, value_col] = np.nan
    df[value_col] = df.groupby(group_cols)[value_col].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    spike_summary = df[group_cols].copy()
    spike_summary["IsSpike"] = spike_mask
    spike_summary = (
        spike_summary.groupby(group_cols)["IsSpike"]
        .sum()
        .reset_index()
        .rename(columns={"IsSpike": "NumSpikes"})
    )
    spike_summary = spike_summary[spike_summary["NumSpikes"] > 0]

    if appliance_df is not None:
        appliance_names = appliance_df.rename(columns={"ID": "ApplianceIDREF"})[
            ["HouseIDREF", "ApplianceIDREF", "Name"]
        ]
        spike_summary = spike_summary.merge(appliance_names, on=group_cols, how="left")

    spike_summary = spike_summary.sort_values("NumSpikes", ascending=False).reset_index(
        drop=True
    )

    return df, spike_summary


def aggregate_hourly_appliance(
    cleaned_consumption_df,
    value_col="Value",
    group_cols=("HouseIDREF", "ApplianceIDREF"),
):
    """
    Agrega consumul curatat de la intervalul de 10 minute la 1 ora per aparat.
    """
    group_cols = list(group_cols)

    hourly = (
        cleaned_consumption_df.groupby(group_cols)[value_col]
        .resample("1h")
        .sum()
        .rename(value_col)
        .reset_index()
    )
    return hourly.set_index("Datetime").sort_index()


def aggregate_hourly_house(hourly_appliance_df, value_col="Value"):
    """
    Calculeaza consumul total al fiecarei case in fiecare ora.
    """
    hourly_house = (
        hourly_appliance_df.reset_index()
        .groupby(["HouseIDREF", "Datetime"])[value_col]
        .sum()
        .reset_index()
    )
    return hourly_house.set_index("Datetime").sort_index()


def main():
    print(f"Se incarca datele din: {DB_PATH}\n")
    house_df, appliance_df, consumption_df = load_raw_tables()

    consumption_df = exclude_site_consumption(consumption_df, appliance_df)
    house_df, consumption_df = convert_time_columns(house_df, consumption_df)

    houses_with_data = sorted(consumption_df["HouseIDREF"].unique().tolist())

    # Conditia 1: gol de consum total (toate aparatele) al casei.
    gap_analysis = analyze_zero_consumption_gaps(consumption_df)
    # Conditia 2 (noua): gol de inactivitate al celui mai problematic
    # consumator esential/major din fiecare casa.
    essential_gap_analysis = analyze_essential_appliance_gaps(consumption_df, appliance_df)

    # Raport de validare pentru TOATE cele 14 case cu date (nu doar valide).
    valid_houses_df = build_house_validation_report(
        houses_with_data, appliance_df, consumption_df, gap_analysis, essential_gap_analysis
    )
    valid_house_ids = valid_houses_df.loc[
        valid_houses_df["Status"] == "VALIDA", "HouseIDREF"
    ].tolist()

    print("-" * 95)
    print(f"RAPORT DE VALIDARE CASE (toate cele {len(houses_with_data)} case cu date)")
    print(f"  Conditia 1: gol consum total casa <= {MAX_ALLOWED_GAP.days} zile")
    print(f"  Conditia 2: niciun aparat esential/major nu are gol > {MAX_APPLIANCE_GAP.days} zile")
    print("-" * 95)
    print(valid_houses_df.to_string(index=False))
    print("-" * 95)
    print(f"Case VALIDE: {len(valid_house_ids)} din {len(houses_with_data)} -> {valid_house_ids}\n")
    
    filtered_consumption_df = consumption_df[consumption_df["HouseIDREF"].isin(valid_house_ids)].copy()
    
    # print(f"Randuri Consumption inainte de filtrare: {len(consumption_df):,}")
    # print(f"Randuri Consumption pastrate: {len(filtered_consumption_df):,}")

    # print("\n" + "-" * 75)
    # print("CURATARE SPIKE-URI DE CONSUM (per aparat, spike-uri IZOLATE, prag Q3 + 3*IQR)")
    # print("-" * 75)

    cleaned_consumption_df, spike_summary = clean_outliers(
        filtered_consumption_df, appliance_df=appliance_df
    )

    # total_spikes = int(spike_summary["NumSpikes"].sum())
    # print(f"Total spike-uri detectate si curatate: {total_spikes:,}")

    # if not spike_summary.empty:
    #     print("\nTop 10 aparate cu cele mai multe anomalii (per instanta casa+aparat):")
    #     top_by_instance = spike_summary.head(10)
    #     for _, row in top_by_instance.iterrows():
    #         name = row.get("Name", "necunoscut")
    #         print(
    #             f"  - Casa {row['HouseIDREF']}, aparat '{name}' "
    #             f"(ApplianceIDREF={row['ApplianceIDREF']}): {row['NumSpikes']} spike-uri"
    #         )

    #     if "Name" in spike_summary.columns:
    #         by_name = (
    #             spike_summary.groupby("Name")["NumSpikes"]
    #             .sum()
    #             .sort_values(ascending=False)
    #             .head(10)
    #         )
    #         print("\nTop 10 tipuri de aparate cu cele mai multe anomalii (agregat pe toate casele):")
    #         for name, count in by_name.items():
    #             print(f"  - {name}: {count} spike-uri")
    # else:
    #     print("Nu au fost detectate spike-uri.")

    hourly_appliance_df = aggregate_hourly_appliance(cleaned_consumption_df)
    hourly_house_df = aggregate_hourly_house(hourly_appliance_df)

    return (
        house_df,
        appliance_df,
        valid_houses_df,
        consumption_df,
        filtered_consumption_df,
        cleaned_consumption_df,
        spike_summary,
        hourly_appliance_df,
        hourly_house_df,
    )


if __name__ == "__main__":
    main()