"""
visualizations.py

Vizualizari interactive (Plotly) pentru datele procesate in
data_processing.py: efectul curatarii spike-urilor, profilul orar al unui
aparat si consumul orar total al unei case.

Ruleaza direct din terminal:
    python visualizations.py

Configurare rapida (vezi constantele de mai jos): casa/casele de analizat
(TARGET_HOUSE_IDS), directorul de iesire (OUTPUT_DIR) si daca graficele se
deschid si in browser pe langa salvarea pe disc (SHOW_IN_BROWSER).
"""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import data_processing as dp

# --- Configurare -----------------------------------------------------------

# ID-ul (sau lista de ID-uri) de case pentru care se genereaza grafice.
# Impliciteste o singura casa; adauga mai multe ID-uri in lista pentru a
# genera cate un set complet de grafice pentru fiecare.
TARGET_HOUSE_IDS = [2000900, 2000901, 2000902, 2000903, 2000904, 2000905, 2000906, 2000907, 2000908, 2000909, 2000910, 2000911, 2000912, 2000913]

# Daca True, fiecare grafic salvat pe disc se deschide si intr-un tab de
# browser. Daca False, graficele sunt doar salvate ca fisiere HTML.
SHOW_IN_BROWSER = False

# Directorul in care se creeaza cate un subfolder per casa (House_<id>/).
OUTPUT_DIR = Path(__file__).resolve().parent / "output_plots"

# Deschidem graficele direct in browser, indiferent de mediul de rulare
# (relevant doar cand SHOW_IN_BROWSER=True).
pio.renderers.default = "browser"


def _sanitize_filename(name):
    """Transforma numele unui aparat intr-un nume de fisier sigur (fara spatii/paranteze)."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "aparat"


def _problem_suffix(duration):
    """Sufix de nume de fisier/titlu pentru un gol de inactivitate problematic (zile intregi)."""
    return f"_PROBLEMA_inactiv_{duration.days}zile"


def _annotate_title(fig, note):
    """Adauga o nota vizibila la titlul unei figuri deja construite."""
    fig.update_layout(title=f"{fig.layout.title.text}  [{note}]")
    return fig


def _save_figure(fig, path):
    """Salveaza figura ca HTML interactiv la `path` si, optional, o deschide in browser."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    print(f"  -> salvat: {path}")
    if SHOW_IN_BROWSER:
        fig.show()


def _filter_date_range(df, start_date=None, end_date=None):
    """
    Filtreaza un DataFrame cu index Datetime (tz-aware) la [start_date, end_date].
    Acceptă string-uri sau obiecte date/datetime; None inseamna "fara limita".
    """
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        if df.index.tz is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if end_ts == end_ts.normalize():
            # S-a dat doar o data calendaristica (fara ora) - o interpretam
            # ca fiind pana la sfarsitul acelei zile, nu miezul noptii de
            # la inceputul ei (altfel ultima zi din interval ar lipsi).
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if df.index.tz is not None and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]

    return df


def plot_spike_cleaning_comparison(raw_df, cleaned_df, house_id, appliance_id, appliance_name):
    """
    Grafic 1: suprapune seria bruta (cu spike-uri) peste seria curatata si
    interpolata, pentru un (house_id, appliance_id) dat - ca sa se vada
    vizual ce anume a fost eliminat de clean_outliers().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raw_sub = raw_df[
        (raw_df["HouseIDREF"] == house_id) & (raw_df["ApplianceIDREF"] == appliance_id)
    ].sort_index()
    cleaned_sub = cleaned_df[
        (cleaned_df["HouseIDREF"] == house_id) & (cleaned_df["ApplianceIDREF"] == appliance_id)
    ].sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=raw_sub.index,
            y=raw_sub["Value"],
            mode="lines+markers",
            name="Brut (cu spike-uri)",
            line=dict(color="firebrick", width=1),
            marker=dict(size=4, symbol="x"),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cleaned_sub.index,
            y=cleaned_sub["Value"],
            mode="lines",
            name="Curatat (interpolat)",
            line=dict(color="royalblue", width=1.5),
        )
    )
    fig.update_layout(
        title=f"Efectul curatarii spike-urilor - Casa {house_id}, '{appliance_name}'",
        xaxis_title="Data/Ora",
        yaxis_title="Energie per 10 min (Wh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_appliance_hourly_profile(
    hourly_appliance_df, house_id, appliance_id, appliance_name, start_date=None, end_date=None
):
    """
    Grafic 2: consumul orar (Wh) al unui aparat specific, pe toata perioada
    disponibila (sau pe o fereastra [start_date, end_date], daca e data).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sub = hourly_appliance_df[
        (hourly_appliance_df["HouseIDREF"] == house_id)
        & (hourly_appliance_df["ApplianceIDREF"] == appliance_id)
    ].sort_index()
    sub = _filter_date_range(sub, start_date, end_date)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=sub["Value"],
            mode="lines",
            name=appliance_name,
            line=dict(color="seagreen", width=1),
        )
    )
    fig.update_layout(
        title=f"Profil orar de consum - Casa {house_id}, '{appliance_name}'",
        xaxis_title="Data/Ora",
        yaxis_title="Energie (Wh)",
        hovermode="x unified",
    )
    return fig


def plot_house_total_hourly_consumption(hourly_house_df, house_id, start_date=None, end_date=None):
    """
    Grafic 3: consumul orar total al unei case (suma tuturor aparatelor
    fizice), pe toata perioada disponibila sau pe o fereastra
    [start_date, end_date] (ex. o saptamana de vara vs. o saptamana de iarna).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sub = hourly_house_df[hourly_house_df["HouseIDREF"] == house_id].sort_index()
    sub = _filter_date_range(sub, start_date, end_date)

    period_label = ""
    if start_date is not None or end_date is not None:
        period_label = f" ({start_date or '...'} -> {end_date or '...'})"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=sub["Value"],
            mode="lines",
            name=f"Casa {house_id}",
            line=dict(color="darkorange", width=1),
        )
    )
    fig.update_layout(
        title=f"Consum orar total - Casa {house_id}{period_label}",
        xaxis_title="Data/Ora",
        yaxis_title="Energie (Wh)",
        hovermode="x unified",
    )
    return fig


def _get_house_data(
    house_id,
    appliance_df,
    consumption_df,
    valid_houses_df,
    raw_consumption_df,
    cleaned_consumption_df,
    spike_summary,
    hourly_appliance_df,
    hourly_house_df,
):
    """
    Returneaza datele necesare graficelor pentru `house_id`.

    Pentru casele VALIDE, foloseste direct rezultatele deja calculate de
    pipeline-ul oficial (`dp.main()`) - eficient, un singur calcul pentru
    toate casele valide deodata.

    Pentru casele INVALIDE (eliminate de `dp.main()` inainte de curatarea
    de spike-uri si agregarea orara, deci absente din datele oficiale),
    rulam ad-hoc, DOAR pentru aceasta casa, acelasi `clean_outliers` +
    `aggregate_hourly_appliance` + `aggregate_hourly_house` ca in pipeline-ul
    oficial - pornind de la `consumption_df` (setul complet, toate cele 14
    case, dinainte de filtrarea de validitate). Graficele rezultate sunt
    deci "in afara analizei oficiale de optimizare", generate doar pentru
    inspectie vizuala la cerere.

    Returns
    -------
    tuple
        (is_valid, status_reason, raw_df, cleaned_df, spike_summary_df, hourly_appliance_df, hourly_house_df)
    """
    status_row = valid_houses_df.loc[valid_houses_df["HouseIDREF"] == house_id]
    is_valid = not status_row.empty and status_row.iloc[0]["Status"] == "VALIDA"
    status_reason = status_row.iloc[0]["MotivInvalidare"] if not status_row.empty else "casa necunoscuta"

    if is_valid:
        return (
            True,
            status_reason,
            raw_consumption_df,
            cleaned_consumption_df,
            spike_summary,
            hourly_appliance_df,
            hourly_house_df,
        )

    house_raw_df = consumption_df[consumption_df["HouseIDREF"] == house_id]
    house_cleaned_df, house_spike_summary = dp.clean_outliers(house_raw_df, appliance_df=appliance_df)
    house_hourly_appliance_df = dp.aggregate_hourly_appliance(house_cleaned_df)
    house_hourly_house_df = dp.aggregate_hourly_house(house_hourly_appliance_df)

    return (
        False,
        status_reason,
        house_raw_df,
        house_cleaned_df,
        house_spike_summary,
        house_hourly_appliance_df,
        house_hourly_house_df,
    )


def generate_house_plots(
    house_id,
    appliance_df,
    consumption_df,
    valid_houses_df,
    raw_consumption_df,
    cleaned_consumption_df,
    spike_summary,
    hourly_appliance_df,
    hourly_house_df,
):
    """
    Genereaza si salveaza (ca HTML) setul complet de grafice pentru o
    singura casa (VALIDA sau INVALIDA - vezi `_get_house_data`), organizate
    astfel in OUTPUT_DIR / House_<house_id>_<VALIDA|INVALIDA> / :
      casa_consum_total[_PROBLEMA_inactiv_Nzile].html      (direct in directorul casei)
      orar/aparat_<Nume>_orar[_PROBLEMA_inactiv_Nzile].html    (profil de consum dupa agregarea orara)
      10min/aparat_<Nume>_10min[_PROBLEMA_inactiv_Nzile].html  (brut + curatat suprapuse, rezolutie 10 min - inainte de agregarea orara)
    cate un fisier per aparat fizic din casa, in fiecare din cele doua subfoldere.

    Sufixul "_PROBLEMA_inactiv_Nzile" apare pe fisierul consumului total
    daca golul de inactivitate al INTREGII case (Conditia 1) depaseste
    MAX_ALLOWED_GAP, respectiv pe fisierele unui aparat (in ambele
    subfoldere) daca acel aparat e esential/major (cf.
    ESSENTIAL_APPLIANCE_KEYWORDS) si golul lui de inactivitate (Conditia 2)
    depaseste MAX_APPLIANCE_GAP - exact criteriile care determina statutul
    VALIDA/INVALIDA al casei, ca sa se vada din numele fisierului ce anume
    a cauzat problema.
    """
    is_valid, status_reason, raw_df, cleaned_df, house_spike_summary, house_hourly_appliance_df, house_hourly_house_df = (
        _get_house_data(
            house_id,
            appliance_df,
            consumption_df,
            valid_houses_df,
            raw_consumption_df,
            cleaned_consumption_df,
            spike_summary,
            hourly_appliance_df,
            hourly_house_df,
        )
    )

    house_raw_df = raw_df[raw_df["HouseIDREF"] == house_id]

    status_label = "VALIDA" if is_valid else "INVALIDA"
    house_dir = OUTPUT_DIR / f"House_{house_id}_{status_label}"
    house_dir.mkdir(parents=True, exist_ok=True)
    # Graficul de consum total al casei ramane direct in house_dir; cele
    # per-aparat sunt organizate in subfoldere separate pe rezolutie.
    orar_dir = house_dir / "orar"
    min10_dir = house_dir / "10min"

    if is_valid:
        print(f"\n=== Casa {house_id} [VALIDA] -> {house_dir} ===")
    else:
        print(f"\n=== Casa {house_id} [INVALIDA: {status_reason}] -> {house_dir} ===")
        print("  (grafice generate ad-hoc, in afara pipeline-ului oficial de analiza)")

    # A) Consum orar total al casei - marcat PROBLEMA daca golul total al
    # casei (Conditia 1) depaseste MAX_ALLOWED_GAP.
    house_gap = dp.analyze_zero_consumption_gaps(house_raw_df)[house_id]["duration"]
    house_has_gap_problem = house_gap > dp.MAX_ALLOWED_GAP

    fig_total = plot_house_total_hourly_consumption(house_hourly_house_df, house_id)
    total_filename = "casa_consum_total"
    if house_has_gap_problem:
        total_filename += _problem_suffix(house_gap)
        _annotate_title(fig_total, f"PROBLEMA: inactiva {house_gap.days} zile")
    _save_figure(fig_total, house_dir / f"{total_filename}.html")

    # B) Cate o pereche de grafice pentru fiecare aparat fizic din casa
    # (Site consumption () e deja exclus din pipeline la sursa):
    #   - 10min: brut + curatat suprapuse (inainte de agregarea orara) -
    #     arata exact ce a facut clean_outliers() pe seria originala.
    #   - orar: profilul dupa agregarea orara.
    # Marcate PROBLEMA doar aparatele esentiale/majore al caror gol de
    # inactivitate (Conditia 2) depaseste MAX_APPLIANCE_GAP.
    appliance_gaps = dp.analyze_appliance_zero_gaps(house_raw_df)

    house_appliances = appliance_df[
        (appliance_df["HouseIDREF"] == house_id)
        & (appliance_df["Name"] != dp.SITE_CONSUMPTION_APPLIANCE_NAME)
    ]
    for _, appliance_row in house_appliances.iterrows():
        appliance_id = appliance_row["ID"]
        appliance_name = appliance_row["Name"]

        has_data = (
            (house_hourly_appliance_df["HouseIDREF"] == house_id)
            & (house_hourly_appliance_df["ApplianceIDREF"] == appliance_id)
        ).any()
        if not has_data:
            continue

        appliance_gap = appliance_gaps.get((house_id, appliance_id))
        appliance_has_problem = (
            appliance_gap is not None
            and dp.is_essential_appliance(appliance_name)
            and appliance_gap["duration"] > dp.MAX_APPLIANCE_GAP
        )
        problem_note = (
            f"PROBLEMA: inactiv {appliance_gap['duration'].days} zile" if appliance_has_problem else None
        )
        problem_file_suffix = _problem_suffix(appliance_gap["duration"]) if appliance_has_problem else ""

        base_name = f"aparat_{_sanitize_filename(appliance_name)}"

        # 10 min: brut vs curatat, inainte de agregarea orara.
        fig_10min = plot_spike_cleaning_comparison(raw_df, cleaned_df, house_id, appliance_id, appliance_name)
        if problem_note:
            _annotate_title(fig_10min, problem_note)
        _save_figure(fig_10min, min10_dir / f"{base_name}_10min{problem_file_suffix}.html")

        # Orar: dupa agregare.
        fig_orar = plot_appliance_hourly_profile(
            house_hourly_appliance_df, house_id, appliance_id, appliance_name
        )
        if problem_note:
            _annotate_title(fig_orar, problem_note)
        _save_figure(fig_orar, orar_dir / f"{base_name}_orar{problem_file_suffix}.html")


def main():
    (
        house_df,
        appliance_df,
        valid_houses_df,
        consumption_df,
        raw_consumption_df,
        cleaned_consumption_df,
        spike_summary,
        hourly_appliance_df,
        hourly_house_df,
    ) = dp.main()

    print("\n" + "-" * 65)
    print("VIZUALIZARI PLOTLY")
    print(f"Case tinta: {TARGET_HOUSE_IDS}")
    print(f"Director de iesire: {OUTPUT_DIR}")
    print(f"Deschidere in browser: {SHOW_IN_BROWSER}")
    print("-" * 65)

    for house_id in TARGET_HOUSE_IDS:
        generate_house_plots(
            house_id,
            appliance_df,
            consumption_df,
            valid_houses_df,
            raw_consumption_df,
            cleaned_consumption_df,
            spike_summary,
            hourly_appliance_df,
            hourly_house_df,
        )


if __name__ == "__main__":
    main()
