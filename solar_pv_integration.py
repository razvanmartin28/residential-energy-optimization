"""
solar_pv_integration.py

Modeleaza o centrala fotovoltaica (PV) cu biblioteca BATEM si suprapune
productia orara peste consumul orar total al unei case din baza de date
iRISE, ca sa vedem cat de bine acopera productia proprie consumul casei
(un prim pas spre analiza NEEG - Net Energy Exchanged with the Grid).

IMPORTANT: ruleaza cu interpretorul din mediul virtual local (contine
BATEM), nu cu python-ul de sistem:
    ./venv/Scripts/python.exe solar_pv_integration.py

Foloseste datele meteo Open-Meteo (prin BATEM), stocate/cache-uite local in
folderul `data/` (configurat in `setup.ini`).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from batem.core.solar import PVplant, MOUNT_TYPES
from batem.core.weather import SWDbuilder

import data_processing as dp



# Casa selectata din iRISE
TARGET_HOUSE_ID = 2000905

# Locatia meteo: Grenoble
WEATHER_LOCATION = "Grenoble"
WEATHER_LATITUDE_DEG = 45.166958
WEATHER_LONGITUDE_DEG = 5.747664

# --- Panou solar ---
# Model: JA Solar JAM54S31-400/MR (panou monocristalin, 400W, half-cell PERC)
# Sursa fisei tehnice: EnergySage
#   https://www.energysage.com/equipment/solar-panels/ja-solar/jam54s31-400-mr-09a1b984/
# Date extrase din fisa tehnica:
#   - Putere nominala (STC):        400 W
#   - Eficienta modul:               20.5 %
#   - Dimensiuni:                    1722 mm (L) x 1134 mm (l) x 30 mm (grosime)
#   - Coeficient termic de putere:  -0.35 %/°C
PANEL_WIDTH_M = 1.134
PANEL_HEIGHT_M = 1.722
PANEL_EFFICIENCY = 0.205
# BATEM asteapta magnitudinea coeficientului termic (valoare pozitiva);
# semnul negativ (pierdere de putere odata cu cresterea temperaturii) e deja
# incorporat in formula interna (Collector.temperature_factor).
PANEL_TEMPERATURE_COEFFICIENT = 0.0035

# 4 panouri x 400W = 1600W = 1.6kWp instalati
NUMBER_OF_PANELS = 4

# Tip montaj: plan simplu (fara umbrire intre randuri), pe sol/acoperis.
MOUNT_TYPE = MOUNT_TYPES.PLAN

# --- Baterie (ESS - Energy Storage System) ---
# Capacitate aleasa proportional cu sistemul PV (1.6kWp): suficient de mare
# cat sa deplaseze cateva ore de productie/consum, fara sa fie nerealist de
# mare pentru o singura casa. SOC_min/SOC_max preiau exact valorile din
# studiul de caz al articolului (Simoiu et al. 2021, Tabelul 1:
# SOC_min=0.2, SOC_max=0.9).
BATTERY_CAPACITY_KWH = 5.0
BATTERY_SOC_INITIAL = 0.2
BATTERY_SOC_MIN = 0.2
BATTERY_SOC_MAX = 0.9

OUTPUT_DIR = Path(__file__).resolve().parent / "output_plots"

# Daca True, graficul se deschide si intr-un tab de browser (pe langa
# salvarea automata ca HTML)
SHOW_IN_BROWSER = False
pio.renderers.default = "browser"


def get_house_consumption_period(hourly_house_df, house_id):
    """
    Extrage seria orara de consum total al casei `house_id`, impreuna cu
    data de inceput/sfarsit a inregistrarilor, ca string-uri "DD/MM/YYYY"
    (formatul cerut de BATEM pentru `from_stringdate`/`to_stringdate`).

    Returns
    -------
    tuple[str, str, pd.DataFrame]
        (start_date, end_date, house_hourly_df) - house_hourly_df contine
        doar randurile casei selectate, indexate dupa Datetime.
    """
    house_hourly_df = hourly_house_df[hourly_house_df["HouseIDREF"] == house_id].sort_index()
    if house_hourly_df.empty:
        raise ValueError(
            f"Casa {house_id} nu are date de consum orar (verifica daca e o casa valida "
            "in raportul de validare din data_processing.py)."
        )

    start_date = house_hourly_df.index.min().strftime("%d/%m/%Y")
    end_date = house_hourly_df.index.max().strftime("%d/%m/%Y")
    return start_date, end_date, house_hourly_df


def build_solar_model(start_date, end_date):
    """
    Descarca/incarca datele meteo Open-Meteo
    pentru Grenoble, pe intervalul [start_date, end_date] (format
    "DD/MM/YYYY"), si construieste modelul de radiatie solara BATEM.
    """
    from batem.core.solar import SolarModel

    print(f"Se incarca/descarca date meteo pentru {WEATHER_LOCATION} "
          f"({start_date} -> {end_date})...")
    builder = SWDbuilder(
        location=WEATHER_LOCATION,
        latitude_north_deg=WEATHER_LATITUDE_DEG,
        longitude_east_deg=WEATHER_LONGITUDE_DEG,
    )
    site_weather_data = builder(from_stringdate=start_date, to_stringdate=end_date)

    solar_model = SolarModel(site_weather_data)
    #print(f"Model solar construit: {len(solar_model.datetimes)} ore de date.")
    return solar_model


def build_pv_plant(solar_model):
    """
    Determina orientarea optima (expunere fata de Sud, panta) pentru
    locatia data si construieste centrala PV: 5 panouri (400W fiecare), montaj tip PLAN.

    Returneaza:
    tuple[PVplant, float, float]
        (pv_plant, best_exposure_deg, best_slope_deg)
    """
    # Cautare euristica pornind de la o inclinare moderata (150 = usor spre
    # cer fata de orizontala, cf. conventiei BATEM: 180=orizontal spre cer).
    best_exposure_deg, best_slope_deg = solar_model.best_direction(
        initial_slope_deg=150, initial_exposure_deg=0
    )
    print(
        f"Orientare optima panouri: expunere={best_exposure_deg:.1f} deg "
        f"(0=Sud, +90=Vest, -90=Est), panta={best_slope_deg:.1f} deg "
        "(180=orizontal spre cer, 90=vertical)"
    )

    pv_plant = PVplant(
        solar_model=solar_model,
        exposure_deg=best_exposure_deg,
        slope_deg=best_slope_deg,
        mount_type=MOUNT_TYPE,
        number_of_panels=NUMBER_OF_PANELS,
        panel_width_m=PANEL_WIDTH_M,
        panel_height_m=PANEL_HEIGHT_M,
        pv_efficiency=PANEL_EFFICIENCY,
        temperature_coefficient=PANEL_TEMPERATURE_COEFFICIENT,
    )

    print(
        f"Centrala PV: {pv_plant.number_of_panels} panouri, "
        f"{pv_plant.surface_pv_m2:.1f} m2, {pv_plant.peak_power_kW:.2f} kWp instalati, "
        f"montaj {pv_plant.mount_type.name}"
    )
    return pv_plant, best_exposure_deg, best_slope_deg


def align_pv_with_consumption(pv_plant, solar_model, house_hourly_df):
    """
    Construieste un DataFrame orar aliniat pe acelasi index temporal, cu
    consumul casei si productia PV, ambele in Wh.

    Nota fizica: `powers_W()` intoarce, per ora, puterea medie generata de
    centrala (W) - la fel ca la Consumption din iRISE 
    """
    pv_index = pd.DatetimeIndex(solar_model.datetimes)
    if pv_index.tz is None:
        pv_index = pv_index.tz_localize(house_hourly_df.index.tz)
    else:
        pv_index = pv_index.tz_convert(house_hourly_df.index.tz)

    pv_series = pd.Series(pv_plant.powers_W(), index=pv_index, name="PV_Wh")
    pv_series = pv_series[~pv_series.index.duplicated(keep="first")].sort_index()

    consumption_series = house_hourly_df["Value"].rename("Consum_Wh")
    consumption_series = consumption_series[~consumption_series.index.duplicated(keep="first")].sort_index()

    combined_df = pd.concat([consumption_series, pv_series], axis=1, join="inner").dropna()
    return combined_df


class BatteryModel:
    """

    Evolutia SOC in discret:
        SOC_{i+1} = SOC_i - p_ESS,i * dt / Q_nom_ESS

    Strategie de control: reguli simple - la fiecare ora:
      - daca productia PV depaseste consumul (surplus), bateria se
        INCARCA cu (o parte din) surplus, niciodata din retea (cf.
        Ec. 24-25 din articol: -p_ESS <= p_PV);
      - daca consumul depaseste productia PV (deficit), bateria se
        DESCARCA pentru a acoperi (o parte din) deficit;
      - in ambele cazuri, puterea e limitata de puterea maxima de
        (des)incarcare si de spatiul/energia ramasa in baterie intre
        SOC_min si SOC_max (Ec. 18).
    """

    def __init__(
        self,
        capacity_kWh,
        soc_initial=0.5,
        soc_min=0.2,
        soc_max=0.9,
        max_charge_power_kW=None,
        max_discharge_power_kW=None,
        dt_h=1.0,
    ):
        """
        Parameters
        ----------
        capacity_kWh : float
            Capacitatea nominala a bateriei, Q_nom_ESS (kWh).
        soc_initial : float
            SOC la inceputul simularii (SOC_initial din articol).
        soc_min, soc_max : float
            Limitele SOC (Ec. 18), pentru protejarea duratei de viata a bateriei.
        max_charge_power_kW, max_discharge_power_kW : float | None
            Puterea maxima de incarcare/descarcare (kW). Daca None, se
            foloseste `capacity_kWh` (adica bateria se poate incarca/
            descarca integral intr-o ora - un C-rate de 1C, o presupunere
            implicita rezonabila pentru o baterie rezidentiala).
        dt_h : float
            Pasul de esantionare (h) - 1h pentru profilul nostru orar.
        """
        if capacity_kWh <= 0:
            raise ValueError("capacity_kWh trebuie sa fie > 0.")
        if not (0 <= soc_min < soc_max <= 1):
            raise ValueError("Trebuie respectat: 0 <= soc_min < soc_max <= 1.")
        if not (soc_min <= soc_initial <= soc_max):
            raise ValueError("soc_initial trebuie sa fie intre soc_min si soc_max.")

        self.capacity_kWh = capacity_kWh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.dt_h = dt_h
        self.max_charge_power_kW = capacity_kWh if max_charge_power_kW is None else max_charge_power_kW
        self.max_discharge_power_kW = capacity_kWh if max_discharge_power_kW is None else max_discharge_power_kW

        self.soc_initial = soc_initial
        self.soc = soc_initial
        # Istoricul simularii, populat de step()/simulate().
        self.soc_history = []
        self.p_ess_history_kW = []

    def reset(self, soc_initial=None):
        """Reseteaza starea bateriei (util pentru a rula mai multe simulari succesive)."""
        self.soc = self.soc_initial if soc_initial is None else soc_initial
        self.soc_history = []
        self.p_ess_history_kW = []

    def _decide_power_kW(self, p_pv_kW, p_load_kW):
        """
        Strategie simpla bazata pe reguli: incarca la surplus, descarca la
        deficit. Intoarce puterea bateriei p_ESS (kW), cu conventia de semn
        stabilita (>0 = descarcare, <0 = incarcare), respectand limitele de
        putere si de SOC disponibil.
        """
        net_surplus_kW = p_pv_kW - p_load_kW  # > 0 = excedent PV, < 0 = deficit

        if net_surplus_kW > 0:
            # Excedent de productie -> incarcam bateria, DOAR din excedentul
            # PV (niciodata din retea - cf. Ec. 24-25 din articol).
            room_left_kWh = (self.soc_max - self.soc) * self.capacity_kWh
            max_charge_from_room_kW = room_left_kWh / self.dt_h if self.dt_h > 0 else 0.0
            charge_power_kW = max(0.0, min(net_surplus_kW, self.max_charge_power_kW, max_charge_from_room_kW))
            return -charge_power_kW  # incarcare => p_ESS negativ

        if net_surplus_kW < 0:
            # Deficit -> descarcam bateria cat sa acoperim deficitul, fara
            # sa descarcam mai mult decat e nevoie.
            deficit_kW = -net_surplus_kW
            available_kWh = (self.soc - self.soc_min) * self.capacity_kWh
            max_discharge_from_soc_kW = available_kWh / self.dt_h if self.dt_h > 0 else 0.0
            discharge_power_kW = max(0.0, min(deficit_kW, self.max_discharge_power_kW, max_discharge_from_soc_kW))
            return discharge_power_kW  # descarcare => p_ESS pozitiv

        return 0.0

    def step(self, p_pv_kW, p_load_kW):
        """
        Avanseaza simularea cu un pas (o ora): decide p_ESS conform
        strategiei, actualizeaza SOC (Ec. 10) si intoarce p_ESS (kW).
        """
        p_ess_kW = self._decide_power_kW(p_pv_kW, p_load_kW)

        # Ec. 10: SOC_{i+1} = SOC_i - p_ESS * dt / Q_nom_ESS
        self.soc = self.soc - (p_ess_kW * self.dt_h) / self.capacity_kWh
        # Clip defensiv pentru erori de rotunjire in virgula mobila (SOC ar
        # trebui sa ramana deja in [soc_min, soc_max] prin constructia lui
        # `_decide_power_kW`, care limiteaza puterea la spatiul disponibil).
        self.soc = min(max(self.soc, self.soc_min), self.soc_max)

        self.p_ess_history_kW.append(p_ess_kW)
        self.soc_history.append(self.soc)
        return p_ess_kW

    def simulate(self, pv_kW_series, load_kW_series):
        """
        Ruleaza simularea bateriei pe intreaga serie temporala data (ex. un
        an intreg de date orare).

        Parameters
        ----------
        pv_kW_series, load_kW_series : array-like, aceeasi lungime
            Puterea PV, respectiv puterea consumata, in kW, per ora.

        Returns
        -------
        tuple[list[float], list[float]]
            (p_ess_kW la fiecare ora, soc la fiecare ora - starea DUPA
            pasul respectiv).
        """
        for p_pv_kW, p_load_kW in zip(pv_kW_series, load_kW_series):
            self.step(p_pv_kW, p_load_kW)
        return list(self.p_ess_history_kW), list(self.soc_history)


def simulate_battery_over_period(combined_df, battery):
    """
    Simuleaza bateria (ESS) ora cu ora, pe toata perioada din `combined_df`,
    folosind strategia cu reguli din `BatteryModel`. Returneaza o copie a
    `combined_df` cu doua coloane noi:
      - "PESS_Wh": puterea bateriei (aceeasi conventie de semn ca p_ESS:
        > 0 descarcare, < 0 incarcare);
      - "SOC": starea de incarcare dupa fiecare ora (fractie in [0, 1]).

    Nota fizica: la rezolutie orara (dt=1h), puterea medie (kW) e numeric
    egala cu energia orei (kWh) - la fel ca peste tot in acest pipeline
    (Consum_Wh, PV_Wh sunt deja energii/ora, nu puteri instantanee).
    """
    pv_kW = (combined_df["PV_Wh"] / 1000).to_numpy()
    load_kW = (combined_df["Consum_Wh"] / 1000).to_numpy()

    p_ess_kW, soc = battery.simulate(pv_kW, load_kW)

    result_df = combined_df.copy()
    result_df["PESS_Wh"] = pd.Series(p_ess_kW, index=combined_df.index) * 1000
    result_df["SOC"] = pd.Series(soc, index=combined_df.index)
    return result_df


def plot_pv_vs_consumption(combined_df, house_id):
    """
    Grafic interactiv Plotly: consum orar total al casei vs. productie PV
    orara, cu o zona vizuala pentru autoconsumul teoretic (fara stocare) -
    la fiecare ora, autoconsum = min(consum, productie PV); ce nu se
    "consuma pe loc" e fie excedent exportat (PV > consum), fie import din
    retea (consum > PV).

    Returneaza
    plotly.graph_objects.Figure
    """
    self_consumption_Wh = combined_df[["Consum_Wh", "PV_Wh"]].min(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=self_consumption_Wh,
            mode="lines",
            name="Autoconsum teoretic (min(consum, PV))",
            fill="tozeroy",
            line=dict(color="rgba(255,193,7,0)", width=0),
            fillcolor="rgba(255,193,7,0.35)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=combined_df["Consum_Wh"],
            mode="lines",
            name="Consum orar total casa (Wh)",
            line=dict(color="firebrick", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=combined_df["PV_Wh"],
            mode="lines",
            name="Productie PV orara (Wh)",
            line=dict(color="seagreen", width=1),
        )
    )

    fig.update_layout(
        title=f"Consum orar vs. productie fotovoltaica - Casa {house_id}",
        xaxis_title="Data/Ora",
        yaxis_title="Energie (Wh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def compute_self_consumption_and_sufficiency(combined_df, with_battery=False):
    """
    Calculeaza Self-Consumption (SC) si Self-Sufficiency (SS) conform
    Simoiu et al. (2021) - formele discrete SC_disc (Ec. 1) si SS_disc
    (Ec. 2), incluzand explicit termenul p_ESS (sarcina compensata de
    baterie):

        SC_disc = sum_i min(p_PV,i, p_Load,i - p_ESS,i) * dt / sum_i p_PV,i * dt
        SS_disc = sum_i min(p_PV,i, p_Load,i - p_ESS,i) * dt / sum_i p_Load,i * dt

    Suma se face pe TOT intervalul (aici: perioada disponibila de date,
    ~1 an) - SC si SS rezulta fiecare ca o singura valoare, nu o serie
    temporala.

    Daca `with_battery=False` (sistem fara ESS), p_ESS,i = 0 pentru toti i,
    si formulele se reduc la varianta simpla (fara stocare):

        SC_disc = sum_i min(p_PV,i, p_Load,i) * dt / sum_i p_PV,i * dt
        SS_disc = sum_i min(p_PV,i, p_Load,i) * dt / sum_i p_Load,i * dt

    Daca `with_battery=True`, `combined_df` trebuie sa contina si coloana
    "PESS_Wh" (produsa de `simulate_battery_over_period`), cu conventia de
    semn stabilita (p_ESS > 0 la descarcare, < 0 la incarcare) - astfel
    "p_Load - p_ESS" scade la descarcare (bateria acopera o parte din
    sarcina) si creste la incarcare (bateria e o sarcina suplimentara).

    Cum puterea medie orara * dt(=1h) e chiar energia orara (Wh) deja
    stocata in coloanele `combined_df`, sumele de mai sus sunt direct sume
    de Wh pe coloanele existente - nu mai e nevoie de o inmultire
    suplimentara cu dt.

    Returns
    -------
    dict
        {'SC': float, 'SS': float, 'self_consumed_kWh': float,
         'pv_kWh': float, 'consumption_kWh': float}
        SC si SS sunt adimensionale, in [0, 1] (0 = deloc, 1 = complet).
    """
    if with_battery:
        net_load_Wh = combined_df["Consum_Wh"] - combined_df["PESS_Wh"]
    else:
        net_load_Wh = combined_df["Consum_Wh"]

    self_consumed_Wh = pd.concat([combined_df["PV_Wh"], net_load_Wh.rename("NetLoad_Wh")], axis=1).min(axis=1).sum()
    pv_Wh = combined_df["PV_Wh"].sum()
    consumption_Wh = combined_df["Consum_Wh"].sum()

    SC = self_consumed_Wh / pv_Wh if pv_Wh > 0 else float("nan")
    SS = self_consumed_Wh / consumption_Wh if consumption_Wh > 0 else float("nan")

    return {
        "SC": SC,
        "SS": SS,
        "self_consumed_kWh": self_consumed_Wh / 1000,
        "pv_kWh": pv_Wh / 1000,
        "consumption_kWh": consumption_Wh / 1000,
    }


def compute_neeg_Wh(combined_df, with_battery=False):
    """
    Calculeaza NEEG (Net Energy Exchanged with the Grid), Ec. 3 din articol:

        NEEG = sum_i |p_PV,i + p_ESS,i - p_Load,i| * dt   [Wh]

    Cu conventia de semn adoptata (p_ESS > 0 la descarcare, < 0 la
    incarcare), semnul "+p_ESS" se ocupa automat de balanta corecta (Ec. 4:
    p_Load = p_G + p_ESS + p_PV): la descarcare, bateria contribuie ca o a
    doua sursa alaturi de PV; la incarcare, "+p_ESS" (negativ) scade
    efectiv din ce ar fi disponibil pentru retea/consum, exact ca un
    consumator suplimentar.

    Fara baterie (`with_battery=False`), p_ESS,i = 0 pentru toti i.

    Returns
    -------
    float
        NEEG in Wh, pe tot intervalul din `combined_df`.
    """
    p_ess_Wh = combined_df["PESS_Wh"] if with_battery else 0.0
    net_Wh = combined_df["PV_Wh"] + p_ess_Wh - combined_df["Consum_Wh"]
    return net_Wh.abs().sum()


def print_annual_summary(combined_df, start_date, end_date):
    """
    Printeaza in terminal rezumatul energetic al perioadei analizate,
    inclusiv Self-Consumption (SC_disc) si Self-Sufficiency (SS_disc) -
    fiecare o singura valoare agregata pe tot intervalul (vezi
    `compute_self_consumption_and_sufficiency`).
    """
    n_days = (combined_df.index.max() - combined_df.index.min()).days
    metrics = compute_self_consumption_and_sufficiency(combined_df)

    print("\n" + "-" * 65)
    print("REZUMAT ENERGETIC")
    print("-" * 65)
    print(f"Perioada analizata:          {start_date} -> {end_date} ({n_days} zile)")
    print(f"Consum total casa:           {metrics['consumption_kWh']:,.1f} kWh")
    print(f"Productie totala PV:         {metrics['pv_kWh']:,.1f} kWh")
    print(f"Energie autoconsumata:       {metrics['self_consumed_kWh']:,.1f} kWh")
    print(
        f"Autoconsum (SC_disc):  {metrics['SC']:.3f}  "
        f"({metrics['SC'] * 100:.1f}% din productia PV e consumata pe loc)"
    )
    print(
        f"Autonomie (SS_disc):  {metrics['SS']:.3f}  "
        f"({metrics['SS'] * 100:.1f}% din consumul casei e acoperit de productia proprie)"
    )
    print("-" * 65)


def plot_battery_soc(combined_with_battery_df, house_id, battery):
    """
    Grafic interactiv Plotly: evolutia SOC a bateriei si puterea ei
    (p_ESS: > 0 descarcare, < 0 incarcare) pe parcursul perioadei simulate.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=combined_with_battery_df.index,
            y=combined_with_battery_df["SOC"] * 100,
            mode="lines",
            name="SOC (%)",
            line=dict(color="royalblue", width=1),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined_with_battery_df.index,
            y=combined_with_battery_df["PESS_Wh"],
            mode="lines",
            name="Putere baterie (Wh) - (+) descarcare, (-) incarcare",
            line=dict(color="darkorange", width=1),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"Evolutia bateriei (ESS, {battery.capacity_kWh:.1f} kWh) - Casa {house_id}",
        xaxis_title="Data/Ora",
        yaxis=dict(title="SOC (%)", range=[0, 100]),
        yaxis2=dict(title="Putere baterie (Wh)", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def print_battery_comparison_summary(combined_df, combined_with_battery_df, battery):
    """
    Printeaza in terminal o comparatie intre scenariul FARA baterie si CU
    baterie: Self-Consumption (SC_disc), Self-Sufficiency (SS_disc) si NEEG
    (Ec. 1, 2, 3 din articol), fiecare ca o singura valoare agregata pe tot
    intervalul simulat.
    """
    metrics_no_ess = compute_self_consumption_and_sufficiency(combined_df, with_battery=False)
    metrics_with_ess = compute_self_consumption_and_sufficiency(combined_with_battery_df, with_battery=True)

    neeg_no_ess_kWh = compute_neeg_Wh(combined_df, with_battery=False) / 1000
    neeg_with_ess_kWh = compute_neeg_Wh(combined_with_battery_df, with_battery=True) / 1000

    print("\n" + "-" * 65)
    print(
        f"SIMULARE BATERIE (ESS) - capacitate {battery.capacity_kWh:.1f} kWh, "
        f"SOC in [{battery.soc_min:.0%}, {battery.soc_max:.0%}], SOC initial {battery.soc_initial:.0%}"
    )
    print("-" * 65)
    print(f"{'':30s}{'Fara ESS':>15s}{'Cu ESS':>15s}")
    print(f"{'Autoconsum (SC_disc)':30s}{metrics_no_ess['SC']:>15.3f}{metrics_with_ess['SC']:>15.3f}")
    print(f"{'Autonomie (SS_disc)':30s}{metrics_no_ess['SS']:>15.3f}{metrics_with_ess['SS']:>15.3f}")
    print(f"{'NEEG (kWh)':30s}{neeg_no_ess_kWh:>15,.1f}{neeg_with_ess_kWh:>15,.1f}")
    print("-" * 65)


def main():
    print(f"Casa selectata pentru integrarea PV: {TARGET_HOUSE_ID}")

    # Consumul orar total al casei
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
    house_status_label = dp.get_house_status_label(valid_houses_df, TARGET_HOUSE_ID)
    start_date, end_date, house_hourly_df = get_house_consumption_period(hourly_house_df, TARGET_HOUSE_ID)
    print(f"\nInterval de consum extras pentru casa {TARGET_HOUSE_ID}: {start_date} -> {end_date}")

    # Date meteo + model de radiatie solara pentru Grenoble
    solar_model = build_solar_model(start_date, end_date)

    # Centrala PV (orientare optima + x panouri de 400W)
    pv_plant, best_exposure_deg, best_slope_deg = build_pv_plant(solar_model)

    # Suprapunerea consum vs. productie si vizualizarea interactiva
    combined_df = align_pv_with_consumption(pv_plant, solar_model, house_hourly_df)
    #print(f"\nRanduri orare aliniate (consum + productie PV): {len(combined_df):,}")

    fig = plot_pv_vs_consumption(combined_df, TARGET_HOUSE_ID)

    house_dir = OUTPUT_DIR / f"House_{TARGET_HOUSE_ID}_{house_status_label}"
    house_dir.mkdir(parents=True, exist_ok=True)
    output_path = house_dir / f"PV_vs_Consumption_House_{TARGET_HOUSE_ID}.html"
    fig.write_html(str(output_path))
    print(f"\nGrafic salvat: {output_path}")
    if SHOW_IN_BROWSER:
        fig.show()

    print_annual_summary(combined_df, start_date, end_date)

    # Nota: simularea bateriei cu o configuratie FIXA/arbitrara (BatteryModel,
    # plot_battery_soc, print_battery_comparison_summary - toate definite mai
    # sus in acest fisier) nu mai e apelata aici. E suprapusa de
    # pv_battery_optimization.py, care gaseste o configuratie optimizata
    # (nu ghicita manual) si genereaza propriul grafic (Optimized_Battery_
    # House_<id>.html, cu subplot de SOC inclus) + propria comparatie in
    # terminal. Functiile raman disponibile mai jos daca vrei sa inspectezi
    # rapid o configuratie custom fara sa rulezi optimizarea completa.


if __name__ == "__main__":
    main()
