# Optimizarea Gestionării Energiei într-o Locuință cu Surse Regenerabile și Stocare

Acest proiect explorează strategii optime pentru gestionarea energiei într-o locuință dotată cu panouri fotovoltaice și un sistem de stocare a energiei (baterie). Scopul principal este minimizarea energiei nete schimbate cu rețeaua (NEEG - Net Energy Exchanged with the Grid).

## Etapele Proiectului

1.  **Procesarea și Curățarea Datelor:** Prelucrarea datelor reale de consum din baza de date iRISE (rezoluție 10 minute), filtrarea caselor cu goluri mari de date și eliminarea anomaliilor (spike-urilor) prin interpolare liniară.
2.  **Integrarea Datelor Meteo și Modelarea PV:** Utilizarea API-ului Open-Meteo și a bibliotecii BATEM pentru a obține date meteo istorice (Grenoble) și a modela producția centralei fotovoltaice, determinând orientarea optimă.
3.  **Modelarea Bateriei (ESS):** Implementarea unui model de baterie (clasa `BatteryModel`) și simularea unei strategii de control bazate pe reguli (greedy): încărcare din excedentul solar și descărcare pentru acoperirea deficitului.
4.  **Optimizarea Traiectoriei de Control:** Formularea și rezolvarea unei probleme de optimizare pentru o perioadă de 24h, folosind algoritmul `differential_evolution`. Vectorul de decizie are 48 de variabile (acțiune și putere orară). S-a demonstrat convergența algoritmului către traiectoria optimă, confirmând eficiența strategiei greedy pe un interval de o zi, fără tarifare dinamică.
5.  **Vizualizări:** Generarea de grafice interactive folosind Plotly pentru analiza consumului, curățarea anomaliilor, balanța de putere și evoluția stării de încărcare (SOC).

## Structura Proiectului

*   `data_processing.py`: Extragerea, validarea, curățarea (spike-uri) și agregarea orară a datelor de consum din iRISE.
*   `solar_pv_integration.py`: Modelarea centralei fotovoltaice (BATEM), a bateriei și calculul indicatorilor de performanță (Autoconsum - SC, Autonomie - SS, NEEG).
*   `pv_battery_optimization.py`: Optimizarea combinată a dimensionării sistemului și a parametrilor de control folosind `scipy.optimize.differential_evolution`.
*   `optimal_control_24h.py`: Configurarea și rularea strategiei baseline (greedy) pe o zi fixată (ex. 1998-12-18), salvând rezultatele pentru comparație.
*   `optimize_control_trajectory.py`: Implementarea funcției obiectiv (NEEG) pentru optimizarea traiectoriei orare (vector de 48 variabile) și generarea graficelor comparative.
*   `visualizations.py`: Generarea graficelor Plotly pentru datele procesate (profiluri de consum, curățarea anomaliilor).

## Rezultate Obtinute

In urma rularii scripturilor, aplicatia genereaza atat rapoarte structurate in consola, cat si grafice interactive salvate automat in directorul `output_plots/`.

### 1. Rezultate Numerice (Afisare in Consola)
La executia pipeline-ului de procesare si optimizare, consola afiseaza etapizat urmatoarele informatii:

* **Raportul de validare a caselor:** Un tabel rezumativ pentru toate locuintele din baza de date, care indica statutul fiecareia (`VALIDA` sau `INVALIDA`), numarul de aparate monitorizate, intervalul de timp disponibil si eventualele motive de invalidare bazate pe golurile de inactivitate (la nivel de casa sau de aparat esential).
* **Configuratia sistemului si orientarea optima:** Parametrii fizici ai centralei fotovoltaice (numar de panouri, putere instalata, unghiurile optime de expunere si panta) si caracteristicile bateriei fixate (capacitate, limite SOC, putere maxima).
* **Traiectoria orara optima pe 24h:** Un tabel detaliat ora cu ora care prezinta consumul locuintei, productia solara, deciziile brute cerute de algoritm (actiunea discreta si puterea in kW), puterea efectiva aplicata bateriei (`PESS_Wh`) dupa proiectia pe constrangerile fizice si evolutia starii de incarcare (`SOC`).
* **Tabelul comparativ final (Baseline vs. Optimizat):** O comparatie directa intre strategia bazata pe reguli (greedy) si solutia gasita de `differential_evolution`, evidentiind indicatorii de performanta: Autoconsumul (`SC_disc`), Autonomia (`SS_disc`), energia neta schimbata cu reteaua (`NEEG`) si diferenta procentuala obtinuta.

### 2. Vizualizari Generate (Plotly HTML/PNG)
Scripturile salveaza automat urmatoarele grafice interactive in folderul dedicat fiecarei case (ex. `output_plots/House_<id>_VALIDA/<data_zilei>/`):

1. **Curatarea anomaliilor (`visualizations.py`):** Grafice comparative (brut vs. curatat) la rezolutie de 10 minute pentru fiecare aparat, evidentiind spike-urile eliminate si inlocuite prin interpolare liniara.
2. **Profilul orar de consum (`visualizations.py`):** Serii temporale cu consumul orar agregat per aparat si consumul total al locuintei.
3. **Balanta de putere pe 24h (`Power_Balance_24h`):** Figura cu doua sub-grafice suprapuse care compara evolutia orara a consumului, productiei PV si puterii bateriei (`p_ESS`) intre varianta optimizata si strategia greedy.
4. **Evolutia starii de incarcare (`SOC_Evolution_24h`):** Figura cu doua sub-grafice care ilustreaza variatia procentuala a `SOC`-ului pe parcursul celor 24 de ore pentru ambele strategii.
5. **Vectorul brut de decizie (`Raw_Decision_Vector_24h`):** Grafic cu axa Y dubla care afiseaza valorile brute ale celor 48 de variabile de decizie (actiunea orara si magnitudinea puterii cerute) inainte de aplicarea regulilor fizice de limitare.
