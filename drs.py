import numpy as np
import streamlit as st
from physik import Windkanal
from flügel import naca_in_windkanal_einbauen
from visualisieren import feld_zu_bild 

st.set_page_config(layout="wide")
st.title("2D-Windkanal: F1-Heckflügel mit DRS")

st.sidebar.header("Simulationseinstellungen")
#naca_code_button = st.sidebar.text_input("NACA Flügelcode", value="2412")
#breite = st.sidebar.slider("Breite des Windkanals (m)", min_value=10, max_value=200, value=160, step=10)
breite = 160
#hoehe = st.sidebar.slider("Höhe des Windkanals (m)", min_value=10, max_value=100, value=70, step=10)
hoehe = 70
#zellgrösse = st.sidebar.slider("Zellgröße (m)", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
zellgrösse = 0.2
#anstellwinkel = st.sidebar.slider("Anstellwinkel des Flügels (Grad)", min_value=-180, max_value=180, value=5, step=1)
anstellwinkel = 5
eintrittsgeschwindigkeit_button = st.sidebar.slider("Eintrittsgeschwindigkeit (m/s)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)  
reset_button = st.sidebar.button("Zurücksetzen")


st.sidebar.header("DRS Simulationseinstellungen")
Klappenwinkel = st.sidebar.slider("DRS Klappenwinkel (Grad)", min_value=0, max_value=60, value=32, step=1)

st.sidebar.header("DRS Steuerung")
spalte_auf, spalte_zu = st.sidebar.columns(2)
drs_auf = spalte_auf.button("DRS öffnen", type="primary")
drs_zu = spalte_zu.button("DRS schließen")




if "drs_stellung" not in st.session_state:
    st.session_state.drs_stellung = 0.0

if "drs_ziel" not in st.session_state:
    st.session_state.drs_ziel = 0.0

if drs_auf:
    st.session_state.drs_ziel = 1.0
if drs_zu:
    st.session_state.drs_ziel= 0.0


if "abtrieb_zu" not in st.session_state:
    st.session_state.abtrieb_zu = 0.0


if "sim" not in st.session_state or reset_button:
    sim = Windkanal(breite=breite, hoehe=hoehe, zellgrösse=zellgrösse)
    sim.setze_wand(eintrittsgeschwindigkeit_button)
    sim.u[:, :] = eintrittsgeschwindigkeit_button
    sim.u[sim.wand == 0] = 0.0
    sim.v[sim.wand == 0] = 0.0
    st.session_state.sim = sim
    st.session_state.schritt = 0



@st.fragment(run_every=0.1)
def simulation_laufen():
    sim = st.session_state.sim


    tempo = 0.1
    ziel = st.session_state.drs_ziel
    if st.session_state.drs_stellung < ziel:
        st.session_state.drs_stellung = min(ziel, st.session_state.drs_stellung + tempo)
    elif st.session_state.drs_stellung > ziel:
        st.session_state.drs_stellung = max(ziel, st.session_state.drs_stellung - tempo)


    klappen_winkel = st.session_state.drs_stellung * Klappenwinkel


    sim.wand = np.ones((sim.nx, sim.ny))
    sim.setze_wand(eintrittsgeschwindigkeit_button)
    
    hauptfluegel = naca_in_windkanal_einbauen(
        naca_code="4412",
        anstellwinkel=185,
        nx=sim.nx,
        ny=sim.ny,
        skalierung=50,
        x_offset=-10,
        y_offset=-10,
        horizontal_spiegeln=True,
    )
    
    drs_klappe = naca_in_windkanal_einbauen(
        naca_code="0012",
        anstellwinkel=-Klappenwinkel + klappen_winkel,
        nx=sim.nx,
        ny=sim.ny,
        skalierung=40,
        x_offset=35,
        y_offset=12,
    )

    sim.wand = np.minimum(sim.wand, hauptfluegel)
    sim.wand = np.minimum(sim.wand, drs_klappe)


    sim.zeitschritt(eintrittsgeschwindigkeit_button)

    st.session_state.schritt += 1
    zeit = st.session_state.schritt * sim.dt

    if st.session_state.drs_stellung == 1.0:
        status_text = ":green[DRS OFFEN]"
    elif st.session_state.drs_stellung == 0.0:
        status_text = ":gray[DRS ZU]"
    else:
        status_text = ":orange[DRS bewegt sich ...]"




    st.write(f"Zeit: **{zeit:.2f}** s | Status: **{status_text}** ")


    F, cL = sim.auftrieb_berechnen(eintrittsgeschwindigkeit_button)

    if st.session_state.drs_stellung == 0.0:
        st.session_state.abtrieb_zu = F

    if abs(st.session_state.abtrieb_zu) > 0:
        
        prozent_änderung = (1.0 - (abs(F) / abs(st.session_state.abtrieb_zu))) * 100.0
    else:
        prozent_änderung = 0.0

    st.write(f"Downforce: **{(F * -1):.2f} N**  |  Auftriebsbeiwert cL: **{cL:.2f}** | Abtriebsreduktion: **{prozent_änderung:.1f}%**")

    
    luft_druck = sim.p[sim.wand == 1]
    druck_grenze = max(abs(np.min(luft_druck)), abs(np.max(luft_druck)), 1.0)

    st.subheader("Druckverteilung")
    druck_bild = feld_zu_bild(sim.p, sim.wand, "RdBu_r", -druck_grenze, druck_grenze)
    
    st.image(druck_bild, use_container_width=True)


simulation_laufen()
