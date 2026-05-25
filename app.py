# -*- coding: utf-8 -*-
"""
Dashboard interactivo - Proyecto Spotify
Proyecto de Ingeniería de Datos

Este archivo se ejecuta con Streamlit y permite interactuar con:
- playlist del grupo
- dataset global de Spotify
- métricas generales
- filtros
- gráficas
- canciones representativas
- arquitectura de datos
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Dashboard Spotify",
    page_icon="🎧",
    layout="wide"
)

COLOR_VERDE = "#1DB954"
COLOR_FONDO = "#0f1117"
COLOR_TARJETA = "#161b22"


# =========================
# ESTILOS VISUALES
# =========================

st.markdown(
    """
    <style>
        .stApp {
            background: #0f1117;
            color: #f5f5f5;
        }

        h1, h2, h3 {
            color: #ffffff;
        }

        .bloque-titulo {
            background: linear-gradient(135deg, #111827, #0b0f14);
            border: 1px solid rgba(29, 185, 84, 0.35);
            border-radius: 22px;
            padding: 28px;
            margin-bottom: 20px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.28);
        }

        .titulo-dashboard {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 6px;
        }

        .subtitulo-dashboard {
            text-align: center;
            color: #a5b4c3;
            font-size: 18px;
            margin-bottom: 0;
        }

        .logo-circulo {
            width: 82px;
            height: 82px;
            margin: 0 auto 14px auto;
            border-radius: 50%;
            background: #1DB954;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 42px;
            font-weight: bold;
            box-shadow: 0 0 35px rgba(29,185,84,0.35);
        }

        .tarjeta-metrica {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 20px;
            min-height: 112px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.20);
        }

        .tarjeta-metrica h4 {
            color: #9ca3af;
            font-size: 15px;
            margin: 0 0 8px 0;
            font-weight: 600;
        }

        .tarjeta-metrica p {
            color: #1DB954;
            font-size: 30px;
            font-weight: 800;
            margin: 0;
        }

        .caja-texto {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 5px solid #1DB954;
            border-radius: 16px;
            padding: 18px 20px;
            color: #d1d5db;
            line-height: 1.6;
        }

        .pregunta {
            background: rgba(29,185,84,0.08);
            border: 1px solid rgba(29,185,84,0.22);
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 10px;
            color: #d1fae5;
        }

        .nota {
            color: #9ca3af;
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
        }

        div[data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# CARGA Y LIMPIEZA DE DATOS
# =========================

@st.cache_data
def cargar_playlist(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga la playlist exportada desde Spotify.
    El archivo tiene separadores especiales al final, por eso se limpia manualmente.
    """

    ruta = Path(ruta_archivo)

    with open(ruta, "r", encoding="utf-8-sig", errors="replace") as archivo:
        texto = archivo.read()

    lineas = texto.splitlines()

    encabezado = lineas[0].rstrip(";")
    columnas = next(csv.reader([encabezado]))

    filas = []

    for linea in lineas[1:]:
        linea = linea.rstrip(";")

        if linea.startswith('"') and linea.endswith('"'):
            linea = linea[1:-1]

        linea = linea.replace('""', '"')

        try:
            fila = next(csv.reader([linea]))
            if len(fila) == len(columnas):
                filas.append(fila)
        except Exception:
            continue

    playlist_original = pd.DataFrame(filas, columns=columnas)

    playlist_original.columns = (
        playlist_original.columns
        .str.replace("﻿", "", regex=False)
        .str.strip()
    )

    playlist_original = playlist_original.rename(columns={
        "Track Name": "track_name",
        "Artist Name(s)": "artists",
        "Duration (ms)": "duration_ms",
        "Genres": "track_genre",
        "Energy": "energy",
        "Tempo": "tempo",
        "Valence": "valence",
        "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness",
        "Loudness": "loudness",
        "Speechiness": "speechiness",
        "Popularity": "popularity"
    })

    variables_playlist = [
        "track_name",
        "artists",
        "energy",
        "tempo",
        "valence",
        "acousticness",
        "instrumentalness",
        "loudness",
        "speechiness",
        "duration_ms",
        "track_genre",
        "popularity"
    ]

    playlist = playlist_original[variables_playlist].copy()

    columnas_numericas = [
        "energy",
        "tempo",
        "valence",
        "acousticness",
        "instrumentalness",
        "loudness",
        "speechiness",
        "duration_ms",
        "popularity"
    ]

    for columna in columnas_numericas:
        playlist[columna] = pd.to_numeric(playlist[columna], errors="coerce")

    playlist["track_genre"] = (
        playlist["track_genre"]
        .astype(str)
        .str.strip()
        .replace(["", "nan", "None"], "sin género registrado")
    )

    playlist = playlist.dropna()
    playlist = playlist.drop_duplicates()

    return playlist


@st.cache_data
def cargar_kaggle(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga y limpia el dataset global de Spotify.
    """

    kaggle = pd.read_csv(ruta_archivo)
    kaggle.columns = kaggle.columns.str.strip()

    variables_kaggle = [
        "energy",
        "tempo",
        "valence",
        "acousticness",
        "instrumentalness",
        "loudness",
        "speechiness",
        "duration_ms",
        "track_genre",
        "popularity"
    ]

    kaggle = kaggle[variables_kaggle].copy()
    kaggle = kaggle.dropna()
    kaggle = kaggle.drop_duplicates()

    return kaggle


def obtener_generos(df: pd.DataFrame) -> pd.Series:
    """
    Separa los géneros cuando vienen separados por coma o punto y coma.
    """

    generos = (
        df["track_genre"]
        .astype(str)
        .str.replace(";", ",", regex=False)
        .str.split(",")
        .explode()
        .str.strip()
    )

    generos = generos[
        (generos != "") &
        (generos.str.lower() != "nan") &
        (generos.str.lower() != "sin género registrado") &
        (generos.str.lower() != "sin genero registrado")
    ]

    return generos


def clasificar_popularidad(valor: float) -> str:
    if valor >= 70:
        return "Mainstream"
    elif valor >= 40:
        return "Popularidad media"
    else:
        return "Alternativa"


def interpretar_perfil(energy: float, valence: float, acousticness: float, popularity: float) -> str:
    """
    Clasificación sencilla del perfil musical.
    """

    perfiles = []

    if energy >= 0.70:
        perfiles.append("energético")
    elif energy >= 0.45:
        perfiles.append("intensidad media")
    else:
        perfiles.append("tranquilo")

    if valence >= 0.60:
        perfiles.append("alegre")
    elif valence >= 0.40:
        perfiles.append("emocional equilibrado")
    else:
        perfiles.append("melancólico")

    if acousticness >= 0.50:
        perfiles.append("acústico")
    else:
        perfiles.append("digital/producido")

    if popularity >= 70:
        perfiles.append("mainstream")
    elif popularity >= 40:
        perfiles.append("popularidad media")
    else:
        perfiles.append("alternativo")

    return " · ".join(perfiles)


def calcular_representativas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula canciones cercanas al promedio del conjunto filtrado.
    """

    if df.empty:
        return pd.DataFrame()

    energy_prom = df["energy"].mean()
    valence_prom = df["valence"].mean()
    acousticness_prom = df["acousticness"].mean()

    temporal = df.copy()

    temporal["distancia_promedio"] = (
        abs(temporal["energy"] - energy_prom) +
        abs(temporal["valence"] - valence_prom) +
        abs(temporal["acousticness"] - acousticness_prom)
    )

    columnas_mostrar = [
        "track_name",
        "artists",
        "track_genre",
        "energy",
        "valence",
        "acousticness",
        "popularity",
        "distancia_promedio"
    ]

    return temporal.sort_values("distancia_promedio").head(10)[columnas_mostrar]


# =========================
# LECTURA DE ARCHIVOS
# =========================

RUTA_PLAYLIST = "Liked_Songs proyecto.csv"
RUTA_KAGGLE = "datasetkaggle.csv"
RUTA_ARQUITECTURA = "arquitectura.png"

playlist = cargar_playlist(RUTA_PLAYLIST)
kaggle = cargar_kaggle(RUTA_KAGGLE)


# =========================
# SIDEBAR / FILTROS
# =========================

st.sidebar.title("🎛️ Filtros del dashboard")
st.sidebar.caption("Interactúa con los datos del grupo.")

generos_playlist = obtener_generos(playlist)
lista_generos = sorted(generos_playlist.unique())

generos_seleccionados = st.sidebar.multiselect(
    "Filtrar por género",
    options=lista_generos,
    default=[]
)

rango_popularidad = st.sidebar.slider(
    "Rango de popularidad",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

rango_energy = st.sidebar.slider(
    "Rango de energy",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01
)

busqueda = st.sidebar.text_input(
    "Buscar canción o artista",
    placeholder="Ej: Bad Bunny, Shakira, rock..."
)

variable_distribucion = st.sidebar.selectbox(
    "Variable para distribución",
    ["energy", "tempo", "valence", "acousticness", "popularity", "speechiness", "instrumentalness"]
)

top_n = st.sidebar.slider(
    "Cantidad de géneros a mostrar",
    min_value=5,
    max_value=20,
    value=10
)


# =========================
# APLICACIÓN DE FILTROS
# =========================

playlist_filtrada = playlist.copy()

playlist_filtrada = playlist_filtrada[
    (playlist_filtrada["popularity"] >= rango_popularidad[0]) &
    (playlist_filtrada["popularity"] <= rango_popularidad[1]) &
    (playlist_filtrada["energy"] >= rango_energy[0]) &
    (playlist_filtrada["energy"] <= rango_energy[1])
]

if generos_seleccionados:
    patron_generos = "|".join(generos_seleccionados)
    playlist_filtrada = playlist_filtrada[
        playlist_filtrada["track_genre"].str.contains(
            patron_generos,
            case=False,
            na=False,
            regex=True
        )
    ]

if busqueda.strip():
    texto = busqueda.strip().lower()
    playlist_filtrada = playlist_filtrada[
        playlist_filtrada["track_name"].str.lower().str.contains(texto, na=False) |
        playlist_filtrada["artists"].str.lower().str.contains(texto, na=False) |
        playlist_filtrada["track_genre"].str.lower().str.contains(texto, na=False)
    ]


# =========================
# ENCABEZADO
# =========================

st.markdown(
    """
    <div class="bloque-titulo">
        <div class="logo-circulo">♫</div>
        <div class="titulo-dashboard">Conjunto de usuarios vs Dataset de Spotify</div>
        <p class="subtitulo-dashboard">
            Dashboard interactivo para analizar playlists del grupo y compararlas con un dataset global.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# VALIDACIÓN DE FILTROS
# =========================

if playlist_filtrada.empty:
    st.warning("No hay canciones que coincidan con los filtros seleccionados. Ajusta los filtros para ver resultados.")
    st.stop()


# =========================
# MÉTRICAS PRINCIPALES
# =========================

cantidad_canciones = len(playlist_filtrada)
cantidad_generos = obtener_generos(playlist_filtrada).nunique()

energy_prom = playlist_filtrada["energy"].mean()
tempo_prom = playlist_filtrada["tempo"].mean()
valence_prom = playlist_filtrada["valence"].mean()
acousticness_prom = playlist_filtrada["acousticness"].mean()
popularity_prom = playlist_filtrada["popularity"].mean()

perfil_general = interpretar_perfil(
    energy_prom,
    valence_prom,
    acousticness_prom,
    popularity_prom
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="tarjeta-metrica">
        <h4>Canciones analizadas</h4>
        <p>{cantidad_canciones}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="tarjeta-metrica">
        <h4>Géneros distintos</h4>
        <p>{cantidad_generos}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="tarjeta-metrica">
        <h4>Energy promedio</h4>
        <p>{energy_prom:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="tarjeta-metrica">
        <h4>Popularidad promedio</h4>
        <p>{popularity_prom:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================
# PESTAÑAS
# =========================

tab_resumen, tab_generos, tab_comparacion, tab_canciones, tab_preguntas, tab_arquitectura = st.tabs([
    "📌 Resumen",
    "🎼 Géneros",
    "📊 Comparación",
    "🎧 Canciones",
    "❓ Preguntas",
    "🏗️ Arquitectura"
])


# =========================
# TAB 1: RESUMEN
# =========================

with tab_resumen:
    st.subheader("📌 Perfil musical del grupo")

    st.markdown(
        f"""
        <div class="caja-texto">
            <strong>Perfil general:</strong> {perfil_general}.<br><br>
            El conjunto analizado presenta una energía promedio de <strong>{energy_prom:.3f}</strong>,
            una valencia promedio de <strong>{valence_prom:.3f}</strong>,
            una acousticness promedio de <strong>{acousticness_prom:.3f}</strong>
            y una popularidad promedio de <strong>{popularity_prom:.1f}</strong>.
            Estos valores permiten interpretar el comportamiento musical del grupo según intensidad,
            emoción, tipo de sonido y cercanía con canciones populares.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Distribución interactiva de variables musicales")

    figura_distribucion = px.histogram(
        playlist_filtrada,
        x=variable_distribucion,
        nbins=25,
        title=f"Distribución de {variable_distribucion}",
        color_discrete_sequence=[COLOR_VERDE]
    )

    figura_distribucion.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO
    )

    st.plotly_chart(figura_distribucion, use_container_width=True)

    st.markdown("### Relación Energy vs Valence")

    figura_dispersion = px.scatter(
        playlist_filtrada,
        x="energy",
        y="valence",
        hover_name="track_name",
        hover_data=["artists", "track_genre", "popularity"],
        color="popularity",
        color_continuous_scale="Greens",
        title="Relación entre intensidad musical y emoción"
    )

    figura_dispersion.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO
    )

    st.plotly_chart(figura_dispersion, use_container_width=True)


# =========================
# TAB 2: GÉNEROS
# =========================

with tab_generos:
    st.subheader("🎼 Diversidad musical")

    generos_filtrados = obtener_generos(playlist_filtrada)
    top_generos = generos_filtrados.value_counts().head(top_n).reset_index()
    top_generos.columns = ["Género", "Cantidad"]

    col_a, col_b = st.columns([2, 1])

    with col_a:
        figura_generos = px.bar(
            top_generos,
            x="Género",
            y="Cantidad",
            title="Géneros más frecuentes en el grupo",
            color="Cantidad",
            color_continuous_scale="Greens"
        )

        figura_generos.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_FONDO,
            plot_bgcolor=COLOR_FONDO,
            xaxis_tickangle=-35
        )

        st.plotly_chart(figura_generos, use_container_width=True)

    with col_b:
        st.markdown(
            f"""
            <div class="caja-texto">
                <strong>Género más frecuente:</strong><br>
                {top_generos.iloc[0]["Género"]}<br><br>
                <strong>Cantidad:</strong><br>
                {int(top_generos.iloc[0]["Cantidad"])} canciones<br><br>
                <strong>Diversidad:</strong><br>
                {cantidad_generos} géneros distintos.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.dataframe(top_generos, use_container_width=True)


# =========================
# TAB 3: COMPARACIÓN
# =========================

with tab_comparacion:
    st.subheader("📊 Comparación grupo vs dataset global")

    variables_comparacion = ["energy", "valence", "acousticness", "speechiness", "instrumentalness", "popularity"]

    comparacion = pd.DataFrame({
        "Variable": variables_comparacion,
        "Playlist grupo": [playlist_filtrada[var].mean() for var in variables_comparacion],
        "Dataset global": [kaggle[var].mean() for var in variables_comparacion]
    })

    comparacion_larga = comparacion.melt(
        id_vars="Variable",
        value_vars=["Playlist grupo", "Dataset global"],
        var_name="Fuente",
        value_name="Promedio"
    )

    figura_comparacion = px.bar(
        comparacion_larga,
        x="Variable",
        y="Promedio",
        color="Fuente",
        barmode="group",
        title="Comparación de promedios musicales",
        color_discrete_map={
            "Playlist grupo": COLOR_VERDE,
            "Dataset global": "#9ca3af"
        }
    )

    figura_comparacion.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO
    )

    st.plotly_chart(figura_comparacion, use_container_width=True)

    st.markdown("### Tabla comparativa")

    comparacion["Diferencia"] = comparacion["Playlist grupo"] - comparacion["Dataset global"]
    st.dataframe(comparacion.round(3), use_container_width=True)

    st.markdown("### Mainstream vs Alternativo")

    clasificacion = playlist_filtrada["popularity"].apply(clasificar_popularidad).value_counts().reset_index()
    clasificacion.columns = ["Clasificación", "Cantidad"]

    orden = ["Mainstream", "Popularidad media", "Alternativa"]
    clasificacion["Clasificación"] = pd.Categorical(clasificacion["Clasificación"], categories=orden, ordered=True)
    clasificacion = clasificacion.sort_values("Clasificación")

    figura_popularidad = px.bar(
        clasificacion,
        x="Clasificación",
        y="Cantidad",
        title="Clasificación de canciones según popularidad",
        color="Clasificación",
        color_discrete_sequence=[COLOR_VERDE, "#93c5fd", "#fbbf24"]
    )

    figura_popularidad.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO
    )

    st.plotly_chart(figura_popularidad, use_container_width=True)


# =========================
# TAB 4: CANCIONES
# =========================

with tab_canciones:
    st.subheader("🎧 Canciones representativas")

    representativas = calcular_representativas(playlist_filtrada)

    st.markdown(
        """
        <div class="caja-texto">
            Las canciones representativas se calculan según su cercanía con los promedios de
            <strong>energy</strong>, <strong>valence</strong> y <strong>acousticness</strong>.
            Por eso, funcionan como ejemplos del comportamiento musical general del grupo filtrado.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(representativas.round(3), use_container_width=True)

    csv_representativas = representativas.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Descargar canciones representativas",
        data=csv_representativas,
        file_name="canciones_representativas_spotify.csv",
        mime="text/csv"
    )

    st.markdown("### Dataset filtrado")

    columnas_tabla = [
        "track_name",
        "artists",
        "track_genre",
        "energy",
        "tempo",
        "valence",
        "acousticness",
        "popularity"
    ]

    st.dataframe(playlist_filtrada[columnas_tabla].round(3), use_container_width=True)

    csv_filtrado = playlist_filtrada[columnas_tabla].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Descargar dataset filtrado",
        data=csv_filtrado,
        file_name="playlist_filtrada_spotify.csv",
        mime="text/csv"
    )


# =========================
# TAB 5: PREGUNTAS
# =========================

with tab_preguntas:
    st.subheader("❓ Preguntas que responde el dashboard")

    preguntas = [
        "¿Cómo es el comportamiento musical del grupo?",
        "¿Qué características predominan en la música que escuchan?",
        "¿Qué tan diverso es el grupo en términos musicales?",
        "¿En qué se diferencia el grupo del dataset global?",
        "¿El grupo escucha música más popular o más alternativa?"
    ]

    for pregunta in preguntas:
        st.markdown(f"<div class='pregunta'>• {pregunta}</div>", unsafe_allow_html=True)

    st.markdown("### Respuestas automáticas según los datos filtrados")

    tendencia_popularidad = clasificar_popularidad(popularity_prom)

    st.markdown(
        f"""
        <div class="caja-texto">
            <strong>1. Comportamiento musical:</strong>
            El grupo presenta un perfil {perfil_general}.<br><br>

            <strong>2. Características predominantes:</strong>
            Predominan valores de energy de {energy_prom:.3f}, valence de {valence_prom:.3f}
            y acousticness de {acousticness_prom:.3f}.<br><br>

            <strong>3. Diversidad musical:</strong>
            Se identifican {cantidad_generos} géneros distintos en el conjunto filtrado.<br><br>

            <strong>4. Diferencia frente al dataset global:</strong>
            La popularidad promedio del grupo es {popularity_prom:.1f}, mientras que
            el dataset global tiene un promedio de {kaggle["popularity"].mean():.1f}.<br><br>

            <strong>5. Popular o alternativa:</strong>
            Según la popularidad promedio, el conjunto se clasifica como
            <strong>{tendencia_popularidad}</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# TAB 6: ARQUITECTURA
# =========================

with tab_arquitectura:
    st.subheader("🏗️ Arquitectura de datos del proyecto")

    st.markdown(
        """
        <div class="caja-texto">
            El proyecto se trabaja en conjunto con la asignatura de Arquitectura de Software.
            La solución integra playlists personales, dataset global de Spotify, almacenamiento en GitHub,
            procesamiento en Python/Google Colab y visualización mediante dashboard web.
            El flujo sigue una lógica ETL: extracción, limpieza, transformación, análisis y visualización.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if Path(RUTA_ARQUITECTURA).exists():
        st.image(RUTA_ARQUITECTURA, caption="Arquitectura de datos del proyecto Spotify", use_container_width=True)
    else:
        st.info("No se encontró la imagen de arquitectura. Agrega el archivo arquitectura.png en el repositorio.")


# =========================
# PIE DE PÁGINA
# =========================

st.markdown(
    """
    <p class="nota">
        Dashboard interactivo del proyecto de Ingeniería de Datos · Spotify · Render · Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
