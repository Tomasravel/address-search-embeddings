import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import io
from preprocessing.maestro import MaestroProcessor
from preprocessing.embedder import MultiEmbedder
from preprocessing.model import Model

# Settings
global_config_path = "configs/base/global.yaml"
with open(global_config_path, "r", encoding="utf-8") as f:
    global_settings = yaml.safe_load(f)

struct_config_path = "configs/base/structured/config.yaml"
with open(struct_config_path, "r", encoding="utf-8") as f:
    struct_settings = yaml.safe_load(f)

st.image(global_settings["LOGO_PATH"], width=150)

@st.cache_resource
def load_maestro_and_embedder():
    "Load Maestro data and embedder models."
    # Load Maestro data
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG"]]
    maestro = MaestroProcessor(
        query_columns=QUERY_COLUMNS, 
        cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
        settings=struct_settings, 
        MAESTRO_PATH=global_settings["MAESTRO_PATH"]
    ).load_data().preprocess().generate_queries().set_query_as_index_hash().split_sin_calle().generate_no_height_street_hashes_list().generate_height_street_hashes_list()

    # Load embedder models
    STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in struct_settings["SEARCHER_CONFIG"]]
    struct_models = [Model(model_path=path, settings=struct_settings) for path in STRUCT_MODEL_PATHS]
    struct_embedder = MultiEmbedder(models=struct_models, maestro=maestro, settings=struct_settings)
    if struct_settings.get("LOAD_EMBEDDINGS", False):
        struct_embedder.load_embeddings()
    else:
        struct_embedder.generate_embeddings()
    struct_embedder.maestro.generate_unique_queries_groups()
    struct_embedder.maestro.generate_first_level_embeddings_matrix()
    return struct_embedder

# Load data and embedder
struct_embedder = load_maestro_and_embedder()

# Streamlit interface
st.title("Buscador de Direcciones")

st.subheader("Provincia", divider="gray")
c1, c2, c3 = st.columns([2, 1, 1])
provincia = c1.text_input("Provincia", key="provincia")
threshold_abs_prov = c2.slider(
    "Umbral de similitud (absoluto) - Provincia", 0.0, 1.0, 0.3, 0.01
)
threshold_rel_prov = c3.slider(
    "Umbral de similitud (relativo) - Provincia", 0.0, 1.0, 0.05, 0.01
)

st.divider()

# ── Municipio / Partido / Localidad ────────────────────────────────────────
st.subheader("Municipio / Partido / Localidad", divider="gray")
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
municipio = c1.text_input("Municipio", key="municipio")
partido   = c2.text_input("Partido", key="partido")
localidad = c3.text_input("Localidad", key="localidad")
threshold_abs_mpl = c4.slider(
    "Umbral de similitud (absoluto) - Municipio, Partido, Localidad", 0.0, 1.0, 0.3, 0.01
)
threshold_rel_mpl = c5.slider(
    "Umbral de similitud (relativo) - Municipio, Partido, Localidad", 0.0, 1.0, 0.05, 0.01
)

st.divider()

# ── Calle ──────────────────────────────────────────────────────────────────
# ── Calle ──────────────────────────────────────────────────────────────────
st.subheader("Calle", divider="gray")
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

calle = c1.text_input("Calle", key="calle")
altura = c2.text_input("Altura", placeholder="Ej: 2450", key="altura")  # NUEVO

threshold_abs_calle = c3.slider(
    "Umbral de similitud (absoluto) - Calle", 0.0, 1.0, 0.3, 0.01
)
threshold_rel_calle = c4.slider(
    "Umbral de similitud (relativo) - Calle", 0.0, 1.0, 0.05, 0.01
)

# --- Estado inicial ---
if "results" not in st.session_state:
    st.session_state.results = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "sugg_idx" not in st.session_state:
    st.session_state.sugg_idx = 0

# Botón de acción (alineado a la izquierda)
btn_col, _ = st.columns([1, 3])
if btn_col.button("Buscar", use_container_width=True):
    calle_query = f"{calle} {altura}".strip() if altura else calle

    results = struct_embedder.get_results(
        PROVINCIA=provincia,
        PARTIDO=partido,
        LOCALIDAD=localidad,
        MUNICIPIO=municipio,
        NOM_CALLE_COMPLETO=calle_query,
        threshold_abs_list=[threshold_abs_prov, threshold_abs_mpl, threshold_abs_calle],
        threshold_rel_list=[threshold_rel_prov, threshold_rel_mpl, threshold_rel_calle]
    )
    st.session_state.results = results

    if results is not None and not results.empty:
        # st.session_state.suggestions = [
        #     f"{row['NOM_CALLE_ABR_C']} {int(row['COD_DESDE'])}–{int(row['COD_HASTA'])}, {row.get('localidad_original','')} {row['PROVINCIA']}"
        #     for _, row in results.iterrows()
        # ]
        suggestions = []
        for _, row in results.iterrows():
            if pd.isna(row["COD_DESDE"]) and pd.isna(row["COD_HASTA"]):
                suggestion = f"{row['NOM_CALLE_ABR_C']}, {row['localidad_original']} {row['PROVINCIA']}"
            elif pd.isna(row["COD_DESDE"]):
                suggestion = f"{row['NOM_CALLE_ABR_C']} hasta {int(row['COD_HASTA'])}, {row['localidad_original']} {row['PROVINCIA']}"
            elif pd.isna(row["COD_HASTA"]):
                suggestion = f"{row['NOM_CALLE_ABR_C']} desde {int(row['COD_DESDE'])}, {row['localidad_original']} {row['PROVINCIA']}"
            else:
                suggestion = f"{row['NOM_CALLE_ABR_C']} {int(row['COD_DESDE'])}–{int(row['COD_HASTA'])}, {row['localidad_original']} {row['PROVINCIA']}"
            suggestions.append(suggestion)
        st.session_state.suggestions = suggestions
        st.session_state.sugg_idx = 0  # reset selección
    else:
        st.session_state.suggestions = []

# --- Render según estado persistido ---
if st.session_state.results is None:
    st.info("Por favor, ingresa una dirección.")
elif st.session_state.results.empty:
    st.warning("No se encontraron direcciones con este modelo.")
else:
    idx = st.radio(
        "Elige una sugerencia:",
        range(len(st.session_state.suggestions)),
        format_func=lambda i: st.session_state.suggestions[i],
        key="sugg_idx",
    )
    row = st.session_state.results.iloc[idx]
    st.write("Calle:", row['NOM_CALLE_ABR_C'])
    st.write("Número:", "" if pd.isna(row["HEIGHT"]) else str(int(row["HEIGHT"])))
    st.write("Municipio:", "" if pd.isna(row["MUNICIPIO"]) else str(row["MUNICIPIO"]))
    st.write("Localidad ingresada:", row['LOCALIDAD'])
    st.write("Localidad oficial:", row.get('localidad_original',''))
    st.write("Barrio:", row.get('BAR_NOMBRE',''))
    st.write("CP:", row['CPA'])
    st.write("Coordenadas:", f"{row['LATITUD']}, {row['LONGITUD']}")
    st.write("Provincia:", row['PROVINCIA'])
    st.write("Partido:", row['PARTIDO'])
    for i in range(struct_embedder.n):
        st.write(f"Similaridad {i}:", f"{row[f'similarity_level_{i}']:.4f}")

    # Botón para limpiar y hacer una nueva búsqueda
    st.button("Nueva búsqueda", on_click=lambda: st.session_state.update(
        results=None, suggestions=[], sugg_idx=0
    ))

st.subheader("Subir un archivo .txt con direcciones")

file_path = st.file_uploader("Sube un archivo TXT delimitado por tabulaciones", type=['txt'])
if file_path is not None:
    process = st.button("Procesar archivo")
    if process:
        try:
            batch_txt_ids = global_settings["BATCH_TXT_ID_COLUMN"]
            df = pd.read_csv(file_path, encoding='latin1', delimiter='\t')
            batch_txt_ids = ['ID']
            df = df.rename(COLUMN_NAMES_MAPPING, axis=1)
            df = df.fillna('').astype(str)
            if "HEIGHT" in df.columns:
                df["NOM_CALLE_COMPLETO"] = df["NOM_CALLE_COMPLETO"] + ' ' + df["HEIGHT"]
            cols = ["PROVINCIA","PARTIDO","LOCALIDAD","MUNICIPIO","NOM_CALLE_COMPLETO","PXL_PREF1974"]
            queries_batch = df[[col for col in cols if col in df.columns]].to_dict(orient='records')
            progress = st.progress(0)
            status = st.empty()

            def _cb(done, total):
                pct = int(done * 100 / total)
                progress.progress(min(pct, 100))
                status.text(f"Procesando: {done}/{total} ({pct}%)")

            max_addresses_to_operate_list=[s["MAX_ADDRESSES_TO_OPERATE"] for s in struct_settings["SEARCHER_CONFIG"]]
            max_addresses_to_show_list=[s["MAX_ADDRESSES_TO_SHOW"] for s in struct_settings["SEARCHER_CONFIG"]]
            max_addresses_to_show_list[-1]=1
            result_df = struct_embedder.get_results_batch(
                queries_batch=queries_batch,
                id_list=df[global_settings["BATCH_TXT_ID_COLUMN"]].iloc[:,0].tolist(),
                threshold_abs_list=[threshold_abs_prov, threshold_abs_mpl, threshold_abs_calle],
                threshold_rel_list=[threshold_rel_prov, threshold_rel_mpl, threshold_rel_calle],
                max_addresses_to_operate_list=max_addresses_to_operate_list,
                max_addresses_to_show_list=max_addresses_to_show_list,
                progress_cb=_cb,           
                progress_every=5           
            )

            st.write("Vista previa del archivo procesado:")
            st.dataframe(result_df.head())

            st.subheader("Descargar archivo procesado")
            output = io.BytesIO()
            result_df.to_csv(output, sep='\t', index=True)
            output.seek(0)
            st.download_button(
                label="Descargar TXT con resultados",
                data=output,
                file_name="direcciones_procesadas.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()


