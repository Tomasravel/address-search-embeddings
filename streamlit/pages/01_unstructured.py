import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import io
from preprocessing.maestro import MaestroProcessor
from preprocessing.embedder import MultiEmbedder
from preprocessing.model import Model
from st_keyup import st_keyup

# Settings
global_config_path = "configs/base/global.yaml"
with open(global_config_path, "r", encoding="utf-8") as f:
    global_settings = yaml.safe_load(f)

unstruct_config_path = "configs/base/unstructured/config.yaml"
with open(unstruct_config_path, "r", encoding="utf-8") as f:
    unstruct_settings = yaml.safe_load(f)

st.image(global_settings["LOGO_PATH"], width=150)

@st.cache_resource
def load_maestro_and_embedder():
    "Load Maestro data and embedder models."
    # Load Maestro data
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in unstruct_settings["SEARCHER_CONFIG"]]
    maestro = MaestroProcessor(
        query_columns=QUERY_COLUMNS,
        cpa_column=global_settings["MAESTRO_CPA_COLUMN"], 
        settings=unstruct_settings,
        MAESTRO_PATH=unstruct_settings["MAESTRO_PROCESSED_PATH"]
    ).load_data().preprocess().generate_queries().set_query_as_index_hash().split_sin_calle().generate_no_height_street_hashes_list().generate_height_street_hashes_list()

    # Load embedder models
    UNSTRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in unstruct_settings["SEARCHER_CONFIG"]]
    unstruct_models = [Model(model_path=path, settings=unstruct_settings) for path in UNSTRUCT_MODEL_PATHS]
    unstruct_embedder = MultiEmbedder(models=unstruct_models, maestro=maestro, settings=unstruct_settings)
    if unstruct_settings.get("LOAD_EMBEDDINGS", False):
        unstruct_embedder.load_embeddings()
    else:
        unstruct_embedder.generate_embeddings()
    
    unstruct_embedder.maestro.generate_unique_queries_groups()
    unstruct_embedder.maestro.generate_first_level_embeddings_matrix()
    return unstruct_embedder

# Load data and embedder
unstruct_embedder = load_maestro_and_embedder()

# Streamlit interface
st.title("Buscador de Direcciones")

# Sliders para ajustar umbral absoluto y relativo
threshold_abs = st.slider("Umbral de similitud (absoluto)", 0.0, 1.0, 0.3, 0.01)
threshold_rel = st.slider("Umbral de similitud (relativo)", 0.0, 1.0, 0.05, 0.01)

# Real-time input with keyup event
user_query = st_keyup("Ingresa una dirección:")

if user_query:
    results = unstruct_embedder.get_results(user_query, threshold_abs_list=[threshold_abs], threshold_rel_list=[threshold_rel])
    if results.empty:
        st.warning("No se encontraron direcciones con este modelo.")
    else:
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
        idx = st.radio("Elige una sugerencia:", range(len(suggestions)),
                    format_func=lambda i: suggestions[i])
        if idx is not None:
            row = results.iloc[idx]
            # --- Formulario minimalista ---
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
            for i in range(unstruct_embedder.n):
                st.write(f"Similaridad {i}:", f"{row[f'similarity_level_{i}']:.4f}")
else:
    st.info("Por favor, ingresa una dirección.")

st.subheader("Subir un archivo .txt con direcciones")

file = st.file_uploader("Sube un archivo TXT delimitado por tabulaciones", type=['txt'])
if file is not None:
    process = st.button("Procesar archivo")
    if process:
        try:
            batch_txt_query_columns = global_settings["BATCH_TXT_QUERY_COLUMNS_UNSTRUCT"]
            batch_txt_ids = global_settings["BATCH_TXT_ID_COLUMN"]
            df = pd.read_csv(file, encoding='latin1', delimiter='\t')
            df = df.drop_duplicates(
                subset=[col for cols in batch_txt_query_columns for col in cols if col in df.columns]
            )
            batch_txt = MaestroProcessor(
                query_columns=batch_txt_query_columns, 
                cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
                settings=unstruct_settings, 
                maestro_df=df,
                APPLY_LOCALIDADES_LINDERAS=False
            ).preprocess().generate_queries().set_query_as_index_hash()
            queries_batch = batch_txt.to_values()
            progress = st.progress(0)
            status = st.empty()

            def _cb(done, total):
                pct = int(done * 100 / total)
                progress.progress(min(pct, 100))
                status.text(f"Procesando: {done}/{total} ({pct}%)")

            result_df = unstruct_embedder.get_results_batch(
                queries_batch=queries_batch,
                id_list=batch_txt.get_columns(global_settings["BATCH_TXT_ID_COLUMN"]).iloc[:,0].tolist(),
                threshold_abs_list=[threshold_abs],
                threshold_rel_list=[threshold_rel],
                max_addresses_to_operate_list=[settings["MAX_ADDRESSES_TO_OPERATE"] for settings in unstruct_settings["SEARCHER_CONFIG"]],
                max_addresses_to_show_list=[1] * unstruct_embedder.n,
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