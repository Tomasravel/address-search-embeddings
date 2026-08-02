import os
import cProfile, pstats, io as sio
import yaml
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from preprocessing.maestro import MaestroProcessor
from preprocessing.model import Model
from preprocessing.embedder import MultiEmbedder

# ---------- Config ----------
STRUCT_CONFIG_PATH = "configs/profiler/configs/structured.yaml"
UNSTRUCT_CONFIG_PATH = "configs/profiler/configs/unstructured.yaml"
GLOBAL_CONFIG_PATH = "configs/profiler/configs/global.yaml"

# Dónde guardar localmente
LOCAL_OUT_DIR = "profiler_reports"  # p.ej. ./profiler_reports

# ---------- Helpers ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_for_filename(s: str, maxlen: int = 60) -> str:
    keep = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    return keep[:maxlen]

def profile_and_save(query: str, suffix: str, embedder, out_dir: str):
    """Ejecuta cProfile sobre una query, genera reporte y lo guarda localmente."""
    pr = cProfile.Profile()
    pr.enable()
    results = embedder.get_results(query, threshold_abs_list=[0.3], threshold_rel_list=[0.05])
    if results.empty:
        st.warning("No se encontraron direcciones con este modelo.")
    else:
        suggestions = [
            f"{row['NOM_CALLE_ABR']}" + f" {row['COD_DESDE']}–{row['COD_HASTA']}" + f", {row['localidad_original']} {row['PROVINCIA']}"
            for _, row in results.iterrows()
        ]
        print(suggestions)
        idx = st.radio("Elige una sugerencia:", range(len(suggestions)),
                    format_func=lambda i: suggestions[i])
        if idx is not None:
            row = results.iloc[idx]
            # --- Formulario minimalista ---
            st.write("Calle:", row['NOM_CALLE_ABR'])
            st.write("Número:", row["HEIGHT"])
            st.write("Municipio:", "" if pd.isna(row["MUNICIPIO"]) else str(row["MUNICIPIO"]))
            st.write("Localidad:", row['LOCALIDAD'])
            st.write("Localidad original:", row.get('localidad_original',''))
            st.write("CP:", row['CPA'])
            st.write("Coordenadas:", f"{row['LATITUD']}, {row['LONGITUD']}")
            st.write("Provincia:", row['PROVINCIA'])
            st.write("Partido:", row['PARTIDO'])
    pr.disable()

    # Top 30 por tiempo acumulado -> .txt
    s = sio.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats()
    report_text = s.getvalue()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = sanitize_for_filename(query.replace(" ", "_"))
    txt_path = os.path.join(out_dir, f"profile_{ts}_{tag}{suffix}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Reporte guardado: {txt_path}")

    # (Opcional) Guardar dump binario .prof para análisis posterior:
    # prof_path = os.path.join(out_dir, f"profile_{ts}_{tag}{suffix}.prof")
    # pr.dump_stats(prof_path)
    # print(f"[OK] Dump binario guardado: {prof_path}")

# ---------- Carga de settings ----------
# with open(STRUCT_CONFIG_PATH, "r", encoding="utf-8") as f:
#     struct_settings = yaml.safe_load(f)
with open(UNSTRUCT_CONFIG_PATH, "r", encoding="utf-8") as f:
    unstruct_settings = yaml.safe_load(f)
with open(GLOBAL_CONFIG_PATH, "r", encoding="utf-8") as f:
    global_settings = yaml.safe_load(f)

QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in unstruct_settings["SEARCHER_CONFIG"]]
UNSTRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in unstruct_settings["SEARCHER_CONFIG"]]

maestro = MaestroProcessor(
    query_columns=QUERY_COLUMNS,
    cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
    settings=unstruct_settings,
    MAESTRO_PATH=unstruct_settings["MAESTRO_PROCESSED_PATH"],
).load_data().set_query_as_index_hash()

unstruct_models = [Model(model_path=path, settings=unstruct_settings) for path in UNSTRUCT_MODEL_PATHS]
unstruct_embedder = MultiEmbedder(models=unstruct_models, maestro=maestro, settings=unstruct_settings).load_embeddings()
unstruct_embedder.maestro.generate_unique_queries_groups()
unstruct_embedder.maestro.generate_first_level_embeddings_matrix()

# ---------- Preparar salida y queries ----------
ensure_dir(LOCAL_OUT_DIR)

items = [
    "tigre bsas torcuato boqueron",              # primera -> _wo_height
    "salta la candelaria el tala 20 de febrero 51",  # segunda -> _w_height
]

# ---------- Ejecutar perfiles y guardar ----------
profile_and_save(items[0], "_wo_height", unstruct_embedder, LOCAL_OUT_DIR)
profile_and_save(items[1], "_w_height",  unstruct_embedder, LOCAL_OUT_DIR)
