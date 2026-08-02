#!/usr/bin/env python3
# validate_performance.py
#%%
import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml

from preprocessing.maestro import MaestroProcessor
from preprocessing.embedder import MultiEmbedder
from preprocessing.model import Model


def build_result_df(
    global_config_path: str,
    unstruct_config_path: str,
    control_config_path: str,
    control_txt_path: str,
    control_sample_n: int,
    random_state: int,
) -> pd.DataFrame:
    # --------- Carga de settings ----------
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)
    with open(unstruct_config_path, "r", encoding="utf-8") as f:
        unstruct_settings = yaml.safe_load(f)
    with open(control_config_path, "r", encoding="utf-8") as f:
        control_settings = yaml.safe_load(f)

    # --------- Maestro base (embeddings) ----------
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in unstruct_settings["SEARCHER_CONFIG"]]
    maestro = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=unstruct_settings,
            MAESTRO_PATH=unstruct_settings["MAESTRO_PROCESSED_PATH"],
            APPLY_LOCALIDADES_LINDERAS=False
        )
        .load_data()
        .preprocess()
        .set_query_as_index_hash()
        .split_sin_calle()
        .generate_no_height_street_hashes_list()
        .generate_height_street_hashes_list()
    )

    # --------- Modelos / embedder ----------
    UNSTRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in unstruct_settings["SEARCHER_CONFIG"]]
    unstruct_models = [Model(model_path=path, settings=unstruct_settings) for path in UNSTRUCT_MODEL_PATHS]
    unstruct_embedder = MultiEmbedder(models=unstruct_models, maestro=maestro, settings=unstruct_settings)
    unstruct_embedder.load_embeddings()
    unstruct_embedder.maestro.load_unique_queries_groups()
    unstruct_embedder.maestro.load_first_level_embeddings_matrix()

    # --------- Lote de control ----------
    control_query_columns = [
        ["C_PROVINCIA", "C_DEPARTAMENTO", "C_LOCALIDAD", "C_MUNICIPIO", "C_CALLE", "C_NUMERO"]
    ]
    cpa_column = "CPA_CONTROL"

    try:
        control_df = pd.read_csv(control_txt_path, encoding="latin1")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {control_txt_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo está vacío: {control_txt_path}")

    if len(control_df) < control_sample_n:
        raise ValueError(
            f"El archivo contiene solo {len(control_df)} filas, "
            f"pero se pidieron {control_sample_n} para el muestreo."
        )

    control_df = (
        control_df
        .sample(n=control_sample_n, random_state=random_state)
        .reset_index(drop=True)
    )
    control_df["C_MUNICIPIO"] = ""

    control_processed = (
        MaestroProcessor(
            query_columns=control_query_columns,
            settings=control_settings,
            maestro_df=control_df,
            APPLY_LOCALIDADES_LINDERAS=False,
        )
        .preprocess()
        .generate_queries()
        .set_query_as_index_hash()
    )

    queries_batch = control_processed.to_values()

    threshold_abs_list = [sc["THRESHOLD_ABS"] for sc in unstruct_settings["SEARCHER_CONFIG"]]
    threshold_rel_list = [sc["THRESHOLD_REL"] for sc in unstruct_settings["SEARCHER_CONFIG"]]
    max_ops_list       = [sc["MAX_ADDRESSES_TO_OPERATE"] for sc in unstruct_settings["SEARCHER_CONFIG"]]
    max_shows_list     = [sc["MAX_ADDRESSES_TO_SHOW"] for sc in unstruct_settings["SEARCHER_CONFIG"]]

    cols_to_drop = ["embedding_0", "embedding_1", "embedding_2"]
    result_df = unstruct_embedder.get_results_batch(
        queries_batch=queries_batch,
        id_list=control_processed.get_columns(cpa_column).tolist(),   # IDs = CPA correctos (GT)
        threshold_abs_list=threshold_abs_list,
        threshold_rel_list=threshold_rel_list,
        max_addresses_to_operate_list=max_ops_list,
        max_addresses_to_show_list=max_shows_list,  
        use_cp=False         
    )
    result_df = result_df.drop(columns=[c for c in cols_to_drop if c in result_df.columns], errors="ignore")

    # Garantizar columna 'CPA' predicha
    if "CPA" not in result_df.columns:
        # Ajustá acá si tu columna predicha se llama distinto
        # p.ej.: result_df.rename(columns={"BEST_CPA": "CPA"}, inplace=True)
        raise KeyError("El DataFrame de resultados no contiene la columna 'CPA' (predicho). Renombrá aquí si difiere.")

    return result_df


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def evaluate_and_exit(result_df: pd.DataFrame, threshold: float, min_samples: int, show_mismatches: int) -> None:
    import sys
    if not isinstance(result_df.index, pd.Index):
        print("✖ Se espera el CPA correcto en el índice.", file=sys.stderr); sys.exit(2)
    if "CPA" not in result_df.columns:
        print('✖ Falta la columna "CPA" con el CPA encontrado.', file=sys.stderr); sys.exit(2)

    df = result_df.copy()
    df["GT"]   = _norm(pd.Series(df.index, index=df.index, name="GT"))
    df["PRED"] = _norm(df["CPA"])

    n_queries = df.index.nunique()
    if n_queries < min_samples:
        print(f"✖ Data insuficiente: {n_queries} < {min_samples}", file=sys.stderr); sys.exit(2)

    # hit@k por índice: ¿al menos un PRED == GT?
    hits_by_idx = df.groupby(level=0).apply(lambda g: (g["PRED"] == g["GT"].iloc[0]).any())
    hit_at_k = hits_by_idx.mean()

    # Diagnóstico
    bad_idx = hits_by_idx[~hits_by_idx].index
    if len(bad_idx) and show_mismatches > 0:
        print("\n--- Ejemplos sin hit@k (hasta 10 índices) ---")
        to_show = (
            df.loc[df.index.isin(bad_idx[:10]), ["GT", "PRED"]]
              .groupby(level=0)
              .head(show_mismatches)
        )
        print(to_show.to_string())

    print(f"\nHit@k = {hit_at_k:.4f}  (threshold = {threshold:.2f}, queries={n_queries})")
    if hit_at_k < threshold:
        print("✖ Performance insuficiente (hit@k < threshold).", file=sys.stderr)
        sys.exit(1)
    print("✓ Performance OK (hit@k ≥ threshold).")


def parse_args():
    p = argparse.ArgumentParser(description="Valida performance (accuracy) del resolutor de CPA")
    p.add_argument("--global-config", default=os.getenv("GLOBAL_CONFIG_PATH", "configs/global.yaml"))
    p.add_argument("--unstruct-config", default=os.getenv("UNSTRUCT_CONFIG_PATH", "configs/unstructured.yaml"))
    p.add_argument("--control-config", default=os.getenv("CONTROL_CONFIG_PATH", "configs/control/unstructured.yaml"))
    p.add_argument("--control-path",  default=os.getenv("CONTROL_TXT_PATH", "datasets/control/lote_control.txt"))
    p.add_argument("--sample-n", type=int, default=int(os.getenv("CONTROL_SAMPLE_N", "100")))
    p.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    p.add_argument("--threshold", type=float, default=float(os.getenv("PERFORMANCE_THRESHOLD", "0.90")))
    p.add_argument("--min-samples", type=int, default=int(os.getenv("MIN_SAMPLES", "1")))
    p.add_argument("--show-mismatches", type=int, default=int(os.getenv("SHOW_MISMATCHES", "10")))
    p.add_argument("--no-exit", action="store_true", help="No hace sys.exit; devuelve código en stdout")
    p.add_argument("--save-csv", default=os.getenv("SAVE_CSV", ""), help="Ruta para guardar el result_df (opcional)")
    return p.parse_args()


def main():
    args = parse_args()
    df = build_result_df(
        global_config_path=args.global_config,
        unstruct_config_path=args.unstruct_config,
        control_config_path=args.control_config,
        control_txt_path=args.control_path,
        control_sample_n=args.sample_n,
        random_state=args.random_state,
    )
    df.to_csv("unstructured_control_results.csv", index=True)
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        df.to_csv(args.save_csv, index=True)
        print(f"Guardado result_df en: {args.save_csv}")
    evaluate_and_exit(
        df, 
        threshold=args.threshold, 
        min_samples=args.min_samples, 
        show_mismatches=args.show_mismatches,
        # no_exit=bool(os.getenv("NO_EXIT")) or getattr(args, "no_exit", False)
    )

#%%
if __name__ == "__main__":
    main()
