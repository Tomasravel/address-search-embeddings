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


COLUMN_NAMES_MAPPING = {
  "C_PROVINCIA": "PROVINCIA",
  "C_DEPARTAMENTO": "PARTIDO",
  "C_LOCALIDAD": "LOCALIDAD",
  "C_MUNICIPIO": "MUNICIPIO",
  "C_CALLE": "NOM_CALLE_COMPLETO",
  "C_NUMERO": "HEIGHT",
  "CP": "PXL_PREF1974"
}

def build_result_df(
    global_config_path: str,
    struct_config_path: str,
    control_config_path: str,
    control_txt_path: str,
    control_sample_n: int,
    random_state: int,
) -> pd.DataFrame:
    # --------- Carga de settings ----------
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)
    with open(struct_config_path, "r", encoding="utf-8") as f:
        struct_settings = yaml.safe_load(f)
    with open(control_config_path, "r", encoding="utf-8") as f:
        control_settings = yaml.safe_load(f)

    # --------- Maestro base (embeddings) ----------
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG"]]
    maestro = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=struct_settings,
            MAESTRO_PATH=struct_settings["MAESTRO_PROCESSED_PATH"],
            APPLY_LOCALIDADES_LINDERAS=False
        ).process(load=True).generate_no_height_street_hashes_list().generate_height_street_hashes_list()
    )

    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"]]
    maestro_solo_localidad = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=struct_settings,
            MAESTRO_PATH=struct_settings["MAESTRO_SOLO_LOCALIDAD_PROCESSED_PATH"]
        ).process_solo_localidad(load=True).generate_no_height_street_hashes_list().generate_height_street_hashes_list()
    )

    # --------- Modelos / embedder ----------
    STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in struct_settings["SEARCHER_CONFIG"]]
    struct_models = [Model(model_path=path, settings=struct_settings) for path in STRUCT_MODEL_PATHS]
    struct_embedder = MultiEmbedder(models=struct_models, maestro=maestro, maestro_solo_localidad=maestro_solo_localidad, settings=struct_settings)
    struct_embedder = struct_embedder.build(load=True)

    # --------- Lote de control ----------
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
    control_df = control_df.rename(COLUMN_NAMES_MAPPING, axis=1)
    control_df = control_df.fillna('').astype(str)
    if "HEIGHT" in control_df.columns:
        control_df["NOM_CALLE_COMPLETO"] = control_df["NOM_CALLE_COMPLETO"] + ' ' + control_df["HEIGHT"]
    
    cols = ["PROVINCIA","PARTIDO","LOCALIDAD","MUNICIPIO","NOM_CALLE_COMPLETO","PXL_PREF1974"]
    queries_batch = control_df[[col for col in cols if col in control_df.columns]].to_dict(orient='records')

    threshold_abs_list = [sc["THRESHOLD_ABS"] for sc in struct_settings["SEARCHER_CONFIG"]]
    threshold_rel_list = [sc["THRESHOLD_REL"] for sc in struct_settings["SEARCHER_CONFIG"]]
    max_ops_list       = [sc["MAX_ADDRESSES_TO_OPERATE"] for sc in struct_settings["SEARCHER_CONFIG"]]
    max_addresses_to_show_list = [settings["MAX_ADDRESSES_TO_SHOW"] for settings in struct_settings["SEARCHER_CONFIG"]]
    cols_to_drop = ["embedding_0", "embedding_1", "embedding_2"]
    result_df = struct_embedder.get_results_batch(
        queries_batch=queries_batch,
        id_list=control_df[cpa_column].tolist(),   # IDs = CPA correctos (GT)
        threshold_abs_list=threshold_abs_list,
        threshold_rel_list=threshold_rel_list,
        max_addresses_to_operate_list=max_ops_list,
        max_addresses_to_show_list=max_addresses_to_show_list,           # top-1
    )
    result_df = result_df.drop(columns=[c for c in cols_to_drop if c in result_df.columns], errors="ignore")

    # Asegurar índice = GT (CPA correcto)
    gt_series = control_df[cpa_column]
    result_df = result_df.copy()
    result_df.index = gt_series.values

    # Garantizar columna 'CPA' predicha
    if "CPA" not in result_df.columns:
        # Ajustá acá si tu columna predicha se llama distinto
        # p.ej.: result_df.rename(columns={"BEST_CPA": "CPA"}, inplace=True)
        raise KeyError("El DataFrame de resultados no contiene la columna 'CPA' (predicho). Renombrá aquí si difiere.")

    return result_df


def normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def evaluate_and_exit(result_df: pd.DataFrame, threshold: float, min_samples: int, show_mismatches: int, no_exit: bool) -> None:
    if not isinstance(result_df.index, pd.Index):
        print("✖ El ground-truth debe estar en el índice del DataFrame.", file=sys.stderr)
        sys.exit(2)
    if "CPA" not in result_df.columns:
        print('✖ Falta la columna "CPA" con el valor predicho/encontrado.', file=sys.stderr)
        sys.exit(2)

    # --- Diagnóstico de calidad de datos ---
    total = len(result_df)
    gt = pd.Series(result_df.index, name="GT", index=result_df.index)
    pred = result_df["CPA"]

    # Normalizá GT (solo GT nulo invalida la fila)
    gt_mask = gt.notna()
    if gt_mask.sum() < min_samples:
        print(f"✖ Data insuficiente de GT no nulo: {gt_mask.sum()} < {min_samples}", file=sys.stderr)
        sys.exit(2)

    # Para que los missing cuenten como ERROR, reemplazamos NaN por marcador
    SENTINEL = "__MISSING__"
    gt_n   = normalize_series(gt.where(gt_mask))             # GT normalizado (solo no nulos)
    pred_n = normalize_series(pred.fillna(SENTINEL))         # Missing pred -> mismatch

    # Accuracy “estricto”: denominador = total de filas con GT válido
    valid_rows = gt_mask
    matches = pd.Series(False, index=result_df.index)
    matches.loc[valid_rows] = gt_n.loc[valid_rows].eq(pred_n.loc[valid_rows])

    # Desglose
    n_gt_valid   = int(gt_mask.sum())
    n_pred_nan   = int(pred.isna().sum())
    n_errors_col = int(result_df.get("error", pd.Series(index=result_df.index, dtype=object)).notna().sum())
    n_mismatches = int((valid_rows & ~matches).sum())

    acc = matches.loc[valid_rows].mean()

    if n_mismatches and show_mismatches > 0:
        mismatches = result_df.loc[valid_rows & ~matches, ["CPA"]].copy()
        mismatches["GT"]   = gt_n.loc[mismatches.index]
        mismatches["PRED"] = pred_n.loc[mismatches.index]
        print("\n--- Muestras de desvíos ---")
        print(mismatches[["GT", "PRED"]].head(show_mismatches).to_string(index=False))

    print(
        f"\nAccuracy CPA = {acc:.4f}  "
        f"(threshold = {threshold:.2f}, n={n_gt_valid}, "
        f"pred_missing={n_pred_nan}, errors={n_errors_col}, mismatches={n_mismatches})"
    )

    if acc < threshold:
        msg = "✖ Performance insuficiente. Fallando validación."
        if no_exit:
            print(msg)
            return 1
        print(msg, file=sys.stderr)
        sys.exit(1)
    print("✓ Performance OK. Supera el umbral.")
    return 0



def parse_args():
    p = argparse.ArgumentParser(description="Valida performance (accuracy) del resolutor de CPA")
    p.add_argument("--global-config", default=os.getenv("GLOBAL_CONFIG_PATH", "configs/global.yaml"))
    p.add_argument("--struct-config", default=os.getenv("STRUCT_CONFIG_PATH", "configs/structured.yaml"))
    p.add_argument("--control-config", default=os.getenv("CONTROL_CONFIG_PATH", "configs/control/structured.yaml"))
    p.add_argument("--control-path",  default=os.getenv("CONTROL_TXT_PATH", "datasets/control/lote_control.txt"))
    p.add_argument("--sample-n", type=int, default=int(os.getenv("CONTROL_SAMPLE_N", "100")))
    p.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    p.add_argument("--threshold", type=float, default=float(os.getenv("PERFORMANCE_THRESHOLD", "0.5")))
    p.add_argument("--min-samples", type=int, default=int(os.getenv("MIN_SAMPLES", "1")))
    p.add_argument("--show-mismatches", type=int, default=int(os.getenv("SHOW_MISMATCHES", "10")))
    p.add_argument("--no-exit", action="store_true", help="No hace sys.exit; devuelve código en stdout")
    p.add_argument("--save-csv", default=os.getenv("SAVE_CSV", ""), help="Ruta para guardar el result_df (opcional)")
    return p.parse_args()


def main():
    args = parse_args()
    df = build_result_df(
        global_config_path=args.global_config,
        struct_config_path=args.struct_config,
        control_config_path=args.control_config,
        control_txt_path=args.control_path,
        control_sample_n=args.sample_n,
        random_state=args.random_state,
    )
    df.to_csv("structured_control_results.csv", index=True)
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        df.to_csv(args.save_csv, index=True)
        print(f"Guardado result_df en: {args.save_csv}")
    evaluate_and_exit(
        df, 
        threshold=args.threshold, 
        min_samples=args.min_samples, 
        show_mismatches=args.show_mismatches,
        no_exit=bool(os.getenv("NO_EXIT")) or getattr(args, "no_exit", False)
    )

#%%
if __name__ == "__main__":
    main()
