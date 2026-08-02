#!/usr/bin/env python3
# analyze_similarity_thresholds.py
#%%
import os
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

DEFAULT_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

def normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def is_filled_str(s: pd.Series) -> pd.Series:
    """
    True si el campo está "completo" (no vacío).
    Trata '', 'nan', 'none', 'null' como vacío.
    """
    x = s.astype(str).str.strip()
    x_low = x.str.lower()
    return (x != "") & (~x_low.isin(["nan", "none", "null"]))

def add_trinomio_count(control_df: pd.DataFrame) -> pd.DataFrame:
    df = control_df.copy()
    for c in ["PARTIDO", "MUNICIPIO", "LOCALIDAD"]:
        if c not in df.columns:
            df[c] = ""
    df["tri_count"] = (
        is_filled_str(df["PARTIDO"]).astype(int) +
        is_filled_str(df["MUNICIPIO"]).astype(int) +
        is_filled_str(df["LOCALIDAD"]).astype(int)
    )
    df["tri_group"] = np.where(df["tri_count"] == 1, "tri_eq_1", "tri_ge_2")
    return df

def describe_series(x: pd.Series, quantiles=DEFAULT_QUANTILES) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        out = {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            **{f"p{int(q*100):02d}": np.nan for q in quantiles},
            "max": np.nan
        }
        return pd.Series(out)

    qs = x.quantile(quantiles).to_dict()
    out = {
        "count": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "min": float(x.min()),
        **{f"p{int(q*100):02d}": float(qs[q]) for q in quantiles},
        "max": float(x.max())
    }
    return pd.Series(out)

def build_stats_table(df: pd.DataFrame, cols: list[str], group_name: str, cohort: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = describe_series(df[c])
        s.name = c
        rows.append(s)
    out = pd.DataFrame(rows)
    out.insert(0, "cohort", cohort)
    out.insert(1, "group", group_name)
    out.insert(2, "metric", out.index)
    out.reset_index(drop=True, inplace=True)
    return out

def pick_thresholds_from_stats(stats_df: pd.DataFrame, cohort: str, group: str) -> dict:
    row = stats_df[(stats_df["cohort"] == cohort) & (stats_df["group"] == group)].iloc[0]
    return {
        "count": int(row["count"]),
        "p01": row["p01"],
        "p05": row["p05"],
        "p10": row["p10"],
        "p25": row["p25"],
        "p50": row["p50"],
    }

# --------- build_result_df: ahora devuelve también control_df ---------
def build_result_df(
    global_config_path: str,
    struct_config_path: str,
    control_config_path: str,
    control_txt_path: str,
    control_sample_n: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)
    with open(struct_config_path, "r", encoding="utf-8") as f:
        struct_settings = yaml.safe_load(f)
    with open(control_config_path, "r", encoding="utf-8") as f:
        control_settings = yaml.safe_load(f)

    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG"]]
    maestro = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=struct_settings,
            MAESTRO_PATH=struct_settings["MAESTRO_PROCESSED_PATH"],
            APPLY_LOCALIDADES_LINDERAS=False
        )
        .load_data()
        .preprocess()
        .set_query_as_index_hash()
        .split_sin_calle()
        .generate_no_height_street_hashes_list()
        .generate_height_street_hashes_list()
    )

    STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in struct_settings["SEARCHER_CONFIG"]]
    struct_models = [Model(model_path=path, settings=struct_settings) for path in STRUCT_MODEL_PATHS]
    struct_embedder = MultiEmbedder(models=struct_models, maestro=maestro, settings=struct_settings)
    struct_embedder = struct_embedder.build(load=True)

    cpa_column = "CPA_CONTROL"

    control_df = pd.read_csv(control_txt_path, encoding="latin1")
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

    # guardamos una copia "query" ya normalizada
    control_df = add_trinomio_count(control_df)

    if "HEIGHT" in control_df.columns:
        control_df["NOM_CALLE_COMPLETO"] = control_df["NOM_CALLE_COMPLETO"] + ' ' + control_df["HEIGHT"]

    cols = ["PROVINCIA","PARTIDO","LOCALIDAD","MUNICIPIO","NOM_CALLE_COMPLETO","PXL_PREF1974"]
    queries_batch = control_df[[col for col in cols if col in control_df.columns]].to_dict(orient='records')

    # Para analizar distribuciones, sacamos filtros del searcher
    threshold_abs_list = [sc["THRESHOLD_ABS"] for sc in struct_settings["SEARCHER_CONFIG"]]
    threshold_rel_list = [sc["THRESHOLD_REL"] for sc in struct_settings["SEARCHER_CONFIG"]]

    max_ops_list       = [sc["MAX_ADDRESSES_TO_OPERATE"] for sc in struct_settings["SEARCHER_CONFIG"]]
    max_addresses_to_show_list = [settings["MAX_ADDRESSES_TO_SHOW"] for settings in struct_settings["SEARCHER_CONFIG"]]

    result_df = struct_embedder.get_results_batch(
        queries_batch=queries_batch,
        id_list=control_df[cpa_column].tolist(),   # IDs = CPA correctos (GT)
        threshold_abs_list=threshold_abs_list,
        threshold_rel_list=threshold_rel_list,
        max_addresses_to_operate_list=max_ops_list,
        max_addresses_to_show_list=max_addresses_to_show_list,           # top-1
    )

    # Índice = GT
    result_df = result_df.copy()
    result_df.index = control_df[cpa_column].values

    if "CPA" not in result_df.columns:
        raise KeyError("El DataFrame de resultados no contiene la columna 'CPA' (predicho).")

    # --- Parche: remover embeddings para que no aparezcan en outputs ---
    cols_to_drop = ["embedding_0", "embedding_1", "embedding_2"]
    result_df = result_df.drop(columns=[c for c in cols_to_drop if c in result_df.columns], errors="ignore")

    # Importante: devolvemos control_df con CPA_CONTROL para mergear por GT
    return result_df, control_df


def analyze_similarities_by_cohort(
    result_df: pd.DataFrame,
    control_df: pd.DataFrame,
    sim_cols: tuple[str, str] = ("similarity_level_1", "similarity_level_2"),
    out_dir: str | None = None,
) -> dict:
    sim1, sim2 = sim_cols
    for c in ["CPA", sim1, sim2, "ambiguous"]:
        if c not in result_df.columns:
            raise KeyError(f"Falta columna '{c}' en result_df.")

    if "CPA_CONTROL" not in control_df.columns:
        raise KeyError("Falta columna 'CPA_CONTROL' en control_df (GT).")
    if "tri_group" not in control_df.columns:
        control_df = add_trinomio_count(control_df)

    # --- armamos df_valid (GT válido) y le pegamos cohort por GT ---
    gt = pd.Series(result_df.index, name="GT", index=result_df.index)
    pred = result_df["CPA"]

    SENTINEL = "__MISSING__"
    gt_mask = gt.notna()

    gt_n = normalize_series(gt.where(gt_mask))
    pred_n = normalize_series(pred.fillna(SENTINEL))

    matches = pd.Series(False, index=result_df.index)
    matches.loc[gt_mask] = gt_n.loc[gt_mask].eq(pred_n.loc[gt_mask])

    df_valid = result_df.loc[gt_mask].copy()
    df_valid["GT"] = gt_n.loc[gt_mask].values
    df_valid["PRED"] = pred_n.loc[gt_mask].values
    df_valid["is_match"] = matches.loc[gt_mask].astype(bool)

    # map GT -> tri_group / tri_count
    tri_map = control_df.set_index("CPA_CONTROL")[["tri_group", "tri_count"]]
    df_valid = df_valid.join(tri_map, how="left")

    if df_valid["tri_group"].isna().any():
        # Si hay duplicados en CPA_CONTROL o algo raro, mejor que quede explícito
        missing = int(df_valid["tri_group"].isna().sum())
        print(f"⚠️  Aviso: {missing} filas no pudieron asociarse a tri_group (join por CPA_CONTROL).")

    # Normalizar ambiguous a booleano (por si viene como "True"/"False")
    if df_valid["ambiguous"].dtype != bool:
        df_valid["ambiguous"] = (
            df_valid["ambiguous"]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "t", "yes", "y"])
        )

    cohorts = [
        ("tri_eq_1", False),
        ("tri_eq_1", True),
        ("tri_ge_2", False),
        ("tri_ge_2", True),
    ]

    all_stats = []
    thresholds_rows = []
    summaries = []

    for tri_group, amb in cohorts:
        cohort = f"{tri_group}__amb_{int(amb)}"  # ej: tri_eq_1__amb_0

        df_c = df_valid[(df_valid["tri_group"] == tri_group) & (df_valid["ambiguous"] == amb)].copy()

        df_match = df_c[df_c["is_match"]].copy()
        df_nomatch = df_c[~df_c["is_match"]].copy()

        stats_match = build_stats_table(df_match, [sim1, sim2], "match", cohort)
        stats_nomatch = build_stats_table(df_nomatch, [sim1, sim2], "no_match", cohort)

        # Heurística culpable en no-match
        s1 = pd.to_numeric(df_nomatch[sim1], errors="coerce")
        s2 = pd.to_numeric(df_nomatch[sim2], errors="coerce")
        failed_level = np.where(s1 <= s2, 1, 2)
        df_nomatch["failed_level"] = failed_level
        df_nomatch["failed_similarity"] = np.where(df_nomatch["failed_level"] == 1, s1, s2)

        df_fail1 = df_nomatch[df_nomatch["failed_level"] == 1].copy()
        df_fail2 = df_nomatch[df_nomatch["failed_level"] == 2].copy()

        stats_fail1 = build_stats_table(df_fail1, ["failed_similarity"], "no_match_failed_lvl1", cohort)
        stats_fail2 = build_stats_table(df_fail2, ["failed_similarity"], "no_match_failed_lvl2", cohort)

        all_stats.append(stats_match)
        all_stats.append(stats_nomatch)
        all_stats.append(stats_fail1)
        all_stats.append(stats_fail2)

        # candidates
        if stats_fail1.iloc[0]["count"] > 0:
            t1 = pick_thresholds_from_stats(stats_fail1, cohort, "no_match_failed_lvl1")
        else:
            t1 = {"count": 0, "p01": np.nan, "p05": np.nan, "p10": np.nan, "p25": np.nan, "p50": np.nan}
        if stats_fail2.iloc[0]["count"] > 0:
            t2 = pick_thresholds_from_stats(stats_fail2, cohort, "no_match_failed_lvl2")
        else:
            t2 = {"count": 0, "p01": np.nan, "p05": np.nan, "p10": np.nan, "p25": np.nan, "p50": np.nan}

        thresholds_rows.append({"cohort": cohort, "group": "no_match_failed_lvl1", **t1})
        thresholds_rows.append({"cohort": cohort, "group": "no_match_failed_lvl2", **t2})

        summaries.append({
            "cohort": cohort,
            "tri_group": tri_group,
            "ambiguous": bool(amb),
            "n_valid_gt": int(df_c.shape[0]),
            "n_match": int(df_match.shape[0]),
            "n_no_match": int(df_nomatch.shape[0]),
            "no_match_failed_lvl1": int(df_fail1.shape[0]),
            "no_match_failed_lvl2": int(df_fail2.shape[0]),
            "match_rate": float(df_match.shape[0] / df_c.shape[0]) if df_c.shape[0] else np.nan
        })

        # guardamos ejemplos por cohort (útil para inspección manual)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            df_nomatch.to_csv(os.path.join(out_dir, f"{cohort}__no_match_with_failed_level.csv"), index=True)
            df_c.to_csv(os.path.join(out_dir, f"{cohort}__valid_rows_with_match_flag.csv"), index=True)

    stats_all = pd.concat(all_stats, ignore_index=True)
    thresholds = pd.DataFrame(thresholds_rows)
    summary_df = pd.DataFrame(summaries)

    if out_dir:
        stats_all.to_csv(os.path.join(out_dir, "similarity_stats_by_cohort.csv"), index=False)
        thresholds.to_csv(os.path.join(out_dir, "threshold_candidates_by_cohort.csv"), index=False)
        summary_df.to_csv(os.path.join(out_dir, "summary_by_cohort.csv"), index=False)

    return {
        "summary_by_cohort": summary_df,
        "stats_by_cohort": stats_all,
        "threshold_candidates_by_cohort": thresholds,
        "df_valid": df_valid,
    }


# --------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Analiza similitudes y propone umbrales, segmentando por trinomio PARTIDO/MUNICIPIO/LOCALIDAD")
    p.add_argument("--global-config", default=os.getenv("GLOBAL_CONFIG_PATH", "configs/global.yaml"))
    p.add_argument("--struct-config", default=os.getenv("STRUCT_CONFIG_PATH", "configs/structured.yaml"))
    p.add_argument("--control-config", default=os.getenv("CONTROL_CONFIG_PATH", "configs/control/structured.yaml"))
    p.add_argument("--control-path",  default=os.getenv("CONTROL_TXT_PATH", "datasets/control/lote_control.txt"))
    p.add_argument("--sample-n", type=int, default=int(os.getenv("CONTROL_SAMPLE_N", "5000")))
    p.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "outputs/similarity_analysis"))
    p.add_argument("--sim1-col", default=os.getenv("SIM1_COL", "similarity_level_1"))
    p.add_argument("--sim2-col", default=os.getenv("SIM2_COL", "similarity_level_2"))
    p.add_argument("--print-top", type=int, default=int(os.getenv("PRINT_TOP", "30")))
    return p.parse_args()


def main():
    args = parse_args()

    result_df, control_df = build_result_df(
        global_config_path=args.global_config,
        struct_config_path=args.struct_config,
        control_config_path=args.control_config,
        control_txt_path=args.control_path,
        control_sample_n=args.sample_n,
        random_state=args.random_state,
    )

    out = analyze_similarities_by_cohort(
        result_df=result_df,
        control_df=control_df,
        sim_cols=(args.sim1_col, args.sim2_col),
        out_dir=args.out_dir
    )

    print("\n== SUMMARY BY COHORT ==")
    print(out["summary_by_cohort"].to_string(index=False))

    print("\n== THRESHOLD CANDIDATES BY COHORT ==")
    print(out["threshold_candidates_by_cohort"].to_string(index=False))

    print("\n== STATS (primeras filas) ==")
    print(out["stats_by_cohort"].head(args.print_top).to_string(index=False))

    print(f"\nGuardados CSV en: {args.out_dir}")


if __name__ == "__main__":
    main()