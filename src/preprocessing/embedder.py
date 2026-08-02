#%%
import numpy as np
from streamlit import columns
import yaml
import sys
import os
import pandas as pd
import json
import re
from tqdm import tqdm
from logs.logger import Logger
from preprocessing.utils import split_query_and_height, safe_concat, split_query_and_height_with_matcher, preprocess_street_with_matcher, cp_candidates
from preprocessing.cts import CP_COLUMN, CPA_COLUMN, SIN_NOMBRE, LOCALIDAD_COLUMN, NOM_APELLIDO_COLUMN, PUNTO_CARDINAL_COLUMN, TIPO_CALLE_COLUMN
from preprocessing.tipo_calle_aliases import TIPO_CALLE_ALIASES
from pathlib import Path

class MultiEmbedder:
    def __init__(self, models, maestro, settings, maestro_solo_localidad=None, maestro_sin_calle=None, base_url_matcher=None, path_url_matcher=None, auth_token_matcher=None, **kwargs):
        # self.embedders = [
        #     Embedder(model, maestro, config_path, **kwargs)
        #     for model in models
        # ]
        self.models = models
        self.maestro = maestro  
        self.maestro_sin_calle = maestro_sin_calle
        self.maestro_solo_localidad = maestro_solo_localidad
        self.n = len(models)
        self.logger = Logger(name=__class__.__name__, settings=settings).get()
        # config
        self.settings = settings
        self.settings.update(kwargs)
        self.base_url_matcher = base_url_matcher
        self.path_url_matcher = path_url_matcher
        self.auth_token_matcher = auth_token_matcher
        self.tipo_calle_alias_map = {
            self._normalize_text_token(k): [self._normalize_text_token(x) for x in v]
            for k, v in TIPO_CALLE_ALIASES.items()
        }
        self.max_tipo_calle_alias_tokens = max(
            len(alias.split()) for alias in self.tipo_calle_alias_map
        ) if self.tipo_calle_alias_map else 1
    
    def _extract_numbers_from_query(self, query: str) -> list[str]:
        return re.findall(r"\d+", str(query))



    def _remove_leading_initials_from_apellido_query(self, query: str) -> str:
        """
        For exact NOM_APELLIDO matching, ignore leading one-letter words
        such as name initials: "J PAGANO" -> "PAGANO".
        """
        tokens = str(query).strip().split()

        while tokens and re.fullmatch(r"[^\W\d_]\.?", tokens[0], flags=re.UNICODE):
            tokens = tokens[1:]

        return " ".join(tokens).strip()

    def _build_nom_apellido_exact_query(self, full_query: str, query_has_height: bool) -> tuple:
        """
        Build the value to search exactly in NOM_APELLIDO_COLUMN.

        If the street query has a height, remove it before checking by surname.
        Then, if the remaining query starts with a one-letter word, ignore that
        token because it is probably the initial of the person's first name.
        """
        if query_has_height:
            query_without_height, height = split_query_and_height(full_query)
            effective_query_has_height = height is not None
        else:
            query_without_height = full_query
            height = None
            effective_query_has_height = False

        apellido_query = self._remove_leading_initials_from_apellido_query(
            query_without_height
        )

        return apellido_query, height, effective_query_has_height

    def _get_exact_nom_apellido_similarities(
        self,
        maestro,
        full_query: str,
        level: int,
        query_has_height: bool,
    ):
        """
        Try to resolve level 2 by exact NOM_APELLIDO_COLUMN match before embeddings.

        Returns:
            tuple(maestro_filtrado, similarities) if there is an exact match,
            otherwise tuple(None, None).
        """
        apellido_query, height, effective_query_has_height = self._build_nom_apellido_exact_query(
            full_query=full_query,
            query_has_height=query_has_height,
        )

        if apellido_query == "":
            return None, None

        apellido_values = list(dict.fromkeys([apellido_query, apellido_query.upper()]))

        maestro_apellido = maestro.filter_by_column(
            NOM_APELLIDO_COLUMN,
            apellido_values,
            level=level,
        )

        if maestro_apellido.empty():
            return None, None

        candidate_idx = pd.Index(
            maestro_apellido.maestro_df.index.get_level_values(level).unique()
        )

        similarities = pd.DataFrame(
            {"similarity": 1.0},
            index=candidate_idx,
        )

        if effective_query_has_height:
            similarities["HEIGHT"] = height
            similarities = self._filter_similarities_by_height(
                maestro_apellido,
                similarities,
                len(similarities),
                height,
                level,
            )
            if similarities.empty:
                self.logger.warning(
                    "No results found after exact NOM_APELLIDO match and height filtering."
                )
                return maestro_apellido, similarities
        else:
            similarities["HEIGHT"] = np.nan

        self.logger.info(
            f"Exact NOM_APELLIDO match found for query '{apellido_query}'."
        )

        return maestro_apellido, similarities

    def _filter_similarities_by_exact_query_numbers(
        self,
        maestro,
        similarities: pd.DataFrame,
        query_numbers: list[str],
        level: int,
    ) -> pd.DataFrame:
        if not query_numbers or similarities.empty:
            return similarities

        query_col = f"query_{level}"

        # Filtramos el maestro solo a los ids candidatos
        maestro_candidates = maestro.filter_by_level(
            similarities.index.tolist(),
            level=level,
        )

        df_candidates = maestro_candidates.get_columns([query_col])

        # Nos quedamos con una query representativa por id del nivel correspondiente
        queries_by_hash = (
            df_candidates
            .groupby(level=level, sort=False)[query_col]
            .first()
            .astype(str)
        )

        candidate_queries = (
            queries_by_hash
            .reindex(similarities.index)
            .fillna("")
            .astype(str)
        )

        mask = pd.Series(True, index=similarities.index)

        for number in query_numbers:
            pattern = rf"(?<!\d){re.escape(number)}(?!\d)"
            mask &= candidate_queries.str.contains(pattern, regex=True, na=False)

        return similarities.loc[mask]
    
    def _filter_similarities_by_height(self, maestro, similarities, top, height, level):
        selected_idx = []
        selected_ids_hash = []

        for idx in similarities.index[:top]:              
            id_hash, cpa, cod_d, cod_h = maestro.get_cod_desde_hasta(idx, height, level)
            if cod_d is None or cod_h is None:
                continue
            selected_idx.append(idx)
            selected_ids_hash.append(id_hash)

        if selected_idx:
            similarities = similarities.loc[selected_idx].copy()
            similarities["id_hash"] = pd.DataFrame(
                selected_ids_hash, index=selected_idx
            )
        else:
            similarities = similarities.iloc[0:0].copy()
        return similarities

    def generate_embeddings(self):
        self.logger.info("Generating embeddings for Maestro queries.")
        for level in range(self.n):
            unique_queries = self.maestro.get_unique_queries_df(level,[f"query_{level}"])[f"query_{level}"]
            embeddings = self.models[level].encode(
                unique_queries.tolist(),
                precision = np.float16
            )
            hash_emb_dict = dict(zip(unique_queries.index, embeddings))
            self.maestro.map(level, hash_emb_dict, f"embedding_{level}")
            if self.settings["SAVE_EMBEDDINGS"]:
                save_path = self.settings["SEARCHER_CONFIG"][level]["SAVE_EMBEDDINGS_PATH"]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, hash_emb_dict)

        return self

    def generate_embeddings_solo_localidad(self):
        self.logger.info("Generating embeddings for Maestro queries.")
        for level in range(self.n):
            unique_queries = self.maestro_solo_localidad.get_unique_queries_df(level,[f"query_{level}"])[f"query_{level}"]
            embeddings = self.models[level].encode(
                unique_queries.tolist(),
                precision = np.float16
            )
            hash_emb_dict = dict(zip(unique_queries.index, embeddings))
            self.maestro_solo_localidad.map(level, hash_emb_dict, f"embedding_{level}")
            if self.settings["SAVE_EMBEDDINGS"]:
                save_path = self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][level]["SAVE_EMBEDDINGS_PATH"]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, hash_emb_dict)

        return self

    def load_embeddings(self):
        for level in range(self.n):
            # Always add ../ prefix to load path
            load_path = self.settings["SEARCHER_CONFIG"][level]["LOAD_EMBEDDINGS_PATH"]
            self.logger.info("Loading embeddings from: {}".format(load_path))
            hash_emb_dict = np.load(load_path, allow_pickle=True).item()
            if len(hash_emb_dict) != len(self.maestro.get_unique_queries_df(level,[f"query_{level}"])):
                self.logger.error("Loaded embeddings do not match the maestro size.")
                return None
            self.maestro.map(level, hash_emb_dict, f"embedding_{level}")
        return self

    def load_embeddings_solo_localidad(self):
        for level in range(self.n):
            # Always add ../ prefix to load path
            load_path = self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][level]["LOAD_EMBEDDINGS_PATH"]
            self.logger.info("Loading embeddings from: {}".format(load_path))
            hash_emb_dict = np.load(load_path, allow_pickle=True).item()
            if len(hash_emb_dict) != len(self.maestro_solo_localidad.get_unique_queries_df(level,[f"query_{level}"])):
                self.logger.error("Loaded embeddings do not match the maestro size.")
                return None
            self.maestro_solo_localidad.map(level, hash_emb_dict, f"embedding_{level}")
        return self

    def load_prefijo_to_pxl_map(self, prefijo_to_pxl_path: str, pxl_set_path: str) -> dict:
        self.logger.info("Loading prefijo_to_pxl_map")
        self.prefijo_to_pxl_map = json.loads(Path(prefijo_to_pxl_path).read_text(encoding="utf-8"))
        self.pxl_set_map = json.loads(Path(pxl_set_path).read_text(encoding="utf-8"))

    # def get_similarities(self, queries, level):
    #     # Encode all queries at once (batch)
    #     query_embeddings = self.models[level].encode(queries)  # shape (q, d)

    #     keys, embeddings = zip(*self.hash_emb_dicts[level].items())
    #     embeddings_matrix = np.stack(embeddings)  # shape (n, d)

    #     # Compute similarity matrix (q x n)
    #     similarity_matrix = np.dot(embeddings_matrix, query_embeddings.T)  # shape (n, q)

    #     # Build output
    #     df = pd.DataFrame(similarity_matrix, index=keys, columns=range(len(queries)))

    #     return df

    def get_similarities(self, maestro, queries, level):
        """
        Compute cosine similarities between queries and maestro embeddings at a specific level.

        Args:
            queries (list of str): Queries to compare.
            level (int or str): Level of embeddings to use (corresponding to 'embedding_{level}' column).

        Returns:
            pd.DataFrame: Similarities matrix (rows: maestro entries, cols: input queries)
        """
        # Encode queries
        query_embeddings = self.models[level].encode(queries)  # shape (q, d)

        # Filtrar los embeddings del nivel correspondiente
        emb_col = f"embedding_{level}"
        maestro_embeddings = maestro.get_unique_queries_df(level, [emb_col])[emb_col]

        keys = maestro_embeddings.index

        if maestro.maestro_embeddings_matrix is not None:
            maestro_embeddings_matrix = maestro.maestro_embeddings_matrix
        else:
            maestro_embeddings_matrix = np.stack(maestro_embeddings.values)  # shape (n, d)

        # Similaridad (n, q)
        similarity_matrix = np.dot(maestro_embeddings_matrix, query_embeddings.T)

        # Devolver DataFrame: filas = maestro entries, columnas = queries
        df = pd.DataFrame(similarity_matrix, index=keys, columns=range(len(queries)))
        return df


    def get_similarities_filtered(
            self, 
            maestro, 
            full_query, 
            level, 
            max_addresses_to_operate, 
            max_addresses_to_show, 
            query_has_height,
            threshold_abs=0,
            threshold_rel=0,
            sin_calle_threshold=0
        ):
        # Split the query into the main query and height
        if query_has_height:
            query_matcher, height_matcher = split_query_and_height_with_matcher(full_query, self.base_url_matcher, self.path_url_matcher, self.auth_token_matcher)
            if height_matcher is not None:
                query = query_matcher
                height = height_matcher
            else:
                self.logger.info("Matcher could not extract height, falling back to local extraction.")
                query, height = split_query_and_height(full_query)
            if height is None:
                self.logger.info("Height could not be extracted from the query.")
                query_has_height = False
        else:
            query = full_query
            height = None
        s = self.get_similarities(maestro, [query], level).iloc[:, 0]  # Series (n,)
        # Caso capital federal
        if level == 1 and self.settings["SEARCHER_CONFIG"][0]["MAX_ADDRESSES_TO_SHOW"]==1 and maestro.get_first_value("PROVINCIA")=="CAPITAL FEDERAL":
            s.iloc[:] = 1 
        # Delete linderas duplicadas
        if maestro.query_hash_to_hash_ori_map.get(level) is not None:
            hash_lindera_to_hash_ori_map = maestro.query_hash_to_hash_ori_map[level]
            idx = s.groupby(hash_lindera_to_hash_ori_map, dropna=False).idxmax()   # índices de s con máxima similitud por grupo
            s = s.loc[idx]

            # s = s[~s.index.isin(hash_lindera_to_hash_ori_df.index[hash_lindera_to_hash_ori_df.index.duplicated(keep='first')])]
        if height is not None:
            # Penalize 0-0 streets
            mask = s.index.isin(maestro.no_height_street_hashes)
            s.loc[mask] = (s.loc[mask] ** 2).clip(upper=0.99)
        # Get top k similarities
        k = min(max_addresses_to_operate, len(s))
        if k <= 0:
            similarities = pd.DataFrame(columns=["similarity"])
        else:
            vals = s.to_numpy()
            idx = np.argpartition(vals, -k)[-k:]               # O(n) aprox
            idx = idx[np.argsort(vals[idx])[::-1]]             # ordena SOLO esos k
            top_index = s.index.take(idx)                      # mapea posiciones → labels
            similarities = pd.DataFrame({"similarity": vals[idx]}, index=top_index)
        # Apply thresholds
        if threshold_abs > 0:
            similarities = similarities[similarities["similarity"] >= threshold_abs]
            if similarities.empty:
                self.logger.warning("No results found after applying absolute threshold.")
                return pd.DataFrame()
        if threshold_rel > 0:
            max_similarity = similarities["similarity"].iloc[0]
            similarities = similarities[max_similarity - similarities["similarity"] <= 1 - threshold_rel]
        if similarities.empty:
            self.logger.warning("No results found after applying relative threshold.")
            return pd.DataFrame()
        # Nuevo filtro: si la query sin altura tiene números,
        # esos números deben aparecer tal cual en la query del candidato.
        query_numbers = self._extract_numbers_from_query(query)

        if query_numbers:
            similarities = self._filter_similarities_by_exact_query_numbers(
                maestro=maestro,
                similarities=similarities,
                query_numbers=query_numbers,
                level=level,
            )

            if similarities.empty:
                self.logger.warning("No results found after filtering by exact query numbers.")
                return pd.DataFrame()
        if query_has_height:
            similarities["HEIGHT"] = height
            similarities = self._filter_similarities_by_height(
            maestro, similarities, max_addresses_to_operate, height, level
            )
            if similarities.empty:
                self.logger.warning("No results found after filtering by height.")
                return pd.DataFrame()
        else:
            similarities["HEIGHT"] = np.nan

        return similarities
    
    def generate_user_query(self, cp=False, args=None, kwargs=None):
        query = []
        if args:
            if isinstance(args[0], list):
                args = args[0]  # Unpack if args is a list of lists
            i_from=0
            i_since=0
            for i in range(len(self.settings["SEARCHER_CONFIG"])):
                i_since+=len(self.settings["SEARCHER_CONFIG"][i]["QUERY_COLUMNS"])
                query_i_list = args[i_from:i_since]
                query_i = safe_concat(*query_i_list)
                if i==len(self.settings["SEARCHER_CONFIG"])-1:
                    # Preprocess street with matcher
                    query_i_matcher = preprocess_street_with_matcher(query_i, self.base_url_matcher, self.path_url_matcher, self.auth_token_matcher)
                    if query_i_matcher != "":
                        query.append(query_i_matcher)
                    else:
                        query.append(query_i)
                        self.logger.warning("Could not preprocess street with matcher, using original query.")
                else:
                    query.append(query_i)
                i_from = i_since
        elif kwargs:
            for i, config in enumerate(self.settings["SEARCHER_CONFIG"]):
                query_i = safe_concat(*[kwargs[col] for col in config["QUERY_COLUMNS"] if col in kwargs])
                if i == len(self.settings["SEARCHER_CONFIG"]) - 1:
                    # Preprocess street with matcher
                    query_i_matcher = preprocess_street_with_matcher(query_i, self.base_url_matcher, self.path_url_matcher, self.auth_token_matcher)
                    if query_i_matcher != "":
                        query.append(query_i_matcher)
                    else:
                        query.append(query_i)
                        self.logger.warning("Could not preprocess street with matcher, using original query.")
                else:
                    query.append(query_i)
        if cp:
            if args:
                query[1] = args[-2]
            elif kwargs:
                query[1] = kwargs.get("PXL_PREF1974", "")
        return query
    
    def _count_full_fields(self, fields: list, query: dict) -> int:
        res = 0
        for field in fields:
            if field in query.keys():
                if query[field]!='':
                    res+=1
        return res
    
    def _solo_completo_localidad(self, fields, query: dict) -> bool:
        if isinstance(query, dict):
            if LOCALIDAD_COLUMN in query.keys():
                return query[LOCALIDAD_COLUMN] != '' and self._count_full_fields(fields, query) == 1
        return False

    def _encontro_localidad(self, results_solo_localidad):
        if isinstance(results_solo_localidad, str):
            if results_solo_localidad=="NOT_LOCALIDAD":
                return False
        else:
            return True
    
    def _has_real_height_mask_for_group(self, group: pd.DataFrame) -> pd.Series:
        """
        True if the candidate street supports height.
        A street is considered to have no height only when both COD_DESDE and
        COD_HASTA are 0.
        """
        if "COD_DESDE" not in group.columns or "COD_HASTA" not in group.columns:
            return pd.Series(False, index=group.index)

        cod_desde = pd.to_numeric(group["COD_DESDE"], errors="coerce").fillna(0)
        cod_hasta = pd.to_numeric(group["COD_HASTA"], errors="coerce").fillna(0)

        return ~((cod_desde == 0) & (cod_hasta == 0))

    def _extract_cardinal_from_street_query(self, *args, **kwargs):
        """
        Detect cardinal point in the street query.
        Returns one of: SUR, NORTE, ESTE, OESTE, or None.
        """
        try:
            query = self.generate_user_query(cp=False, args=args, kwargs=kwargs)
            if not query:
                return None
            street_query = str(query[-1]).strip().lower()
        except Exception:
            return None

        tokens = re.findall(r"\b\w+\b", street_query, flags=re.UNICODE)

        token_map = {
            "s": "SUR",
            "sur": "SUR",
            "n": "NORTE",
            "norte": "NORTE",
            "e": "ESTE",
            "este": "ESTE",
            "o": "OESTE",
            "oe": "OESTE",
            "oeste": "OESTE",
        }

        detected = [token_map[t] for t in tokens if t in token_map]

        if not detected:
            return None

        # Si aparecen varios distintos, mejor no resolver por este criterio
        detected_unique = list(dict.fromkeys(detected))
        if len(detected_unique) == 1:
            return detected_unique[0]

        return None
    
    def _normalize_text_token(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().upper())

    def _extract_tipo_calle_from_street_query(self, *args, **kwargs) -> set[str]:
        """
        Detect street type from the beginning of the street query.
        Returns a set of canonical street types, because one alias may map
        to more than one canonical type.

        If no explicit street type is detected, default to {"CALLE"}.
        """
        if not self.tipo_calle_alias_map:
            return {"CALLE"}

        try:
            query = self.generate_user_query(cp=False, args=args, kwargs=kwargs)
            if not query:
                return {"CALLE"}
            street_query = self._normalize_text_token(query[-1])
        except Exception:
            return {"CALLE"}

        tokens = re.findall(r"[A-ZÁÉÍÓÚÜÑ0-9]+", street_query, flags=re.UNICODE)
        if not tokens:
            return {"CALLE"}

        max_n = min(self.max_tipo_calle_alias_tokens, len(tokens))

        for n in range(max_n, 0, -1):
            candidate = " ".join(tokens[:n])
            if candidate in self.tipo_calle_alias_map:
                return set(self.tipo_calle_alias_map[candidate])

        return {"CALLE"}
    
    def _mark_ambiguity(self, thr: float, df: pd.DataFrame) -> pd.DataFrame:
        if thr <= 0 or df is None or df.empty or "result_similarity" not in df.columns:
            if df is None:
                return df
            df = df.copy()
            df["ambiguous"] = False
            return df

        if len(df) < 2:
            df = df.copy()
            df["ambiguous"] = False
            return df

        df = df.copy()
        df["ambiguous"] = False

        s = pd.to_numeric(df["result_similarity"], errors="coerce")

        diff_next = (s - s.shift(-1)).abs()
        diff_prev = (s - s.shift(1)).abs()
        amb = ((diff_next < thr) | (diff_prev < thr)).fillna(False)

        df.loc[amb, "ambiguous"] = True
        return df
    
    def _resolve_ambiguity(
        self,
        *args,
        df: pd.DataFrame,
        use_cp=True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Resolve ambiguous result groups using ordered criteria.

        Priority:
        1) Cardinal point in street query.
        2) Street type in street query.
        3) Original locality vs bordering locality.
        4) Real street height.
        5) Original street name vs synonym.
        6) Postal code (only if use_cp=True).
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        if "ambiguous" not in df.columns or not df["ambiguous"].any():
            return df

        rows_to_drop = []

        query_cardinal = self._extract_cardinal_from_street_query(*args, **kwargs)
        query_street_types = self._extract_tipo_calle_from_street_query(*args, **kwargs)

        cp_cands = set()
        if use_cp and CP_COLUMN in df.columns:
            try:
                query_cp = self.generate_user_query(cp=True, args=args, kwargs=kwargs)
                user_cp = query_cp[1]
                if user_cp:
                    cp_cands = set(
                        cp_candidates(user_cp, self.prefijo_to_pxl_map, self.pxl_set_map) or []
                    )
            except Exception:
                cp_cands = set()

        ambiguous_mask = df["ambiguous"].fillna(False).tolist()
        idx_list = list(df.index)

        start = None
        for pos, is_amb in enumerate(ambiguous_mask + [False]):
            if is_amb and start is None:
                start = pos

            elif not is_amb and start is not None:
                end = pos
                group_positions = list(range(start, end))
                group_idx = [idx_list[p] for p in group_positions]
                group = df.loc[group_idx]

                resolved = False

                # --- Criterio 1: punto cardinal ---
                if (not resolved) and query_cardinal and (PUNTO_CARDINAL_COLUMN in group.columns):
                    cardinal_match_mask = (
                        group[PUNTO_CARDINAL_COLUMN].astype(str).str.upper().str.strip()
                        == query_cardinal
                    )

                    if cardinal_match_mask.sum() == 1:
                        winner_idx = group[cardinal_match_mask].index[0]
                        losers_idx = [i for i in group.index if i != winner_idx]
                        rows_to_drop.extend(losers_idx)
                        df.loc[winner_idx, "ambiguous"] = False
                        resolved = True

                # --- Criterio 2: tipo de calle ---
                if (not resolved) and query_street_types and (TIPO_CALLE_COLUMN in group.columns):
                    street_type_match_mask = (
                        group[TIPO_CALLE_COLUMN].astype(str).str.upper().str.strip()
                        .isin(query_street_types)
                    )

                    if street_type_match_mask.sum() == 1:
                        winner_idx = group[street_type_match_mask].index[0]
                        losers_idx = [i for i in group.index if i != winner_idx]
                        rows_to_drop.extend(losers_idx)
                        df.loc[winner_idx, "ambiguous"] = False
                        resolved = True

                # --- Criterio 3: localidad original vs lindera ---
                if (not resolved) and LOCALIDAD_COLUMN in group.columns and "localidad_original" in group.columns:
                    localidad_match_mask = (
                        group[LOCALIDAD_COLUMN].astype(str).str.upper().str.strip()
                        ==
                        group["localidad_original"].astype(str).str.upper().str.strip()
                    )

                    if localidad_match_mask.sum() == 1:
                        winner_idx = group[localidad_match_mask].index[0]
                        losers_idx = [i for i in group.index if i != winner_idx]
                        rows_to_drop.extend(losers_idx)
                        df.loc[winner_idx, "ambiguous"] = False
                        resolved = True

                # --- Criterio 4: altura real ---
                if (not resolved) and ("HEIGHT" in group.columns):
                    user_entered_height = group["HEIGHT"].notna().any()

                    if user_entered_height:
                        has_real_height_mask = self._has_real_height_mask_for_group(group)

                        if has_real_height_mask.sum() == 1:
                            winner_idx = has_real_height_mask[has_real_height_mask].index[0]
                            losers_idx = [i for i in group.index if i != winner_idx]
                            rows_to_drop.extend(losers_idx)
                            df.loc[winner_idx, "ambiguous"] = False
                            resolved = True

                # --- Criterio 5: nombre original vs sinónimo ---
                if (not resolved) and (SIN_NOMBRE in group.columns):
                    original_name_mask = (
                        group[SIN_NOMBRE].astype(str).str.upper().str.strip() == "C"
                    )

                    if original_name_mask.sum() == 1:
                        winner_idx = group[original_name_mask].index[0]
                        losers_idx = [i for i in group.index if i != winner_idx]
                        rows_to_drop.extend(losers_idx)
                        df.loc[winner_idx, "ambiguous"] = False
                        resolved = True

                # --- Criterio 6: CP ---
                if (not resolved) and use_cp and cp_cands and CP_COLUMN in group.columns:
                    cp_match_mask = group[CP_COLUMN].astype(str).isin(cp_cands)

                    if cp_match_mask.sum() == 1:
                        winner_idx = group[cp_match_mask].index[0]
                        losers_idx = [i for i in group.index if i != winner_idx]
                        rows_to_drop.extend(losers_idx)
                        df.loc[winner_idx, "ambiguous"] = False

                start = None

        if rows_to_drop:
            df = df.drop(index=rows_to_drop)

        return df


    def _get_results_core(
            self, 
            query,
            threshold_abs_list=None, 
            threshold_rel_list=None, 
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            result_similarity_levels=None,
            solo_localidad=False,
            cp=False
        ):
        """Get results for structured queries."""
        if cp and query[1] == "":
            self.logger.warning("No CP provided for CP filtering.")
            return pd.DataFrame()
        self.logger.info(f"Generated query: {query}")
        if len(query) != self.n:
            self.logger.error("Number of queries does not match number of levels.")
            return None
        if result_similarity_levels is None:
            result_similarity_levels = self.settings["RESULT_SIMILARITY_LEVELS"]
        if solo_localidad:
            maestro = self.maestro_solo_localidad
        else: 
            maestro = self.maestro
        for level, (query_i, settings) in enumerate(zip(query, self.settings["SEARCHER_CONFIG"])):
            if threshold_abs_list is not None:
                threshold_abs = threshold_abs_list[level]
            else:
                threshold_abs = settings["THRESHOLD_ABS"]
            if threshold_rel_list is not None:
                threshold_rel = threshold_rel_list[level]
            else:
                threshold_rel = settings["THRESHOLD_REL"]
            if max_addresses_to_operate_list is not None:
                max_addresses_to_operate = max_addresses_to_operate_list[level]
            else:
                max_addresses_to_operate = settings["MAX_ADDRESSES_TO_OPERATE"]
            if max_addresses_to_show_list is not None:
                max_addresses_to_show = max_addresses_to_show_list[level]
            else:
                max_addresses_to_show = settings["MAX_ADDRESSES_TO_SHOW"]

            self.logger.info(f"Processing level {level} with query: {query_i}")
            self.logger.info(f"Thresholds - Abs: {threshold_abs}, Rel: {threshold_rel}, Max to Operate: {max_addresses_to_operate}, Max to Show: {max_addresses_to_show}")
            similarities = None

            if cp and level == 1:
                similarities = pd.DataFrame(columns=["similarity"])
                maestro = maestro.filter_by_column(CP_COLUMN, [query_i], level=level)
                if maestro.empty():
                    return pd.DataFrame()

            elif level == 2:
                maestro_apellido, similarities = self._get_exact_nom_apellido_similarities(
                    maestro=maestro,
                    full_query=query_i,
                    level=level,
                    query_has_height=settings["HAS_HEIGHT"],
                )

                if maestro_apellido is not None:
                    if similarities.empty:
                        return pd.DataFrame()
                    maestro = maestro_apellido

            # Fallback común: aplica a los niveles normales y a level == 2
            # cuando no hubo match exacto por NOM_APELLIDO_COLUMN.
            if similarities is None:
                similarities = self.get_similarities_filtered(
                    maestro=maestro,
                    full_query=query_i,
                    level=level,
                    max_addresses_to_operate=max_addresses_to_operate,
                    max_addresses_to_show=max_addresses_to_show,
                    query_has_height=settings["HAS_HEIGHT"],
                    threshold_abs=threshold_abs,
                    threshold_rel=threshold_rel
                )
                if similarities.empty:
                    if solo_localidad and level == 1:
                        return "NOT_LOCALIDAD" # No encontro localidad
                    else:
                        return pd.DataFrame()
                maestro = maestro.filter_by_level(similarities.index, level=level)

            maestro.map(level, similarities["similarity"], f"similarity_level_{level}")
        similarities_columns = similarities.columns.tolist()
        maestro_columns = maestro.get_column_names()
        similarities_columns_to_return = list(set(similarities_columns) & set(self.settings["COLUMNS_TO_RETURN"]))
        maestro_columns_to_return = list(set(maestro_columns) & set(self.settings["COLUMNS_TO_RETURN"]) - set(similarities_columns_to_return))
        maestro_columns_to_return += [f"similarity_level_{i}" for i in range(self.n)]
        # Caso tiene altura
        if "id_hash" in similarities.columns:
            results = pd.merge(
                similarities,
                maestro.maestro_df,
                on="id_hash"
            ).drop(columns="similarity")  # Drop the last similarity column as it is redundant
        # Caso sin altura
        else:
            results = pd.merge(
                similarities,
                maestro.get_unique_queries_df(level, maestro_columns_to_return),
                left_index=True,
                right_index=True
            ).drop(columns="similarity")  # Drop the last similarity column as it is redundant
        # Promedio las similaridades de los niveles indicados
        results["result_similarity"] = results[[f"similarity_level_{i}" for i in result_similarity_levels]].mean(axis=1)
        if SIN_NOMBRE in results.columns:
            mask = results["SIN_NOMBRE"] == "S"
            results.loc[mask, "result_similarity"] -= 0.01
        results = results.sort_values(by="result_similarity", ascending=False)
        results = results.drop_duplicates(subset=CPA_COLUMN, keep="first")
        results = results.iloc[:min(max_addresses_to_show, len(similarities))]
        return results
    
    def _get_results_with_cp(
            self, 
            *args, 
            threshold_abs_list=None, 
            threshold_rel_list=None, 
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            result_similarity_levels=None,
            **kwargs
        ):
        if max_addresses_to_show_list is not None:
            max_addresses_to_show = max_addresses_to_show_list[-1]
        else:
            max_addresses_to_show = self.settings["SEARCHER_CONFIG"][-1]["MAX_ADDRESSES_TO_SHOW"]
        query = self.generate_user_query(cp=True, args=args, kwargs=kwargs)
        user_cp = query[1]
        cands = cp_candidates(user_cp, self.prefijo_to_pxl_map, self.pxl_set_map)
        if not cands:
            return pd.DataFrame()

        res_list = []
        for cp_cand in cands:
            res = self._get_results_core(
                query=[query[0], cp_cand, query[2]],
                threshold_abs_list=threshold_abs_list,
                threshold_rel_list=threshold_rel_list,
                max_addresses_to_operate_list=max_addresses_to_operate_list,
                max_addresses_to_show_list=max_addresses_to_show_list,
                result_similarity_levels=result_similarity_levels,
                cp=True,
            )
            if res is not None and not res.empty:
                res_list.append(res)

        if not res_list:
            return pd.DataFrame()

        results = pd.concat(res_list, ignore_index=True)

        # dedupe + sort + topK
        results = results.drop_duplicates(subset=CPA_COLUMN, keep="first")
        results = results.sort_values(by="result_similarity", ascending=False)
        results = results.head(max_addresses_to_show)

        return results
    
    def _get_results_without_cp(
            self, 
            *args, 
            threshold_abs_list=None, 
            threshold_rel_list=None, 
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            result_similarity_levels=None,
            solo_localidad=False,
            **kwargs
        ):
        query = self.generate_user_query(cp=False, args=args, kwargs=kwargs)
        return self._get_results_core(
            query=query, 
            threshold_abs_list=threshold_abs_list, 
            threshold_rel_list=threshold_rel_list, 
            max_addresses_to_operate_list=max_addresses_to_operate_list, 
            max_addresses_to_show_list=max_addresses_to_show_list,
            result_similarity_levels=result_similarity_levels,
            cp=False,
            solo_localidad=solo_localidad
        )
    
    def _get_results_solo_localidad(
            self, 
            *args, 
            threshold_abs_list=None, 
            threshold_rel_list=None, 
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            result_similarity_levels=None,
            **kwargs
    ):
        return self._get_results_without_cp(
                *args, 
                threshold_abs_list=threshold_abs_list, 
                threshold_rel_list=threshold_rel_list, 
                max_addresses_to_operate_list=max_addresses_to_operate_list, 
                max_addresses_to_show_list=max_addresses_to_show_list, 
                result_similarity_levels=result_similarity_levels,
                solo_localidad=True,
                **kwargs
            )
    
    def build(self, load=True):
        if load:
            if self.maestro_solo_localidad is not None:
                self.load_embeddings_solo_localidad()
                self.maestro_solo_localidad.load_unique_queries_groups(solo_localidad=True)
                self.maestro_solo_localidad.load_first_level_embeddings_matrix(solo_localidad=True)
            self.load_embeddings()
            self.maestro.load_unique_queries_groups()
            self.maestro.load_first_level_embeddings_matrix()            
        else:
            if self.maestro_solo_localidad is not None:
                self.generate_embeddings_solo_localidad()
                self.maestro_solo_localidad.generate_unique_queries_groups(solo_localidad=True)
                self.maestro_solo_localidad.generate_first_level_embeddings_matrix(solo_localidad=True)
            self.generate_embeddings()
            self.maestro.generate_unique_queries_groups()
            self.maestro.generate_first_level_embeddings_matrix()
        if self.settings.get("LOAD_PREFIJO_TO_PXL_MAP", False): 
            self.load_prefijo_to_pxl_map(self.settings["PREFIJO_TO_PXL_MAP_PATH"], self.settings["PXL_SET_PATH"])

        return self
        
    def get_results(
            self, 
            *args, 
            threshold_abs_list=None, 
            mlp_threshold_abs_relaxed=None,
            solo_localidad_threshold_abs=None,
            threshold_rel_list=None, 
            ambiguous_threshold=None,
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            result_similarity_levels=None,
            use_cp=True,
            **kwargs
        ):
        """Get results for structured queries, with or without CP filtering."""
        if max_addresses_to_show_list is not None:
            max_addresses_to_show = max_addresses_to_show_list[-1]
        else:
            max_addresses_to_show = self.settings["SEARCHER_CONFIG"][-1]["MAX_ADDRESSES_TO_SHOW"]
        if ambiguous_threshold is None:
            ambiguous_threshold = self.settings["AMBIGUOUS_THRESHOLD"]
        if self.settings.get("USE_MLP_ABS_THRESHOLD_RELAXED", False):
            if mlp_threshold_abs_relaxed is None:
                mlp_threshold_abs_relaxed = self.settings["SEARCHER_CONFIG"][1]["THRESHOLD_ABS_RELAXED"]
            cant_mlp_full_fields = self._count_full_fields(
                fields=["MUNICIPIO","PARTIDO","LOCALIDAD"],
                query=kwargs
                )
            if cant_mlp_full_fields <= 1:
                threshold_abs_list[1] = mlp_threshold_abs_relaxed
        if self._solo_completo_localidad(fields=["MUNICIPIO","PARTIDO","LOCALIDAD"], query=kwargs):
            if solo_localidad_threshold_abs is None:
                solo_localidad_threshold_abs = self.settings["SOLO_LOCALIDAD_THRESHOLD_ABS"]
            if solo_localidad_threshold_abs is not None:
                threshold_abs_list_solo_localidad = [threshold_abs_list[0], solo_localidad_threshold_abs, threshold_abs_list[2]]
            else:
                threshold_abs_list_solo_localidad = threshold_abs_list
            results_solo_localidad = self._get_results_solo_localidad(
                *args, 
                threshold_abs_list=threshold_abs_list_solo_localidad, 
                threshold_rel_list=threshold_rel_list, 
                max_addresses_to_operate_list=max_addresses_to_operate_list, 
                max_addresses_to_show_list=max_addresses_to_show_list,
                result_similarity_levels=result_similarity_levels,
                **kwargs
            )
            if self._encontro_localidad(results_solo_localidad):
                results_solo_localidad = self._mark_ambiguity(ambiguous_threshold, results_solo_localidad)
                results_solo_localidad = self._resolve_ambiguity(
                        *args,
                        df = results_solo_localidad,
                        use_cp=False,
                        **kwargs
                    )
                return results_solo_localidad
        if use_cp:
            results_with_cp = self._get_results_with_cp(
                *args, 
                threshold_abs_list=threshold_abs_list, 
                threshold_rel_list=threshold_rel_list, 
                max_addresses_to_operate_list=max_addresses_to_operate_list, 
                max_addresses_to_show_list=max_addresses_to_show_list,
                result_similarity_levels=result_similarity_levels,
                **kwargs
            )
            results_with_cp = self._mark_ambiguity(
                ambiguous_threshold, 
                results_with_cp
            )
            results_with_cp = self._resolve_ambiguity(
                *args,
                df = results_with_cp,
                **kwargs
            )
        else:
            results_with_cp = pd.DataFrame()
        results_without_cp = self._get_results_without_cp(
            *args, 
            threshold_abs_list=threshold_abs_list, 
            threshold_rel_list=threshold_rel_list, 
            max_addresses_to_operate_list=max_addresses_to_operate_list, 
            max_addresses_to_show_list=max_addresses_to_show_list,
            result_similarity_levels=result_similarity_levels,
            **kwargs
        )
        results_without_cp = self._mark_ambiguity(ambiguous_threshold, results_without_cp)
        results_without_cp = self._resolve_ambiguity(
                *args,
                df = results_without_cp,
                use_cp=False,
                **kwargs
            )
        results = pd.concat([results_without_cp, results_with_cp])
        if results.empty:
            return pd.DataFrame()
        results = results.drop_duplicates(subset=CPA_COLUMN, keep="first")
        return results

    def get_results_batch(
            self, 
            queries_batch, 
            threshold_abs_list=None, 
            mlp_threshold_abs_relaxed = None,
            threshold_rel_list=None, 
            ambiguous_threshold=None,
            max_addresses_to_operate_list=None, 
            max_addresses_to_show_list=None, 
            id_list=None,
            progress_cb=None,
            use_cp=True,          
            progress_every=1 
        ):
        """
        Get results for a batch of structured queries.

        Args:
            queries_batch (list of dict or list of str): Each item can be a dictionary with keys matching the query columns or a string.
            id_list (list, optional): List of IDs to filter results by. If None, no filtering is applied.

        Returns:
            pd.DataFrame: DataFrame with results for all queries in the batch.
        """
        if not id_list:
            id_list = range(len(queries_batch))
        elif len(id_list) != len(queries_batch):
            self.logger.error("Length of id_list does not match length of queries_batch.")
            return None
        total = len(id_list)          # <-- NUEVO
        i = 0 
        results = []
        if isinstance(queries_batch[0], dict) or isinstance(queries_batch[0], str) or isinstance(queries_batch[0], list):
            for query, id_ in tqdm(zip(queries_batch, id_list)):
                try:
                    if isinstance(query, dict):
                            result = self.get_results(
                            **query, 
                            threshold_abs_list=threshold_abs_list, 
                            mlp_threshold_abs_relaxed = mlp_threshold_abs_relaxed,
                            threshold_rel_list=threshold_rel_list,
                            ambiguous_threshold=ambiguous_threshold,
                            max_addresses_to_operate_list=max_addresses_to_operate_list,
                            max_addresses_to_show_list=max_addresses_to_show_list,
                            use_cp=use_cp
                            )
                            result = result.iloc[:min(self.settings["BATCH_MAX_ADDRESSES_TO_SHOW"], len(result))]
                    else:
                        result = self.get_results(
                            query, 
                            threshold_abs_list=threshold_abs_list, 
                            mlp_threshold_abs_relaxed = mlp_threshold_abs_relaxed,
                            threshold_rel_list=threshold_rel_list,
                            ambiguous_threshold=ambiguous_threshold,
                            max_addresses_to_operate_list=max_addresses_to_operate_list,
                            max_addresses_to_show_list=max_addresses_to_show_list,
                            use_cp=use_cp
                        )
                        result = result.iloc[:min(self.settings["BATCH_MAX_ADDRESSES_TO_SHOW"], len(result))]
                    if result.empty:
                        columns = list(set(self.settings["COLUMNS_TO_RETURN"] + [f"similarity_level_{i}" for i in range(self.n)]))
                        result = pd.DataFrame([ [np.nan] * len(columns) ], columns=columns)
                    result.index = [id_] * len(result)  # Set index to id_
                    results.append(result)
                    i += 1
                    if progress_cb and (i % progress_every == 0 or i == total):
                        try:
                            progress_cb(i, total)
                        except Exception:
                            pass
                except Exception as e:
                    import traceback
                    # Build a concise error description and full traceback
                    err_msg = f"{type(e).__name__}: {e}"
                    tb = traceback.format_exc()

                    original_columns = list(set(self.settings["COLUMNS_TO_RETURN"] + [f"similarity_level_{j}" for j in range(self.n)]))
                    # Add error info columns so callers can inspect what failed
                    columns = original_columns + ["error", "traceback"]
                    # Create a row with NaNs for data columns and error info in the new columns
                    row = [np.nan] * len(original_columns) + [err_msg, tb]
                    result = pd.DataFrame([row], columns=columns)
                    result.index = [id_] * len(result)  # Set index to id_
                    results.append(result)
        else:
            self.logger.error("queries_batch must be a list of dictionaries or strings.")
            return None
        return pd.concat(results)

    def filter(self, query_hashes, level):
        """Filter the hash_emb_dict to only include specified query hashes."""
        self.maestro.filter_by_level(query_hashes, level)
        return self


#%%
if __name__ == "__main__":
    # from address_searcher import Searcher, AddressSearcherStructured
    from preprocessing.maestro import MaestroProcessor
    from preprocessing.embedder import MultiEmbedder
    from preprocessing.model import Model
    import yaml
    import os
    os.chdir("/home/iaadmin/gonza/ubidata-intelligence")
    import pandas as pd
    struct_config_path = "configs/base/structured/config.yaml"
    with open(struct_config_path, "r", encoding="utf-8") as f:
        struct_settings = yaml.safe_load(f)

    unstruct_config_path = "configs/base/unstructured/config.yaml"
    with open(unstruct_config_path, "r", encoding="utf-8") as f:
        unstruct_settings = yaml.safe_load(f)

    global_config_path = "configs/base/global.yaml"
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG"]]
    maestro = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=struct_settings,
            MAESTRO_PATH="/home/iaadmin/gonza/ubidata-intelligence/datasets/maestros/base_maestro.txt",
            APPLY_LOCALIDADES_LINDERAS=True
        ).process(load=False).generate_no_height_street_hashes_list().generate_height_street_hashes_list()
    )

    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"]]
    maestro_solo_localidad = (
        MaestroProcessor(
            query_columns=QUERY_COLUMNS,
            cpa_column=global_settings["MAESTRO_CPA_COLUMN"],
            settings=struct_settings,
            MAESTRO_PATH="/home/iaadmin/gonza/ubidata-intelligence/datasets/maestros/base_maestro.txt",
            APPLY_LOCALIDADES_LINDERAS=True
        ).process_solo_localidad(load=False).generate_no_height_street_hashes_list().generate_height_street_hashes_list()
    )
    STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in struct_settings["SEARCHER_CONFIG"]]
    struct_models = [Model(model_path=path, settings=struct_settings) for path in STRUCT_MODEL_PATHS]
    struct_embedder = MultiEmbedder(models=struct_models, maestro=maestro, maestro_solo_localidad=maestro_solo_localidad, settings=struct_settings)
    struct_embedder = struct_embedder.build(load=False)
    # struct_embedder.get_results_batch(
    #     [{
    #         "PROVINCIA":"SALTA",
    #         "PARTIDO":"",
    #         "LOCALIDAD":"",
    #         "MUNICIPIO":"",
    #         "PXL_PREF1974":"4126",
    #         "NOM_CALLE_COMPLETO":"feb",
    #     },
    #     {
    #         "PROVINCIA":"BUENOS AIRES",
    #         "PARTIDO":"",
    #         "LOCALIDAD":"",
    #         "MUNICIPIO":"",
    #         "PXL_PREF1974":"1619",
    #         "NOM_CALLE_COMPLETO":"remedios",
    #     }]
    # )
    # struct_embedder.get_results(
    #     **{'PROVINCIA': 'RÍO NEGRO',
    #     'PARTIDO': '',
    #     'LOCALIDAD': 'LAMARQUE',
    #     'MUNICIPIO': '',
    #     'NOM_CALLE_COMPLETO': 'PASAJE PAGANO 1172',
    #     'PXL_PREF1974': '1612'},
    #     threshold_abs_list=[0, 0, 0],
    #     threshold_rel_list=[0, 0, 0],
    #     max_addresses_to_operate_list=[1,1,1]
    # )
    # struct_embedder.get_results(
    #     **{'PROVINCIA': 'CAPITAL FEDERAL',
    #     'PARTIDO': '',
    #     'LOCALIDAD': '',
    #     'MUNICIPIO': '',
    #     'NOM_CALLE_COMPLETO': 'PASAJE PAGANO',
    #     'PXL_PREF1974': '1612'},
    #     threshold_abs_list=[0, 0.5, 0],
    #     threshold_rel_list=[0, 0, 0],
    #     max_addresses_to_operate_list=[1,1,1]
    # )
    # struct_embedder.get_results(
    #         **{'PROVINCIA': 'SALTA',
    #     'PARTIDO': '',
    #     'LOCALIDAD': 'LA CANDELARIA',
    #     'MUNICIPIO': '',
    #     'NOM_CALLE_COMPLETO': '21 DE FEBRERO',
    #     'PXL_PREF1974': '1612'},
    #     threshold_abs_list=[0, 0, 0],
    #     threshold_rel_list=[0, 0, 0],
    #     max_addresses_to_operate_list=[1,10,1],
    #         cp=False,
    #     )

    struct_embedder.get_results(
            **{'PROVINCIA': 'BUENOS AIRES',
        'PARTIDO': '',
        'LOCALIDAD': 'EL TALA',
        'MUNICIPIO': '',
        'NOM_CALLE_COMPLETO': '21 FEBRERO',
        'PXL_PREF1974': ''},
        threshold_abs_list=[0, 0, 0],
        threshold_rel_list=[0, 0, 0],
        max_addresses_to_operate_list=[1,10,1],
            cp=False,
        )