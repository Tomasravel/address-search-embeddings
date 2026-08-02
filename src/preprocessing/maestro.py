import os
import numpy as np
import pandas as pd
import copy
import json
from logs.logger import Logger
from preprocessing.utils import remove_accents, generate_queries, filter_df_by_index_level
from preprocessing.cts import NOM_CALLE_ABR_COLUMN, NOM_LOCA_ABR_COLUMN, TIPO_CALLE_COLUMN, NOM_CALLE_COLUMN, SIN_LOCA_COLUMN, LOCALIDAD_COLUMN, PARTIDO_COLUMN, NOM_APELLIDO_COLUMN, NOM_APELLIDO_ORIGINAL_COLUMN, CP_COLUMN

EMBEDDINGS_MATRIX_PATH_SETTING = "EMBEDDINGS_MATRIX_PATH"
STORE_EMBEDDINGS_MATRIX_SETTING = "STORE_EMBEDDINGS_MATRIX"
SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING = "SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH"

class MaestroProcessor:
    "Class to process Maestro data."
    def __init__(
            self, 
            query_columns, 
            settings, 
            cpa_column=None, 
            maestro_df=pd.DataFrame(), 
            sin_calle_df=pd.DataFrame(), 
            height_street_hashes = [], 
            no_height_street_hashes=[], 
            query_hash_to_hash_ori_map={},
            **kwargs):
        self.logger = Logger(name=__class__.__name__, settings=settings).get()
        # config
        self.settings = settings
        self.settings.update(kwargs)
        
        self.query_columns = query_columns
        if maestro_df.empty:
            self.maestro_path = self.settings["MAESTRO_PATH"]
        self.levels = len(query_columns)
        self.maestro_df = maestro_df    
        self.cpa_column = cpa_column
        self.maestro_embeddings_matrix = None
        self.sin_calle_df = sin_calle_df
        self.no_height_street_hashes = no_height_street_hashes
        self.height_street_hashes = height_street_hashes
        self.query_hash_to_hash_ori_map = query_hash_to_hash_ori_map

    def load_data(self):
        "Load the Maestro data from the CSV file."
        self.logger.info("Loading Maestro data from %s", self.maestro_path)
        maestro_df = pd.read_csv(
            self.maestro_path, 
            encoding="latin1", 
            delimiter=",", 
            low_memory=False
        )
        self.maestro_df = maestro_df

        return self

    def generate_queries(self):
        "Set the query column as index for faster access."
        # Generate queries from specified columns
        self.logger.debug(f"Generating queries from columns: {self.query_columns}")
        for i in range(len(self.query_columns)):
            self.maestro_df[f"query_{i}"] = generate_queries(
                self.maestro_df, 
                self.query_columns[i], 
                unique_cols=self.settings["SEARCHER_CONFIG"][i].get("UNIQUE_COLUMNS")
            )
        return self

    def generate_queries_solo_localidad(self):
        "Set the query column as index for faster access."
        # Generate queries from specified columns
        self.logger.debug(f"Generating queries from columns: {self.query_columns}")
        for i in range(len(self.query_columns)):
            self.maestro_df[f"query_{i}"] = generate_queries(
                self.maestro_df, 
                self.query_columns[i], 
                unique_cols=self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][i].get("UNIQUE_COLUMNS")
            )
        return self

    def set_query_as_index_hash(self):
        query_hash_to_hash_ori_map = {}
        if self.cpa_column is not None:
            self.maestro_df["_CPA2"] = self.maestro_df[self.cpa_column].str[-3:-1]
        else:
            self.maestro_df["_CPA2"] = ""
        for level in range(len(self.query_columns)):
            if f"query_hash_{level}" not in self.maestro_df.columns:
                cols = self.settings["SEARCHER_CONFIG"][level].get("ID_COLUMNS")
                self.maestro_df[f"query_hash_{level}"] = pd.util.hash_pandas_object(
                    self.maestro_df[cols],
                    index=False
                )
            if "ID_COLUMNS_ORI" in self.settings["SEARCHER_CONFIG"][level]:
                if f"query_hash_{level}_ori" not in self.maestro_df.columns:
                    cols_ori = self.settings["SEARCHER_CONFIG"][level].get("ID_COLUMNS_ORI")
                    self.maestro_df[f"query_hash_{level}_ori"] = pd.util.hash_pandas_object(
                        self.maestro_df[cols_ori],
                        index=False
                    )
                # Construimos el mapa
                serie_map = (
                    self.maestro_df[[f"query_hash_{level}_ori", f"query_hash_{level}"]]
                    .drop_duplicates()
                    .set_index(f"query_hash_{level}")
                )[f"query_hash_{level}_ori"].copy()

                # ✔️ Chequeo de duplicados en el índice
                if serie_map.index.duplicated().any():
                    dups = serie_map.index[serie_map.index.duplicated()].unique()
                    self.logger.error(
                        f"Duplicated indices found in query_hash_to_hash_ori_map[{level}]: {dups.tolist()}"
                    )
                    raise ValueError(f"Duplicated query_hash indices detected for level '{level}'")
                                                
                query_hash_to_hash_ori_map[level] = serie_map
        self.query_hash_to_hash_ori_map = query_hash_to_hash_ori_map
        self.maestro_df.drop(columns=["_CPA2"], inplace=True)
        self.maestro_df.set_index([f"query_hash_{level}" for level in range(len(self.query_columns))], inplace=True)
        self.maestro_df.sort_index(inplace=True)
        return self
    
    def set_query_as_index_hash_solo_localidad(self):
        query_hash_to_hash_ori_map = {}
        if self.cpa_column is not None:
            self.maestro_df["_CPA2"] = self.maestro_df[self.cpa_column].str[-3:-1]
        else:
            self.maestro_df["_CPA2"] = ""
        for level in range(len(self.query_columns)):
            if f"query_hash_{level}" not in self.maestro_df.columns:
                cols = self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][level].get("ID_COLUMNS")
                self.maestro_df[f"query_hash_{level}"] = pd.util.hash_pandas_object(
                    self.maestro_df[cols],
                    index=False
                )
            if "ID_COLUMNS_ORI" in self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][level]:
                if f"query_hash_{level}_ori" not in self.maestro_df.columns:
                    cols_ori = self.settings["SEARCHER_CONFIG_SOLO_LOCALIDAD"][level].get("ID_COLUMNS_ORI")
                    self.maestro_df[f"query_hash_{level}_ori"] = pd.util.hash_pandas_object(
                        self.maestro_df[cols_ori],
                        index=False
                    )
                # Construimos el mapa
                serie_map = (
                    self.maestro_df[[f"query_hash_{level}_ori", f"query_hash_{level}"]]
                    .drop_duplicates()
                    .set_index(f"query_hash_{level}")
                )[f"query_hash_{level}_ori"].copy()

                # ✔️ Chequeo de duplicados en el índice
                if serie_map.index.duplicated().any():
                    dups = serie_map.index[serie_map.index.duplicated()].unique()
                    self.logger.error(
                        f"Duplicated indices found in query_hash_to_hash_ori_map[{level}]: {dups.tolist()}"
                    )
                    raise ValueError(f"Duplicated query_hash indices detected for level '{level}'")
                                                
                query_hash_to_hash_ori_map[level] = serie_map

        self.query_hash_to_hash_ori_map = query_hash_to_hash_ori_map
        self.maestro_df.drop(columns=["_CPA2"], inplace=True)
        self.maestro_df.set_index([f"query_hash_{level}" for level in range(len(self.query_columns))], inplace=True)
        self.maestro_df.sort_index(inplace=True)
        return self
    
    def load_unique_queries_groups(self, solo_localidad=False):
        """Loads the unique queries groups from path."""

        path = self.settings["UNIQUE_QUERIES_GROUPS_PATH"]

        self.unique_queries_groups = {}

        for level in range(self.levels):
            if solo_localidad:
                file_path = os.path.join(path, f"solo-localidad-level_{level}.parquet")
            else:
                file_path = os.path.join(path, f"level_{level}.parquet")
            self.unique_queries_groups[level] = pd.read_parquet(file_path)
    
    def generate_no_height_street_hashes_list(self):
        """Generates no height street hashes list for all levels.

        Used to speed up no height street searches.

        **Note**: Must be called after loading embeddings or in general after
        all needed columns are generated for the `maestro_df`.

        Returns:
            Self class for method chaining.
        """
        maestro_df = self.maestro_df
        no_height_df = maestro_df[(maestro_df['COD_HASTA'] == 0) & (maestro_df['COD_DESDE'] == 0)]
        self.no_height_street_hashes = no_height_df.index.get_level_values(self.levels - 1).unique().tolist()

        return self
    
    def generate_height_street_hashes_list(self):
        """Generates height street hashes list for all levels.
        Used to speed up height street searches.

        Returns:
            Self class for method chaining.
        """
        maestro_df = self.maestro_df
        height_df = maestro_df[(maestro_df['COD_HASTA'] != 0) | (maestro_df['COD_DESDE'] != 0)]
        self.height_street_hashes = height_df.index.get_level_values(self.levels - 1).unique().tolist()    
        return self
    
    def generate_unique_queries_groups(self, solo_localidad=False):
        """Generates unique queries groups.

        Generates and stores unique queries groups per level on hash columns
        to speed up the `get_unique_queries_df` method.
        
        **Note**: Must be called after loading embeddings or in general after
        all needed columns are generated for the `maestro_df`.

        Returns:
            Self class for method chaining.
        """

        # for unstructured queries, unique queries groups are preprocessed
        self.unique_queries_groups = {}

        for level in range(self.levels):
            self.unique_queries_groups[level] = self.maestro_df.groupby(f'query_hash_{level}').head(1)

        if self.settings.get("SAVE_UNIQUE_QUERIES_GROUPS", False):
            for level in range(self.levels):
                path = self.settings["UNIQUE_QUERIES_GROUPS_PATH"]
                os.makedirs(path, exist_ok=True)
                if solo_localidad:
                    path = os.path.join(path, f"solo-localidad-level_{level}.parquet")
                else:
                    path = os.path.join(path, f"level_{level}.parquet")
                self.unique_queries_groups[level].to_parquet(path)

        return self
    
    def generate_first_level_embeddings_matrix(self, solo_localidad=False):
        """
        Generates embeddings matrix for the first level embeddings.

        Used to speed up np.stack on `get_similarities`.

        **Note**: Must be called after loading embeddings or in general after
        all needed columns are generated for the `maestro_df`.

        Raises:
            KeyError: If path setting is not set.
        """

        embedding_col = "embedding_0"

        maestro_embeddings = self.get_unique_queries_df(0, [embedding_col])[embedding_col]

        self.maestro_embeddings_matrix = np.stack(maestro_embeddings.values)

        if STORE_EMBEDDINGS_MATRIX_SETTING in self.settings:

            if self.settings[STORE_EMBEDDINGS_MATRIX_SETTING]:
                if solo_localidad:
                    if SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING in self.settings:

                        np.save(
                            self.settings[SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING],
                            self.maestro_embeddings_matrix
                        )
                    else:
                        raise KeyError(
                            f"if {STORE_EMBEDDINGS_MATRIX_SETTING} is True, "
                            f"{SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING} must be set."
                        )
                else:
                    if EMBEDDINGS_MATRIX_PATH_SETTING in self.settings:

                        np.save(
                            self.settings[EMBEDDINGS_MATRIX_PATH_SETTING],
                            self.maestro_embeddings_matrix
                        )
                    else:
                        raise KeyError(
                            f"if {STORE_EMBEDDINGS_MATRIX_SETTING} is True, "
                            f"{EMBEDDINGS_MATRIX_PATH_SETTING} must be set."
                        )

    
    def load_first_level_embeddings_matrix(self, solo_localidad=False):
        """
        Loads the embeddings matrix for the first level embeddings.

        Used to speed up np.stack on `get_similarities`.

        **Note**: Must be called after loading embeddings or in general after
        all needed columns are generated for the `maestro_df`.

        Raises:
            KeyError: If path setting is not set.
        """
        if solo_localidad:
            if SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING not in self.settings:
                raise KeyError(f"missing setting {SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING}")
            
            path = self.settings[SOLO_LOCALIDAD_EMBEDDINGS_MATRIX_PATH_SETTING]

            self.maestro_embeddings_matrix = np.load(path)
        else:
            if EMBEDDINGS_MATRIX_PATH_SETTING not in self.settings:
                raise KeyError(f"missing setting {EMBEDDINGS_MATRIX_PATH_SETTING}")
            
            path = self.settings[EMBEDDINGS_MATRIX_PATH_SETTING]

            self.maestro_embeddings_matrix = np.load(path)
    
    def get_unique_queries_df(self, level=0, columns=[], use_calculated=False):
        """
        Return a DataFrame with unique entries at the specified MultiIndex level.
        
        Args:
            level (int or str): The index level to deduplicate on.
            columns (list): List of column names to include in the result.

        Returns:
            pd.DataFrame: A DataFrame with the specified index level as the only index,
                        and no duplicate values on that level.
        """
        # Get name of the level if it's passed as int
        if isinstance(level, int):
            level_name = self.maestro_df.index.names[level]
        else:
            level_name = level

        calculated_cols = [
            c for c in columns
            if str(c).startswith("similarity_level_")
        ]

        if calculated_cols:
            use_calculated = True
            
        if not hasattr(self, 'unique_queries_groups') or use_calculated:
            return self.maestro_df.groupby(level_name, sort=False)[columns].first()

        if level == 0:
            unique_queries_df: pd.DataFrame = self.unique_queries_groups[0][columns]
        else:

            hash_levels_value = [
                self.maestro_df.index.unique(level=i) for i in range(level)
            ]

            mask = self.unique_queries_groups[level].index.get_level_values(0).isin(hash_levels_value[0])

            for i in range(1, level):
                mask &= self.unique_queries_groups[level].index.get_level_values(i).isin(hash_levels_value[i])

            unique_queries_df: pd.DataFrame = self.unique_queries_groups[level][mask][columns]

        unique_queries_df = unique_queries_df.droplevel([i for i in range(self.levels) if i != level])

        return unique_queries_df

    def get_index(self):
        """
        Get the index of the Maestro DataFrame.
        
        Returns:
            pd.Index: The index of the Maestro DataFrame.
        """
        if self.maestro_df is not None:
            return self.maestro_df.index
        else:
            self.logger.error("Maestro DataFrame is not loaded.")
            return None
    
    def add_column(self, column_name, data):
        """
        Add a new column to the Maestro DataFrame.
        
        Args:
            column_name (str): The name of the new column.
            data (pd.Series or list): The data to add to the new column.
        """
        if self.maestro_df is not None:
            self.maestro_df[column_name] = data
        else:
            self.logger.error("Maestro DataFrame is not loaded. Cannot add column.")
    
    def get_column_names(self):
        return self.maestro_df.columns if self.maestro_df is not None else None
    
    def get_columns(self, columns=None):
        """
        Get specific columns from the Maestro DataFrame.
        
        Args:
            columns (list): List of column names to retrieve. If None, returns all columns.
        
        Returns:
            pd.DataFrame: DataFrame containing the specified columns.
        """
        if self.maestro_df is not None:
            if columns is None:
                return self.maestro_df
            else:
                return self.maestro_df[columns]
        else:
            self.logger.error("Maestro DataFrame is not loaded. Cannot get columns.")
            return None
        
    def get_first_value(self, column_name):
        """
        Get the first value of a specific column from the Maestro DataFrame.
        
        Args:
            column_name (str): The name of the column.
        Returns:
            The first value of the specified column.
        """
        return self.get_columns([column_name]).iloc[0, 0]

    def to_values(self):
        """
        Convert the Maestro DataFrame to JSON format.
        
        Args:
            columns (list): List of columns to include in the JSON output. If None, all columns are included.
        
        Returns:
            str: JSON string representation of the DataFrame.
        """
    
        res = []
        for _, row in self.maestro_df.iterrows():
            query = []
            for level in range(len(self.query_columns)):
                query.append(row[f"query_{level}"])
                for _ in range(1, len(self.query_columns[level])):
                    query.append("")
            res.append(query)
        return res

    def to_csv(self, path):
        """
        Convert the Maestro DataFrame to CSV format.
        
        Args:
            path (str): The file path to save the CSV.
        """
        if self.maestro_df is not None:
            self.maestro_df.to_csv(path, index=True, encoding="latin1", sep=",")
        else:
            self.logger.error("Maestro DataFrame is not loaded. Cannot convert to CSV.")

    def map(self, level, data, column_name):
        """
        Map a function or data to a specific level of the Maestro DataFrame.

        Args:
            level (int or str): The level of the query to map.
            data (dict, pd.Series, callable): The data or function to apply.
            column_name (str): Name of the column to create/update.
        """
        if self.maestro_df is None:
            self.logger.error("Maestro DataFrame not loaded.")
            return

        # Normalize level to its index name
        if isinstance(level, int):
            if level < 0 or level >= len(self.query_columns):
                self.logger.error("Invalid level.")
                return
            level_name = self.maestro_df.index.names[level]
        else:
            if level not in self.maestro_df.index.names:
                self.logger.error("Invalid level.")
                return
            level_name = level

        # Update main Maestro DataFrame
        self.maestro_df[column_name] = (
            self.maestro_df.index
            .get_level_values(level_name)
            .map(data)
        )
        if column_name.startswith("similarity_level_"):
            return
        # Keep unique_queries_groups consistent, if they already exist
        if hasattr(self, "unique_queries_groups"):

            n_levels = len(self.unique_queries_groups)

            for level in range(n_levels):
                self.unique_queries_groups[level][column_name] = (
                    self.unique_queries_groups[level]
                        .index
                        .get_level_values(level)
                        .map(data)
                )
    
    def filter_by_level(self, query_hashes, level=0):
        maestro_df = filter_df_by_index_level(
            self.maestro_df, 
            values=query_hashes, 
            level=level
        )
        new_instance = MaestroProcessor(
            query_columns=self.query_columns,
            cpa_column=self.cpa_column,
            maestro_df=maestro_df,
            height_street_hashes=self.height_street_hashes,
            no_height_street_hashes=self.no_height_street_hashes,
            query_hash_to_hash_ori_map=self.query_hash_to_hash_ori_map,
            settings=self.settings
        )

        if hasattr(self, 'unique_queries_groups'):
            # Warning: This is a shallow copy. If the unique queries groups are
            # modified, the original will be modified too. In theory the unique
            # queries groups are not modified.
            new_instance.unique_queries_groups = dict(self.unique_queries_groups)

        return new_instance

    def filter(self, query_hashes):
        return self.filter_by_level(query_hashes, level=0)
    
    def filter_by_column(self, column_name, values, level=0):
        maestro_df_filtered = self.maestro_df[self.maestro_df[column_name].isin(values)]
        query_hashes = maestro_df_filtered.index.get_level_values(level).unique().tolist()
        return self.filter_by_level(query_hashes, level=level)

    def preprocess(self):
        "Preprocess the Maestro DataFrame."
        self.logger.info("Starting Maestro data processing...")
        maestro_df = self.maestro_df
        maestro_df = maestro_df.map(remove_accents)
        maestro_df = maestro_df.drop_duplicates()
        # Data types
        if "COD_DESDE" in maestro_df.columns:
            maestro_df['COD_DESDE'] = maestro_df['COD_DESDE'].fillna(-1).astype(np.int64)
        if "COD_HASTA" in maestro_df.columns:
            maestro_df['COD_HASTA'] = maestro_df['COD_HASTA'].fillna(0).astype(np.int64)
        if CP_COLUMN in maestro_df.columns:
            maestro_df[CP_COLUMN] = maestro_df[CP_COLUMN].fillna("").astype(str)
        all_query_columns = [col[i] for col in self.query_columns for i in range(len(col)) if col[i] in maestro_df.columns]
        maestro_df[all_query_columns] = maestro_df[all_query_columns].fillna("").astype(str)
        for level in range(self.levels):
            query_col = f"query_{level}"
            if query_col in maestro_df.columns:
                maestro_df[query_col] = maestro_df[query_col].fillna("").astype(str)
        
        # NOM_APELLIDO PREPROCESS
        if NOM_APELLIDO_COLUMN in maestro_df.columns:
            # Preservamos el apellido original (post remove_accents) para devolverlo,
            # y guardamos la version normalizada por generate_queries para el matching exacto.
            maestro_df[NOM_APELLIDO_ORIGINAL_COLUMN] = maestro_df[NOM_APELLIDO_COLUMN]
            maestro_df[NOM_APELLIDO_COLUMN] = generate_queries(maestro_df, [NOM_APELLIDO_COLUMN])
        # SIN_NOMBRE
        if "SIN_NOMBRE" in maestro_df.columns and "NOM_CALLE_ABR_C" not in maestro_df.columns:
            oficial = (
                maestro_df.loc[maestro_df["SIN_NOMBRE"] == "C", [self.cpa_column, NOM_CALLE_ABR_COLUMN]]
                .drop_duplicates(self.cpa_column)                   # por si hay repetidas
                .set_index(self.cpa_column)[NOM_CALLE_ABR_COLUMN]  # elijo la columna oficial
            )

            # 2) Creo la columna con el nombre oficial para TODAS las filas (C y S)
            maestro_df["NOM_CALLE_ABR_C"] = maestro_df[self.cpa_column].map(oficial)

        # LOCALIDAD
        if LOCALIDAD_COLUMN in maestro_df.columns: 
            if "localidad_original" not in maestro_df.columns:
                maestro_df['localidad_original'] = maestro_df['LOCALIDAD']
            # Localidades linderas
            if self.settings["APPLY_LOCALIDADES_LINDERAS"]:
                self.logger.info("Applying localidades linderas to Maestro DataFrame...")
                loc_lind_path = self.settings["LOCALIDADES_LINDERAS_PATH"]
                loc_lind_df = pd.read_excel(loc_lind_path, dtype=str, usecols=['PARTIDO_1','LOCALIDAD_1','LOCALIDAD_'])
                extra = maestro_df.merge(
                    loc_lind_df,
                    left_on=[PARTIDO_COLUMN, LOCALIDAD_COLUMN],
                    right_on=['PARTIDO_1', "LOCALIDAD_1"],
                    how='inner'
                )
                extra = extra.assign(
                    LOCALIDAD=extra['LOCALIDAD_'], 
                    localidad_original=extra['LOCALIDAD_1']
                )
                extra = extra[maestro_df.columns]
                maestro_df = pd.concat([maestro_df, extra], ignore_index=True)
            else:
                self.logger.info("Skipped localidades linderas application.")

            # SIN_LOCA
            if SIN_LOCA_COLUMN in maestro_df.columns:
                self.logger.info("Expanding Maestro DataFrame for SIN_LOCA entries...")
                df_extra = maestro_df[maestro_df[SIN_LOCA_COLUMN].notna()].copy()
                df_extra[LOCALIDAD_COLUMN] = df_extra[SIN_LOCA_COLUMN]
                maestro_df = pd.concat([maestro_df, df_extra], ignore_index=True)
                maestro_df.drop(columns=[SIN_LOCA_COLUMN], inplace=True)
            
            # NOM_LOCA_ABR
            if NOM_LOCA_ABR_COLUMN in maestro_df.columns:
                self.logger.info("Expanding Maestro DataFrame for NOM_LOCA_ABR entries...")
                df_extra = maestro_df[(maestro_df[NOM_LOCA_ABR_COLUMN].notna()) & (maestro_df[NOM_LOCA_ABR_COLUMN] != maestro_df[LOCALIDAD_COLUMN])].copy()
                df_extra[LOCALIDAD_COLUMN] = df_extra[NOM_LOCA_ABR_COLUMN]
                maestro_df = pd.concat([maestro_df, df_extra], ignore_index=True)
                maestro_df.drop(columns=[NOM_LOCA_ABR_COLUMN], inplace=True)
            
            # NOM_CALLE_COMPLETO
            if TIPO_CALLE_COLUMN in maestro_df.columns and NOM_CALLE_COLUMN in maestro_df.columns and NOM_CALLE_ABR_COLUMN in maestro_df.columns and "NOM_CALLE_COMPLETO" not in maestro_df.columns:
                maestro_df.loc[maestro_df["TIPO_CALLE"]!="CALLE", "NOM_CALLE_COMPLETO"] = maestro_df[TIPO_CALLE_COLUMN] + ' ' + maestro_df[NOM_CALLE_COLUMN]
                maestro_df.fillna({"NOM_CALLE_COMPLETO": maestro_df[NOM_CALLE_COLUMN]}, inplace=True)
                df_extra = maestro_df[(maestro_df[NOM_CALLE_ABR_COLUMN] != maestro_df["NOM_CALLE_COMPLETO"])].copy()
                df_extra["NOM_CALLE_COMPLETO"] = df_extra[NOM_CALLE_ABR_COLUMN]
                maestro_df = pd.concat([maestro_df, df_extra], ignore_index=True)

        # Indexes for faster access
        maestro_df["id_hash"] = pd.util.hash_pandas_object(
            maestro_df, 
            index=False
        )
        # Drop unused columns
        maestro_df.drop(columns=[
            "SIN_LOCA",
            "CATEG_MUNI",
            "NOM_APELLIDO_LOCA", 
            "NOM_NOMBRE_LOCA", 
            "ABR_ANTEPONEN_LOCA", 
            "NOM_LOCA_ABR", 
            "NOM_CALLE_2", 
            "BAR_TIPO", 
            "BAR_TIPYNOM"
        ], errors="ignore", inplace=True)
        self.maestro_df = maestro_df

        return self

    def process(self, load=False):
        "Process the Maestro data."
        if load:
            self.load_data().preprocess().set_query_as_index_hash().split_sin_calle().generate_no_height_street_hashes_list().generate_height_street_hashes_list()
        else: 
            self.load_data().preprocess().generate_queries().set_query_as_index_hash().split_sin_calle()
        self.logger.info("Maestro data processing complete.")

        return self
    
    def process_solo_localidad(self, load=False):
        "Process the Maestro data."
        if load:
            self.load_data().preprocess().set_query_as_index_hash_solo_localidad().split_sin_calle().generate_no_height_street_hashes_list().generate_height_street_hashes_list()
        else: 
            self.load_data().preprocess().generate_queries_solo_localidad().set_query_as_index_hash_solo_localidad().split_sin_calle()
        self.logger.info("Maestro data processing complete.")

        return self
    
    def split_sin_calle(self):
        self.logger.info("Splitting Maestro DataFrame into with and without street names...")
        maestro_df = self.maestro_df
        self.sin_calle_df = maestro_df[maestro_df['CPA'].str[-3] == "X"]
        maestro_df = maestro_df[maestro_df['CPA'].str[-3] != "X"]
        self.maestro_df = maestro_df
        if self.settings.get("SIN_CALLE_PROCESSED_PATH", None):
            self.sin_calle_df.to_csv(self.settings["SIN_CALLE_PROCESSED_PATH"], index=False, encoding="latin1", sep=",")
            self.logger.info(f"Saved sin calle DataFrame to {self.settings['SIN_CALLE_PROCESSED_PATH']}")
        return self
    
    def get_cod_desde_hasta(self, query_id, height=None, level=0):
        if level == 0:
            try:
                df_filtered = self.maestro_df.loc[[query_id]]
            except KeyError:
                df_filtered = pd.DataFrame()
        else:
            df_filtered = self.maestro_df.xs(query_id, level=level, drop_level=False)
        
        cod_desde = df_filtered["COD_DESDE"]
        cod_hasta = df_filtered["COD_HASTA"]

        if df_filtered.empty:
            self.logger.warning(f"No records found for query_id: {query_id}")
            return (None, None, None, None)
        if height is None:
            # If height is not provided, return the first row's COD_DESDE and COD_HASTA
            return (df_filtered["id_hash"].iloc[0], df_filtered[self.cpa_column].iloc[0], cod_desde.iloc[0], cod_hasta.iloc[0])
        if query_id in self.no_height_street_hashes and query_id not in self.height_street_hashes:
            return (df_filtered["id_hash"].iloc[0], df_filtered[self.cpa_column].iloc[0], cod_desde.iloc[0], cod_hasta.iloc[0])

        # 1) Rango que contiene a 'height'
        mask = (cod_desde <= height) & (height <= cod_hasta)

        # 2) Paridad (mucho más rápido con bitwise que con %):
        #    “mismo parity” equivale a ((x ^ h) & 1) == 0
        mask &= (((cod_desde ^ height) & 1) == 0) & (((cod_hasta ^ height) & 1) == 0)

        if not mask.any():
            if query_id in self.no_height_street_hashes:
                mask = (cod_desde==0) | (cod_hasta==0)
            else:
                return (None, None, None, None)

        # Si hay varias filas válidas, tomamos la primera en orden actual
        i = int(mask.argmax())
        row = df_filtered.iloc[i]
        return (row["id_hash"], row[self.cpa_column], int(row["COD_DESDE"]), int(row["COD_HASTA"]))
    
    def copy(self):
        """
        Return a deep copy of the current MaestroProcessor instance.
        Ensures that self.maestro_df, self.settings, and related attributes
        are fully independent in the copy.
        """
        new_instance = MaestroProcessor(config_path=self.config_path)
        
        new_instance.settings = copy.deepcopy(self.settings)
        new_instance.maestro_path = self.maestro_path

        new_instance.maestro_df = (
            self.maestro_df.copy(deep=True) if self.maestro_df is not None else None
        )

        new_instance.logger = self.logger

        if hasattr(self, 'unique_queries_groups'):
            new_instance.unique_queries_groups = copy.deepcopy(self.unique_queries_groups)

        return new_instance
    
    def empty(self):
        return len(self.maestro_df)==0
        

if __name__ == "__main__":
    import yaml
    struct_config_path = "configs/base/structured/config.yaml"
    with open(struct_config_path, "r", encoding="utf-8") as f:
        struct_settings = yaml.safe_load(f)

    unstruct_config_path = "configs/base/unstructured/config.yaml"
    with open(unstruct_config_path, "r", encoding="utf-8") as f:
        unstruct_settings = yaml.safe_load(f)

    global_config_path = "configs/base/global.yaml"
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)

    MAESTRO_PATH = "datasets/maestros/debug.txt"
    QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in struct_settings["SEARCHER_CONFIG"]]
    STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in struct_settings["SEARCHER_CONFIG"]]
    maestro = MaestroProcessor(query_columns=QUERY_COLUMNS,cpa_column=global_settings["MAESTRO_CPA_COLUMN"],settings=struct_settings, MAESTRO_PATH=MAESTRO_PATH).process()
    maestro=maestro.generate_queries()
    maestro=maestro.set_query_as_index_hash()