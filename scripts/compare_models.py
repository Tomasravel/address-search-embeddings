#%%
import pandas as pd
import yaml
import os 
from src.preprocessing.maestro import MaestroProcessor
from src.preprocessing.embedder import MultiEmbedder
from src.preprocessing.model import Model

validation_config_path = "configs/base/validation.yaml"
with open(validation_config_path, "r", encoding="utf-8") as f:
    validation_settings = yaml.safe_load(f)
#%%

QUERY_COLUMNS = [col["QUERY_COLUMNS"] for col in validation_settings["SEARCHER_CONFIG"]]
maestro = MaestroProcessor(
    query_columns=QUERY_COLUMNS, settings=validation_settings, MAESTRO_PATH=validation_settings["MAESTRO_PATH"]
).process().generate_queries().set_query_as_index_hash()
#%%
# Load embedder models
STRUCT_MODEL_PATHS = [col["MODEL_PATH"] for col in validation_settings["SEARCHER_CONFIG"]]
struct_models = [Model(model_path=path, settings=validation_settings) for path in STRUCT_MODEL_PATHS]
struct_embedder = MultiEmbedder(models=struct_models, maestro=maestro, settings=validation_settings).generate_embeddings()
#%%
batch_txt_query_columns = validation_settings["VALIDATION_TXT_QUERY_COLUMNS_STRUCT"]
validation_df = pd.read_csv(os.path.join("..",validation_settings["VALIDATION_CSV_PATH"]), encoding='latin1').drop_duplicates(subset=[col for cols in batch_txt_query_columns for col in cols])
validation_df.index = range(len(validation_df))
batch_txt = MaestroProcessor(
    query_columns=batch_txt_query_columns, settings=validation_settings, maestro_df=validation_df
).preprocess().generate_queries().set_query_as_index_hash()
#%%
queries_batch = batch_txt.to_json(orient="values", columns=[f"query_{i}" for i in range(len(batch_txt_query_columns))])
result_df = struct_embedder.get_results_batch(
    queries_batch=queries_batch,
    id_list=list(validation_df.index),
    threshold_abs_list=[searcher_config["THRESHOLD_ABS"] for searcher_config in validation_settings["SEARCHER_CONFIG"]],
    threshold_rel_list=[searcher_config["THRESHOLD_REL"] for searcher_config in validation_settings["SEARCHER_CONFIG"]],
    max_addresses_to_operate_list=[searcher_config["MAX_ADDRESSES_TO_OPERATE"] for searcher_config in validation_settings["SEARCHER_CONFIG"]],
    max_addresses_to_show_list=[1] * struct_embedder.n
)
#%%
if __name__ == "__main__":
    ...