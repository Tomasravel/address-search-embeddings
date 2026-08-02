import pandas as pd
import random
import yaml
import nlpaug.augmenter.char as nac
import json
import os
import sys
import shutil
from train.constants import LANGUAGE, ColumnCategories as ColCat
from logs.logger import Logger
from preprocessing.utils import remove_accents, preprocess_query, generate_queries
from datetime import datetime

def contains_abbrevation(text, dict):
    """ Checks if any word in the text is present in the abbreviation dictionary."""
    if not isinstance(text, str):
        return False
    else:
        return any(word in dict for word in text.split())
    
def abbreviate_address(address, dic, abb_word_max):
    """Abbreviates a random number of words (up to abb_word_max) in the address based on the provided dictionary."""
    if not isinstance(address, str):
        return address
    words = address.split()
    abbreviable_indices = [i for i, word in enumerate(words) if word in dic]
    if not abbreviable_indices:
        return address
    num_to_abbreviate = random.randint(1, min(abb_word_max, len(abbreviable_indices)))
    indices_to_abbreviate = set(random.sample(abbreviable_indices, num_to_abbreviate))
    abbreviated_words = [
        dic[word] if i in indices_to_abbreviate else word
        for i, word in enumerate(words)
    ]
    return " ".join(abbreviated_words)

def mask_words(text: str, max_words_to_delete: int) -> str:
    """ Masks a random number of words in the text.
    If the text has fewer words than max_words_to_delete, it is left unchanged.
    """
    words = text.split()
    num_words = len(words)
    num_to_mask = random.randint(1, min(max_words_to_delete, num_words - 1))
    indices_to_mask = set(random.sample(range(num_words), num_to_mask))
    masked_words = [
        "" if i in indices_to_mask else word
        for i, word in enumerate(words)
    ]
    return " ".join(masked_words)

def mask_row(row, columns, max_cols_to_delete,  not_mask_columns):
    # Only consider columns that are not null for masking
    notna_indices = [i for i, col in enumerate(columns) if pd.notna(row[col]) and col not in not_mask_columns]
    n_notna = len(notna_indices)
    n_to_mask = random.randint(1, min(max_cols_to_delete, n_notna - 1))
    mask_indices = set(random.sample(notna_indices, n_to_mask))
    masked = [
        "" if i in mask_indices else row.fillna("")[columns[i]]
        for i in range(len(columns))
    ]
    masked_row = " ".join(masked).strip()
    masked_row_preprocessed = preprocess_query(masked_row)
    return masked_row_preprocessed


def augment_abbreviations(series, abbreviations, sample_frac, abb_word_max):
    """ Applies abbreviation to a random fraction of non-null values in the series"""
    result = pd.Series(index=series.index, dtype=object)
    notna_mask = series.notna() & series.apply(lambda x: contains_abbrevation(x, abbreviations))
    sampled_idx = series[notna_mask].sample(frac=sample_frac).index
    result.loc[sampled_idx] = series.apply(lambda x: abbreviate_address(x, abbreviations, abb_word_max))
    return result

def augment_keyboard_mistakes(series: pd.Series, sample_frac: float, aug) -> pd.Series:
    """
    Applies `aug.augment(x)` to a random fraction of non-null values in the series,
    leaving the rest as NaN.

    Parameters:
        series (pd.Series): Column to apply augmentation on.
        sample_frac (float): Fraction (0-1) of non-nulls to modify.
        aug: Augmenter with a `.augment(x)` method.

    Returns:
        pd.Series with augmentations on part of the values, and NaN on the rest.
    """
    result = pd.Series(index=series.index, dtype=object)
    notna_mask = series.notna() & series.apply(lambda x: len(x) > 0)
    sampled_idx = series[notna_mask].sample(frac=sample_frac).index
    result.loc[sampled_idx] = series.loc[sampled_idx].apply(lambda x: aug.augment(x)[0])
    return result

def augment_mask_columns(df, columns, sample_frac, max_cols_to_delete, min_notna_cols_to_operate, not_mask_columns):
    """
    Masks (empties) a random number of columns (up to max_cols_to_delete) for each row,
    among the specified columns. Returns a Series with the (possibly masked) column values
    concatenated by space for each row.

    Parameters:
        df (pd.DataFrame): The dataframe containing the columns.
        columns (list): List of column names to consider for masking.
        sample_frac (float): Fraction of rows to apply masking.
        max_cols_to_delete (int): Maximum number of columns to mask per row.

    Returns:
        pd.Series: Each value is the concatenation (by space) of the (possibly masked) columns.
    """
    result = pd.Series(index=df.index, dtype=object)
    # Filter out columns that should not be masked
    columns_to_mask = [col for col in columns if col not in not_mask_columns]
    mask_notna = df[columns_to_mask].notna().sum(axis=1) >= min_notna_cols_to_operate
    sampled_idx = df[mask_notna].sample(frac=sample_frac).index
    if not sampled_idx.any():
        return result
    # Apply masking to sampled rows
    result.loc[sampled_idx] = df.loc[sampled_idx].apply(lambda row: mask_row(row, columns, max_cols_to_delete, not_mask_columns), axis=1)
    return result

def augment_mask_words(series: pd.Series, sample_frac, max_words_to_delete, min_words_to_operate) -> pd.Series:
    """ Masks a random number of words in each string of the series.
    If the string has fewer words than min_words_to_operate, it is left unchanged.
    """
    result = pd.Series(index=series.index, dtype=object)
    len_words = series.str.split().str.len()
    mask_notna_min_words = series.notna() & (len_words >= min_words_to_operate)
    sampled_idx = series[mask_notna_min_words].sample(frac=sample_frac).index
    result.loc[sampled_idx] = series.loc[sampled_idx].apply(lambda x: mask_words(x, max_words_to_delete))
    return result

def shuffle_row(row, columns):
    """ Shuffles the values in the specified columns of a row.
    Returns a string with the shuffled values concatenated by space.
    """
    shuffled_values = [row[col] for col in columns if pd.notna(row[col])]
    if len(shuffled_values) > 1:
        random_choice = random.choice(shuffled_values[:-1]) 
        random.shuffle(shuffled_values)
        shuffled_values = [value for value in shuffled_values if value != random_choice] + [random_choice]
        return " ".join(shuffled_values).strip()
    else:
        raise ValueError("Cannot shuffle a row with less than 2 non-null values in the specified columns.")

def augment_shuffle_columns(df, columns, sample_frac):
    """ Shuffles the values in the specified columns for a random fraction of rows.
    Returns a Series with the shuffled values.
    """
    result = pd.Series(index=df.index, dtype=object)
    mask_notna = df[columns].notna().sum(axis=1) >= 2
    sampled_idx = df[mask_notna].sample(frac=sample_frac).index
    if not sampled_idx.any():
        return result
    result.loc[sampled_idx] = df.loc[sampled_idx].apply(lambda row: shuffle_row(row, columns), axis=1)
    return result

class FinetuningDataset():
    """ Class for loading and augmenting a dataset for fine-tuning.
    This class reads a dataset from a CSV file, applies text augmentations to the anchor and positive columns,
    and saves the augmented dataset to a specified output directory.
    The augmentations include keyboard mistakes and abbreviations."""
    def __init__(self, settings):
        # config
        self.settings = settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.settings["OUTPUT_DIR"], f"{self.settings['NAME']}_{timestamp}")
        self.logger = Logger(name=__class__.__name__, settings=settings).get()

        self.ANCHOR_NAME = ColCat.ANCHOR.value
        self.original_columns = {ColCat.ANCHOR: None, ColCat.POSITIVE: [], ColCat.NEGATIVE: []}
        self.columns = {ColCat.ANCHOR: None, ColCat.POSITIVE: [], ColCat.NEGATIVE: []}
        self.df = None

    def load_dataset(self) -> None:
        self.logger.info("Loading dataset: %s", self.settings["DATASET_PATH"])
        
        if self.settings["DELIMITER"]=='\t':
            df = pd.read_csv(self.settings["DATASET_PATH"], delimiter='\t', encoding='latin1', low_memory=False)
        else:
            df = pd.read_csv(self.settings["DATASET_PATH"], encoding='latin1', low_memory=False)
        df = df.map(remove_accents)
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
        self.logger.info("Dataset loaded. Rows: %d  Cols: %s",
                         len(df), list(df.columns))
        # detect columns
        # 1. anchor
        df[self.ANCHOR_NAME] = generate_queries(
            df, 
            self.settings["ANCHOR_COL"]
        )
        self.columns[ColCat.ANCHOR] = [self.ANCHOR_NAME]
        self.original_columns[ColCat.ANCHOR] = [self.ANCHOR_NAME]

        # 2. positive columns
        for pc_name, pc_columns  in self.settings["POSITIVE_COLS"].items():
            df[pc_name] = generate_queries(
                df, 
                pc_columns
            )
            self.columns[ColCat.POSITIVE].append(pc_name)
            self.original_columns[ColCat.POSITIVE].append(pc_name)

        # 3. negative columns
        for nc_name, nc_columns in self.settings["NEGATIVE_COLS"].items():
            df[nc_name] = generate_queries(
                df,
                nc_columns
            )
            self.columns[ColCat.NEGATIVE].append(nc_name)
            self.original_columns[ColCat.NEGATIVE].append(nc_name)

        self.logger.info(
            "Detected → anchor: '%s'  positives: %s  negatives: %s",
            self.ANCHOR_NAME, self.columns[ColCat.POSITIVE], self.columns[ColCat.NEGATIVE] or "—"
        )
        columns =  self.columns[ColCat.ANCHOR] + self.columns[ColCat.POSITIVE] + self.columns[ColCat.NEGATIVE]
        df = df.drop_duplicates(subset=columns)
        if isinstance(self.settings["NOTNULL_COLS"], list):
            df = df.dropna(subset=self.settings["NOTNULL_COLS"])
            self.logger.info("Dropped rows with NaN in columns: %s", self.settings["NOTNULL_COLS"])
        
        self.df = df

    def augment_dataset(self) -> None:
        """ Augments the dataset by applying text augmentation to the anchor and positive columns.
        """
        self.logger.info("Applying text augmentation…")
        df = self.df

        # Keyboard Augmentation
        aug = nac.KeyboardAug(
            lang=LANGUAGE,
            aug_char_max=self.settings["KEYBOARD_MAX_CHARS_TO_CHANGE"],
            aug_word_max=self.settings["KEYBOARD_MAX_WORDS_TO_CHANGE"],
        )

        # Abbreviations
        with open(self.settings["ABBREVIATIONS_PATH"], 'r', encoding='utf-8') as f:
            abbreviations_json = json.load(f)

        ####### AUGMENT WITH KEYBOARD MISTAKES AND ABBREVIATIONS #######    
        ### Augment anchor and positive columns ###
        # Augment positive and anchor (as positive) columns
        original_columns = self.original_columns
        for cat, col_names in original_columns.items():
            for col in col_names:
                # Augment keyboard mistakes
                self.logger.info("Augmenting column '%s' with keyboard mistakes and abbreviations", col)
                df[f"{col}_keyboard"] = augment_keyboard_mistakes(
                    df[col],
                    self.settings["AUGMENTATION_KEYBOARD_MISTAKE_PROBABILITY"],
                    aug
                )
                self.columns[cat].append(f"{col}_keyboard")

                # Augment abbreviations
                self.logger.info("Augmenting column '%s' with abbreviations", col)
                df[f"{col}_abbreviations"] = augment_abbreviations(
                    df[col],
                    abbreviations_json,
                    self.settings["AUGMENTATION_ABBREVIATIONS_PROBABILITY"],
                    self.settings["KEYBOARD_MAX_WORDS_TO_CHANGE"]
                )
                self.columns[cat].append(f"{col}_abbreviations")

                # Augment shuffling
                self.logger.info("Augmenting column '%s' with shuffling", col)
                if col in self.settings["SHUFFLE_COLUMNS"]:
                    columns_to_shuffle = self.settings[f"{cat.name}_COLS"][col]
                    df[f"{col}_shuffled"] = augment_shuffle_columns(
                        df,
                        columns_to_shuffle,
                        self.settings["AUGMENTATION_SHUFFLE_PROBABILITY"]
                    )
                    self.columns[cat].append(f"{col}_shuffled")
                
                # Augment masking
                self.logger.info("Augmenting column '%s' with masking", col)
                if col in self.settings["MASKING_COLUMNS"].keys():
                    if self.settings["MASKING_COLUMNS"][col]["mask_type"] == "words":
                        df[f"{col}_masked"] = augment_mask_words(
                            df[col],
                            self.settings["AUGMENTATION_MASKING_PROBABILITY"],
                            self.settings["MASKING_MAX_WORDS_TO_DELETE"],
                            self.settings["MASKING_MIN_WORDS_TO_OPERATE"]
                        )
                        self.columns[cat].append(f"{col}_masked")

                    elif self.settings["MASKING_COLUMNS"][col]["mask_type"] == "columns":
                        columns_to_mask = self.settings[f"{cat.name}_COLS"][col]
                        df[f"{col}_masked"] = augment_mask_columns(
                            df,
                            columns_to_mask,
                            self.settings["AUGMENTATION_MASKING_PROBABILITY"],
                            self.settings["MASKING_COLUMNS"][col]["max_cols_to_delete"],
                            self.settings["MASKING_COLUMNS"][col]["min_notna_cols_to_operate"],
                            self.settings["MASKING_COLUMNS"][col].get("not_mask_columns", [])
                        )
                        self.columns[cat].append(f"{col}_masked")
                    
                    else:
                        self.logger.warning("Unknown mask type for column '%s': %s. Skipping masking.", col, self.settings["MASKING_COLUMNS"][col]["mask_type"])
                        continue

        df = df.drop_duplicates()

        self.logger.info("Augmentation complete. New columns added: %s",
                         [col 
                          for col in self.columns[ColCat.ANCHOR] + self.columns[ColCat.POSITIVE] + self.columns[ColCat.NEGATIVE]
                          if col not in self.original_columns[ColCat.ANCHOR] + self.original_columns[ColCat.POSITIVE] + self.original_columns[ColCat.NEGATIVE]])

        # Save the augmented DataFrame
        self.df = df
        return df

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) != 2:
        print("Usage: python3 src/train/preprocessing/finetuning_dataset.py path/al/train_config.yaml")
        sys.exit(1)

    train_config_path = sys.argv[1]
    data = FinetuningDataset(train_config_path)
    data.load_dataset()
    dirty_df = data.augment_dataset()
    # Save the augmented DataFrame
    output_path = data.out_dir
    os.makedirs(output_path, exist_ok=True)
    dirty_df.to_csv(os.path.join(output_path, "dirty_dataset.csv"), index=False)
    shutil.copy(train_config_path, os.path.join(output_path, "dataset_config.yaml"))
    print(dirty_df)