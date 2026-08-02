import unicodedata
import re
import pandas as pd
import requests

def matcher_request(query: str, BASE: str, PATH: str, AUTH_TOKEN: str):
    if BASE is not None and PATH is not None and AUTH_TOKEN is not None:
        r = requests.post(
        BASE + PATH,
        headers={"Authorization": AUTH_TOKEN, "Content-Type": "application/json"},
        json={"text": query},
        timeout=10,
        )
        return r.status_code, r
    else:
        return None, None

def preprocess_street_with_matcher(query: str, BASE: str, PATH: str, AUTH_TOKEN: str):
    status_code, response = matcher_request(query, BASE, PATH, AUTH_TOKEN)
    if status_code == 200:
        r_json = response.json()["list"]
        parts = []
        for item in r_json:
            if item["qualifier"] in ["STREET", "NUMBER"]:
                parts.append(item["value"].lower())
        return " ".join(parts)
    else:
        return ""

def split_query_and_height_with_matcher(query: str, BASE: str, PATH: str, AUTH_TOKEN: str):
    status_code, response = matcher_request(query, BASE, PATH, AUTH_TOKEN)
    if status_code == 200:
        height = None
        parts_query = []
        r_json = response.json()["list"]

        # Construimos la query sin altura (solo las partes "STREET")
        for item in r_json:
            if item["qualifier"] == "STREET":
                parts_query.append(item["value"].lower())
            if item["qualifier"] == "NUMBER":
                height = int(item["value"])

        # Armamos la query sin altura
        query_wo_height = " ".join(parts_query)

        # --- NUEVA LÓGICA: reordenar ---
        if height is not None:
            # Buscamos la altura en el texto original (convertido a str por seguridad)
            height_str = str(height)
            # Dividimos en dos: antes y después de la altura
            if height_str in query:
                before, after = query.split(height_str, 1)
                after = after.strip()
                if after:  # si hay algo después de la altura
                    # movemos esa parte al principio
                    query_wo_height = f"{after} {query_wo_height}".strip()

        return query_wo_height, height

    else:
        return None, None


def remove_accents(text):
    if isinstance(text, str):
        text = text.replace("ñ", "__enie_min__").replace("Ñ", "__enie_may__")

        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

        text = text.replace("__enie_min__", "ñ").replace("__enie_may__", "Ñ")
        return text

    return text

def remove_entrecalles(query):
    # Eliminar los patrones "E/ número Y número", "entre número y número", o "e número y número"
    if not isinstance(query, str):
        return ""

    # Elimina patrones de intersección entre calles (entre 2 calles)
    query = re.sub(r"\b(e\s\d+\s\d+|e\s\d+\sy\s\d+|entre\s\d+\sy\s\d+|e/\s\d+\s\d+|e/\s\d+\sy\s\d+|entre\s\d+\s/\s\d+|e\d+\sy\s\d+)\b", "", query)
    query = re.sub(r"\b(e.\s\d+\s\d+|e.\s\d+\sy\s\d+|e./\s\d+\s\d+|e./\s\d+\sy\s\d+|e.\d+\sy\s\d+)\b", "", query)
    # Si aparece "entre ... y ..." pero NO "entre rios", eliminar todo desde "entre" hasta la primera palabra después de "y"
    if re.search(r"\bentre\s+.+?\s+y\s+\S+\b", query) and not re.search(r"\bentre\s+rios\b", query, re.IGNORECASE):
        query = re.sub(r"\bentre\s+.+?\s+y\s+\S+\b", "", query).strip()
    return query

import re

def preprocess_query(query):
    if not isinstance(query, str):
        return ""
    
    # Convert to lowercase
    query = query.lower()
    
    # Remove accents
    query = remove_accents(query)
    
    # Remove "entre calles" patterns
    query = remove_entrecalles(query)
    
    # Normalize spaces
    query = re.sub(r'\s+', ' ', query).strip()

    # Replace full words
    query = re.sub(r'\bnorte\b', 'n', query)
    query = re.sub(r'\bsur\b', 's', query)
    # "caba" -> "capital federal" para subir la similaridad
    query = re.sub(r'\bcaba\b', 'capital federal', query)

    # Phonetic normalizations
    query = re.sub(r'z', 's', query)
    query = re.sub(r'y', 'i', query)
    # Reemplazar "c" por "s" cuando va seguida de "e" o "i" (ce/ci -> se/si)
    query = re.sub(r'c(?=[ei])', 's', query)

    return query


def generate_queries(df, columns, unique_cols=None):
    """
    Concatena columnas en un string por fila.

    Regla de deduplicación (a nivel valor completo, NO por palabra):
    - Si dos (o más) columnas listadas en `unique_cols` tienen el MISMO valor,
      solo se conserva la PRIMERA aparición de ese valor entre esas columnas.
    - Si una columna NO está en `unique_cols`, nunca se deduplica contra otras (se incluye siempre).
    - Se respeta el orden de `columns`.

    Ejemplos:
      columns = ["localidad","municipio"]
      unique_cols = ["localidad","municipio"]

        localidad="la matanza", municipio="la matanza" → "la matanza"
        localidad="la matanza", municipio="la boca"    → "la matanza la boca"

      unique_cols = ["localidad"] (municipio fuera) y ambos = "la matanza" → "la matanza la matanza"
    """

    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        return pd.Series([""] * len(df), index=df.index)

    # Validación mínima para evitar fallos silenciosos
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas inexistentes: {missing}")

    unique_set = set(unique_cols or [])
    data = df[columns].fillna("").astype(str)

    def _row_to_query(row):
        seen_unique_values = set()  # valores ya vistos SOLO de columnas en unique_cols
        parts = []
        for col in columns:
            val = row[col]
            if not val:
                continue

            if col in unique_set:
                # deduplicamos si el valor ya apareció en alguna col de unique_cols
                if val in seen_unique_values:
                    continue
                seen_unique_values.add(val)

            # columnas fuera de unique_cols nunca se deduplican
            parts.append(val)

        return " ".join(parts)

    queries = data.apply(_row_to_query, axis=1)
    queries = queries.str.replace(",", "", regex=False)
    queries = queries.replace(r"\s+", " ", regex=True)
    queries = queries.apply(lambda x: preprocess_query(x))
    return queries



def extract_height(address):
    # Initial text cleaning
    address = address.lower()

    # Remove intersection patterns between streets (between 2 streets)
    address = re.sub(r"\b(e\s\d+\s\d+|e\s\d+\sy\s\d+|entre\s\d+\sy\s\d+|e/\s\d+\s\d+|e/\s\d+\sy\s\d+|entre\s\d+\s/\s\d+|e\d+\sy\s\d+)\b", "", address)
    address = re.sub(r"\b(e\.\s\d+\s\d+|e\.\s\d+\sy\s\d+|e\./\s\d+\s\d+|e\./\s\d+\sy\s\d+|e\.\d+\sy\s\d+)\b", "", address)
    # If "entre ... y ..." appears but NOT "entre rios", remove everything from "entre" up to the first word after "y"
    if re.search(r"\bentre\s+.+?\s+y\s+\S+\b", address) and not re.search(r"\bentre\s+rios\b", address, re.IGNORECASE):
        address = re.sub(r"\bentre\s+.+?\s+y\s+\S+\b", "", address).strip()

    # Define common patterns for floors and apartments
    # Nota: el marcador ordinal (º/°) es OBLIGATORIO en "[1-9](º|°)"; si fuera opcional
    # cualquier dígito suelto 1-9 (p. ej. la altura "4" en "3 de febrero 4") se descartaría
    # como si fuera un piso cuando el nombre de calle también tiene un número.
    floor_apartment_patterns = r"\b(\d+[a-zA-Z]|[a-zA-Z]\d+|PB|piso|dpto|departamento|Depto|°|[1-9](º|°))\b"

    # Tokenize the address and find numeric candidates
    tokens = address.split()
    numeric_tokens = [token for token in tokens if token.isdigit()]

    # If there are no numeric tokens, return None
    if not numeric_tokens:
        return None

    # Exclude tokens that match floor/apartment patterns
    clean_tokens = []
    for token in numeric_tokens:
        # Find the position of the number in the original address
        match = re.search(rf"{token} ([a-zA-Z])\b", address)
        
        # If the number is followed by a letter, exclude it
        if match and len(numeric_tokens) > 1:
            continue
        
        # If it does not match floor/apartment patterns, include it
        if not re.search(floor_apartment_patterns, token) or len(numeric_tokens) == 1:
            clean_tokens.append(token)

    # Handle ambiguous cases
    for token in reversed(clean_tokens):
        token_index = tokens.index(token)
        
        # Check if the token is preceded by a likely street name
        if token_index > 0:
            preceding_token = tokens[token_index - 1]
        else:
            preceding_token = ""
        if len(tokens) > token_index + 1:
            next_token = tokens[token_index + 1]
        else:
            next_token = ""

        if preceding_token not in ("av", "avenida", "calle", "de", "y", "esq", "esquina", "tel", "telefono", "piso", "dpto", "depto", "departamento") and next_token not in ("y", "de"):
            return token  # Most likely the street number

    return None

def split_query_and_height(query):
    """
    Preprocess the query to extract the height (street number) and return the query without it.
    Returns a tuple: (query_reordered_without_height, height)
    """
    query = preprocess_query(query)
    height = extract_height(query)

    if not height:
        return query, None

    # Dividimos la query en la parte antes y después de la altura
    before, after = query.split(height, 1)
    # Rearmamos: lo que venía después de la altura va al principio
    query_wo_height = (after.strip() + " " + before.strip()).strip()

    query_wo_height = preprocess_query(query_wo_height)
    return query_wo_height, int(height)

def safe_concat(*args):
    return ' '.join([preprocess_query(str(x)) for x in args if pd.notna(x) and str(x).strip() != '']).strip()

def filter_df_by_index_level(df, values, level):
    """
    Filtra un DataFrame MultiIndex por valores en un nivel específico del índice.

    Args:
        df (pd.DataFrame): El DataFrame a filtrar (debe tener un MultiIndex).
        values (list, set o array-like): Los valores a conservar en ese nivel del índice.
        level (int or str): Nivel del índice por el que se va a filtrar (puede ser nombre o posición).

    Returns:
        pd.DataFrame: El DataFrame filtrado.
    """
    if not isinstance(df.index, pd.MultiIndex):
        if level != 0:
            raise ValueError("El DataFrame no tiene MultiIndex, solo se puede usar level=0")
        return df.loc[df.index.isin(values)]

    if level == 0 and values.shape[0] == 1:
        return df.xs(values.item(), level=0, drop_level=False)

    # MultiIndex
    index = df.index
    levels = index.nlevels
    slicers = [slice(None)] * levels

    if isinstance(level, str):
        level = index.names.index(level)

    slicers[level] = values
    idx = pd.IndexSlice[tuple(slicers)]
    return df.loc[idx]

def normalize_cp(cp: str, prefijo_to_pxl: dict) -> str:
    if cp is None:
        return cp

    cp_norm = (
        str(cp)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )

    return prefijo_to_pxl.get(cp_norm, cp_norm)

from typing import Tuple, Dict, Set

def cp_candidates(cp: str, prefijo_to_pxl: Dict[str, str], pxl_set: Set[str]) -> Tuple[str, ...]:
    if cp is None:
        return tuple()

    cp_norm = (
        str(cp).strip().upper().replace(" ", "").replace("-", "")
    )

    is_google = cp_norm in prefijo_to_pxl
    is_pxl = cp_norm in pxl_set

    if is_google and is_pxl:
        # Caso ambigüo: hacer dos búsquedas
        return (cp_norm, prefijo_to_pxl[cp_norm])

    if is_google:
        return (prefijo_to_pxl[cp_norm],)

    # No es google: buscás directo por lo que ingresó (asumiendo que tu sistema acepta PXL)
    return (cp_norm,)
