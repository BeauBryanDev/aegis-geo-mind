
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
 
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
 
 
DATASET_ID = "daven3/geosignal"

GEO_TYPES = {"geo", "geoqa", "self"}
 
NOISY_CATEGORY_PREFIXES = (
    "gso.wikipedia.",
    "gso.wordnet.",
)
 
MIN_ANSWER_LEN = 15  
MAX_ANSWER_LEN_DEFAULT = 2500  
 
NOISY_INSTRUCTION_PATTERNS = (
    r"related paper",
)
 
BROKEN_OUTPUT_PATTERNS = (
    r"\b(?:is|means) that \d+\b",
)
 
NOISY_OUTPUT_PATTERNS = (
    r"no corresponding information",
)
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Explorar y filtrar daven3/geosignal")
    
    p.add_argument("--out", type=str, default="./geosignal_clean",
                    help="Carpeta de salida")
    p.add_argument("--hf-token", type=str, default=None,
                    help="Token de Hugging Face ||  la env var HF_TOKEN")
    p.add_argument("--keep-wikipedia-wordnet", action="store_true",
                    help="No descartar las categorias gso.wikipedia.* / gso.wordnet.*")
    p.add_argument("--val-split", type=float, default=0.02,
                    help="Fraccion para validacion (default 0.02)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system-prompt", type=str,
                    default="You are an expert geologist specializing in Earth Sciences.",
                    help="System message aplicado a TODAS las filas exportadas. ")
    p.add_argument("--max-output-chars", type=int, default=MAX_ANSWER_LEN_DEFAULT,
                    help=f"Descarta filas con output mas largo que esto "
                          f"(default {MAX_ANSWER_LEN_DEFAULT}, ~p99 observado). "
                          f"Pasar 0 para no aplicar tope.")
    p.add_argument("--keep-related-paper", action="store_true",
                    help="No descartar filas cuya instruccion pide bibliografia "
                          "de un mineral (patron 'related paper')")
    p.add_argument("--keep-broken-artifacts", action="store_true",
                    help="No descartar filas con el artefacto de plantilla "
                          "roto ('is/means that <numero>')")
    p.add_argument("--keep-noinfo-placeholders", action="store_true",
                    help="No descartar filas cuyo output es un placeholder "
                          "de campo vacio ('No Corresponding Information')")
    
    return p.parse_args()
 
 
def authenticate(token: str | None):
    
    token = token or os.environ.get("HF_TOKEN")
    
    if not token:
        
        print("ADVERTENCIA: no se encontro HF_TOKEN. Si el dataset requiere "
              "autenticacion la descarga puede fallar. Segui adelante igual.",
              file=sys.stderr)
        return
    
    login(token=token)
 
 
def load_raw():
    
    print(f"Descargando {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    print(f"Filas totales: {len(df)}")
    
    return df
 
 
def explore(df: pd.DataFrame) -> str:
    
    lines = []
    lines.append(f"Total filas: {len(df)}")
    lines.append("")
 
    lines.append("Distribucion por 'type' ")
    
    type_counts = df["type"].value_counts(dropna=False)
    
    for t, c in type_counts.items():
        
        lines.append(f"  {t!s:15s} {c:6d}  ({100*c/len(df):.1f}%)")
        
    lines.append("")
 
    lines.append("Top 30 prefijo  'category' *antes de '.'* ")
    
    cat_prefixes = (
        df["category"].dropna().astype(str)
        .str.split(".").str[0]
        .value_counts()
        .head(30)
    )
    for prefix, c in cat_prefixes.items():
        
        lines.append(f"  {prefix!s:20s} {c:6d}")
        
    lines.append("")
 
    lines.append("  Longitud de 'output' *chars* ")
    out_len = df["output"].fillna("").str.len()
    lines.append(f"  min={out_len.min()}  mediana={out_len.median():.0f}  "
                  f"media={out_len.mean():.0f}  max={out_len.max()}")
    lines.append(f"  filas con output vacio : (<{MIN_ANSWER_LEN} chars): "
                  f"{(out_len < MIN_ANSWER_LEN).sum()}")
    lines.append("")
 
    report = "\n".join(lines)
    
    print(report)
    
    return report
 
 
def is_noisy_category(category: str, keep_wikipedia_wordnet: bool) -> bool:
    
    if keep_wikipedia_wordnet:
        return False
    
    if not isinstance(category, str):
        return False
    
    return category.startswith(NOISY_CATEGORY_PREFIXES)
 
 
def clean_text(x) -> str:
    
    if not isinstance(x, str):
        return ""
    
    x = x.strip()
    # Colapsar espacios/saltos de linea repetidos que aparecen en el CSV fuente
    x = re.sub(r"\n{3,}", "\n\n", x)
    x = re.sub(r"[ \t]{2,}", " ", x)
    
    return x
 
 
def filter_and_clean(df: pd.DataFrame, 
                       keep_wikipedia_wordnet: bool,
                       max_output_chars: int = MAX_ANSWER_LEN_DEFAULT,
                       drop_related_paper: bool = True,
                       drop_broken_artifacts: bool = True,
                       drop_noinfo_placeholders: bool = True
                       ) -> pd.DataFrame:
    
    before = len(df)
 
    df = df[df["type"].isin(GEO_TYPES)].copy()
    print(f"Luego de filtrar por type in {GEO_TYPES}: {len(df)} / {before}")
 
    #  Descartar   (wikipedia/wordnet genericos)
    mask_noisy = df["category"].apply(
        
        lambda c: is_noisy_category(c, keep_wikipedia_wordnet)
    )
    n_noisy = mask_noisy.sum()
    df = df[~mask_noisy].copy()
    
    print(f"Descartadas {n_noisy} filas de categorias wikipedia/wordnet genericas")
 
    # Descartar filas tipo "related paper" (listas de bibliografia sin
    # valor conversacional, detectadas :: type == 'geo' / metaearth.rruff.qa)
    if drop_related_paper:
        
        mask_refpaper = df["instruction"].str.contains(
            "|".join(NOISY_INSTRUCTION_PATTERNS), case=False, na=False, regex=True
        )
        n_refpaper = mask_refpaper.sum()
        df = df[~mask_refpaper].copy()
        
        print(f"Descartadas {n_refpaper} filas de patron 'related paper' (bibliografia)")
 
    #  Descartar artefactos de plantilla rota ("is/means that <numero>")
    if drop_broken_artifacts:
        
        mask_broken = df["output"].str.contains(
            "|".join(BROKEN_OUTPUT_PATTERNS), case=False, na=False, regex=True
        )
        
        n_broken = mask_broken.sum()
        df = df[~mask_broken].copy()
        
        print(f"Descartadas {n_broken} filas con artefacto de plantilla roto")
 
    # Descartar placeholder "sin informacion" (campo vacio en el
    # catalogo de minerales convertido en pseudo-respuesta)
    if drop_noinfo_placeholders:
        
        mask_noinfo = df["output"].str.contains(
            "|".join(NOISY_OUTPUT_PATTERNS), case=False, na=False, regex=True
        )
        n_noinfo = mask_noinfo.sum()
        df = df[~mask_noinfo].copy()
        
        print(f"Descartadas {n_noinfo} filas con placeholder de 'sin informacion'")
 
    #  Limpiar texto
    for col in ("instruction", "input", "output"):
        df[col] = df[col].apply(clean_text)
 
    # Descartar respuestas vacias o demasiado cortas para ser utiles
    df = df[df["output"].str.len() >= MIN_ANSWER_LEN].copy()
    print(f"Tras descartar outputs < {MIN_ANSWER_LEN} chars: {len(df)}")
 
    # Descartar outliers de longitud (colas largas: listas de referencias,
    # texto scrapeado sin cortar, etc. Se descartan enteras, no  truncar.
    if max_output_chars and max_output_chars > 0:
        
        before_len_filter = len(df)
        
        df = df[df["output"].str.len() <= max_output_chars].copy()
        
        print(f"Descartadas {before_len_filter - len(df)} filas con output > "
              f"{max_output_chars} chars")
 
    #  Deduplicar por instruction + input + output 
    dedup_key = (
        df["instruction"].fillna("") + "||"
            + df["input"].fillna("") + "||"
            + df["output"].fillna("")
            )
    
    before_dedup = len(df)
    
    df = df[~dedup_key.duplicated()].copy()
    
    print(f"Deduplicadas {before_dedup - len(df)} filas repetidas -> {len(df)} finales")
 
    return df.reset_index(drop=True)
 
 
def _clean_meta_value(v):
    """Evita volcar NaN/floats indeseables al JSON ."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    
    return v
 
 
def to_chat_format(row, system_prompt: str = "") -> dict:
    """Convierte instruction/input/output a formato ChatML (system/user/assistant)
    Correcto para QLoRA SFT con Qwen2.5-7B-Instruct . """
    
    user_content = row["instruction"]
    
    if row.get("input"):
    
        user_content = f"{user_content}\n\n{row['input']}"
    
    messages = []
    
    if system_prompt:
    
        messages.append({"role": "system", "content": system_prompt})
        
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": row["output"]})
    
    return {
        "messages": messages,
        "meta": {
            "type": _clean_meta_value(row.get("type")),
            "category": _clean_meta_value(row.get("category")),
            "source": DATASET_ID,
        },
    }
 
 
def write_jsonl(df: pd.DataFrame, path: Path, system_prompt: str = ""):
    
    with open(path, "w", encoding="utf-8") as f:
        
        for _, row in df.iterrows():
            
            f.write(json.dumps(
                to_chat_format(row, system_prompt),
                                ensure_ascii=False
                                ) + "\n")
 
 
def main():
    
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    authenticate(args.hf_token)
 
    df_raw = load_raw()
    df_raw.to_parquet(out_dir / "geosignal_raw.parquet", index=False)
 
    report = explore(df_raw)
    (out_dir / "exploration_report.txt").write_text(report, encoding="utf-8")
 
    df_clean = filter_and_clean(
        df_raw,
        args.keep_wikipedia_wordnet,
        max_output_chars=args.max_output_chars,
        drop_related_paper=not args.keep_related_paper,
        drop_broken_artifacts=not args.keep_broken_artifacts,
        drop_noinfo_placeholders=not args.keep_noinfo_placeholders,
    )
 
    # Split train/val
    df_clean = df_clean.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(df_clean) * args.val_split))
    df_val = df_clean.iloc[:n_val]
    df_train = df_clean.iloc[n_val:]
 
    write_jsonl(df_train, out_dir / "geosignal_geo_train.jsonl", args.system_prompt)
    write_jsonl(df_val, out_dir / "geosignal_geo_val.jsonl", args.system_prompt)
    
    df_clean.to_parquet(out_dir / "geosignal_geo_clean.parquet", index=False)
 
    print("\n--- Resumen final ---")
    print(f"Crudo:        {len(df_raw)} filas")
    print(f"Limpio total: {len(df_clean)} filas")
    print(f"  train: {len(df_train)}")
    print(f"  val:   {len(df_val)}")
    print(f"\nArchivos escritos en: {out_dir.resolve()}")
    print("  - geosignal_raw.parquet          (dataset crudo completo)")
    print("  - exploration_report.txt         (conteos por type/category)")
    print("  - geosignal_geo_clean.parquet     (filtrado, sin split, formato tabular)")
    print("  - geosignal_geo_train.jsonl       (formato chat, listo para SFT)")
    print("  - geosignal_geo_val.jsonl         (formato chat, validacion)")
 
 
if __name__ == "__main__":
    
    main()