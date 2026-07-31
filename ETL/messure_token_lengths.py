
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
 
import pandas as pd
from transformers import AutoTokenizer
 
 
CANDIDATE_MAX_LENGTHS = [512, 1024, 1536, 2048, 3072, 4096]
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Medir longitud en tokens del corpus")
    p.add_argument("--input", type=str, nargs="+", required=True,
                    help="Uno o mas archivos JSONL en formato ChatML")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Modelo/tokenizer a usar (baja solo el tokenizer)")
    p.add_argument("--by-source", action="store_true",
                    help="Reportar distribucion tambien por archivo de origen. "
                          "Requiere que los JSONL originales tengan info de "
                          "fuente, o pasar --input archivo por archivo.")
    return p.parse_args()
 
 
def load_examples(paths: list[str]) -> list[tuple[str, dict]]:
    """Devuelve lista de (source_file, example)."""
    out: list[tuple[str, dict]] = []
    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            print(f"ERROR: no existe {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    out.append((p.name, ex))
                except json.JSONDecodeError:
                    continue
    print(f"Cargados {len(out)} ejemplos de {len(paths)} archivo(s)")
    return out
 
 
def render_and_count(examples: list[tuple[str, dict]], tokenizer) -> pd.DataFrame:
    """Aplica el chat template real de Qwen y cuenta tokens por ejemplo.
    Tambien mide la porcion assistant sola (util para saber cuanto contexto
    queda para el input si acotamos max_seq_length)."""
    rows = []
    for i, (src, ex) in enumerate(examples):
        messages = ex["messages"]
        # Renderizado completo tal como lo vera el modelo en training
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
 
        # Renderizado solo hasta antes del assistant (para saber cuanto pesa
        # el prompt vs la respuesta)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
 
        rows.append({
            "source": src,
            "total_tokens": len(full_ids),
            "prompt_tokens": len(prompt_ids),
            "assistant_tokens": len(full_ids) - len(prompt_ids),
        })
        if (i + 1) % 5000 == 0:
            print(f"  procesados {i+1}/{len(examples)}")
    return pd.DataFrame(rows)
 
 
def report_distribution(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"Distribucion de tokens - {label} ({len(df)} ejemplos)")
    print(f"{'='*60}")
 
    for col in ("total_tokens", "prompt_tokens", "assistant_tokens"):
        s = df[col]
        print(f"\n  {col}:")
        print(f"    min={s.min()}  mediana={s.median():.0f}  "
              f"media={s.mean():.0f}  max={s.max()}")
        for pct in (50, 75, 90, 95, 99, 99.5, 99.9):
            print(f"    p{pct}: {s.quantile(pct/100):.0f}")
 
    print("\n  Cobertura por max_seq_length candidato (sobre total_tokens):")
    n = len(df)
    for L in CANDIDATE_MAX_LENGTHS:
        n_ok = (df["total_tokens"] <= L).sum()
        n_trunc = n - n_ok
        pct_cov = 100 * n_ok / n
        print(f"    max_seq_length={L:5d}  cubre {n_ok:6d}/{n} "
              f"({pct_cov:.2f}%)  truncados: {n_trunc}")
 
 
def main():
    args = parse_args()
 
    print(f"Cargando tokenizer de {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}, "
          f"model_max_length: {tokenizer.model_max_length}")
 
    examples = load_examples(args.input)
    df = render_and_count(examples, tokenizer)
 
    report_distribution(df, "TODO EL CORPUS")
 
    if args.by_source:
        for src, sub in df.groupby("source"):
            report_distribution(sub, f"source = {src}")
 
    # Recomendacion basada en datos: sugerir el menor candidato que cubra >= 99%
    print("\n" + "="*60)
    print("RECOMENDACION")
    print("="*60)
    for L in CANDIDATE_MAX_LENGTHS:
        pct = 100 * (df["total_tokens"] <= L).sum() / len(df)
        if pct >= 99.0:
            print(f"  El menor umbral que cubre >=99% es max_seq_length={L} "
                  f"({pct:.2f}% cubierto).")
            print(f"  Con esto se truncan {(df['total_tokens'] > L).sum()} ejemplos.")
            break
    else:
        print(f"  Ninguno de los candidatos {CANDIDATE_MAX_LENGTHS} cubre >=99%. "
              f"Considerar subir el candidato mas alto.")
 
 
 
if __name__ == "__main__":
    
    
    main()
 