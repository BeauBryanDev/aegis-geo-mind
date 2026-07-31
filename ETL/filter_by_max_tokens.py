
import argparse
import json
import sys
from pathlib import Path
 
from transformers import AutoTokenizer
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out-dir", required=True,
                    help="Directorio LOCAL (recomendado /content/..., no Drive) "
                          "donde se escriben los archivos finales")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    return p.parse_args()
 
 
def count_tokens(messages, tokenizer) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
 
 
def filter_file(path: Path, tokenizer, max_len: int, label: str) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    n_invalid = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [SKIP] {path.name}:{i} JSON invalido: {e}", file=sys.stderr)
                n_invalid += 1
                continue
            n_tok = count_tokens(ex["messages"], tokenizer)
            if n_tok <= max_len:
                kept.append(ex)
            else:
                dropped.append({"n_tokens": n_tok, "example": ex})
 
    print(f"\n{label}: {len(kept)} conservados, {len(dropped)} descartados "
          f"(>{max_len} tokens), {n_invalid} invalidos")
    return kept, dropped
 
 
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Cargando tokenizer de {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
 
    train_kept, train_dropped = filter_file(
        Path(args.train), tokenizer, args.max_seq_length, "TRAIN"
    )
    val_kept, val_dropped = filter_file(
        Path(args.val), tokenizer, args.max_seq_length, "VAL"
    )
 
    train_out = out_dir / "aegis_geo_mind_train_final.jsonl"
    val_out = out_dir / "aegis_geo_mind_val_final.jsonl"
    dropped_out = out_dir / "dropped_examples.jsonl"
 
    with open(train_out, "w", encoding="utf-8") as f:
        for ex in train_kept:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
 
    with open(val_out, "w", encoding="utf-8") as f:
        for ex in val_kept:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
 
    # Guardar lo descartado para auditoria, por si despues quieren revisar
    with open(dropped_out, "w", encoding="utf-8") as f:
        for item in train_dropped + val_dropped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
 
    print(f"\n=== Resumen final ===")
    print(f"  train: {len(train_kept)} filas -> {train_out}")
    print(f"  val:   {len(val_kept)} filas -> {val_out}")
    print(f"  descartados (auditoria): {len(train_dropped) + len(val_dropped)} -> {dropped_out}")
    print(f"\n  max_seq_length definitivo para el SFTConfig: {args.max_seq_length}")
 
 
if __name__ == "__main__":
    
    
    main()
 