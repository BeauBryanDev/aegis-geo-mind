
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
 
 
def parse_args():
    
    p = argparse.ArgumentParser(description="Unir datasets ChatML para Aegis-Geo-Mind")
    
    p.add_argument("--input", type=str, nargs="+", required=True,
                    help="Uno o mas archivos .jsonl en formato ChatML")
    p.add_argument("--out", type=str, default="./aegis_geo_mind_corpus")
    p.add_argument("--val-split", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-assistant-len", type=int, default=10,
                    help="Descarta ejemplos  respuesta del assistant "
                          "tenga menos caracteres que esto")
    p.add_argument("--oversample", type=str, nargs="*", default=None,
                    help="Pares 'nombre_de_archivo:factor' para repetir una "
                          "fuente N veces DENTRO DE TRAIN ")
    
    return p.parse_args()
 
 
def validate_example(ex: dict, source: str, line_no: int) -> str | None:
    """Devuelve un mensaje de error si el ejemplo es invalido, o None si esta bien."""
    if "messages" not in ex or not isinstance(ex["messages"], list):
        
        return f"{source}:{line_no} -> falta 'messages' o no es una lista"
 
    roles = [m.get("role") for m in ex["messages"]]
    
    if "user" not in roles or "assistant" not in roles:
        
        return f"{source}:{line_no} -> falta role 'user' o 'assistant'"
 
    for m in ex["messages"]:
        
        if m.get("role") not in ("system", "user", "assistant"):
            
            return f"{source}:{line_no} -> role desconocido: {m.get('role')!r}"
        
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            
            return f"{source}:{line_no} -> content vacio o no es string (role={m.get('role')})"
 
    # El ultimo mensaje debe ser del assistant SFT format. 
    if ex["messages"][-1]["role"] != "assistant":
        
        return f"{source}:{line_no} -> el ultimo mensaje no es del assistant"
 
    return None
 
 
def dedup_key(ex: dict) -> str:
    """Hash sobre el contenido de user + assistant."""
    user_parts = [m["content"] for m in ex["messages"] if m["role"] == "user"]
    assistant_parts = [m["content"] for m in ex["messages"] if m["role"] == "assistant"]
    
    raw = "||".join(user_parts) + "###" + "||".join(assistant_parts)
    
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
 
 
def load_and_validate(path: Path) -> list[dict]:
    
    examples = []
    n_invalid = 0
    
    with open(path, encoding="utf-8") as f:
        
        for i, line in enumerate(f, start=1):
            
            line = line.strip()
            
            if not line:
                continue
            
            try:
                ex = json.loads(line)
                
            except json.JSONDecodeError as e:
                
                print(f"  [{path.name}:{i}] JSON invalido, se descarta: {e}", file=sys.stderr)
                n_invalid += 1
                continue
            
            err = validate_example(ex, path.name, i)
            
            if err:
                print(f"  [SKIP] {err}", file=sys.stderr)
                n_invalid += 1
                continue
            
            ex["_source_file"] = path.name
            
            examples.append(ex)
            
    print(f"{path.name}: {len(examples)} ejemplos validos, {n_invalid} descartados")
    
    return examples
 
 
def main():
    
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed)
 
    all_examples: list[dict] = []
    
    for input_path in args.input:
        
        p = Path(input_path)
        
        if not p.exists():
            
            print(f"ERROR: no existe {p}", file=sys.stderr)
            
            sys.exit(1)
            
        all_examples.extend(load_and_validate(p))
 
    print(f"\nTotal cargado  : {len(all_examples)}")
 
    # Filtrar respuestas demasiado cortas
    before = len(all_examples)
    
    all_examples = [
        ex for ex in all_examples
        if len(ex["messages"][-1]["content"]) >= args.min_assistant_len
    ]
    print(f"Descartados por respuesta corta {args.min_assistant_len} chars: "
          f"{before - len(all_examples)}")
 
    # Deduplicar 
    seen = set()
    deduped = []
    
    n_dup = 0
    
    for ex in all_examples:
        
        k = dedup_key(ex)
        
        if k in seen: 
            
            n_dup += 1
            continue
        
        seen.add(k)
        deduped.append(ex)
        
    print(f"Duplicados cruzados entre fuentes eliminados: {n_dup}")
    print(f"Total final (unico, antes de oversampling): {len(deduped)}")
 
    # Reporte de composicion por fuente (sobre el pool unico, sin oversample)
    print(" Composicion por archivo de origen (sin oversampling) ")
    counts: dict[str, int] = {}
    
    for ex in deduped:
        
        counts[ex["_source_file"]] = counts.get(ex["_source_file"], 0) + 1
        
    for src, c in sorted(counts.items(), key=lambda x: -x[1]):
        
        print(f"  {src:40s} {c:6d}  ({100*c/len(deduped):.1f}%)")
 
    # Shuffle y split SOBRE EL POOL UNICO ) , evitar  leakage train/val.
    random.shuffle(deduped)
    
    n_val = max(1, int(len(deduped) * args.val_split))
    val_set = deduped[:n_val]
    train_set = deduped[n_val:]
 
    # Oversampling: parsear "archivo:factor" y aplicar SOLO sobre train_set
    oversample_map: dict[str, int] = {}
    
    if args.oversample:
        
        for spec in args.oversample:
            
            if ":" not in spec:
                print(f"ERROR: --oversample espera 'archivo:factor', recibido: {spec!r}",
                      file=sys.stderr)
                
                sys.exit(1)
            fname, factor_str = spec.rsplit(":", 1)
            
            try:
                
                factor = int(factor_str)
                
            except ValueError:
                
                print(f"ERROR: factor de oversample invalido en {spec!r}", file=sys.stderr)
                sys.exit(1)
                
            oversample_map[fname] = factor
 
    if oversample_map:
        
        print("Aplicando oversampling (solo dentro de train ")
        extra: list[dict] = []
        
        for src_name, factor in oversample_map.items():
            
            matching = [ex for ex in train_set if ex["_source_file"] == src_name]
            
            if not matching:
                
                print(f"  ADVERTENCIA: '{src_name}' no aparece en train_set, "
                      f"revisa el nombre de archivo", file=sys.stderr)
                continue
            
            n_repeats = factor - 1 
            
            if n_repeats > 0:
                
                extra.extend(matching * n_repeats)
                
            print(f"  {src_name}: {len(matching)} originales x{factor} "
                  f"-> {len(matching) * factor} en train")
            
        train_set = train_set + extra
        
        random.shuffle(train_set)
 
    # Reporte final de composicion de train (post oversampling)
    print("\n  Composicion final de TRAIN  ")
    
    train_counts: dict[str, int] = {}
    
    for ex in train_set:
        
        train_counts[ex["_source_file"]] = train_counts.get(ex["_source_file"], 0) + 1
        
    for src, c in sorted(train_counts.items(), key=lambda x: -x[1]):
        
        print(f"  {src:40s} {c:6d}  ({100*c/len(train_set):.1f}%)")
 
    def strip_internal_fields(ex: dict) -> dict:
        
        return {"messages": ex["messages"]}
 
    train_path = out_dir / "aegis_geo_mind_train.jsonl"
    val_path = out_dir / "aegis_geo_mind_val.jsonl"
 
    with open(train_path, "w", encoding="utf-8") as f:
        
        for ex in train_set:
            
            f.write(json.dumps(strip_internal_fields(ex), ensure_ascii=False) + "\n")
 
    with open(val_path, "w", encoding="utf-8") as f:
        
        for ex in val_set:
            
            f.write(json.dumps(strip_internal_fields(ex), ensure_ascii=False) + "\n")
 
    print(f" Escrito:  {train_path} {len(train_set)} ejemplosb \n"
          f"  {val_path}  ({len(val_set)} ejemplos ")
 
 
 
if __name__ == "__main__":
    
    
    main()
 