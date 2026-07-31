
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
 
 
SYSTEM_PROMPT = "You are an expert geologist specializing in Earth Sciences."
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--petroleum", required=True,
                    help="Ruta al petroleum_geology.json (JSON array o JSONL)")
    p.add_argument("--train", required=True,
                    help="Ruta al train JSONL actual")
    p.add_argument("--val", required=True,
                    help="Ruta al val JSONL (solo para deduplicar contra el, NO se modifica)")
    p.add_argument("--out", required=True,
                    help="Ruta de salida para el train ampliado (puede ser la misma que --train)")
    p.add_argument("--factor", type=int, default=5,
                    help="Factor de oversampling para petroleum (default 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize-system", action="store_true", default=True,
                    help="Reemplazar el system prompt de petroleum por el estandar del corpus")
    p.add_argument("--no-backup", action="store_true",
                    help="No hacer backup del train original antes de sobrescribir")
    return p.parse_args()
 
 
def load_flexible(path: Path) -> list[dict]:
    """Carga JSON array o JSONL, lo que sea."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Intentar como JSON array primero
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON no es lista")
        return data
    # Sino, JSONL
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [SKIP] {path.name}:{i} JSON invalido: {e}", file=sys.stderr)
    return out
 
 
def validate(ex: dict, source: str, i: int) -> str | None:
    if "messages" not in ex or not isinstance(ex["messages"], list):
        return f"{source}:{i} sin 'messages' o no es lista"
    roles = [m.get("role") for m in ex["messages"]]
    if "user" not in roles or "assistant" not in roles:
        return f"{source}:{i} falta user o assistant"
    for m in ex["messages"]:
        if m.get("role") not in ("system", "user", "assistant"):
            return f"{source}:{i} role invalido {m.get('role')!r}"
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return f"{source}:{i} content vacio en role {m.get('role')}"
    if ex["messages"][-1]["role"] != "assistant":
        return f"{source}:{i} ultimo mensaje no es assistant"
    return None
 
 
def dedup_key(ex: dict) -> str:
    """Hash sobre user + assistant (ignora system para deduplicar bien
    aunque el system prompt haya cambiado)."""
    user = "||".join(m["content"] for m in ex["messages"] if m["role"] == "user")
    asst = "||".join(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    return hashlib.sha256((user + "###" + asst).encode("utf-8")).hexdigest()
 
 
def normalize_system(ex: dict, system_prompt: str) -> dict:
    """Asegura que el ejemplo tenga exactamente UN system message con el
    prompt estandar, al principio."""
    non_system = [m for m in ex["messages"] if m["role"] != "system"]
    ex["messages"] = [{"role": "system", "content": system_prompt}] + non_system
    return ex
 
 
def main():
    args = parse_args()
    random.seed(args.seed)
 
    petroleum_path = Path(args.petroleum)
    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)
 
    for p in (petroleum_path, train_path, val_path):
        if not p.exists():
            print(f"ERROR: no existe {p}", file=sys.stderr)
            sys.exit(1)
 
    # 1. Cargar los tres
    print("=== Cargando ===")
    petroleum_raw = load_flexible(petroleum_path)
    train_raw = load_flexible(train_path)
    val_raw = load_flexible(val_path)
    print(f"  petroleum: {len(petroleum_raw)} ejemplos")
    print(f"  train:     {len(train_raw)} ejemplos")
    print(f"  val:       {len(val_raw)} ejemplos (no se modifica)")
 
    # 2. Validar petroleum
    print("\n=== Validando petroleum ===")
    petroleum = []
    n_invalid = 0
    for i, ex in enumerate(petroleum_raw, start=1):
        err = validate(ex, petroleum_path.name, i)
        if err:
            print(f"  [SKIP] {err}", file=sys.stderr)
            n_invalid += 1
            continue
        petroleum.append(ex)
    print(f"  validos: {len(petroleum)}, descartados: {n_invalid}")
 
    # 3. Normalizar system prompt
    if args.normalize_system:
        n_changed = 0
        for ex in petroleum:
            has_std = any(
                m["role"] == "system" and m["content"] == SYSTEM_PROMPT
                for m in ex["messages"]
            )
            if not has_std:
                n_changed += 1
            normalize_system(ex, SYSTEM_PROMPT)
        print(f"  normalizado system prompt en {n_changed}/{len(petroleum)} filas")
 
    # 4. Dedup cruzado contra val (data leakage)
    val_keys = {dedup_key(ex) for ex in val_raw}
    before = len(petroleum)
    petroleum = [ex for ex in petroleum if dedup_key(ex) not in val_keys]
    n_leak = before - len(petroleum)
    if n_leak > 0:
        print(f"  ADVERTENCIA: {n_leak} filas de petroleum coincidian con val (leakage evitado)")
    else:
        print(f"  sin overlap con val")
 
    # 5. Dedup contra train
    train_keys = {dedup_key(ex) for ex in train_raw}
    before = len(petroleum)
    petroleum_new = [ex for ex in petroleum if dedup_key(ex) not in train_keys]
    n_already = before - len(petroleum_new)
    if n_already > 0:
        print(f"  {n_already} filas de petroleum ya estaban en train (se ignoran)")
    petroleum = petroleum_new
    print(f"  petroleum unicas para agregar: {len(petroleum)}")
 
    if not petroleum:
        print("\nNada nuevo para agregar. Saliendo sin tocar el train.")
        sys.exit(0)
 
    # 6. Oversample x factor y agregar
    added = petroleum * args.factor
    print(f"\n=== Agregando ===")
    print(f"  {len(petroleum)} originales x {args.factor} = {len(added)} filas al train")
 
    combined = train_raw + added
    random.shuffle(combined)
    print(f"  train total: {len(train_raw)} -> {len(combined)} (+{len(added)})")
 
    # 7. Backup del train original si vamos a sobrescribir
    if out_path.resolve() == train_path.resolve() and not args.no_backup:
        backup = train_path.with_suffix(train_path.suffix + ".bak")
        shutil.copy2(train_path, backup)
        print(f"\n  backup creado: {backup}")
 
    # 8. Escribir
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in combined:
            # Solo guardar el campo messages, sin metadata interna
            clean = {"messages": ex["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
 
    print(f"\n=== Listo ===")
    print(f"  escrito: {out_path}")
    print(f"  filas totales: {len(combined)}")
    print(f"  val NO fue modificado: {val_path}")
 
 
if __name__ == "__main__":
    
    
    main()