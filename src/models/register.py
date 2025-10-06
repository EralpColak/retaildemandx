
import json, shutil, pathlib, time
from datetime import datetime

ART = pathlib.Path("artifacts")
MODELS = ART / "models"
REPORTS = ART / "reports"
REGISTRY = ART / "registry" / "staging"   

def load_summary():
    summ = REPORTS / "summary.json"
    if not summ.exists():
        raise FileNotFoundError("summary.json not found. Run evaluate stage first.")
    with open(summ, "r") as f:
        return json.load(f)

def collect_models():
    items = []
    for hdir in sorted(MODELS.glob("h*/")):
        m = hdir / "model.pkl"
        v = hdir / "val_preds.csv"
        if m.exists():
            items.append({"h": hdir.name, "model": str(m), "val": str(v) if v.exists() else None})
    if not items:
        raise FileNotFoundError("No trained models under artifacts/models/*")
    return items

def main():
    summary = load_summary()
    items = collect_models()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = REGISTRY / ts
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for it in items:
        h = it["h"]   
        dst_dir = target / h
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["model"], dst_dir / "model.pkl")
        if it["val"]:
            shutil.copy2(it["val"], dst_dir / "val_preds.csv")
        copied.append({"h": h, "path": str(dst_dir / "model.pkl")})

    manifest = {
        "version": ts,
        "created_utc": ts,
        "source_models_dir": str(MODELS),
        "metrics_summary": summary,   
        "models": copied
    }
    with open(target / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


    current = REGISTRY / "current"

    if current.exists() or current.is_symlink():
        try:
            if current.is_symlink():
                current.unlink()
            else:
                shutil.rmtree(current)
        except FileNotFoundError:
            pass


    current.parent.mkdir(parents=True, exist_ok=True)

    target_abs = target.resolve()

    try:
        
        current.symlink_to(target_abs, target_is_directory=True)
        print(f"↪ current (symlink) -> {target_abs}")
    except Exception as e:
        print(f"[WARN] symlink failed ({e}), copying instead...")
        shutil.copytree(target_abs, current)
        print(f"↪ current (copy) -> {current.resolve()}")

    print(f"Registered to {target}")
    print(f"↪ current -> {current.resolve()}")
    print(f"Metrics summary: {summary}")

if __name__ == "__main__":
    main() 