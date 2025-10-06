# src/models/evaluate.py
import json, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ART_DIR = pathlib.Path("artifacts")
MODELS_DIR = ART_DIR / "models"
REPORT_DIR = ART_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return 100.0 * np.mean(diff)

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def load_val_preds():
    """Tüm ufuklar için val_preds.csv dosyalarını birleştir."""
    frames = []
    for hdir in sorted(MODELS_DIR.glob("h*/")):
        csv_path = hdir / "val_preds.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["ts"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No val_preds.csv files found in artifacts/models/*")
    out = pd.concat(frames, ignore_index=True)
    return out

def aggregate_metrics(df):
    """H bazında ve genel metrikler."""
    rows = []
    for h, g in df.groupby("h"):
        rows.append({
            "h": int(h),
            "rmse": float(rmse(g["y_true"].values, g["y_pred"].values)),
            "smape": float(smape(g["y_true"].values, g["y_pred"].values)),
            "wape": float(wape(g["y_true"].values, g["y_pred"].values)),
        })
    m_df = pd.DataFrame(rows).sort_values("h")
    summary = {
        "rmse_mean_h1_24": float(m_df["rmse"].mean()),
        "smape_mean_h1_24": float(m_df["smape"].mean()),
        "wape_mean_h1_24": float(m_df["wape"].mean()),
    }
    return m_df, summary

def plot_metrics_per_h(m_df):
    """H ufkuna göre metrik trendleri (PNG)."""
    fig = plt.figure(figsize=(8,4))
    plt.plot(m_df["h"], m_df["rmse"], marker="o", label="RMSE")
    plt.plot(m_df["h"], m_df["smape"], marker="o", label="sMAPE")
    plt.plot(m_df["h"], m_df["wape"], marker="o", label="WAPE")
    plt.xlabel("Horizon (hours ahead)")
    plt.legend()
    plt.title("Validation metrics per horizon")
    out = REPORT_DIR / "metrics_per_h.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out

def plot_example_series(df, store="S1", sku="SKU1"):
    """Seçili store/sku için val aralığında birkaç h çizimi."""
    sample = df[(df["store_id"]==store) & (df["sku"]==sku)]
    # En küçük horizonları çizmek daha anlamlı
    sample_h = sample[sample["h"].isin([1, 6, 12, 24])]
    fig = plt.figure(figsize=(10,5))
    for h, g in sample_h.groupby("h"):
        g = g.sort_values("ts")
        plt.plot(g["ts"], g["y_true"], label=f"y_true (h={h})", alpha=0.5)
        plt.plot(g["ts"], g["y_pred"], label=f"y_pred (h={h})", linestyle="--")
    plt.title(f"Validation predictions — {store}/{sku}")
    plt.xlabel("Time")
    plt.ylabel("Sales")
    plt.legend(ncol=2)
    out = REPORT_DIR / f"series_{store}_{sku}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out

def save_html_report(m_df, summary, figs):
    """Basit bir HTML raporu yaz."""
    html_path = REPORT_DIR / "report.html"
    rows = "".join(
        f"<tr><td>{int(r.h)}</td><td>{r.rmse:.3f}</td><td>{r.smape:.2f}%</td><td>{r.wape:.3f}</td></tr>"
        for r in m_df.itertuples()
    )
    fig_tags = "".join([f'<img src="{p.name}" style="max-width:100%;"/>' for p in figs])
    html = f"""
    <html><head><meta charset="utf-8"><title>RetailDemandX – Validation Report</title></head>
    <body>
    <h1>RetailDemandX – Validation Report</h1>
    <h2>Summary</h2>
    <ul>
      <li>RMSE mean (h1-24): {summary['rmse_mean_h1_24']:.3f}</li>
      <li>sMAPE mean (h1-24): {summary['smape_mean_h1_24']:.2f}%</li>
      <li>WAPE mean (h1-24): {summary['wape_mean_h1_24']:.3f}</li>
    </ul>
    <h2>Metrics per horizon</h2>
    <table border="1" cellspacing="0" cellpadding="6">
      <tr><th>h</th><th>RMSE</th><th>sMAPE</th><th>WAPE</th></tr>
      {rows}
    </table>
    <h2>Figures</h2>
    {fig_tags}
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

def main():
    df = load_val_preds()
    m_df, summary = aggregate_metrics(df)
    m_df.to_csv(REPORT_DIR / "metrics_per_h.csv", index=False)
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig1 = plot_metrics_per_h(m_df)
    # bir örnek çizim — istersen değiştir
    fig2 = plot_example_series(df, store="S1", sku="SKU1")

    html = save_html_report(m_df, summary, [fig1, fig2])

    print("Evaluation done.")
    print(f"Saved: {fig1.name}, {fig2.name}, metrics_per_h.csv, summary.json, report.html")

if __name__ == "__main__":
    main()