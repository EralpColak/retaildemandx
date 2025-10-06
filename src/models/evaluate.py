# src/models/evaluate.py
import json, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ART_DIR = pathlib.Path("artifacts")
MODELS_DIR = ART_DIR / "models"
REPORT_DIR = ART_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- metrics ---------------- #
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return 100.0 * np.mean(diff)


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ---------------- IO helpers ---------------- #
def load_val_preds():
    """Tüm ufuklar için val_preds.csv dosyalarını birleştirir. Yoksa boş df döner."""
    frames = []
    cols = ["h", "ts", "store_id", "sku", "y_true", "y_pred"]
    for hdir in sorted(MODELS_DIR.glob("h*/")):
        csv_path = hdir / "val_preds.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["ts"])
                # beklenen kolonlar yoksa atla
                if not set(cols).issubset(df.columns):
                    continue
                frames.append(df[cols])
            except Exception:
                # bozuk dosya vs. ise sessiz geç
                continue
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out


def aggregate_metrics(df):
    """H bazında ve genel metrikler. Boşsa NaN özet döner."""
    if df.empty:
        m_df = pd.DataFrame(columns=["h", "rmse", "smape", "wape"])
        summary = {
            "rmse_mean_h1_24": float("nan"),
            "smape_mean_h1_24": float("nan"),
            "wape_mean_h1_24": float("nan"),
        }
        return m_df, summary

    rows = []
    for h, g in df.groupby("h"):
        y_true = g["y_true"].values
        y_pred = g["y_pred"].values
        if len(y_true) == 0:
            r, s, w = float("nan"), float("nan"), float("nan")
        else:
            r = float(rmse(y_true, y_pred))
            s = float(smape(y_true, y_pred))
            w = float(wape(y_true, y_pred))
        rows.append({"h": int(h), "rmse": r, "smape": s, "wape": w})

    m_df = pd.DataFrame(rows).sort_values("h")
    # NaN'leri es geçerek ortalama
    def _nanmean(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    summary = {
        "rmse_mean_h1_24": _nanmean(m_df["rmse"]),
        "smape_mean_h1_24": _nanmean(m_df["smape"]),
        "wape_mean_h1_24": _nanmean(m_df["wape"]),
    }
    return m_df, summary


# ---------------- plots ---------------- #
def plot_metrics_per_h(m_df):
    """H ufkuna göre metrik trendleri (PNG). Veri yoksa None döner."""
    if m_df.empty:
        return None
    fig = plt.figure(figsize=(8, 4))
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
    """Seçili store/sku için val aralığında birkaç h çizimi. Veri yoksa None."""
    if df.empty:
        return None
    sample = df[(df["store_id"] == store) & (df["sku"] == sku)]
    if sample.empty:
        return None
    sample_h = sample[sample["h"].isin([1, 6, 12, 24])]
    if sample_h.empty:
        return None

    fig = plt.figure(figsize=(10, 5))
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


# ---------------- report ---------------- #
def save_html_report(m_df, summary, figs):
    """Basit bir HTML raporu yaz. Veri yoksa bilgilendirici mesaj üret."""
    html_path = REPORT_DIR / "report.html"

    if m_df.empty:
        rows_html = "<tr><td colspan='4'>No validation data available.</td></tr>"
    else:
        rows_html = "".join(
            f"<tr><td>{int(r.h)}</td><td>{(r.rmse if pd.notna(r.rmse) else 'NaN'):.3f}</td>"
            f"<td>{(r.smape if pd.notna(r.smape) else float('nan')):.2f}%</td>"
            f"<td>{(r.wape if pd.notna(r.wape) else float('nan')):.3f}</td></tr>"
            for r in m_df.itertuples()
        )

    fig_tags = "".join(
        [f'<img src="{p.name}" style="max-width:100%;"/>' for p in figs if p is not None]
    )
    if not fig_tags:
        fig_tags = "<p>No figures to display.</p>"

    def _fmt(x, fmt):
        return fmt.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "NaN"

    html = f"""
    <html><head><meta charset="utf-8"><title>RetailDemandX – Validation Report</title></head>
    <body>
    <h1>RetailDemandX – Validation Report</h1>
    <h2>Summary</h2>
    <ul>
      <li>RMSE mean (h1-24): {_fmt(summary.get('rmse_mean_h1_24'), '{:.3f}')}</li>
      <li>sMAPE mean (h1-24): {_fmt(summary.get('smape_mean_h1_24'), '{:.2f}%')}</li>
      <li>WAPE mean (h1-24): {_fmt(summary.get('wape_mean_h1_24'), '{:.3f}')}</li>
    </ul>
    <h2>Metrics per horizon</h2>
    <table border="1" cellspacing="0" cellpadding="6">
      <tr><th>h</th><th>RMSE</th><th>sMAPE</th><th>WAPE</th></tr>
      {rows_html}
    </table>
    <h2>Figures</h2>
    {fig_tags}
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ---------------- main ---------------- #
def main():
    df = load_val_preds()
    m_df, summary = aggregate_metrics(df)

    # artefaktlar (csv/json) — veri boş olsa da yaz
    m_df.to_csv(REPORT_DIR / "metrics_per_h.csv", index=False)
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # görseller (varsa üret)
    fig1 = plot_metrics_per_h(m_df)
    fig2 = plot_example_series(df, store="S1", sku="SKU1")

    # html rapor
    html = save_html_report(m_df, summary, [fig1, fig2])

    print("Evaluation done.")
    created = ["metrics_per_h.csv", "summary.json", "report.html"]
    if fig1: created.append(pathlib.Path(fig1).name)
    if fig2: created.append(pathlib.Path(fig2).name)
    print("Saved:", ", ".join(created))


if __name__ == "__main__":
    main()