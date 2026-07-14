"""
generate_training_curves.py
----------------------------
Genera las figuras oficiales de las curvas de entrenamiento para la
documentación de tesis (Sesión 05) a partir del historial CSV.

Fuente de datos
---------------
dataset/reports/training/training_history.csv

Figuras generadas en docs/figuras/
-----------------------------------
Figura_02_Curva_Loss_Entrenamiento.png
    Curva de pérdida total: train_loss vs val_loss por época.

Figura_03_Curva_Accuracy_Validacion.png
    Curva de accuracy por tarea y media: accuracy_{color,estilo,
    elemento,posicion} + mean_accuracy, expresadas en porcentaje.

Dependencias
------------
Únicamente matplotlib y pandas (ya instalados en el proyecto).

Uso
---
    python src/nlp/documentation/generate_training_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CSV_PATH     = _PROJECT_ROOT / "dataset" / "reports" / "training" / "training_history.csv"
_OUT_DIR      = _PROJECT_ROOT / "docs" / "figuras"

# ── Estilo global ─────────────────────────────────────────────────────────────
_BG      = "#F8FAFC"
_GRID    = "#E2E8F0"
_SPINE   = "#CBD5E1"
_TEXT    = "#0F172A"
_SUBTEXT = "#64748B"

# Paleta figura 02 — Loss
_COL_TRAIN = "#2563EB"   # azul
_COL_VAL   = "#DC2626"   # rojo

# Paleta figura 03 — Accuracy por tarea
_TASK_COLORS = {
    "color":    "#F59E0B",   # ámbar
    "estilo":   "#10B981",   # esmeralda
    "elemento": "#8B5CF6",   # violeta
    "posicion": "#EC4899",   # rosa
    "mean":     "#1E3A8A",   # azul oscuro (media, línea destacada)
}


# ── Utilidades de estilo ──────────────────────────────────────────────────────

def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Aplica el estilo profesional común a ambas figuras."""
    ax.set_facecolor(_BG)
    ax.set_xlabel(xlabel, fontsize=11, color=_SUBTEXT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=_SUBTEXT, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', color=_TEXT, pad=14)
    ax.tick_params(colors=_SUBTEXT, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8, linestyle='--', alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_SPINE)
        spine.set_linewidth(0.8)


def _integer_xaxis(ax: plt.Axes, epochs: pd.Series) -> None:
    """Fuerza ticks enteros en el eje X (épocas)."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlim(epochs.min() - 0.3, epochs.max() + 0.3)


def _annotate_best(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    label: str,
    color: str,
    above: bool = True,
) -> None:
    """Añade una anotación de punto destacado (mejor época)."""
    offset = (0, 12) if above else (0, -16)
    ax.annotate(
        label,
        xy=(x_val, y_val),
        xytext=offset,
        textcoords='offset points',
        ha='center', va='bottom' if above else 'top',
        fontsize=8.5, color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=color, alpha=0.85, linewidth=0.8),
    )


# ── Figura 02 — Curva de pérdida ──────────────────────────────────────────────

def generate_loss_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Genera la figura de curvas de pérdida (train vs val).

    Parameters
    ----------
    df : pd.DataFrame
        Historial de entrenamiento leído desde el CSV.
    output_path : Path
        Ruta PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)

    epochs = df["epoch"]

    # ── Líneas principales ────────────────────────────────────────────────────
    ax.plot(
        epochs, df["train_loss"],
        color=_COL_TRAIN, linewidth=2.2, marker='o', markersize=6,
        label="Train Loss", zorder=3,
    )
    ax.plot(
        epochs, df["val_loss"],
        color=_COL_VAL, linewidth=2.2, marker='s', markersize=6,
        linestyle='--', label="Val Loss", zorder=3,
    )

    # ── Relleno de área entre curvas ──────────────────────────────────────────
    ax.fill_between(
        epochs, df["train_loss"], df["val_loss"],
        alpha=0.07, color=_COL_VAL,
    )

    # ── Anotación del mínimo de val_loss ──────────────────────────────────────
    best_idx = df["val_loss"].idxmin()
    best_ep  = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_loss"]
    _annotate_best(
        ax, best_ep, best_val,
        f"Mejor val_loss\n{best_val:.4f}  (época {best_ep})",
        _COL_VAL, above=True,
    )
    ax.axvline(best_ep, color=_COL_VAL, linewidth=0.8,
               linestyle=':', alpha=0.5, zorder=1)

    # ── Pérdidas individuales por tarea (líneas tenues) ───────────────────────
    task_colors_light = {
        "color":    "#93C5FD",
        "estilo":   "#86EFAC",
        "elemento": "#C4B5FD",
        "posicion": "#F9A8D4",
    }
    for task, col in task_colors_light.items():
        ax.plot(
            epochs, df[f"val_loss_{task}"],
            color=col, linewidth=1.0, linestyle=':', alpha=0.7,
            label=f"Val loss {task}",
        )

    # ── Estilo y guardado ─────────────────────────────────────────────────────
    _style_axes(
        ax,
        xlabel="Época",
        ylabel="Pérdida (CrossEntropyLoss)",
        title="Curva de Pérdida — Entrenamiento y Validación",
    )
    _integer_xaxis(ax, epochs)

    legend = ax.legend(
        fontsize=8.5, framealpha=0.92, edgecolor=_SPINE,
        loc='upper right', ncol=2,
    )
    legend.get_frame().set_facecolor(_BG)

    ax.text(
        0.01, 0.02,
        "Pérdidas individuales por tarea (líneas punteadas): color / estilo / elemento / posicion",
        transform=ax.transAxes, fontsize=7.5, color=_SUBTEXT, style='italic',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Figura 03 — Curva de accuracy ─────────────────────────────────────────────

def generate_accuracy_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Genera la figura de curvas de accuracy por tarea.

    Convierte los valores de accuracy (0–1) a porcentaje (0–100).

    Parameters
    ----------
    df : pd.DataFrame
        Historial de entrenamiento leído desde el CSV.
    output_path : Path
        Ruta PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)

    epochs = df["epoch"]

    # ── Líneas por tarea ──────────────────────────────────────────────────────
    task_cfg = [
        ("accuracy_color",    "color",    "Color",    "o", 1.8, "-"),
        ("accuracy_estilo",   "estilo",   "Estilo",   "s", 1.8, "-"),
        ("accuracy_elemento", "elemento", "Elemento", "^", 1.8, "-"),
        ("accuracy_posicion", "posicion", "Posición", "D", 1.8, "-"),
    ]
    for col, task, label, marker, lw, ls in task_cfg:
        ax.plot(
            epochs, df[col] * 100,
            color=_TASK_COLORS[task], linewidth=lw, marker=marker,
            markersize=6, linestyle=ls, label=label, zorder=3,
        )

    # ── Línea de media ────────────────────────────────────────────────────────
    ax.plot(
        epochs, df["mean_accuracy"] * 100,
        color=_TASK_COLORS["mean"], linewidth=2.8, marker='P',
        markersize=8, linestyle='--', label="Mean Accuracy",
        zorder=4,
    )

    # ── Anotación del máximo de mean_accuracy ─────────────────────────────────
    best_idx  = df["mean_accuracy"].idxmax()
    best_ep   = df.loc[best_idx, "epoch"]
    best_mean = df.loc[best_idx, "mean_accuracy"] * 100
    _annotate_best(
        ax, best_ep, best_mean,
        f"Mean Acc máx.\n{best_mean:.1f}%  (época {best_ep})",
        _TASK_COLORS["mean"], above=False,
    )
    ax.axvline(best_ep, color=_TASK_COLORS["mean"], linewidth=0.8,
               linestyle=':', alpha=0.5, zorder=1)

    # ── Línea de referencia al 100 % ──────────────────────────────────────────
    ax.axhline(100, color=_GRID, linewidth=1.0, linestyle='-', zorder=1)

    # ── Rango Y con margen ────────────────────────────────────────────────────
    y_min = max(0, df[[
        "accuracy_color", "accuracy_estilo",
        "accuracy_elemento", "accuracy_posicion",
        "mean_accuracy",
    ]].min().min() * 100 - 5)
    ax.set_ylim(y_min, 105)

    # ── Formateo del eje Y como porcentaje ────────────────────────────────────
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── Estilo y guardado ─────────────────────────────────────────────────────
    _style_axes(
        ax,
        xlabel="Época",
        ylabel="Accuracy (%)",
        title="Curva de Accuracy por Tarea — Conjunto de Validación",
    )
    _integer_xaxis(ax, epochs)

    legend = ax.legend(
        fontsize=9, framealpha=0.92, edgecolor=_SPINE,
        loc='lower right',
    )
    legend.get_frame().set_facecolor(_BG)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    """Lee el historial CSV y genera las dos figuras de curvas."""
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el historial de entrenamiento:\n  {_CSV_PATH}\n"
            "Ejecuta primero train.py (Fase D)."
        )

    df = pd.read_csv(_CSV_PATH)
    print(f"Historial cargado: {len(df)} épocas  —  {_CSV_PATH.name}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figura 02 — Loss ──────────────────────────────────────────────────────
    path_loss = _OUT_DIR / "Figura_02_Curva_Loss_Entrenamiento.png"
    generate_loss_figure(df, path_loss)
    size_kb = path_loss.stat().st_size / 1024
    print(f"[OK] {path_loss.name}  ({size_kb:.1f} KB)")

    # ── Figura 03 — Accuracy ──────────────────────────────────────────────────
    path_acc = _OUT_DIR / "Figura_03_Curva_Accuracy_Validacion.png"
    generate_accuracy_figure(df, path_acc)
    size_kb = path_acc.stat().st_size / 1024
    print(f"[OK] {path_acc.name}  ({size_kb:.1f} KB)")

    print(f"\nFiguras guardadas en: {_OUT_DIR.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
