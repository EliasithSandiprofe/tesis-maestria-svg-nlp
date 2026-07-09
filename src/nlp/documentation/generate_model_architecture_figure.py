"""
generate_model_architecture_figure.py
--------------------------------------
Genera la figura de arquitectura del modelo MultiTaskDistilBERT
para el documento de tesis de Maestría SVG-NLP.

Salida
------
docs/figuras/Figura_01_Arquitectura_MultiTaskDistilBERT.png

Dependencias
------------
Únicamente matplotlib (ya instalado en el entorno del proyecto).
No requiere seaborn, Graphviz, ni cargar el modelo DistilBERT.

Uso
---
    python src/nlp/documentation/generate_model_architecture_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Paleta de colores ─────────────────────────────────────────────────────────
_C = {
    "bg":      "#F8FAFC",   # fondo figura
    "inp_f":   "#DBEAFE",   # Entrada: azul hielo
    "inp_e":   "#1D4ED8",
    "tok_f":   "#EDE9FE",   # Tokenizer: lavanda
    "tok_e":   "#5B21B6",
    "enc_f":   "#1E3A8A",   # Encoder: azul profundo
    "enc_e":   "#172554",
    "enc_gl":  "#60A5FA",   # Borde decorativo interno del encoder
    "cls_f":   "#4C1D95",   # CLS: púrpura
    "cls_e":   "#2E1065",
    "dro_f":   "#6D28D9",   # Dropout: violeta
    "dro_e":   "#4C1D95",
    "hed_f":   "#92400E",   # Cabezas: ámbar oscuro
    "hed_e":   "#78350F",
    "out_f":   "#064E3B",   # Salidas: verde esmeralda
    "out_e":   "#022C22",
    "arr":     "#374151",   # Flechas y líneas
    "lbl":     "#64748B",   # Etiquetas secundarias
    "dot":     "#374151",   # Puntos de unión
}


# ── Primitivas de dibujo ──────────────────────────────────────────────────────

def _box(
    ax: plt.Axes,
    cx: float, cy: float, w: float, h: float,
    title: str,
    sub: str | None = None,
    fc: str = "#fff", ec: str = "#000", tc: str = "black",
    fs: float = 10.5, fw: str = "normal", lw: float = 1.5,
) -> None:
    """Dibuja una caja redondeada con título y subtítulo opcional.

    Parameters
    ----------
    ax : Axes
        Ejes sobre los que dibujar.
    cx, cy : float
        Centro de la caja (coordenadas de axes, 0‒1).
    w, h : float
        Ancho y alto de la caja.
    title : str
        Texto principal centrado.
    sub : str | None
        Texto secundario en cursiva (se coloca debajo del título).
    fc, ec, tc : str
        Color de relleno, borde y texto.
    fs : float
        Tamaño de fuente del título.
    fw : str
        Peso de fuente del título (``'bold'`` o ``'normal'``).
    lw : float
        Grosor del borde.
    """
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, zorder=3,
    ))

    if sub:
        dy = h * 0.23
        ax.text(cx, cy + dy, title,
                ha='center', va='center', fontsize=fs, color=tc,
                fontweight=fw, transform=ax.transAxes, zorder=4)
        ax.text(cx, cy - dy, sub,
                ha='center', va='center', fontsize=max(fs - 1.8, 7.5),
                color=tc, fontstyle='italic', alpha=0.88,
                transform=ax.transAxes, zorder=4)
    else:
        ax.text(cx, cy, title,
                ha='center', va='center', fontsize=fs, color=tc,
                fontweight=fw, transform=ax.transAxes, zorder=4)


def _arr(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    color: str = _C["arr"],
    lw: float = 1.7,
    ms: float = 14,
) -> None:
    """Dibuja una flecha entre dos puntos (coordenadas de axes)."""
    ax.annotate(
        '',
        xy=(x2, y2), xytext=(x1, y1),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='-|>', color=color,
            lw=lw, mutation_scale=ms,
        ),
        zorder=2,
    )


def _hline(
    ax: plt.Axes,
    x0: float, x1: float, y: float,
    color: str = _C["arr"], lw: float = 2.0,
) -> None:
    """Dibuja una línea horizontal (coordenadas de axes)."""
    ax.plot(
        [x0, x1], [y, y],
        color=color, lw=lw, transform=ax.transAxes,
        solid_capstyle='round', zorder=2,
    )


def _note(
    ax: plt.Axes,
    x: float, y: float, text: str,
    color: str = _C["lbl"], fs: float = 8.5, ha: str = 'left',
) -> None:
    """Añade una etiqueta de anotación secundaria."""
    ax.text(x, y, text,
            ha=ha, va='center', fontsize=fs, color=color,
            fontstyle='italic', transform=ax.transAxes, zorder=4)


# ── Función principal ─────────────────────────────────────────────────────────

def generate_figure(output_path: Path) -> None:
    """Construye y guarda la figura de arquitectura.

    Parameters
    ----------
    output_path : Path
        Ruta completa (PNG) del archivo de salida.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(_C["bg"])
    ax.set_facecolor(_C["bg"])

    # ── TÍTULO ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.966,
            "Arquitectura del modelo MultiTaskDistilBERT",
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='#0F172A', transform=ax.transAxes)
    ax.text(0.5, 0.932,
            "Clasificación multi-tarea para la interpretación de prompts SVG-NLP"
            "  ·  Fine-tuning mediante hard parameter sharing",
            ha='center', va='center', fontsize=9.5, color='#475569',
            fontstyle='italic', transform=ax.transAxes)

    # ── GEOMETRÍA ────────────────────────────────────────────────────────
    XC  = 0.500   # eje central
    BW  = 0.390   # ancho de cajas principales
    BH  = 0.058   # alto de cajas principales
    EH  = 0.125   # alto encoder
    DH  = 0.050   # alto dropout

    Y_PROMPT = 0.847
    Y_TOK    = 0.756
    Y_ENC    = 0.626
    Y_CLS    = 0.502
    Y_DRO    = 0.424
    Y_SPLIT  = 0.349
    Y_HEAD   = 0.230
    Y_OUT    = 0.083
    OH       = 0.064   # alto cajas salida

    HEAD_XS = [0.096, 0.356, 0.618, 0.882]
    HW      = 0.178    # ancho cabezas / salidas

    # ── 1. PROMPT ─────────────────────────────────────────────────────────
    _box(ax, XC, Y_PROMPT, BW, BH,
         "Prompt de entrada",
         sub='"Camiseta con montaña, color negro, estilo minimalista, zona centrada"',
         fc=_C["inp_f"], ec=_C["inp_e"], tc="#1E3A8A",
         fs=11, fw='bold')

    _arr(ax, XC, Y_PROMPT - BH / 2, XC, Y_TOK + BH / 2)
    _note(ax, XC + BW / 2 + 0.018,
          (Y_PROMPT - BH / 2 + Y_TOK + BH / 2) / 2,
          "texto plano (str)")

    # ── 2. TOKENIZER ─────────────────────────────────────────────────────
    _box(ax, XC, Y_TOK, BW, BH,
         "DistilBertTokenizerFast",
         sub="distilbert-base-uncased  ·  vocabulario: 30 522 tokens  ·  max_length = 128",
         fc=_C["tok_f"], ec=_C["tok_e"], tc="#2E1065",
         fs=11, fw='bold')

    _arr(ax, XC, Y_TOK - BH / 2, XC, Y_ENC + EH / 2)
    _note(ax, XC + 0.510 / 2 + 0.018,
          (Y_TOK - BH / 2 + Y_ENC + EH / 2) / 2,
          "input_ids  ·  attention_mask  ·  (batch × 128)")

    # ── 3. DISTILBERT ENCODER ────────────────────────────────────────────
    # Borde exterior punteado (visual de "bloque preentrenado")
    ax.add_patch(FancyBboxPatch(
        (XC - 0.262, Y_ENC - EH / 2 - 0.006), 0.524, EH + 0.012,
        boxstyle="round,pad=0.010",
        facecolor='none', edgecolor=_C["enc_gl"],
        linewidth=1.2, linestyle='--', alpha=0.55,
        transform=ax.transAxes, zorder=2,
    ))
    _box(ax, XC, Y_ENC, 0.510, EH,
         "DistilBERT Encoder  —  parámetros compartidos",
         sub="6 Transformer blocks  ·  12 attn heads  ·  hidden_size = 768  ·  66 M parámetros",
         fc=_C["enc_f"], ec=_C["enc_e"], tc="white",
         fs=11.5, fw='bold', lw=2.5)

    _arr(ax, XC, Y_ENC - EH / 2, XC, Y_CLS + BH / 2)
    _note(ax, XC + 0.510 / 2 + 0.018,
          (Y_ENC - EH / 2 + Y_CLS + BH / 2) / 2,
          "last_hidden_state  ·  (batch × 128 × 768)")

    # ── 4. REPRESENTACIÓN [CLS] ───────────────────────────────────────────
    _box(ax, XC, Y_CLS, BW, BH,
         "Representación  [CLS]  —  Primer token",
         sub="hidden_state[:, 0, :]  →  vector de 768 dimensiones",
         fc=_C["cls_f"], ec=_C["cls_e"], tc="white",
         fs=11, fw='bold', lw=2)

    _arr(ax, XC, Y_CLS - BH / 2, XC, Y_DRO + DH / 2)
    _note(ax, XC + BW / 2 + 0.018,
          (Y_CLS - BH / 2 + Y_DRO + DH / 2) / 2,
          "(batch × 768)")

    # ── 5. DROPOUT ────────────────────────────────────────────────────────
    _box(ax, XC, Y_DRO, 0.225, DH,
         "Dropout  (p = 0.3)",
         fc=_C["dro_f"], ec=_C["dro_e"], tc="white",
         fs=10, fw='normal', lw=1.5)

    # Flecha dropout → punto de distribución
    _arr(ax, XC, Y_DRO - DH / 2, XC, Y_SPLIT + 0.006)

    # ── LÍNEA HORIZONTAL DE DISTRIBUCIÓN ─────────────────────────────────
    _hline(ax, HEAD_XS[0], HEAD_XS[-1], Y_SPLIT)

    # Puntos de unión en la línea horizontal (ramificaciones)
    for hx in HEAD_XS:
        ax.plot(hx, Y_SPLIT, 'o',
                markersize=5.5, color=_C["dot"],
                transform=ax.transAxes, zorder=5)

    # Etiqueta de la distribución
    _note(ax, HEAD_XS[-1] + 0.013, Y_SPLIT,
          "→  4 tareas", fs=9, color=_C["arr"])

    # ── 6. CABEZAS DE CLASIFICACIÓN + SALIDAS ────────────────────────────
    HEAD_INFO = [
        ("Color Head",    "Linear(768 → 6)"),
        ("Estilo Head",   "Linear(768 → 5)"),
        ("Elemento Head", "Linear(768 → 10)"),
        ("Posición Head", "Linear(768 → 4)"),
    ]
    OUT_INFO = [
        ("color",    "6 clases  (azul, negro…)"),
        ("estilo",   "5 clases  (vintage…)"),
        ("elemento", "10 clases  (montaña…)"),
        ("posicion", "4 clases  (centrado…)"),
    ]

    for hx, (htitle, hsub), (otask, osub) in zip(HEAD_XS, HEAD_INFO, OUT_INFO):
        # Flecha rama → cabeza
        _arr(ax, hx, Y_SPLIT, hx, Y_HEAD + BH / 2 + 0.002, lw=1.6)

        # Caja cabeza de clasificación
        _box(ax, hx, Y_HEAD, HW, BH, htitle, sub=hsub,
             fc=_C["hed_f"], ec=_C["hed_e"], tc="white",
             fs=9.5, fw='bold', lw=2)

        # Flecha cabeza → salida
        _arr(ax, hx, Y_HEAD - BH / 2, hx, Y_OUT + OH / 2 + 0.003, lw=1.6)

        # Caja de predicción final
        _box(ax, hx, Y_OUT, HW, OH,
             f"Predicción:  {otask}", sub=osub,
             fc=_C["out_f"], ec=_C["out_e"], tc="white",
             fs=9.5, fw='bold', lw=2)

    # ── LEYENDA (banda inferior) ──────────────────────────────────────────
    legend_items = [
        (_C["inp_f"], _C["inp_e"], "#1E3A8A", "Entrada / Tokenización"),
        (_C["enc_f"], _C["enc_e"], "white",   "Encoder DistilBERT compartido"),
        (_C["cls_f"], _C["cls_e"], "white",   "Representación [CLS] + Dropout"),
        (_C["hed_f"], _C["hed_e"], "white",   "Cabezas de clasificación"),
        (_C["out_f"], _C["out_e"], "white",   "Predicciones finales"),
    ]
    lx_start = 0.070
    lx = lx_start
    ly = 0.027
    spacing = 0.183
    for i, (fc, ec, tc, label) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch(
            (lx, ly - 0.011), 0.018, 0.022,
            boxstyle="round,pad=0.003",
            facecolor=fc, edgecolor=ec, linewidth=1,
            transform=ax.transAxes, zorder=5,
        ))
        ax.text(lx + 0.022, ly, label,
                ha='left', va='center', fontsize=8, color='#374151',
                transform=ax.transAxes, zorder=5)
        lx += spacing

    # ── GUARDAR ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    """Punto de entrada: genera la figura y muestra la ruta."""
    output_path = (
        _PROJECT_ROOT / "docs" / "figuras"
        / "Figura_01_Arquitectura_MultiTaskDistilBERT.png"
    )
    print("Generando figura de arquitectura MultiTaskDistilBERT...")
    generate_figure(output_path)
    size_kb = output_path.stat().st_size / 1024
    print(f"Figura guardada en : {output_path}")
    print(f"Tamaño             : {size_kb:.1f} KB  (dpi=300)")


if __name__ == "__main__":
    main()
