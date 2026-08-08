"""One look for every chart in the project, and the rupee axis formatter.

Three decisions live in this file, and all three exist for a reason.

**1. The Agg backend, chosen before pyplot is ever imported.**

matplotlib picks a *backend* — the thing that actually draws — when ``pyplot``
is first imported. The default tries to open a window. On a CI runner, in a
Docker container, or over SSH there is no window, and the import either crashes
or hangs. ``Agg`` draws into memory instead. It cannot show you anything on
screen, which is exactly the point: the plots in this project are **files and
figures returned to the caller**, never pop-ups. There is no ``plt.show()``
anywhere in ``paybands.plots``, and there never should be.

**2. Every plot function takes an optional ``ax`` and returns ``(fig, ax)``.**

A function that creates its own figure can only ever produce one chart. A
function that will draw into an axes *you* hand it composes — coverage next to
band width, raw distribution next to logged, six fairness panels in a grid. It
costs three lines (:func:`new_axes`) and it is the difference between a plotting
module and a pile of scripts.

**3. Rupees are formatted in lakhs and crores.**

₹1.5M is a correct number written in the wrong number system. An Indian reader
parses ₹15L instantly and ₹1,500,000 slowly. A chart whose axis its audience has
to translate is a chart nobody reads carefully.

The palette is colourblind-safe by measurement, not by taste — see
:data:`SERIES`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# The backend must be chosen BEFORE pyplot is imported — after that it is baked
# in for the life of the process. This is the one place in the package where an
# import is deliberately not at the top of the file.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# ─────────────────────────────────────────────────────────── the palette

#: Categorical colours, used **in this order, never cycled**.
#:
#: These four were checked with a colour-vision-deficiency simulator rather than
#: eyeballed: every pair stays at least ΔE 9 apart under deuteranopia and
#: protanopia (red-green colourblindness, ~8% of men), and at least ΔE 16 apart
#: under normal vision. "They look different to me" is not a check — roughly one
#: reader in twelve does not see what you see.
#:
#: Four is the cap on purpose. A fifth hue fails the separation test, and a chart
#: needing nine colours is a chart that should have been small multiples.
SERIES: tuple[str, ...] = (
    "#2a78d6",  # blue    — the main series
    "#eb6834",  # orange  — the comparison series
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
)

#: Status colours. **Reserved** — never reuse one as "series 5". They mean
#: good / needs-attention / wrong, and they always ship with a written label,
#: because colour alone cannot carry meaning to every reader.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

#: Chrome. Text and structure, deliberately quiet so the data is the loudest
#: thing on the page.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Where :func:`save` puts a bare filename. Already gitignored — plots are
#: derived artefacts, regenerated from code, and a repo full of stale PNGs is a
#: repo where nobody knows which PNG matches which model.
DEFAULT_OUTPUT_DIR = Path("reports")

#: Default figure size, in inches. Wide enough for a rupee axis without the tick
#: labels colliding, short enough to sit in a README without dominating it.
DEFAULT_FIGSIZE = (7.0, 4.5)

#: Two panels side by side (raw vs log, promised vs actual).
WIDE_FIGSIZE = (11.0, 4.5)


#: rcParams applied by :func:`apply_style`. Nothing here is decorative: every
#: entry either removes something that wasn't carrying information (top and right
#: spines, heavy gridlines) or makes something readable (sizes, tick colours).
RC_PARAMS: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 110,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    # The box around a plot is four lines, of which two carry scale information
    # and two are decoration. Drop the decoration.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": SECONDARY_INK,
    "axes.labelsize": 10,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.grid": True,
    "axes.axisbelow": True,  # gridlines behind the data, always
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.size": 10,
    "figure.autolayout": True,
}


def apply_style() -> None:
    """Install the project look as matplotlib's defaults.

    This mutates global state, which is why it is a function you call rather
    than something that happens on import. :func:`new_axes` calls it when it
    creates a figure of its own — and pointedly does *not* when you hand it an
    axes, because a figure you built is a figure you style.
    """
    plt.rcParams.update(RC_PARAMS)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    **kwargs: object,
):
    """Styled ``plt.subplots`` — the way this package builds multi-panel figures.

    Exists so that no other module has to import ``pyplot`` directly. That is not
    tidiness: an ``import matplotlib.pyplot`` sorted to the top of another file
    would run *before* this module's ``matplotlib.use("Agg")``, silently picking
    the wrong backend. One import point, one backend decision.
    """
    apply_style()
    return plt.subplots(nrows, ncols, figsize=figsize or DEFAULT_FIGSIZE, **kwargs)  # type: ignore[arg-type]


def new_axes(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Get a ``(figure, axes)`` to draw on, creating one only if needed.

    Every plotting function in this package starts with this line. Pass ``ax``
    and the drawing lands in your existing multi-panel figure; pass nothing and
    you get a styled standalone chart.

    Args:
        ax: An existing axes to draw into, or ``None`` to make a new figure.
        figsize: Size in inches for the new figure. Ignored when ``ax`` is given
            — the figure already has a size and it isn't ours to change.
    """
    if ax is not None:
        return ax.get_figure(), ax  # type: ignore[return-value]
    apply_style()
    fig, new_ax = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE)
    return fig, new_ax


# ────────────────────────────────────────────────────── rupees on an axis


def rupee_label(value: float, _pos: int | None = None) -> str:
    """Format a rupee amount the way an Indian reader says it out loud.

    ::

        1_500_000   →  ₹15L
        12_000_000  →  ₹1.2Cr
        750_000     →  ₹7.5L
        48_000      →  ₹48,000

    One lakh is 100,000 and one crore is 100 lakh (10,000,000). Below a lakh we
    fall back to plain digit grouping — and note that Western and Indian grouping
    are *identical* below five digits (₹48,000 either way), so no special case is
    needed there. They only diverge at ₹1,00,000, which is exactly where the lakh
    branch takes over.

    The unused ``_pos`` argument is matplotlib's tick position; ``FuncFormatter``
    passes it and we ignore it. That is what lets this same function serve as
    both a formatter and something you can call in a f-string.
    """
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))

    if magnitude >= 1_00_00_000:
        return f"{sign}₹{_trim(magnitude / 1_00_00_000)}Cr"
    if magnitude >= 1_00_000:
        return f"{sign}₹{_trim(magnitude / 1_00_000)}L"
    return f"{sign}₹{magnitude:,.0f}"


def _trim(number: float) -> str:
    """One decimal place, but drop it when it is a bare zero.

    ₹15L reads better than ₹15.0L, and on an axis the extra character is the
    difference between labels that fit and labels that overlap.
    """
    text = f"{number:.1f}"
    return text[:-2] if text.endswith(".0") else text


def rupee_formatter() -> FuncFormatter:
    """Axis formatter producing ₹15L / ₹1.2Cr. Use on any salary axis."""
    return FuncFormatter(rupee_label)


def percent_formatter(decimals: int = 0) -> FuncFormatter:
    """Axis formatter turning a *fraction* into a percentage: 0.8 → ``80%``.

    Coverage, pay gaps and error rates are all naturally fractions in the code
    and all naturally percentages on the page. Converting at the axis rather than
    in the data means nothing downstream has to remember which scale it is on.
    """
    return FuncFormatter(lambda value, _pos: f"{value * 100:.{decimals}f}%")


def rupee_axis(ax: Axes, which: str = "x") -> None:
    """Put lakh/crore labels on an axis, and silence the minor ticks.

    The minor-tick step matters on a log scale: matplotlib will happily label
    every minor decade subdivision, producing ₹2L ₹3L ₹4L ₹5L crushed together
    between the majors.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_formatter(rupee_formatter())
    axis.set_minor_formatter(NullFormatter())


# ─────────────────────────────────────────────────────────── annotation


def note(ax: Axes, text: str, *, loc: str = "upper right", color: str = SECONDARY_INK) -> None:
    """Write a short explanatory note inside the axes.

    Every plot in this package carries the sentence that says what it *means* —
    "over-confident: the band is too narrow", "n = 36, treat this point as
    noise". A chart that needs the surrounding prose to be interpretable stops
    being evidence the moment someone screenshots it.
    """
    positions = {
        "upper right": (0.98, 0.97, "right", "top"),
        "upper left": (0.02, 0.97, "left", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left": (0.02, 0.03, "left", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        color=color,
        linespacing=1.45,
        zorder=10,
        # A near-opaque plate in the surface colour behind the words. Notes sit
        # inside the axes (so they survive being composed into someone else's
        # multi-panel figure, where anything drawn *outside* the axes would land
        # on the neighbouring panel) — and a note you cannot read because a
        # boxplot whisker runs through it is a note that isn't doing its job.
        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.88, "pad": 3.0},
    )


# ──────────────────────────────────────────────────────── writing to disk


def save(fig: Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Write a figure to disk. **The only function in this package that does.**

    No plotting function writes a file as a side effect. That rule sounds fussy
    until you import a plotting module inside a test suite and find PNGs
    scattered through your repo, or run a notebook cell twice and silently
    overwrite the figure you were about to show someone. Drawing and saving are
    different decisions, so they are different functions.

    Args:
        fig: The figure to write.
        path: Where to write it. A bare filename (``"coverage.png"``) lands in
            :data:`DEFAULT_OUTPUT_DIR`; anything with a directory in it is used
            exactly as given. Missing parent directories are created.
        dpi: Dots per inch. 150 is crisp in a README without being a 4MB file.

    Returns:
        The path actually written, so callers can log or assert on it.
    """
    target = Path(path)
    if target.parent == Path("."):
        target = DEFAULT_OUTPUT_DIR / target
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return target
