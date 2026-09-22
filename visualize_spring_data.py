"""
Miniscope Spring 2026 Dataset Overview Figure
Publication-quality visualization of recording sessions.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 1. Parse all recording sessions ──────────────────────────────────────────

BASE = Path("/Users/sanskriti/Downloads/Miniscope_Spring_Data")

DATE_MAP = {
    "4_29_Recordings":  "Apr 29",
    "5_12_Recordings":  "May 12",
    "5_14_Recordings":  "May 14",
    "5_20_Recordings":  "May 20",
    "5_22_Recordings":  "May 22",
}

def parse_session(date_folder, session_name):
    """Return a dict of metadata parsed from the folder name, or None to skip."""
    name = session_name.lower()

    # ── quality flags ──
    # Truly discard: test runs, accident recordings, unlabelled/ambiguous sessions
    DISCARD_TOKS = ["test_run", "testrun", "14_34_26", "14_49_58",
                    "16_00_09", "16_04_03", "16_07_40", "16_10_37",
                    "16_53_33", "16_59_23", "14_18_59_test", "16_11_58_testrun",
                    "16_14_31_bad_test"]
    if any(tok in name for tok in DISCARD_TOKS):
        return None
    # folders with ONLY a timestamp and nothing else (e.g. "14_34_26")
    if re.fullmatch(r'\d{2}_\d{2}_\d{2}', session_name):
        return None

    # Keep but flag caveated sessions
    has_caveat = any(tok in name for tok in [
        "kindofbad", "noresponse", "novideo", "nomovement",
        "blinking", "onlyedges", "onlysmalledges", "only_edge",
        "accidentphone", "notmoving", "_bad_", "bad_diseased",
        "topleftedge", "small area", "norespon",
    ])
    # Sessions explicitly marked bad with no usable data label
    if "bad_test" in name:
        return None

    quality = "With Caveats" if has_caveat else "Good"

    # ── condition  (use simple substring after splitting on _ and , ) ──
    tokens = re.split(r'[_,\s]+', name)
    def has_tok(prefixes):
        return any(any(t.startswith(p) for p in prefixes) for t in tokens)

    if has_tok(["diseas", "disease"]):
        condition = "Diseased"
    elif has_tok(["control", "contol"]):
        condition = "Control"
    else:
        return None  # unidentifiable

    # ── modality ──
    if any(t in ("fluor", "flour") or t.startswith("fluor") or t.startswith("flour")
           for t in tokens):
        modality = "Fluorescence"
    elif "bf" in tokens or any(t.startswith("brightfield") for t in tokens):
        modality = "Brightfield"
    else:
        modality = "Unknown"

    # ── pacing ──
    hz_match  = re.search(r'([\d.]+)\s*hz', name)
    bpm_match = re.search(r'pacing[_]?(\d+)|[_](\d+)(?:_|$)', name)

    if "spontan" in name or "spon" in name or "spontaneous" in name:
        pacing = "Spontaneous"
    elif hz_match:
        hz = float(hz_match.group(1))
        pacing = f"{hz} Hz"
    elif date_folder == "5_22_Recordings":
        bpm = None
        if bpm_match:
            for g in [bpm_match.group(1), bpm_match.group(2)]:
                if g:
                    candidate = int(g)
                    if candidate in (5, 10, 15, 20):
                        bpm = candidate
                        break
        # Also check tail number tokens like _10, _15, _20, _5
        if bpm is None:
            tail = re.search(r'[_](\d{1,2})(?:_5v)?$', name)
            if tail:
                candidate = int(tail.group(1))
                if candidate in (5, 10, 15, 20):
                    bpm = candidate
        if bpm is not None:
            pacing = f"{bpm} BPM"
        else:
            # withpacer but unknown rate
            pacing = "Paced (unknown)"
    else:
        pacing = "Unknown"

    # ── recording duration from timestamps ──
    ts_path = BASE / date_folder / session_name / "My_V4_Miniscope" / "timeStamps.csv"
    duration_s = None
    if ts_path.exists():
        try:
            lines = ts_path.read_text().strip().splitlines()
            last = lines[-1].split(",")
            duration_s = int(last[1]) / 1000.0
        except Exception:
            pass

    # ── subject number ──
    # Use [a-z]* (not \w*) so we don't consume the digit via underscore/digit backtrack
    sub_match = re.search(r'(?:control[a-z]*|contol[a-z]*|diseas[a-z]*)(\d)', name)
    subject_id = f"{condition[0]}{sub_match.group(1)}" if sub_match else f"{condition[0]}?"

    return dict(
        date=DATE_MAP[date_folder],
        date_folder=date_folder,
        session=session_name,
        condition=condition,
        modality=modality,
        pacing=pacing,
        quality=quality,
        subject_id=subject_id,
        duration_s=duration_s,
    )


sessions = []
for date_folder in sorted(BASE.iterdir()):
    if not date_folder.is_dir() or date_folder.name.startswith("."):
        continue
    df_name = date_folder.name
    if df_name not in DATE_MAP:
        continue
    for session_dir in sorted(date_folder.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        rec = parse_session(df_name, session_dir.name)
        if rec:
            sessions.append(rec)

print(f"Parsed {len(sessions)} usable recording sessions.")
for s in sessions:
    print(f"  {s['date']}  {s['condition']:8}  {s['modality']:14}  {s['pacing']:12}  "
          f"{s['quality']:16}  {s['subject_id']}  dur={s['duration_s']}")


# ── 2. Build summary tables ───────────────────────────────────────────────────

def count(pred): return sum(1 for s in sessions if pred(s))

conds      = ["Control", "Diseased"]
modalities = ["Brightfield", "Fluorescence"]
dates      = ["Apr 29", "May 12", "May 14", "May 20", "May 22"]

# condition × modality matrix
cond_mod = np.array([[count(lambda s, c=c, m=m: s['condition']==c and s['modality']==m)
                       for m in modalities] for c in conds])

# pacing distribution
all_pacings = sorted(set(s['pacing'] for s in sessions))
pacing_order = ["Spontaneous", "0.5 Hz", "1.0 Hz", "5 BPM", "10 BPM", "15 BPM", "20 BPM",
                "Paced (unknown)", "Unknown"]
pacing_order = [p for p in pacing_order if p in all_pacings]

pacing_by_cond = {c: [count(lambda s, c=c, p=p: s['condition']==c and s['pacing']==p)
                        for p in pacing_order] for c in conds}

# recordings per date per condition
date_cond = np.array([[count(lambda s, d=d, c=c: s['date']==d and s['condition']==c)
                        for c in conds] for d in dates])

# quality breakdown
qualities  = ["Good", "With Caveats"]
qual_cond  = np.array([[count(lambda s, c=c, q=q: s['condition']==c and s['quality']==q)
                         for q in qualities] for c in conds])

# per-subject session count
subjects = sorted(set(s['subject_id'] for s in sessions))
subj_counts = {subj: count(lambda s, subj=subj: s['subject_id']==subj)
               for subj in subjects}


# ── 3. Style constants ────────────────────────────────────────────────────────

CTL_COLOR  = "#2E86AB"   # teal-blue
DIS_COLOR  = "#E84855"   # coral-red
BF_SHADE   = 0.75        # darker tint for BF
FLU_SHADE  = 1.0         # full saturation for Fluor

GOOD_COLOR = "#3BB273"
CAV_COLOR  = "#F4A261"

def tint(hex_color, factor):
    """Lighten a hex color by factor (1=original, <1=darker, >1=lighter towards white)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    if factor > 1:
        r = int(r + (255 - r) * (factor - 1))
        g = int(g + (255 - g) * (factor - 1))
        b = int(b + (255 - b) * (factor - 1))
    else:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       180,
})

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", va="top", ha="left")


# ── 4. Build figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("white")

# outer title
fig.suptitle("Miniscope Spring 2026 — Dataset Overview",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.955,
         f"n = {len(sessions)} recording sessions  ·  "
         f"5 collection dates  ·  "
         f"Control: {count(lambda s: s['condition']=='Control')}  ·  "
         f"Diseased: {count(lambda s: s['condition']=='Diseased')}",
         ha="center", va="top", fontsize=9, color="#555555")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97,
                       top=0.91, bottom=0.09,
                       hspace=0.48, wspace=0.38)


# ── Panel A: donut – total Control vs Diseased ────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
totals = [count(lambda s, c=c: s['condition']==c) for c in conds]
wedges, texts = ax_a.pie(
    totals,
    colors=[CTL_COLOR, DIS_COLOR],
    startangle=90,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=9),
)
# Annotate counts manually inside each wedge
for i, (wedge, val) in enumerate(zip(wedges, totals)):
    angle = (wedge.theta1 + wedge.theta2) / 2
    x = 0.68 * np.cos(np.radians(angle))
    y = 0.68 * np.sin(np.radians(angle))
    ax_a.text(x, y, str(val), ha="center", va="center",
              fontsize=13, fontweight="bold", color="white")

# centre annotation
ax_a.text(0, 0, f"{sum(totals)}\nsessions", ha="center", va="center",
          fontsize=10, fontweight="bold", color="#333333", linespacing=1.4)
ax_a.legend(wedges, conds, loc="lower center", bbox_to_anchor=(0.5, -0.12),
            ncol=2, frameon=False, handlelength=1.2)
ax_a.set_title("A   Condition Split", loc="left", pad=6)


# ── Panel B: grouped bar – modality × condition ───────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
x = np.arange(len(modalities))
w = 0.32
bars_ctl = ax_b.bar(x - w/2, cond_mod[0], w,
                    color=CTL_COLOR, label="Control", zorder=3)
bars_dis = ax_b.bar(x + w/2, cond_mod[1], w,
                    color=DIS_COLOR, label="Diseased", zorder=3)

for bar in list(bars_ctl) + list(bars_dis):
    h = bar.get_height()
    if h > 0:
        ax_b.text(bar.get_x() + bar.get_width()/2, h + 0.25,
                  str(int(h)), ha="center", va="bottom", fontsize=8.5,
                  fontweight="bold", color=bar.get_facecolor())

ax_b.set_xticks(x)
ax_b.set_xticklabels(["Brightfield\n(Mechanical)", "Fluorescence\n(Optical)"])
ax_b.set_ylabel("# Sessions")
ax_b.set_ylim(0, max(cond_mod.max() * 1.3, 4))
ax_b.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_b.legend(frameon=False, loc="upper right")
ax_b.set_title("B   Imaging Modality", loc="left", pad=6)
ax_b.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)


# ── Panel C: horizontal stacked bar – pacing × condition ─────────────────────
ax_c = fig.add_subplot(gs[0, 2])
y = np.arange(len(pacing_order))
h = 0.35
bars_c = ax_c.barh(y + h/2, pacing_by_cond["Control"],   h, color=CTL_COLOR, label="Control",  zorder=3)
bars_d = ax_c.barh(y - h/2, pacing_by_cond["Diseased"],  h, color=DIS_COLOR, label="Diseased", zorder=3)

for bar in list(bars_c) + list(bars_d):
    w_val = bar.get_width()
    if w_val > 0:
        ax_c.text(w_val + 0.1, bar.get_y() + bar.get_height()/2,
                  str(int(w_val)), va="center", ha="left", fontsize=7.5,
                  fontweight="bold")

ax_c.set_yticks(y)
ax_c.set_yticklabels(pacing_order, fontsize=8)
ax_c.set_xlabel("# Sessions")
ax_c.set_xlim(0, max(max(pacing_by_cond["Control"]), max(pacing_by_cond["Diseased"])) * 1.45 + 1)
ax_c.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_c.legend(frameon=False, loc="lower right")
ax_c.set_title("C   Pacing Protocol", loc="left", pad=6)
ax_c.grid(axis="x", linewidth=0.5, alpha=0.4, zorder=0)



# ── Panel D: timeline – sessions per collection date ─────────────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
x = np.arange(len(dates))
w = 0.3
bars_dc = ax_d.bar(x - w/2, date_cond[:, 0], w, color=CTL_COLOR, label="Control",  zorder=3)
bars_dd = ax_d.bar(x + w/2, date_cond[:, 1], w, color=DIS_COLOR, label="Diseased", zorder=3)

# total above each pair
for i, (nc, nd) in enumerate(date_cond):
    tot = nc + nd
    ax_d.text(i, max(nc, nd) + 0.35, f"n={tot}",
              ha="center", va="bottom", fontsize=8, color="#555555")

for bar in list(bars_dc) + list(bars_dd):
    h_val = bar.get_height()
    if h_val > 0:
        ax_d.text(bar.get_x() + bar.get_width()/2, h_val/2,
                  str(int(h_val)), ha="center", va="center",
                  fontsize=8, fontweight="bold", color="white")

ax_d.set_xticks(x)
ax_d.set_xticklabels(dates)
ax_d.set_ylabel("# Sessions")
ax_d.set_ylim(0, date_cond.max() * 1.5 + 1.5)
ax_d.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_d.legend(frameon=False)
ax_d.set_title("D   Collection Timeline", loc="left", pad=6)
ax_d.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)



# ── Panel E: quality + subject summary ───────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

# Heatmap: subject × modality session count
ctrl_subjects  = sorted(s for s in subjects if s.startswith("C"))
dis_subjects   = sorted(s for s in subjects if s.startswith("D"))
ordered_subj   = ctrl_subjects + dis_subjects

mat = np.array([[count(lambda s, subj=subj, m=m: s['subject_id']==subj and s['modality']==m)
                 for m in modalities] for subj in ordered_subj])

cmap_custom = matplotlib.colors.LinearSegmentedColormap.from_list(
    "wb", ["#f7f7f7", "#3f8ac2"])
im = ax_e.imshow(mat, aspect="auto", cmap=cmap_custom,
                 vmin=0, vmax=mat.max() if mat.max() > 0 else 1)

for i in range(len(ordered_subj)):
    for j in range(len(modalities)):
        val = mat[i, j]
        ax_e.text(j, i, str(val), ha="center", va="center",
                  fontsize=9, fontweight="bold",
                  color="white" if val > mat.max() * 0.55 else "#333333")

ax_e.set_xticks([0, 1])
ax_e.set_xticklabels(["BF", "Fluor"], fontsize=8)
ax_e.set_yticks(range(len(ordered_subj)))

yticklabels = []
for s in ordered_subj:
    cond_label = "Ctl" if s.startswith("C") else "Dis"
    num = s[1:] if s[1:] != "?" else "?"
    yticklabels.append(f"{cond_label} {num}")
ax_e.set_yticklabels(yticklabels, fontsize=7.5)

# color y-tick labels by condition
for tick, subj in zip(ax_e.get_yticklabels(), ordered_subj):
    tick.set_color(CTL_COLOR if subj.startswith("C") else DIS_COLOR)

# divider line between control and diseased rows
if ctrl_subjects:
    ax_e.axhline(len(ctrl_subjects) - 0.5, color="#aaaaaa", linewidth=1.2, linestyle="--")

cb = plt.colorbar(im, ax=ax_e, fraction=0.046, pad=0.12)
cb.set_label("Sessions", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax_e.set_title("E   Sessions per Subject\n& Modality", loc="left", pad=6)



# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "BF = Brightfield (mechanical contrast)  ·  Fluor = Fluorescence (GCaMP / calcium indicator)  ·  "
         "BPM = electrical pacing beats per minute  ·  Hz = optical pacing frequency",
         ha="center", va="bottom", fontsize=7, color="#888888", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = Path("/Users/sanskriti/Documents/GitHub/miniscope_fullstack/plots/spring_data_overview.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}")
plt.close()
