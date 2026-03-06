"""
Unified experiment configuration for all comparison scripts.

All experiment paths, drug labels, colors, and shared settings live here.
Individual scripts import from this module instead of defining their own configs.
"""

# ============================================================================
# DRUG LABELS & COLORS
# ============================================================================

DRUG_LABELS = ["Control", "Thar 0.1 nM", "DOF 100 nM", "QUAN 30 nM"]

# Color mapping: drug keyword -> (trend_color, raw_color)
COLORS = {
    "control": ("#999999", "#cccccc"),
    "thar":    ("#2E8B57", "#90d4aa"),
    "dof":     ("#7B2D8E", "#c490d4"),
    "quan":    ("#2E6DB4", "#7fb3e0"),
}

# Per-drug colors (keyed by exact drug label)
DRUG_COLORS = {
    "Control":      ("#999999", "#cccccc"),
    "Thar 0.1 nM":  ("#2E8B57", "#90d4aa"),
    "DOF 100 nM":   ("#7B2D8E", "#c490d4"),
    "QUAN 30 nM":   ("#2E6DB4", "#7fb3e0"),
}

# Flat list of bar-chart colors (one per drug, same order as DRUG_LABELS)
BAR_COLORS = ["#999999", "#2E8B57", "#7B2D8E", "#2E6DB4"]


def get_experiment_colors(label):
    """Return (trend_color, raw_color) based on keywords in the label.
    Falls back to control (grey) if no drug keyword matches."""
    label_lower = label.lower()
    for key in ("thar", "dof", "quan"):
        if key in label_lower:
            return COLORS[key]
    return COLORS["control"]


# ============================================================================
# SHARED SETTINGS
# ============================================================================

ROI_INDEX = 1            # Which ROI to plot (1-based)
CLIP_SEC = 8             # Show / analyse first N seconds (0 = all)
SMOOTH_WINDOW_SEC = 0.15
SMOOTH_POLYORDER = 2


# ============================================================================
# FLUORESCENT EXPERIMENTS — for transient comparison & decay50
# ============================================================================

# Each entry: (label, path_to_fluorescence_traces_csv)
FLUORESCENT_EXPERIMENTS = [
    ("Control 0.5 Hz",  "plots/_16_53_20_Control2_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("Thar 0.1 nM",     "plots/_15_18_35_0.1nM_Thar4_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("DOF 100 nM",      "plots/_16_01_01_100nM_DOF1_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("QUAN 30 nM",      "plots/_17_25_00_QUAN3_30nM_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
]

# Decay50 uses the same experiments but with "Control" label (no "0.5 Hz" suffix)
DECAY50_EXPERIMENTS = [
    ("Control",     "plots/_16_53_20_Control2_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("Thar 0.1 nM", "plots/_15_18_35_0.1nM_Thar4_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("DOF 100 nM",  "plots/_16_01_01_100nM_DOF1_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
    ("QUAN 30 nM",  "plots/_17_25_00_QUAN3_30nM_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv"),
]


# ============================================================================
# MECHANICAL EXPERIMENTS — for mechanical transient comparison
# ============================================================================

# Each value is a list of folder paths containing contractility CSVs.
MECHANICAL_EXPERIMENTS = {
    "Control": [
        "plots/batch_results_quan/mechanical/16_57_23_Control3_Bright_0.5Hz_30fps_0",
        "plots/batch_results_quan/mechanical/15_30_12_Control1_Brightfield_0.5hz_30fps_0",
        "plots/batch_results_quan/mechanical/16_52_04_Control2_Bright_0.5Hz_30fps_0",
    ],
    "Thar 0.1 nM": [
        "plots/0",  # 15_12_31_0.1nM_Thar4_Brightfield_0.5Hz_30fps
    ],
    "DOF 100 nM": [
        "/Users/sanskriti/Downloads/Dof Data 02_19/16_03_51_100nM_DOF3_Brightfield_0.5Hz_30fps",
        "/Users/sanskriti/Downloads/Dof Data 02_19/16_02_19_100nM_DOF3_Brightfield_0.5Hz_30fps",
    ],
    "QUAN 30 nM": [
        "plots/batch_results_quan/mechanical/17_21_02_QUAN2_30nM_Bright_0.5Hz_30fps_0",
        "plots/batch_results_quan/mechanical/17_46_15_QUAN3_3uM_Bright_0.5Hz_30fps_0",
        "plots/batch_results_quan/mechanical/17_18_47_QUAN1_30nM_Bright_0.5Hz_30fps_0",
        "plots/batch_results_quan/mechanical/17_23_34_QUAN3_30nM_Bright_0.5Hz_30fps_0",
    ],
}


# ============================================================================
# IBI EXPERIMENTS — expanded recordings for beat-to-beat variability stats
# ============================================================================

BATCH_FLUO = "plots/batch_results_quan/fluorescent"

IBI_EXPERIMENTS = {
    "Control": [
        "plots/_16_53_20_Control2_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/15_29_18_Control1_Fluoresent_0.5hz_30fps_0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/16_58_14_Control3_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/16_59_37_Control3_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/17_02_57_Control4_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
    ],
    "Thar 0.1 nM": [
        "plots/_15_18_35_0.1nM_Thar4_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv",
    ],
    "DOF 100 nM": [
        "plots/_16_01_01_100nM_DOF1_Fluoresent_0.5Hz_25fps_vid0/fluorescence_traces.csv",
        "/Users/sanskriti/Downloads/Dof Data 02_19 2/GOOD_16_03_51_100nM_DOF3_Brightfield_0.5Hz_30fps/fluorescence_traces.csv",
        "/Users/sanskriti/Downloads/Dof Data 02_19 2/GREAT_16_02_19_100nM_DOF3_Brightfield_0.5Hz_30fps/fluorescence_traces.csv",
    ],
    "QUAN 30 nM": [
        "plots/_17_25_00_QUAN3_30nM_Flo_0.5Hz_25fps_vid0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/17_19_31_QUAN1_30nM_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/17_22_11_QUAN2_30nM_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
        f"{BATCH_FLUO}/17_24_37_QUAN3_30nM_Flo_0.5Hz_25fps_0/fluorescence_traces.csv",
    ],
}
