"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        QML-BASED TRAFFIC MANAGEMENT IN 6G NETWORKS                         ║
║        Final Semester Research Project — Complete Runnable Model            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Framework : PennyLane + PyTorch + Scikit-learn                             ║
║  Task      : 6G Network Slice Classification (eMBB / URLLC / mMTC)         ║
║  Model     : Hybrid Variational Quantum Circuit (VQC) Neural Network        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN (Google Colab — FREE, Recommended):                             ║
║  ──────────────────────────────────────────────                             ║
║  1. Open https://colab.research.google.com/                                 ║
║  2. Upload this file  →  File > Upload notebook  OR paste into a cell       ║
║  3. Run this ONE cell first to install:                                     ║
║       !pip install pennylane pennylane-lightning torch scikit-learn         ║
║                matplotlib seaborn pandas numpy tqdm                         ║
║  4. Then run the script:  !python qml_6g_traffic_v2.py                      ║
║                                                                             ║
║  HOW TO RUN (Local Machine):                                                ║
║  ───────────────────────────                                                ║
║  pip install pennylane pennylane-lightning torch scikit-learn               ║
║              matplotlib seaborn pandas numpy tqdm                           ║
║  python qml_6g_traffic_v2.py                                               ║
║                                                                             ║
║  OUTPUTS (auto-saved in same folder):                                       ║
║    01_training_curves.png          — Loss & Accuracy per epoch              ║
║    02_confusion_matrix.png         — QML confusion matrix (normalised)      ║
║    03_precision_recall_f1.png      — Per-class P / R / F1 bar chart         ║
║    04_roc_curves.png               — ROC curves + AUC for all 3 classes     ║
║    05_model_comparison.png         — QML vs RF vs SVM vs GBM radar chart    ║
║    06_feature_analysis.png         — Scatter, correlation, importance       ║
║    07_quantum_circuit.png          — VQC circuit diagram                    ║
║    08_per_class_metrics.png        — Precision/Recall per class per model   ║
║    09_learning_rate_schedule.png   — LR decay curve                         ║
║    10_full_dashboard.png           — Combined dashboard (all plots)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  0.  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import pennylane as qml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = "../Outputs/qml_6g_outputs_3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette (dark theme) ─────────────────────────────────────────────
BG     = '#0d0d1f'
BG2    = '#1a1a2e'
BG3    = '#0a0a1a'
TC     = '#00e5ff'   # title cyan
TX     = '#e0e0e0'   # general text
GC     = '#2a2a3a'   # grid

C_EMBB  = '#ff6b6b'
C_URLLC = '#4ecdc4'
C_MMTC  = '#ffd93d'
C_QML   = '#6bcb77'
C_RF    = '#ff9f43'
C_SVM   = '#a29bfe'
C_GBM   = '#fd79a8'
CLRS    = [C_EMBB, C_URLLC, C_MMTC]
NAMES   = ['eMBB', 'URLLC', 'mMTC']

plt.rcParams.update({
    'text.color': TX, 'axes.labelcolor': TX,
    'xtick.color': TX, 'ytick.color': TX,
    'axes.edgecolor': '#444', 'figure.facecolor': BG,
    'axes.facecolor': BG2, 'font.family': 'monospace'
})

# ═══════════════════════════════════════════════════════════════════════════════
#  1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
N_QUBITS   = 4
N_LAYERS   = 3       # increased for better expressibility
N_CLASSES  = 3
N_FEATURES = 8
EPOCHS     = 30      # set low for testing; increase to 30-40 for final results
BATCH_SIZE = 32
LR         = 0.005
N_SAMPLES  = 900     # more data → better convergence

HEADER = "═" * 68
print(HEADER)
print("  QML-BASED TRAFFIC MANAGEMENT IN 6G NETWORKS")
print("  Hybrid Variational Quantum Circuit (VQC) Classifier")
print(HEADER)
print(f"  Qubits       : {N_QUBITS}")
print(f"  VQC Layers   : {N_LAYERS}")
print(f"  Classes      : {N_CLASSES}  (eMBB | URLLC | mMTC)")
print(f"  Features     : {N_FEATURES}")
print(f"  Epochs       : {EPOCHS}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Learning rate: {LR}")
print(f"  Samples      : {N_SAMPLES}")
print(HEADER)

# ═══════════════════════════════════════════════════════════════════════════════
#  2.  DATASET  —  structured after Kim & Choi (2022/2023) 5G Traffic Dataset
# ═══════════════════════════════════════════════════════════════════════════════
"""
SOURCE DATASET
──────────────
  Title   : 5G Traffic Datasets
  Authors : Yong-Hoon Choi, Daegyeom Kim, Myeongjin Ko
  DOI     : 10.21227/ewhk-n061  (IEEE DataPort, 2023)
  Kaggle  : https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets
  License : CC BY 4.0

COLLECTION METHOD
─────────────────
  • Device    : Samsung Galaxy A90 5G (Qualcomm Snapdragon X50 5G modem)
  • Tool      : PCAPdroid (Android packet sniffer — no root required)
  • Operator  : Major South Korean mobile operator (5G live network)
  • Period    : May – October 2022  |  Total duration: 328 hours
  • Format    : CSV, timestamp-mapped time-series, packet-header level
  • No mixed background traffic — each application captured in isolation

ORIGINAL TRAFFIC APPLICATIONS IN THE DATASET
─────────────────────────────────────────────
  eMBB  (high-throughput video/streaming):
    ├── Netflix         — OTT video streaming (downlink-heavy)
    ├── Amazon Prime    — OTT video streaming
    ├── AfreecaTV       — Live streaming platform
    ├── Zoom            — Video conferencing (symmetric)
    ├── MS Teams        — Video conferencing
    └── Google Meet     — Video conferencing

  URLLC  (low-latency, time-sensitive interactive):
    ├── Cloud Gaming    — Remote server renders game, streams video + input
    └── Mobile Gaming   — Interactive online game (Roblox – Collect All Pets)

  mMTC  (metaverse / many-device / background):
    ├── Zepeto          — Social metaverse (light, bursty presence updates)
    └── Roblox Metaverse— Social metaverse exploration (low-bitrate idle)

  NOTE: In 5G/6G research, metaverse idle traffic and background IoT-like
  device signalling are assigned to mMTC because they exhibit the defining
  properties: low per-device throughput, high connection count, and latency
  tolerance. See Hsieh et al. ICC 2021 and TRACTOR (arXiv 2312.07896).

REAL CSV COLUMNS (packet-header time-series from PCAPdroid)
────────────────────────────────────────────────────────────
  timestamp_us        — Packet capture timestamp (microseconds, Unix epoch)
  frame_len_bytes     — Raw packet frame length in bytes  (payload + headers)
  ip_proto            — IP protocol number: 6=TCP, 17=UDP
  src_port            — Source port number
  dst_port            — Destination port number
  tcp_flags           — TCP control flags byte (SYN/ACK/FIN/RST …)
  inter_arrival_us    — Time since previous packet in the same flow (µs)
  direction           — 0=uplink (UE→network), 1=downlink (network→UE)

  DERIVED / FLOW-LEVEL FEATURES (computed per 1-second window):
  throughput_kbps     — Total bytes in window × 8 / 1000
  avg_pkt_size_bytes  — Mean frame_len_bytes in window
  pkt_inter_arr_mean_us — Mean inter-arrival time in window
  pkt_inter_arr_std_us  — Std dev of inter-arrival time (jitter proxy)
  flow_pkt_count      — Packets per second in window
  dl_ul_ratio         — Downlink/Uplink byte ratio in window
  bitrate_mean_kbps   — Rolling mean bitrate (from dataset statistical summary)
  bitrate_std_kbps    — Rolling std dev of bitrate (from dataset stat summary)

HOW TO USE THE REAL DATASET INSTEAD OF THIS SIMULATION
───────────────────────────────────────────────────────
  Step 1 — Download:
    kaggle datasets download -d kimdaegyeom/5g-traffic-datasets
    unzip 5g-traffic-datasets.zip -d ./5g_data/

  Step 2 — Load a single traffic type CSV (e.g. Netflix):
    df_netflix = pd.read_csv('./5g_data/Netflix.csv')

  Step 3 — Add slice labels to each application file:
    EMBB_APPS   = ['Netflix','AmazonPrime','AfreecaTV','Zoom','MSTeams','GoogleMeet']
    URLLC_APPS  = ['CloudGaming','MobileGaming']
    MMTC_APPS   = ['Zepeto','Roblox']

    dfs = []
    for app in EMBB_APPS:
        d = pd.read_csv(f'./5g_data/{app}.csv')
        d['traffic_class'] = 0          # eMBB
        dfs.append(d)
    for app in URLLC_APPS:
        d = pd.read_csv(f'./5g_data/{app}.csv')
        d['traffic_class'] = 1          # URLLC
        dfs.append(d)
    for app in MMTC_APPS:
        d = pd.read_csv(f'./5g_data/{app}.csv')
        d['traffic_class'] = 2          # mMTC
        dfs.append(d)
    df_real = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)

  Step 4 — Select the 8 features below and replace df = generate_5g_dataset()
           with df = df_real[feat_cols + ['traffic_class']]

SIMULATION FIDELITY
───────────────────
  The synthetic values below are calibrated against the statistical summaries
  published in the IEEE DataPort documentation and the companion paper:
    "ML-Based 5G Traffic Generation for Practical Simulations Using Open
     Datasets", Choi et al., IEEE Communications Magazine, vol. 61, no. 9,
     Sep 2023, doi: 10.1109/MCOM.001.2200679

  Key calibration targets:
  ┌──────────────────────┬──────────────┬───────────────┬─────────────────┐
  │ Feature              │ eMBB         │ URLLC         │ mMTC            │
  ├──────────────────────┼──────────────┼───────────────┼─────────────────┤
  │ throughput_kbps      │ ~8000–25000  │ ~3000–8000    │ ~50–500         │
  │ avg_pkt_size_bytes   │ ~900–1400    │ ~200–600      │ ~80–300         │
  │ inter_arrival_mean_us│ ~100–800     │ ~20–200       │ ~5000–50000     │
  │ inter_arrival_std_us │ ~200–1500    │ ~10–100       │ ~2000–30000     │
  │ flow_pkt_count/s     │ ~200–1200    │ ~80–400       │ ~2–30           │
  │ dl_ul_ratio          │ ~8–25 (DL)   │ ~0.8–2.5      │ ~1–5            │
  │ bitrate_mean_kbps    │ ~5000–20000  │ ~2000–7000    │ ~20–200         │
  │ bitrate_std_kbps     │ ~1000–5000   │ ~500–2000     │ ~5–50           │
  └──────────────────────┴──────────────┴───────────────┴─────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET PATH  ← change this to wherever you extracted the Kaggle zip
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_PATH = r"/5G_Traffic_Datasets"

# ── Slice label mapping ───────────────────────────────────────────────────────
# Based on ITU IMT-2020 / 3GPP TS 22.261 service categories.
# Each key is the CSV filename stem (without .csv); value is the integer label.
#
#  eMBB  (0) — enhanced Mobile Broadband
#              High sustained throughput, large packets, high DL/UL ratio.
#              Sources: live streaming + video conferencing applications.
#
#  URLLC (1) — Ultra-Reliable Low-Latency Communications
#              Low inter-arrival time, tight jitter, near-symmetric DL/UL.
#              Sources: cloud gaming + mobile gaming (interactive, real-time).
#
#  mMTC  (2) — massive Machine-Type Communications
#              Very low throughput, sparse bursty packets, latency-tolerant.
#              Sources: metaverse idle/background presence traffic.
#
SLICE_MAP = {
    # ── eMBB ──────────────────────────────────────────────────────────────────
    'NaverNOW'   : 0,   # Naver NOW live streaming
    'AfreecaTV'  : 0,   # AfreecaTV live streaming
    'YouTube'    : 0,   # YouTube Live streaming
    'Zoom'       : 0,   # Zoom video conferencing
    'MSteams'    : 0,   # Microsoft Teams conferencing
    'GoogleMeet' : 0,   # Google Meet conferencing
    # ── URLLC ─────────────────────────────────────────────────────────────────
    'Cloudgaming': 1,   # Cloud gaming (server-rendered, streamed)
    'Mobilegame' : 1,   # Mobile game connected to the Internet
    # ── mMTC ──────────────────────────────────────────────────────────────────
    'Zepeto'     : 2,   # Zepeto metaverse (15 h camping session)
    'Roblox'     : 2,   # Roblox metaverse (auto-clicker idle, 25 h)
}

# ── Raw CSV columns present in the Kim & Choi dataset ────────────────────────
# The CSV files contain packet-header time-series from PCAPdroid captures.
# Column names below match the actual headers in the distributed CSV files.
#
#   time          — packet capture timestamp (seconds, relative to flow start)
#   length        — IP packet length in bytes (frame payload, no Ethernet hdr)
#   ip_proto      — IP protocol: 6=TCP, 17=UDP
#   src_port      — source port
#   dst_port      — destination port
#   flags         — TCP flags byte (0 for UDP)
#
# We derive 8 flow-level features per 1-second sliding window:
#
FEAT_COLS = [
    'throughput_kbps',        # window bytes × 8 / 1000
    'avg_pkt_size_bytes',     # mean packet length in window
    'inter_arrival_mean_us',  # mean inter-packet gap in window (µs)
    'inter_arrival_std_us',   # std dev of inter-packet gap — jitter proxy (µs)
    'flow_pkt_count',         # packets per second in window
    'dl_ul_ratio',            # downlink / uplink byte ratio (dst_port heuristic)
    'bitrate_mean_kbps',      # rolling 5-window mean of throughput_kbps
    'bitrate_std_kbps',       # rolling 5-window std of throughput_kbps
]

feat_cols  = FEAT_COLS
N_FEATURES = len(feat_cols)   # 8, matches quantum circuit encoding


# ── Feature extraction helpers ────────────────────────────────────────────────

def extract_features_from_csv(path: str, label: int,
                               window_sec: float = 1.0,
                               max_windows: int = 300) -> pd.DataFrame:
    """
    Load one application CSV and extract sliding-window flow features.

    Parameters
    ----------
    path        : full path to the CSV file
    label       : integer slice class (0=eMBB, 1=URLLC, 2=mMTC)
    window_sec  : observation window size in seconds (default 1 s)
    max_windows : maximum rows to extract per file (caps memory usage)

    Returns
    -------
    pd.DataFrame with columns = FEAT_COLS + ['traffic_class']
    """
    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(path)
    except Exception as e:
        print(f"      [WARN] Could not read {path}: {e}")
        return pd.DataFrame()

    # Normalise column names to lowercase with no leading/trailing spaces
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # ── Identify timestamp and length columns (handle naming variations) ──────
    time_col = next((c for c in df_raw.columns
                     if c in ('time', 'timestamp', 'time_s', 'relative_time')), None)
    len_col  = next((c for c in df_raw.columns
                     if c in ('length', 'len', 'pkt_len', 'frame_len',
                              'ip_len', 'packet_length')), None)
    port_col = next((c for c in df_raw.columns
                     if c in ('dst_port', 'dport', 'dst port', 'destination_port')), None)

    if time_col is None or len_col is None:
        print(f"      [WARN] {path}: required columns not found "
              f"(cols={list(df_raw.columns)[:8]}…) — skipping")
        return pd.DataFrame()

    df_raw = df_raw[[time_col, len_col] +
                    ([port_col] if port_col else [])].copy()
    df_raw.columns = (['time', 'length'] +
                      (['dst_port'] if port_col else []))
    df_raw = df_raw.dropna(subset=['time', 'length'])
    df_raw['time']   = pd.to_numeric(df_raw['time'],   errors='coerce')
    df_raw['length'] = pd.to_numeric(df_raw['length'], errors='coerce')
    df_raw = df_raw.dropna().sort_values('time').reset_index(drop=True)

    if len(df_raw) < 10:
        return pd.DataFrame()

    # ── Sliding 1-second windows ───────────────────────────────────────────────
    rows = []
    t_start = df_raw['time'].iloc[0]
    t_end   = df_raw['time'].iloc[-1]
    t       = t_start

    while t + window_sec <= t_end and len(rows) < max_windows:
        mask = (df_raw['time'] >= t) & (df_raw['time'] < t + window_sec)
        win  = df_raw[mask]

        if len(win) < 2:
            t += window_sec
            continue

        lengths = win['length'].values.astype(float)

        # inter-arrival times
        times_w = win['time'].values.astype(float)
        iats_us = np.diff(times_w) * 1e6          # seconds → microseconds

        # DL/UL heuristic: dst_port < 1024 → uplink (client→server request)
        if 'dst_port' in win.columns:
            uplink_bytes   = win.loc[win['dst_port'] < 1024, 'length'].sum()
            downlink_bytes = win.loc[win['dst_port'] >= 1024, 'length'].sum()
            dl_ul = (downlink_bytes / uplink_bytes) if uplink_bytes > 0 else float(downlink_bytes)
        else:
            dl_ul = 1.0   # unknown — assume symmetric

        throughput = lengths.sum() * 8 / 1000.0   # kbps

        rows.append({
            'throughput_kbps'       : throughput,
            'avg_pkt_size_bytes'    : float(np.mean(lengths)),
            'inter_arrival_mean_us' : float(np.mean(iats_us)) if len(iats_us) > 0 else 0.0,
            'inter_arrival_std_us'  : float(np.std(iats_us))  if len(iats_us) > 1 else 0.0,
            'flow_pkt_count'        : float(len(win)),
            'dl_ul_ratio'           : float(np.clip(dl_ul, 0.1, 100.0)),
        })
        t += window_sec

    if not rows:
        return pd.DataFrame()

    df_feat = pd.DataFrame(rows)

    # rolling bitrate mean/std (window size = 5 rows)
    df_feat['bitrate_mean_kbps'] = (df_feat['throughput_kbps']
                                    .rolling(5, min_periods=1).mean())
    df_feat['bitrate_std_kbps']  = (df_feat['throughput_kbps']
                                    .rolling(5, min_periods=1).std()
                                    .fillna(0.0))

    df_feat['traffic_class'] = label
    return df_feat[feat_cols + ['traffic_class']]


def load_real_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Walk DATASET_PATH, find CSV files whose stems match SLICE_MAP,
    extract features from each, and concatenate into one DataFrame.

    Falls back to the synthetic generator if no files are found or if
    fewer than 90 valid windows could be extracted (30 per class minimum).
    """
    import os

    print(f"      Scanning: {dataset_path}")
    all_frames = []
    found      = []
    missing    = []

    for stem, label in SLICE_MAP.items():
        # Try both exact case and common variations
        candidates = [
            os.path.join(dataset_path, f"{stem}.csv"),
            os.path.join(dataset_path, f"{stem.lower()}.csv"),
            os.path.join(dataset_path, f"{stem.upper()}.csv"),
        ]
        # Also search one level deep in sub-folders
        try:
            for entry in os.scandir(dataset_path):
                if entry.is_dir():
                    candidates.append(os.path.join(entry.path, f"{stem}.csv"))
                    candidates.append(os.path.join(entry.path, f"{stem.lower()}.csv"))
        except PermissionError:
            pass

        matched = next((p for p in candidates if os.path.isfile(p)), None)

        if matched:
            found.append((stem, matched, label))
        else:
            missing.append(stem)

    if missing:
        print(f"      [INFO] Files not found (will be skipped): {missing}")

    for stem, path, label in found:
        slice_name = {0:'eMBB', 1:'URLLC', 2:'mMTC'}[label]
        print(f"      Loading  [{slice_name:5s}]  {os.path.basename(path)} ...", end=' ')
        df_app = extract_features_from_csv(path, label, max_windows=400)
        if df_app.empty:
            print("0 windows extracted — skipped")
        else:
            print(f"{len(df_app)} windows")
            all_frames.append(df_app)

    if not all_frames:
        return pd.DataFrame()

    df_combined = pd.concat(all_frames, ignore_index=True)

    # Hard clip for physically impossible values
    df_combined['throughput_kbps']       = df_combined['throughput_kbps'].clip(lower=1)
    df_combined['avg_pkt_size_bytes']    = df_combined['avg_pkt_size_bytes'].clip(lower=20, upper=1500)
    df_combined['inter_arrival_mean_us'] = df_combined['inter_arrival_mean_us'].clip(lower=0.1)
    df_combined['inter_arrival_std_us']  = df_combined['inter_arrival_std_us'].clip(lower=0.0)
    df_combined['flow_pkt_count']        = df_combined['flow_pkt_count'].clip(lower=1)
    df_combined['dl_ul_ratio']           = df_combined['dl_ul_ratio'].clip(lower=0.05, upper=200)
    df_combined['bitrate_mean_kbps']     = df_combined['bitrate_mean_kbps'].clip(lower=0.1)
    df_combined['bitrate_std_kbps']      = df_combined['bitrate_std_kbps'].clip(lower=0.0)

    return df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


def balance_classes(df: pd.DataFrame, min_per_class: int = 50) -> pd.DataFrame:
    """
    Under-sample majority classes so all three classes have equal size.
    Keeps at least min_per_class rows per class; raises if any class is below.
    """
    counts = df['traffic_class'].value_counts()
    n_each = counts.min()
    if n_each < min_per_class:
        raise ValueError(
            f"Too few samples in smallest class ({n_each}). "
            f"Need at least {min_per_class}. "
            f"Check that the CSV files loaded correctly."
        )
    balanced = (df.groupby('traffic_class', group_keys=False)
                  .apply(lambda g: g.sample(n_each, random_state=42)))
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Load dataset ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading 5G Traffic Dataset (Kim & Choi 2022)...")
print(f"      Path    : {DATASET_PATH}")
print( "      Source  : https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets")

df_raw = load_real_dataset(DATASET_PATH)

if df_raw.empty or df_raw['traffic_class'].nunique() < N_CLASSES:
    # ── Fallback: synthetic data if real files can't be read ─────────────────
    print("\n      [FALLBACK] Real files could not be loaded — using synthetic data.")
    print("      Check DATASET_PATH and CSV column names.")

    def _synth(n=900):
        rng = np.random.default_rng(42); data=[]; labels=[]
        spc = n // N_CLASSES
        for _ in range(spc):
            data.append([np.clip(rng.normal(14000,4000),10,None),
                         np.clip(rng.normal(1100,200),40,1500),
                         np.clip(rng.normal(350,120),1,None),
                         np.clip(rng.normal(800,250),1,None),
                         np.clip(rng.normal(650,180),1,None),
                         np.clip(rng.normal(15.0,4.0),0.1,None),
                         np.clip(rng.normal(12000,3500),1,None),
                         np.clip(rng.normal(2800,700),0.1,None)]); labels.append(0)
        for _ in range(spc):
            data.append([np.clip(rng.normal(5500,1200),10,None),
                         np.clip(rng.normal(380,100),40,1500),
                         np.clip(rng.normal(80,25),1,None),
                         np.clip(rng.normal(45,12),1,None),
                         np.clip(rng.normal(220,60),1,None),
                         np.clip(rng.normal(1.6,0.4),0.1,None),
                         np.clip(rng.normal(4800,1000),1,None),
                         np.clip(rng.normal(900,250),0.1,None)]); labels.append(1)
        for _ in range(spc):
            data.append([np.clip(rng.normal(180,80),10,None),
                         np.clip(rng.normal(160,60),40,1500),
                         np.clip(rng.normal(22000,8000),1,None),
                         np.clip(rng.normal(15000,5000),1,None),
                         np.clip(rng.normal(12,5),1,None),
                         np.clip(rng.normal(2.5,0.8),0.1,None),
                         np.clip(rng.normal(120,50),1,None),
                         np.clip(rng.normal(30,12),0.1,None)]); labels.append(2)
        d = pd.DataFrame(data, columns=feat_cols)
        d['traffic_class'] = labels
        return d.sample(frac=1, random_state=42).reset_index(drop=True)

    df = _synth(N_SAMPLES)
    print(f"      Mode    : Synthetic fallback")
else:
    # ── Balance and cap total sample count ───────────────────────────────────
    df = balance_classes(df_raw)
    # cap at N_SAMPLES rows to keep training time reasonable
    cap = N_SAMPLES // N_CLASSES
    df  = (df.groupby('traffic_class', group_keys=False)
             .apply(lambda g: g.head(cap))
             .sample(frac=1, random_state=42)
             .reset_index(drop=True))
    print(f"      Mode    : Real CSV data")

print(f"      Shape   : {df.shape}")
print(f"      Classes : {df['traffic_class'].value_counts().sort_index().to_dict()}")
print(f"      Mapping : 0=eMBB  (NaverNOW / AfreecaTV / YouTube / Zoom / Teams / Meet)")
print(f"                1=URLLC (Cloudgaming / Mobilegame)")
print(f"                2=mMTC  (Zepeto / Roblox)")
print()
print("  Feature statistics:")
print(df[feat_cols].describe().round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
#  3.  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Preprocessing...")
X = df[feat_cols].values
y = df['traffic_class'].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_angle = np.clip(X_sc, -np.pi, np.pi)   # map to valid rotation-angle range

X_tr, X_te, y_tr, y_te = train_test_split(
    X_angle, y, test_size=0.25, stratify=y, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

dataset    = TensorDataset(X_tr_t, y_tr_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)   # drop last incomplete batch → prevents BatchNorm crash on batch_size=1

print(f"      Train samples : {len(X_tr)}")
print(f"      Test  samples : {len(X_te)}")
print(f"      Features      : {N_FEATURES}  (throughput, pkt_size, IAT, jitter, pkt_count, DL/UL, bitrate_mean, bitrate_std)")
print(f"                       →  angle-encoded into {N_QUBITS} qubits")

# ═══════════════════════════════════════════════════════════════════════════════
#  4.  QUANTUM CIRCUIT  (Variational Quantum Circuit — VQC)
# ═══════════════════════════════════════════════════════════════════════════════
"""
VQC Design:
  ┌─────────────────────────────────────────────────────────────────┐
  │  ENCODING LAYER                                                 │
  │    AngleEmbedding(inputs[:4], rotation='Y')                     │
  │    → Ry(θᵢ)|0⟩  for each qubit i                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  VARIATIONAL LAYERS  ×  N_LAYERS                                │
  │    StronglyEntanglingLayers(weights)                            │
  │    → Rz, Ry, Rz on each qubit  +  CNOT ring entanglement        │
  ├─────────────────────────────────────────────────────────────────┤
  │  MEASUREMENT                                                    │
  │    PauliZ expectation  ⟨Z⟩ᵢ  on each qubit                     │
  │    → N_QUBITS scalar outputs ∈ [−1, +1]                         │
  └─────────────────────────────────────────────────────────────────┘

WHY WE BYPASS TorchLayer:
  TorchLayer internally calls _to_qfunc_output_type which wraps the circuit
  output back into a pnp.tensor (PennyLane numpy tensor) regardless of the
  interface setting on older PL versions. That wrapper calls .numpy() on a
  torch tensor that still has requires_grad=True → RuntimeError.

  The fix: use a raw qml.device + plain qnode with numpy inputs/outputs,
  then wrap the entire batch loop inside a custom torch.autograd.Function
  that handles the torch ↔ numpy boundary explicitly and cleanly computes
  parameter-shift gradients manually. This is version-agnostic and works on
  PennyLane ≥ 0.20 with any PyTorch version.
"""

# ── Raw PennyLane device (no interface lock-in) ──────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)

# Plain qnode — NO interface argument so PL uses its default numpy/autograd.
# Inputs and weights are plain numpy arrays; outputs are plain numpy arrays.
# The torch↔numpy boundary is handled explicitly in QuantumFunction below.
@qml.qnode(dev, diff_method='parameter-shift')
def _qcircuit_raw(inputs, weights):
    """Raw qnode — returns a list (PennyLane default for multiple observables)."""
    qml.AngleEmbedding(inputs[:N_QUBITS], wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def quantum_circuit_np(inputs, weights):
    """
    Wraps _qcircuit_raw and converts its list output to np.ndarray so that
    arithmetic (subtraction, dot product) works in the parameter-shift backward.
    inputs  : np.ndarray (N_QUBITS,)
    weights : np.ndarray (N_LAYERS, N_QUBITS, 3)
    returns : np.ndarray (N_QUBITS,)  — PauliZ expectation values
    """
    return np.array(_qcircuit_raw(inputs, weights), dtype=np.float64)


def quantum_batch_forward(inputs_np, weights_np):
    """Run the circuit on every sample in the batch. Pure numpy → numpy."""
    return np.array([quantum_circuit_np(x, weights_np) for x in inputs_np],
                    dtype=np.float32)          # shape (batch, N_QUBITS)


def parameter_shift_grad(inputs_np, weights_np):
    """
    Compute dL/d(weights) using the parameter-shift rule:
      grad[i] = 0.5 * (f(w + π/2 at i) − f(w − π/2 at i))
    Returns array of same shape as weights_np, averaged over the batch.
    """
    grad = np.zeros_like(weights_np)
    flat = weights_np.flatten()
    for idx in range(len(flat)):
        w_plus  = flat.copy(); w_plus[idx]  += np.pi / 2
        w_minus = flat.copy(); w_minus[idx] -= np.pi / 2
        f_plus  = quantum_batch_forward(inputs_np, w_plus .reshape(weights_np.shape))
        f_minus = quantum_batch_forward(inputs_np, w_minus.reshape(weights_np.shape))
        grad.flat[idx] = np.mean(0.5 * (f_plus - f_minus))
    return grad                                # shape (N_LAYERS, N_QUBITS, 3)


class QuantumFunction(torch.autograd.Function):
    """
    Custom autograd Function that bridges PyTorch and PennyLane cleanly.

    forward : torch tensors  →  detach to numpy  →  run VQC  →  return torch tensor
    backward: receive grad from post-net  →  parameter-shift  →  return torch grads

    This avoids every version-dependent code path inside TorchLayer / qnode
    interface dispatch that triggers the numpy() / grad crash.
    """

    @staticmethod
    def forward(ctx, inputs, weights):
        # Detach safely before converting to numpy
        inputs_np  = inputs.detach().cpu().numpy().astype(np.float64)
        weights_np = weights.detach().cpu().numpy().astype(np.float64)

        out_np = quantum_batch_forward(inputs_np, weights_np)  # (batch, N_QUBITS)

        # Save for backward
        ctx.save_for_backward(inputs, weights)
        ctx._inputs_np  = inputs_np
        ctx._weights_np = weights_np

        return torch.tensor(out_np, dtype=torch.float32,
                            requires_grad=inputs.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        inputs_np  = ctx._inputs_np
        weights_np = ctx._weights_np

        # grad_output: (batch, N_QUBITS) — backprop signal from post-net
        grad_np = grad_output.detach().cpu().numpy()  # (batch, N_QUBITS)

        # ── Gradient w.r.t. weights via parameter-shift ──────────────────────
        # Full parameter-shift is O(n_params × batch), which is slow but exact.
        # We use a chain-rule approximation: weight_grad ≈ grad_output^T · J
        # where J is the Jacobian from parameter-shift, averaged over the batch.
        # This gives the correct gradient direction for Adam optimisation.
        w_grad_np = np.zeros_like(weights_np)
        flat = weights_np.flatten()
        for idx in range(len(flat)):
            w_plus  = flat.copy(); w_plus[idx]  += np.pi / 2
            w_minus = flat.copy(); w_minus[idx] -= np.pi / 2
            jac_col = (
                quantum_batch_forward(inputs_np, w_plus .reshape(weights_np.shape)) -
                quantum_batch_forward(inputs_np, w_minus.reshape(weights_np.shape))
            ) * 0.5                              # shape (batch, N_QUBITS)
            # chain rule: sum over batch and output dims
            w_grad_np.flat[idx] = np.sum(grad_np * jac_col)

        # ── Gradient w.r.t. inputs (needed for pre-net backprop) ─────────────
        i_grad_np = np.zeros_like(inputs_np)
        for b in range(inputs_np.shape[0]):
            for feat in range(N_QUBITS):         # only first N_QUBITS features encoded
                x_plus  = inputs_np[b].copy(); x_plus[feat]  += np.pi / 2
                x_minus = inputs_np[b].copy(); x_minus[feat] -= np.pi / 2
                jac_col_i = (
                    quantum_circuit_np(x_plus,  weights_np) -
                    quantum_circuit_np(x_minus, weights_np)
                ) * 0.5                          # shape (N_QUBITS,)
                i_grad_np[b, feat] = np.dot(grad_np[b], jac_col_i)

        return (torch.tensor(i_grad_np, dtype=torch.float32),
                torch.tensor(w_grad_np, dtype=torch.float32))


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  HYBRID QUANTUM-CLASSICAL NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
"""
Full Hybrid Architecture:
  Input(8) → Linear(8→4) → Tanh → BN  ← Classical Pre-processing
           → QuantumFunction (VQC)      ← Custom autograd quantum layer
           → Linear(4→32) → ReLU → Dropout(0.3)
           → Linear(32→16) → ReLU
           → Linear(16→N_CLASSES)       ← Classical Post-processing
           → CrossEntropyLoss
"""

class QuantumLayer(nn.Module):
    """Wraps QuantumFunction as an nn.Module with learnable VQC weights."""

    def __init__(self):
        super().__init__()
        # Initialise VQC weights with small random values (near identity)
        init = (torch.rand(N_LAYERS, N_QUBITS, 3) - 0.5) * 0.2
        self.vqc_weights = nn.Parameter(init)

    def forward(self, x):
        # x : (batch, N_QUBITS)  float32
        return QuantumFunction.apply(x, self.vqc_weights)


class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(N_FEATURES, N_QUBITS),
            nn.Tanh(),
            nn.BatchNorm1d(N_QUBITS),
        )
        self.qnn = QuantumLayer()
        self.post = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, N_CLASSES),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.qnn(x)
        x = self.post(x)
        return x

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  TRAINING  — every epoch is printed
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Training Hybrid QNN (VQC)...")
print("─" * 90)
print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  "
      f"{'Time/Ep':>8}  {'ETA':>10}")
print("─" * 90)

model     = HybridQNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

# History tracking
hist = {
    'train_loss': [], 'train_acc': [],
    'val_loss'  : [], 'val_acc'  : [],
    'lr'        : [],
    'precision' : [], 'recall'   : [], 'f1': [],
}

t0          = time.time()
epoch_times = []   # seconds per epoch — used for ETA

for epoch in range(1, EPOCHS + 1):
    t_epoch_start = time.time()

    # ── Training pass ──────────────────────────────────────────────────────
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss    += loss.item() * len(xb)
        ep_correct += (logits.argmax(1) == yb).sum().item()
        ep_total   += len(yb)

    scheduler.step()

    tr_loss = ep_loss / ep_total
    tr_acc  = ep_correct / ep_total

    # ── Validation pass ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_logits = model(X_te_t)
        val_loss   = criterion(val_logits, y_te_t).item()
        val_preds  = val_logits.argmax(1).numpy()

    val_acc = accuracy_score(y_te, val_preds)
    prec    = precision_score(y_te, val_preds, average='macro', zero_division=0)
    rec     = recall_score   (y_te, val_preds, average='macro', zero_division=0)
    f1      = f1_score       (y_te, val_preds, average='macro', zero_division=0)
    cur_lr  = optimizer.param_groups[0]['lr']

    # Store history
    hist['train_loss'].append(tr_loss)
    hist['train_acc'] .append(tr_acc)
    hist['val_loss']  .append(val_loss)
    hist['val_acc']   .append(val_acc)
    hist['lr']        .append(cur_lr)
    hist['precision'] .append(prec)
    hist['recall']    .append(rec)
    hist['f1']        .append(f1)

    # ── Per-epoch timing & ETA ─────────────────────────────────────────────
    epoch_sec = time.time() - t_epoch_start
    epoch_times.append(epoch_sec)
    avg_sec   = sum(epoch_times) / len(epoch_times)   # rolling average
    remaining = EPOCHS - epoch
    eta_sec   = avg_sec * remaining

    # Format ETA as mm:ss or hh:mm:ss
    def _fmt(s):
        s = int(s)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

    timing_str = f"  {epoch_sec:5.1f}s/ep  ETA {_fmt(eta_sec)}"

    # ── Print every epoch ──────────────────────────────────────────────────
    marker = ' ◄ best' if val_acc == max(hist['val_acc']) else ''
    print(f"  {epoch:5d}  {tr_loss:10.4f}  {tr_acc*100:8.2f}%  "
          f"{val_loss:8.4f}  {val_acc*100:6.2f}%  {cur_lr:10.6f}"
          f"{timing_str}{marker}")

elapsed = time.time() - t0
avg_per_epoch = elapsed / EPOCHS
print("─" * 90)
print(f"  Training complete in {elapsed:.1f}s  "
      f"(avg {avg_per_epoch:.1f}s/epoch)")
print(f"  Best Val Accuracy : {max(hist['val_acc'])*100:.2f}%  "
      f"(Epoch {hist['val_acc'].index(max(hist['val_acc']))+1})")

# ═══════════════════════════════════════════════════════════════════════════════
#  7.  CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Training classical baselines...")

rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
svm = SVC(kernel='rbf', C=10, probability=True, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                  max_depth=4, random_state=42)

rf .fit(X_tr, y_tr);  rf_preds  = rf .predict(X_te)
svm.fit(X_tr, y_tr);  svm_preds = svm.predict(X_te)
gbm.fit(X_tr, y_tr);  gbm_preds = gbm.predict(X_te)

# Final QML predictions
model.eval()
with torch.no_grad():
    final_logits = model(X_te_t)
    qml_preds    = final_logits.argmax(1).numpy()
    qml_proba    = F.softmax(final_logits, dim=1).numpy()

rf_proba  = rf .predict_proba(X_te)
svm_proba = svm.predict_proba(X_te)
gbm_proba = gbm.predict_proba(X_te)

def metrics(y_true, y_pred, name):
    return {
        'name'      : name,
        'accuracy'  : accuracy_score (y_true, y_pred) * 100,
        'precision' : precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall'    : recall_score   (y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1'        : f1_score       (y_true, y_pred, average='macro', zero_division=0) * 100,
    }

results = [
    metrics(y_te, qml_preds,  'Hybrid QNN (VQC)'),
    metrics(y_te, rf_preds,   'Random Forest'),
    metrics(y_te, svm_preds,  'SVM (RBF)'),
    metrics(y_te, gbm_preds,  'Gradient Boost'),
]

print()
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("  " + "─" * 60)
for r in results:
    print(f"  {r['name']:<22} {r['accuracy']:>8.2f}%  {r['precision']:>8.2f}%  "
          f"{r['recall']:>7.2f}%  {r['f1']:>7.2f}%")

print()
print("[5/7] Classification Report — Hybrid QNN (VQC):")
print(classification_report(y_te, qml_preds, target_names=NAMES, digits=4))

# ═══════════════════════════════════════════════════════════════════════════════
#  8.  VISUALISATIONS  — 10 separate report-quality figures
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/7] Generating report-quality figures...")

def save(fig, fname, tight=True):
    path = os.path.join(OUT_DIR, fname)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      Saved → {path}")

epochs_x = list(range(1, EPOCHS + 1))

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 01 — Training Curves (Loss + Accuracy per epoch)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG2)
ax.plot(epochs_x, hist['train_loss'], color=C_EMBB,  lw=2, label='Train Loss')
ax.plot(epochs_x, hist['val_loss'],   color=C_URLLC, lw=2, label='Val Loss',  linestyle='--')
ax.fill_between(epochs_x, hist['train_loss'], alpha=0.10, color=C_EMBB)
ax.fill_between(epochs_x, hist['val_loss'],   alpha=0.10, color=C_URLLC)
ax.set_title('Training & Validation Loss', color=TC, fontsize=13, pad=10)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.6)
ax.set_xlim(1, EPOCHS)

ax = axes[1]; ax.set_facecolor(BG2)
ax.plot(epochs_x, [v*100 for v in hist['train_acc']], color=C_EMBB,  lw=2, label='Train Acc')
ax.plot(epochs_x, [v*100 for v in hist['val_acc']],   color=C_URLLC, lw=2, label='Val Acc', linestyle='--')
ax.plot(epochs_x, [v*100 for v in hist['f1']],        color=C_MMTC,  lw=1.5, label='Val F1 (macro)', linestyle=':')
ax.fill_between(epochs_x, [v*100 for v in hist['val_acc']], alpha=0.12, color=C_URLLC)
best_ep = hist['val_acc'].index(max(hist['val_acc'])) + 1
ax.axvline(best_ep, color=C_QML, lw=1.5, linestyle='--', alpha=0.8, label=f'Best Epoch ({best_ep})')
ax.set_title('Training & Validation Accuracy', color=TC, fontsize=13, pad=10)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy / F1 (%)')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.6)
ax.set_xlim(1, EPOCHS)

fig.suptitle('VQC Training Curves — All Epochs', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '01_training_curves.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 02 — Confusion Matrix (raw counts + normalised side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
cm     = confusion_matrix(y_te, qml_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

for ax, data, fmt, title in [
    (axes[0], cm,     'd',    'Confusion Matrix — Raw Counts'),
    (axes[1], cm_pct, '.1f',  'Confusion Matrix — Normalised (%)'),
]:
    ax.set_facecolor(BG2)
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=NAMES, yticklabels=NAMES,
                ax=ax, linewidths=1, linecolor='#333',
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, color=TC, fontsize=13, pad=10)
    ax.set_ylabel('True Label', color=TX)
    ax.set_xlabel('Predicted Label', color=TX)

fig.suptitle('QML Traffic Classifier — Confusion Matrices', color='white',
             fontsize=14, y=1.02, fontweight='bold')
save(fig, '02_confusion_matrix.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 03 — Precision / Recall / F1  per class, per model
# ─────────────────────────────────────────────────────────────────────────────
models_dict = {
    'Hybrid QNN\n(VQC)': qml_preds,
    'Random\nForest'   : rf_preds,
    'SVM (RBF)'        : svm_preds,
    'Gradient\nBoosting': gbm_preds,
}
model_colors = [C_QML, C_RF, C_SVM, C_GBM]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
metric_labels = ['Precision', 'Recall', 'F1 Score']
metric_fns    = [precision_score, recall_score, f1_score]

for col, (metric_name, metric_fn) in enumerate(zip(metric_labels, metric_fns)):
    ax = axes[col]; ax.set_facecolor(BG2)
    x_pos = np.arange(N_CLASSES)
    width = 0.18
    for j, (mname, mpreds) in enumerate(models_dict.items()):
        per_class = metric_fn(y_te, mpreds, average=None, zero_division=0)
        offset    = (j - 1.5) * width
        bars = ax.bar(x_pos + offset, per_class * 100, width,
                      label=mname.replace('\n',' '), color=model_colors[j],
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=6.5, color='white', fontweight='bold')
    ax.set_title(f'Per-Class {metric_name}', color=TC, fontsize=12, pad=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(NAMES)
    ax.set_ylabel(f'{metric_name} (%)')
    ax.set_ylim(0, 115)
    ax.legend(facecolor=BG3, edgecolor='#444', fontsize=8)
    ax.grid(axis='y', color=GC, alpha=0.5)

fig.suptitle('Precision / Recall / F1 — Per Class & Per Model', color='white',
             fontsize=14, y=1.02, fontweight='bold')
save(fig, '03_precision_recall_f1.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 04 — ROC Curves (one-vs-rest, all 3 classes)
# ─────────────────────────────────────────────────────────────────────────────
y_te_bin = label_binarize(y_te, classes=[0, 1, 2])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor(BG)

# Left: QML ROC per class
ax = axes[0]; ax.set_facecolor(BG2)
for cls, (cname, clr) in enumerate(zip(NAMES, CLRS)):
    fpr, tpr, _ = roc_curve(y_te_bin[:, cls], qml_proba[:, cls])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=clr, lw=2.5, label=f'{cname}  AUC={roc_auc:.4f}')
ax.plot([0,1],[0,1], color='#555', lw=1.5, linestyle='--', label='Random (AUC=0.5)')
ax.fill_between([0,1],[0,1], alpha=0.05, color='white')
ax.set_title('ROC Curves — Hybrid QNN (VQC)', color=TC, fontsize=13, pad=8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.5)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

# Right: Macro AUC comparison across all models
ax = axes[1]; ax.set_facecolor(BG2)
all_probas = {'Hybrid QNN\n(VQC)': qml_proba, 'Random\nForest': rf_proba,
              'SVM (RBF)': svm_proba, 'Gradient\nBoosting': gbm_proba}
all_colors = [C_QML, C_RF, C_SVM, C_GBM]

for (mname, proba), clr in zip(all_probas.items(), all_colors):
    for cls in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:, cls], proba[:, cls])
        ax.plot(fpr, tpr, color=clr, lw=1.5, alpha=0.6)
    # macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_te_bin[:,c], proba[:,c])[0] for c in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:,c], proba[:,c])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= N_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=clr, lw=3,
            label=f'{mname.replace(chr(10)," ")} (macro AUC={macro_auc:.4f})')

ax.plot([0,1],[0,1],'--', color='#555', lw=1.5, label='Random')
ax.set_title('ROC — All Models (macro avg)', color=TC, fontsize=13, pad=8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(facecolor=BG3, edgecolor='#444', fontsize=9)
ax.grid(True, color=GC, alpha=0.5)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

fig.suptitle('ROC Curves & AUC Analysis', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '04_roc_curves.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 05 — Radar / Spider Chart Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=False))
fig.patch.set_facecolor(BG)

# Left: grouped bar — Accuracy / Precision / Recall / F1
ax = axes[0]; ax.set_facecolor(BG2)
metric_keys = ['accuracy','precision','recall','f1']
metric_disp = ['Accuracy','Precision','Recall','F1']
x = np.arange(len(metric_keys))
width = 0.2
for j, (r, clr) in enumerate(zip(results, [C_QML, C_RF, C_SVM, C_GBM])):
    vals = [r[k] for k in metric_keys]
    offset = (j - 1.5) * width
    bars = ax.bar(x + offset, vals, width, label=r['name'],
                  color=clr, edgecolor='white', linewidth=0.5, alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metric_disp)
ax.set_ylim(0, 115)
ax.set_title('All Metrics — Model Comparison', color=TC, fontsize=13, pad=8)
ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, edgecolor='#444', fontsize=9)
ax.grid(axis='y', color=GC, alpha=0.5)

# Right: radar chart
ax_radar = fig.add_subplot(1, 2, 2, polar=True)
ax_radar.set_facecolor(BG3)
categories = metric_disp
N_cat = len(categories)
angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
angles += angles[:1]
ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_thetagrids(np.degrees(angles[:-1]), categories, color=TX, fontsize=10)
ax_radar.set_ylim(0, 105)
ax_radar.set_yticks([20,40,60,80,100])
ax_radar.set_yticklabels(['20','40','60','80','100'], color='#888', fontsize=7)
ax_radar.grid(color=GC, alpha=0.6)

for r, clr in zip(results, [C_QML, C_RF, C_SVM, C_GBM]):
    vals = [r[k] for k in metric_keys]
    vals += vals[:1]
    ax_radar.plot(angles, vals, 'o-', color=clr, lw=2.5, label=r['name'])
    ax_radar.fill(angles, vals, alpha=0.08, color=clr)

ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                facecolor=BG3, edgecolor='#444', fontsize=9)
ax_radar.set_title('Radar Chart', color=TC, fontsize=13, pad=20)

fig.suptitle('Model Performance Comparison', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '05_model_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 06 — Feature Analysis (scatter, correlation heatmap, importance)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Scatter: bandwidth vs latency
ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG2)
for cls, (nm, clr) in enumerate(zip(NAMES, CLRS)):
    m = df['traffic_class'] == cls
    ax.scatter(df.loc[m,'throughput_kbps'], df.loc[m,'inter_arrival_mean_us'],
               alpha=0.4, s=15, c=clr, label=nm)
ax.set_title('Throughput vs IAT', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Throughput (kbps)'); ax.set_ylabel('IAT Mean (µs)')
ax.legend(facecolor=BG3, edgecolor='#444'); ax.grid(True, color=GC, alpha=0.4)

# Correlation heatmap
ax = fig.add_subplot(gs[1]); ax.set_facecolor(BG2)
corr = df[feat_cols].corr()
mask = np.zeros_like(corr, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#222',
            xticklabels=['Tput','PktSz','IAT\nmean','IAT\nstd','PktCnt','DL/UL','BR\nmean','BR\nstd'],
            yticklabels=['Tput','PktSz','IAT\nmean','IAT\nstd','PktCnt','DL/UL','BR\nmean','BR\nstd'],
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', color=TC, fontsize=12, pad=8)

# Feature importance
ax = fig.add_subplot(gs[2]); ax.set_facecolor(BG2)
imp   = rf.feature_importances_
order = np.argsort(imp)
colors_imp = plt.cm.plasma(np.linspace(0.15, 0.9, N_FEATURES))
ax.barh(range(N_FEATURES), imp[order], color=colors_imp)
ax.set_yticks(range(N_FEATURES))
_short = ['Tput','PktSz','IAT\nmean','IAT\nstd','PktCnt','DL/UL','BR\nmean','BR\nstd']
ax.set_yticklabels([_short[i] for i in order], fontsize=8)
ax.set_title('Feature Importance (RF)', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Importance Score')
ax.grid(axis='x', color=GC, alpha=0.5)

fig.suptitle('Feature Analysis', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '06_feature_analysis.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 07 — Quantum Circuit Diagram  (fully dynamic: works for any N_QUBITS)
# ─────────────────────────────────────────────────────────────────────────────
# Layout constants — scale with N_QUBITS so diagram is never cramped
_fig_h   = max(7, N_QUBITS * 1.2)          # taller figure for more qubits
_x_wire  = 18.0                             # total wire length
_x_enc   = 0.5                              # x-start of encoding gate
_gw      = 0.85                             # gate width
_gh      = min(0.44, 0.9 / max(N_QUBITS,1))# gate height scales down if many qubits
_lyr_gap = (_x_wire - _x_enc - _gw - 2.2) / N_LAYERS   # space per layer
_x_meas  = _x_wire - 1.8                   # x-start of measurement gate

# Colour palette — one colour per qubit, cycles gracefully beyond 4
_base_qcols = [C_EMBB, C_URLLC, C_MMTC, C_QML, '#c084fc', '#34d399',
               '#f97316', '#38bdf8']
qcols = [_base_qcols[i % len(_base_qcols)] for i in range(N_QUBITS)]

# Feature short labels for encoding gates (one per qubit)
_enc_labels = ['Throughput', 'IAT mean', 'Pkt Size', 'DL/UL',
               'Pkt Count', 'BR mean', 'IAT std', 'BR std']

fig, ax = plt.subplots(figsize=(_x_wire + 2, _fig_h))
fig.patch.set_facecolor(BG3); ax.set_facecolor(BG3); ax.axis('off')
ax.set_xlim(-0.8, _x_wire + 0.5)
ax.set_ylim(-1.8, N_QUBITS + 0.6)
ax.set_title(
    f'Variational Quantum Circuit (VQC) — {N_QUBITS} Qubits, {N_LAYERS} Layers\n'
    f'AngleEmbedding (Ry) → StronglyEntanglingLayers × {N_LAYERS} → PauliZ Measurement',
    color=TC, fontsize=13, pad=14, fontweight='bold')

for i in range(N_QUBITS):
    y = N_QUBITS - 1 - i
    clr = qcols[i]
    lbl = _enc_labels[i] if i < len(_enc_labels) else f'f{i}'

    # ── Qubit wire ────────────────────────────────────────────────────────────
    ax.plot([-0.1, _x_wire + 0.2], [y, y], color='#3a3a5a', lw=1.8, zorder=1)

    # ── |0⟩ label ─────────────────────────────────────────────────────────────
    ax.text(-0.15, y, '|0⟩', color=clr, fontsize=10,
            ha='right', va='center', fontweight='bold')
    ax.text(_x_enc - 0.05, y + _gh*0.85, f'q{i}',
            color=clr, fontsize=7.5, ha='center', alpha=0.85)

    # ── Encoding gate (Ry) ────────────────────────────────────────────────────
    r = plt.Rectangle((_x_enc, y - _gh/2), _gw, _gh,
                       fc='#102040', ec='#00e5ff', lw=1.8, zorder=2)
    ax.add_patch(r)
    ax.text(_x_enc + _gw/2, y, f'Ry\n{lbl}',
            color='white', fontsize=6.2, ha='center', va='center',
            zorder=3, fontweight='bold')

    # ── Variational layers (Rz + Ry per layer) ───────────────────────────────
    for lyr in range(N_LAYERS):
        xg  = _x_enc + _gw + 0.3 + lyr * _lyr_gap
        xg2 = xg + _gw * 0.92

        r1 = plt.Rectangle((xg,  y - _gh/2), _gw*0.85, _gh,
                            fc='#2a0e3f', ec='#ff6b6b', lw=1.3, zorder=2)
        ax.add_patch(r1)
        ax.text(xg + _gw*0.425, y, f'Rz\nL{lyr+1}',
                color='white', fontsize=6, ha='center', va='center', zorder=3)

        r2 = plt.Rectangle((xg2, y - _gh/2), _gw*0.85, _gh,
                            fc='#0e2f1f', ec='#4ecdc4', lw=1.3, zorder=2)
        ax.add_patch(r2)
        ax.text(xg2 + _gw*0.425, y, f'Ry\nL{lyr+1}',
                color='white', fontsize=6, ha='center', va='center', zorder=3)

    # ── PauliZ measurement ────────────────────────────────────────────────────
    rm = plt.Rectangle((_x_meas, y - _gh/2), _gw * 1.1, _gh,
                        fc='#3f2800', ec='#ffd93d', lw=1.8, zorder=2)
    ax.add_patch(rm)
    ax.text(_x_meas + _gw*0.55, y, '⟨Z⟩\nmeas',
            color='white', fontsize=7, ha='center', va='center',
            zorder=3, fontweight='bold')

# ── CNOT ring entanglement arrows (per layer) ─────────────────────────────────
for lyr in range(N_LAYERS):
    xc = _x_enc + _gw + 0.3 + lyr * _lyr_gap + _gw * 1.85
    arrow_kw = dict(arrowstyle='->', color='#ffd93d', lw=1.6)
    ctrl_r = max(3, 7 - N_QUBITS) * 0.8   # wrap-around arc radius scales

    for i in range(N_QUBITS - 1):
        y1 = N_QUBITS - 1 - i
        y2 = y1 - 1
        gap = _gh / 2 + 0.05
        ax.annotate('', xy=(xc, y2 + gap), xytext=(xc, y1 - gap),
                    arrowprops=dict(**arrow_kw))
        ax.plot(xc, y1 - gap, 'o', color='#ffd93d', ms=7, zorder=4)
    # wrap-around: last qubit → first qubit
    ax.annotate('', xy=(xc, N_QUBITS - 1 + _gh/2 + 0.05),
                xytext=(xc, 0 - _gh/2 - 0.05),
                arrowprops=dict(**arrow_kw,
                                connectionstyle=f'arc3,rad={-0.35 - N_QUBITS*0.03}'))
    ax.plot(xc, 0 - _gh/2 - 0.05, 'o', color='#ffd93d', ms=7, zorder=4)

# ── Legend ────────────────────────────────────────────────────────────────────
_ly = -1.35
ax.text(_x_enc,          _ly, '■ Ry Angle Encoding',      color='#00e5ff', fontsize=8.5)
ax.text(_x_enc + 4.2,    _ly, '■ Rz + Ry Variational',   color='#ff6b6b', fontsize=8.5)
ax.text(_x_enc + 9.0,    _ly, '⬤ → CNOT Ring',           color='#ffd93d', fontsize=8.5)
ax.text(_x_enc + 13.0,   _ly, '■ ⟨Z⟩ Measurement',       color='#ffd93d', fontsize=8.5)

save(fig, '07_quantum_circuit.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 08 — Per-class Precision & Recall heatmap table
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(BG)

for ax_idx, (metric_fn, mname) in enumerate([(precision_score, 'Precision'), (recall_score, 'Recall')]):
    ax = axes[ax_idx]; ax.set_facecolor(BG2)
    table_data = []
    row_labels = []
    for mname2, mpreds in [('Hybrid QNN (VQC)', qml_preds),
                            ('Random Forest',    rf_preds),
                            ('SVM (RBF)',        svm_preds),
                            ('Gradient Boost',   gbm_preds)]:
        row = metric_fn(y_te, mpreds, average=None, zero_division=0) * 100
        table_data.append(row)
        row_labels.append(mname2)
    table_data = np.array(table_data)
    sns.heatmap(table_data, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=NAMES, yticklabels=row_labels,
                ax=ax, linewidths=0.5, linecolor='#222',
                annot_kws={'size': 12, 'weight': 'bold'},
                vmin=85, vmax=100, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Per-Class {mname} (%) — All Models', color=TC, fontsize=13, pad=8)
    ax.set_xlabel('Traffic Class'); ax.set_ylabel('Model')

fig.suptitle('Precision & Recall Heatmaps', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '08_per_class_metrics.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 09 — LR Schedule + Precision/Recall/F1 over epochs
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG2)
ax.plot(epochs_x, hist['lr'], color=C_MMTC, lw=2.5)
ax.fill_between(epochs_x, hist['lr'], alpha=0.15, color=C_MMTC)
ax.set_title('Learning Rate Schedule (Cosine Annealing)', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
ax.grid(True, color=GC, alpha=0.5)
ax.set_xlim(1, EPOCHS)

ax = axes[1]; ax.set_facecolor(BG2)
ax.plot(epochs_x, [v*100 for v in hist['precision']], color=C_EMBB,  lw=2, label='Precision (macro)')
ax.plot(epochs_x, [v*100 for v in hist['recall']],    color=C_URLLC, lw=2, label='Recall (macro)')
ax.plot(epochs_x, [v*100 for v in hist['f1']],        color=C_MMTC,  lw=2, label='F1 (macro)')
ax.set_title('Precision / Recall / F1 per Epoch', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.5)
ax.set_xlim(1, EPOCHS)

fig.suptitle('Training Dynamics', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '09_training_dynamics.png')

# ─────────────────────────────────────────────────────────────────────────────
#  FIG 10 — Full Dashboard (all key plots combined)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 20))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

def dash_ax(pos, title):
    ax = fig.add_subplot(pos); ax.set_facecolor(BG2)
    ax.set_title(title, color=TC, fontsize=10, pad=6)
    ax.grid(True, color=GC, alpha=0.4)
    return ax

# Row 0
ax = dash_ax(gs[0,0], 'Train/Val Loss')
ax.plot(epochs_x, hist['train_loss'], color=C_EMBB,  lw=2, label='Train')
ax.plot(epochs_x, hist['val_loss'],   color=C_URLLC, lw=2, label='Val', ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[0,1], 'Train/Val Accuracy')
ax.plot(epochs_x, [v*100 for v in hist['train_acc']], color=C_EMBB,  lw=2, label='Train')
ax.plot(epochs_x, [v*100 for v in hist['val_acc']],   color=C_URLLC, lw=2, label='Val', ls='--')
ax.axvline(best_ep, color=C_QML, lw=1.5, ls='--', alpha=0.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Acc (%)'); ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[0,2], 'P/R/F1 per Epoch')
ax.plot(epochs_x, [v*100 for v in hist['precision']], color=C_EMBB,  lw=1.8, label='Precision')
ax.plot(epochs_x, [v*100 for v in hist['recall']],    color=C_URLLC, lw=1.8, label='Recall')
ax.plot(epochs_x, [v*100 for v in hist['f1']],        color=C_MMTC,  lw=1.8, label='F1')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score (%)'); ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[0,3], 'LR Schedule')
ax.plot(epochs_x, hist['lr'], color=C_MMTC, lw=2)
ax.fill_between(epochs_x, hist['lr'], alpha=0.15, color=C_MMTC)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')

# Row 1
ax = fig.add_subplot(gs[1,0]); ax.set_facecolor(BG2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=NAMES, yticklabels=NAMES,
            ax=ax, linewidths=0.8, linecolor='#333',
            annot_kws={'size': 11, 'weight': 'bold'}, cbar=False)
ax.set_title('Confusion Matrix', color=TC, fontsize=10, pad=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

ax = fig.add_subplot(gs[1,1]); ax.set_facecolor(BG2)
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=NAMES, yticklabels=NAMES,
            ax=ax, linewidths=0.8, linecolor='#333',
            annot_kws={'size': 10, 'weight': 'bold'}, cbar=False)
ax.set_title('Confusion Matrix (%)', color=TC, fontsize=10, pad=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

ax = dash_ax(gs[1,2], 'ROC Curves (QML)')
for cls, (cname, clr) in enumerate(zip(NAMES, CLRS)):
    fpr, tpr, _ = roc_curve(y_te_bin[:,cls], qml_proba[:,cls])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=clr, lw=2, label=f'{cname} AUC={roc_auc:.3f}')
ax.plot([0,1],[0,1],'--', color='#555', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(facecolor=BG3, fontsize=8)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

ax = dash_ax(gs[1,3], 'Model Comparison')
metric_keys2 = ['accuracy','precision','recall','f1']
x2 = np.arange(len(metric_keys2)); w2 = 0.18
for j,(r,clr) in enumerate(zip(results,[C_QML,C_RF,C_SVM,C_GBM])):
    off = (j-1.5)*w2
    ax.bar(x2+off, [r[k] for k in metric_keys2], w2,
           label=r['name'].split('(')[0].strip(), color=clr, edgecolor='w', lw=0.4, alpha=0.9)
ax.set_xticks(x2); ax.set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=9)
ax.set_ylim(0,115); ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, fontsize=7, ncol=2)

# Row 2
ax = dash_ax(gs[2,0], 'Throughput vs IAT')
for cls,(nm,clr) in enumerate(zip(NAMES,CLRS)):
    m = df['traffic_class']==cls
    ax.scatter(df.loc[m,'throughput_kbps'], df.loc[m,'inter_arrival_mean_us'],
               alpha=0.35, s=12, c=clr, label=nm)
ax.set_xlabel('Throughput (kbps)'); ax.set_ylabel('IAT Mean (µs)'); ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[2,1], 'Avg Packet Size Distribution')
for cls,(nm,clr) in enumerate(zip(NAMES,CLRS)):
    m = df['traffic_class']==cls
    ax.hist(df.loc[m,'avg_pkt_size_bytes'], bins=25, alpha=0.6, color=clr, label=nm, edgecolor='w', lw=0.2)
ax.set_xlabel('Avg Packet Size (bytes)'); ax.set_ylabel('Count'); ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[2,2], 'Feature Importance (RF)')
imp   = rf.feature_importances_; order = np.argsort(imp)
ax.barh(range(N_FEATURES), imp[order], color=plt.cm.plasma(np.linspace(0.15,0.9,N_FEATURES)))
ax.set_yticks(range(N_FEATURES))
_sl = ['Tput','PktSz','IAT\nmean','IAT\nstd','PktCnt','DL/UL','BR\nmean','BR\nstd']
ax.set_yticklabels([_sl[i] for i in order], fontsize=7)
ax.set_xlabel('Importance')

# Architecture box
ax = fig.add_subplot(gs[2,3]); ax.set_facecolor(BG3); ax.axis('off')
ax.set_title('Hybrid QNN Architecture', color=TC, fontsize=10, pad=6)
arch = [('INPUT',   '8 Network Features',             '#264653'),
        ('Pre-Net', 'Linear(8→4)+Tanh+BN',            '#2a9d8f'),
        ('VQC',     f'{N_QUBITS}q × {N_LAYERS}L VQC', '#e9c46a'),
        ('Post-Net','Linear(4→32→16→3)',               '#e76f51'),
        ('OUTPUT',  'eMBB | URLLC | mMTC',            '#264653')]
for k,(n,d,c) in enumerate(arch):
    yp = 0.88 - k*0.19
    r = plt.Rectangle((0.05,yp-0.08),0.9,0.14,fc=c,ec='white',lw=1,
                       transform=ax.transAxes,zorder=2,alpha=0.9)
    ax.add_patch(r)
    ax.text(0.5,yp,f'{n}: {d}',color='white',fontsize=7.5,
            ha='center',va='center',transform=ax.transAxes,fontweight='bold',zorder=3)
    if k < len(arch)-1:
        ax.annotate('',xy=(0.5,yp-0.09),xytext=(0.5,yp-0.02),
                    xycoords='axes fraction',textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',color='white',lw=1.8))

fig.suptitle(
    'QML-BASED TRAFFIC MANAGEMENT IN 6G NETWORKS\n'
    'Hybrid Variational Quantum Circuit (VQC) — Complete Research Dashboard',
    color='white', fontsize=15, fontweight='bold', y=1.005)

save(fig, '10_full_dashboard.png')

# ═══════════════════════════════════════════════════════════════════════════════
#  9.  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[7/7] All figures saved to: ./{OUT_DIR}/")
print()
print(HEADER)
print("  FINAL RESULTS SUMMARY")
print(HEADER)
print(f"  Dataset       : {len(df)} samples from real 5G CSV files")
print(f"  Features      : {N_FEATURES}  (throughput, pkt_size, IAT_mean, IAT_std, ")
print(f"                    pkt_count, DL/UL_ratio, bitrate_mean, bitrate_std)")
print(f"  Architecture  : Hybrid QNN  (Classical → VQC → Classical)")
print(f"  Quantum Part  : {N_QUBITS}-qubit VQC | {N_LAYERS} StronglyEntangling layers")
print(f"  Optimizer     : Adam + Cosine Annealing LR")
print(f"  Epochs trained: {EPOCHS}  (all printed above)")
print()
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("  " + "─" * 60)
for r in results:
    tag = "  ◄ QML" if r['name'].startswith('Hybrid') else ""
    print(f"  {r['name']:<22} {r['accuracy']:>8.2f}%  {r['precision']:>8.2f}%  "
          f"{r['recall']:>7.2f}%  {r['f1']:>7.2f}%{tag}")
print()
print("  OUTPUT FILES:")
for i, fname in enumerate([
    '01_training_curves.png       — Loss & Accuracy ALL epochs',
    '02_confusion_matrix.png      — Raw + Normalised confusion matrices',
    '03_precision_recall_f1.png   — Per-class P/R/F1 bar charts',
    '04_roc_curves.png            — ROC + AUC all classes & models',
    '05_model_comparison.png      — Grouped bar + Radar chart',
    '06_feature_analysis.png      — Scatter + Correlation + Importance',
    '07_quantum_circuit.png       — Full VQC gate diagram',
    '08_per_class_metrics.png     — P/R heatmaps all models',
    '09_training_dynamics.png     — LR schedule + P/R/F1 curves',
    '10_full_dashboard.png        — Combined report dashboard',
]):
    print(f"    {fname}")
print(HEADER)
print("  ✅  COMPLETE — ready to use in your research report!")
print(HEADER)