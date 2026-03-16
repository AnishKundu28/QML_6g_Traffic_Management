"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        QML-BASED TRAFFIC MANAGEMENT IN 6G NETWORKS                         ║
║        Final Semester Research Project — Complete Runnable Model            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Framework : PennyLane + PyTorch + Scikit-learn                             ║
║  Task      : 6G Network Slice Classification (eMBB / URLLC / mMTC)         ║
║  Model     : Hybrid Variational Quantum Circuit (VQC) Neural Network        ║
║  Dataset   : Kim & Choi 5G Traffic Dataset (Wireshark CSV captures)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN (terminal — recommended):                                       ║
║    cd D:/Projects/QML_6G/5G_Traffic_Datasets                                ║
║    python QML_6G_final.py                                                   ║
║                                                                             ║
║  INSTALL DEPENDENCIES (once):                                               ║
║    pip install pennylane pennylane-lightning torch scikit-learn             ║
║                matplotlib seaborn pandas numpy                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  0.  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os, warnings, time, re as _re
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import pennylane as qml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

np.random.seed(42)
torch.manual_seed(42)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "../Outputs/qml_6g_outputs_5"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dark theme colour palette ─────────────────────────────────────────────────
BG, BG2, BG3 = '#0d0d1f', '#1a1a2e', '#0a0a1a'
TC  = '#00e5ff'   # title cyan
TX  = '#e0e0e0'   # general text
GC  = '#2a2a3a'   # grid lines

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
#  1.  CONFIGURATION  ← edit these values
# ═══════════════════════════════════════════════════════════════════════════════
N_QUBITS   = 4
N_LAYERS   = 5
N_CLASSES  = 3
N_FEATURES = 8
EPOCHS     = 40
BATCH_SIZE = 32
LR         = 0.004
N_SAMPLES  = 1200

# ── Class weights: penalise missing URLLC (low recall observed on real data) ──
# URLLC recall was 38.7% in the 5-epoch run. Weight 2.5 pushes the loss to
# focus harder on URLLC errors without overwhelming the other two classes.
CLASS_WEIGHTS = torch.tensor([1.0, 2.5, 1.0])

HEADER = "═" * 68
print(HEADER)
print("  QML-BASED TRAFFIC MANAGEMENT IN 6G NETWORKS")
print("  Hybrid Variational Quantum Circuit (VQC) Classifier")
print(HEADER)
print(f"  Qubits       : {N_QUBITS}  (2^{N_QUBITS} = {2**N_QUBITS}-dim state space)")
print(f"  VQC Layers   : {N_LAYERS}")
print(f"  Classes      : {N_CLASSES}  (eMBB | URLLC | mMTC)")
print(f"  Features     : {N_FEATURES}")
print(f"  Epochs       : {EPOCHS}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Learning rate: {LR}  (cosine annealing → 1e-4)")
print(f"  Samples      : {N_SAMPLES}  ({N_SAMPLES//N_CLASSES} per class)")
print(f"  Class weights: eMBB={CLASS_WEIGHTS[0]:.1f}  URLLC={CLASS_WEIGHTS[1]:.1f}  mMTC={CLASS_WEIGHTS[2]:.1f}")
print(HEADER)

# ═══════════════════════════════════════════════════════════════════════════════
#  2.  DATASET  —  Kim & Choi (2022/2023) 5G Traffic Dataset
#                  Wireshark packet-capture CSVs
# ═══════════════════════════════════════════════════════════════════════════════
"""
CSV FORMAT (identical across all files — Wireshark export)
──────────────────────────────────────────────────────────
  Columns: No. | Time | Source | Destination | Protocol | Length | Info

  Time   : "2022-09-27 13:08:31.564846"  (absolute datetime)
  Source : sender IP   e.g. "10.215.173.1"   (private = phone = uplink)
  Dest   : receiver IP e.g. "112.217.128.200" (public  = server = downlink)
  Length : IP packet size in bytes

FOLDER STRUCTURE → SLICE MAPPING
─────────────────────────────────
  Game_Streaming/   → URLLC (1)  GeForce_Now, KT_GameBox
  Online_Game/      → URLLC (1)  Battleground, Teamfight_Tactics
  Live_Streaming/   → eMBB  (0)  AfreecaTV, Naver_NOW, YouTube_Live
  Stored_Streaming/ → eMBB  (0)  Netflix, Amazon_Prime, YouTube
  Video_Conferencing→ eMBB  (0)  Zoom, MS_Teams, Google_Meet
  Metaverse/        → mMTC  (2)  Roblox, Zepeto

DERIVED FEATURES (8, per 1-second sliding window)
──────────────────────────────────────────────────
  throughput_kbps        total bytes × 8 / 1000
  avg_pkt_size_bytes     mean packet length
  inter_arrival_mean_us  mean inter-packet gap (µs) — latency proxy
  inter_arrival_std_us   std dev of gap (µs)       — jitter proxy
  flow_pkt_count         packets per second
  dl_ul_ratio            downlink_bytes / uplink_bytes (IP direction heuristic)
  bitrate_mean_kbps      5-window rolling mean of throughput
  bitrate_std_kbps       5-window rolling std  of throughput
"""

DATASET_PATH = r"/5G_Traffic_Datasets"

FOLDER_SLICE_MAP = {
    'stored_streaming'  : 0,
    'live_streaming'    : 0,
    'video_conferencing': 0,
    'game_streaming'    : 1,
    'online_game'       : 1,
    'metaverse'         : 2,
}

FEAT_COLS = [
    'throughput_kbps',
    'avg_pkt_size_bytes',
    'inter_arrival_mean_us',
    'inter_arrival_std_us',
    'flow_pkt_count',
    'dl_ul_ratio',
    'bitrate_mean_kbps',
    'bitrate_std_kbps',
]
feat_cols  = FEAT_COLS
N_FEATURES = len(feat_cols)   # 8

# ── Private IP detector (phone = uplink source) ───────────────────────────────
def _is_private(ip):
    try:
        parts = [int(x) for x in str(ip).split('.')]
        if len(parts) != 4:
            return False
        a, b = parts[0], parts[1]
        return (a == 10 or a == 127 or
                (a == 172 and 16 <= b <= 31) or
                (a == 192 and b == 168) or
                (a == 192 and b == 0))
    except Exception:
        return False

# ── Single-file loader ────────────────────────────────────────────────────────
def _load_one_csv(path, label, max_windows=150):
    try:
        df_raw = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    df_raw.columns = df_raw.columns.str.strip()
    needed = {'Time', 'Length', 'Source', 'Destination'}
    if not needed.issubset(set(df_raw.columns)):
        return pd.DataFrame()

    df_raw['_t'] = pd.to_datetime(df_raw['Time'], errors='coerce')
    df_raw = df_raw.dropna(subset=['_t'])
    if df_raw.empty:
        return pd.DataFrame()

    t0 = df_raw['_t'].iloc[0]
    df_raw['_ts'] = (df_raw['_t'] - t0).dt.total_seconds()
    df_raw['_len'] = pd.to_numeric(df_raw['Length'], errors='coerce').fillna(0)
    df_raw = df_raw[df_raw['_len'] > 0].reset_index(drop=True)
    if len(df_raw) < 10:
        return pd.DataFrame()

    df_raw['_uplink'] = df_raw['Source'].apply(_is_private)
    df_raw = df_raw.sort_values('_ts').reset_index(drop=True)

    rows  = []
    t_end = df_raw['_ts'].iloc[-1]
    t     = 0.0

    while t + 1.0 <= t_end and len(rows) < max_windows:
        mask = (df_raw['_ts'] >= t) & (df_raw['_ts'] < t + 1.0)
        win  = df_raw[mask]
        if len(win) < 2:
            t += 1.0
            continue

        lengths = win['_len'].values.astype(np.float64)
        iats_us = np.diff(win['_ts'].values.astype(np.float64)) * 1e6

        up_bytes   = win.loc[ win['_uplink'], '_len'].sum()
        down_bytes = win.loc[~win['_uplink'], '_len'].sum()
        dl_ul = float(down_bytes / up_bytes) if up_bytes > 0 else float(down_bytes + 1.0)

        rows.append({
            'throughput_kbps'       : float(lengths.sum() * 8.0 / 1000.0),
            'avg_pkt_size_bytes'    : float(np.mean(lengths)),
            'inter_arrival_mean_us' : float(np.mean(iats_us)),
            'inter_arrival_std_us'  : float(np.std(iats_us)) if len(iats_us) > 1 else 0.0,
            'flow_pkt_count'        : float(len(win)),
            'dl_ul_ratio'           : float(np.clip(dl_ul, 0.01, 1000.0)),
        })
        t += 1.0

    if not rows:
        return pd.DataFrame()

    df_f = pd.DataFrame(rows)
    roll = df_f['throughput_kbps'].rolling(5, min_periods=1)
    df_f['bitrate_mean_kbps'] = roll.mean()
    df_f['bitrate_std_kbps']  = roll.std().fillna(0.0)
    df_f['traffic_class']     = label

    df_f['throughput_kbps']       = df_f['throughput_kbps'].clip(lower=1.0)
    df_f['avg_pkt_size_bytes']    = df_f['avg_pkt_size_bytes'].clip(lower=20.0, upper=1500.0)
    df_f['inter_arrival_mean_us'] = df_f['inter_arrival_mean_us'].clip(lower=0.01)
    df_f['inter_arrival_std_us']  = df_f['inter_arrival_std_us'].clip(lower=0.0)
    df_f['flow_pkt_count']        = df_f['flow_pkt_count'].clip(lower=1.0)
    df_f['dl_ul_ratio']           = df_f['dl_ul_ratio'].clip(lower=0.01, upper=1000.0)
    df_f['bitrate_mean_kbps']     = df_f['bitrate_mean_kbps'].clip(lower=0.1)
    df_f['bitrate_std_kbps']      = df_f['bitrate_std_kbps'].clip(lower=0.0)

    return df_f[feat_cols + ['traffic_class']]

# ── Full folder-tree loader ───────────────────────────────────────────────────
def load_real_dataset(root):
    all_frames  = []
    slice_names = {0: 'eMBB', 1: 'URLLC', 2: 'mMTC'}
    win_counts  = {0: 0, 1: 0, 2: 0}
    file_counts = {0: 0, 1: 0, 2: 0}

    try:
        top_entries = sorted(os.scandir(root), key=lambda e: e.name)
    except FileNotFoundError:
        print(f"      [ERROR] Path not found: {root}")
        return pd.DataFrame()

    for top in top_entries:
        if not top.is_dir():
            continue
        key = top.name.lower().replace('-', '_')
        if key not in FOLDER_SLICE_MAP:
            continue
        label = FOLDER_SLICE_MAP[key]

        for dirpath, _, filenames in os.walk(top.path):
            csv_files = sorted(f for f in filenames if f.lower().endswith('.csv'))
            if not csv_files:
                continue
            app_name = os.path.basename(dirpath)
            print(f"      [{slice_names[label]:5s}]  {top.name}/{app_name}/"
                  f"  ({len(csv_files)} files) ...", end=' ', flush=True)

            app_frames = []
            for fname in csv_files:
                df_f = _load_one_csv(os.path.join(dirpath, fname), label)
                if not df_f.empty:
                    app_frames.append(df_f)
                    file_counts[label] += 1

            if app_frames:
                df_app = pd.concat(app_frames, ignore_index=True)
                win_counts[label] += len(df_app)
                all_frames.append(df_app)
                print(f"{len(df_app)} windows")
            else:
                print("0 windows — skipped")

    if not all_frames:
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"\n      Windows loaded:")
    for lbl, name in slice_names.items():
        print(f"        {name:5s} ({lbl}): {win_counts[lbl]:5d} windows  "
              f"({file_counts[lbl]} CSV files)")
    return df_all.sample(frac=1, random_state=42).reset_index(drop=True)


def balance_and_cap(df, max_per_class=300):
    counts = df['traffic_class'].value_counts()
    n_each = min(counts.min(), max_per_class)
    if n_each < 30:
        raise ValueError(f"Smallest class has only {n_each} windows — check CSV files.")
    balanced = (df.groupby('traffic_class', group_keys=False)
                  .apply(lambda g: g.sample(n_each, random_state=42)))
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def _synth_fallback(n=900):
    """Used only if real data fails to load."""
    rng = np.random.default_rng(42); rows = []; labels = []
    spc = n // N_CLASSES
    params = [
        [(14000,4000),(1100,200),(350,120),(800,250),(650,180),(15.0,4.0),(12000,3500),(2800,700)],
        [(5500,1200),(380,100),(80,25),(45,12),(220,60),(1.6,0.4),(4800,1000),(900,250)],
        [(180,80),(160,60),(22000,8000),(15000,5000),(12,5),(2.5,0.8),(120,50),(30,12)],
    ]
    clips = [(1,None),(20,1500),(0.01,None),(0,None),(1,None),(0.01,None),(0.1,None),(0,None)]
    for lbl, plist in enumerate(params):
        for _ in range(spc):
            row = [float(np.clip(rng.normal(mu, sd), lo, hi))
                   for (mu, sd), (lo, hi) in zip(plist, clips)]
            rows.append(row); labels.append(lbl)
    d = pd.DataFrame(rows, columns=feat_cols)
    d['traffic_class'] = labels
    return d.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Load ──────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading 5G Traffic Dataset (Kim & Choi 2022)...")
print(f"      Path   : {DATASET_PATH}")
print( "      Source : https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets")
print( "      Format : Wireshark CSV  (No.|Time|Source|Destination|Protocol|Length|Info)")
print()

df_raw = load_real_dataset(DATASET_PATH)

if df_raw.empty or df_raw['traffic_class'].nunique() < N_CLASSES:
    print("\n      [FALLBACK] Real data load failed — using synthetic data.")
    df    = _synth_fallback(N_SAMPLES)
    _mode = "Synthetic fallback"
else:
    df    = balance_and_cap(df_raw, max_per_class=N_SAMPLES // N_CLASSES)
    _mode = "Real Wireshark CSV data"

print(f"\n      Mode    : {_mode}")
print(f"      Shape   : {df.shape}")
print(f"      Classes : {df['traffic_class'].value_counts().sort_index().to_dict()}")
print(f"      Mapping : 0=eMBB  (Stored/Live Streaming + Video Conferencing)")
print(f"                1=URLLC (Game Streaming + Online Game)")
print(f"                2=mMTC  (Metaverse)")
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
X_angle = np.clip(X_sc, -np.pi, np.pi)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_angle, y, test_size=0.25, stratify=y, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

dataset    = TensorDataset(X_tr_t, y_tr_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)  # drop incomplete batch → safe for BatchNorm

print(f"      Train samples : {len(X_tr)}")
print(f"      Test  samples : {len(X_te)}")
print(f"      Features      : {N_FEATURES}  angle-encoded into {N_QUBITS} qubits")
print(f"      Batches/epoch : {len(dataloader)}  (batch_size={BATCH_SIZE}, drop_last=True)")

# ═══════════════════════════════════════════════════════════════════════════════
#  4.  QUANTUM CIRCUIT  (Variational Quantum Circuit — VQC)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Architecture:
  AngleEmbedding (Ry rotations on first N_QUBITS features)
  → StronglyEntanglingLayers × N_LAYERS  (Rz-Ry-Rz per qubit + CNOT ring)
  → PauliZ measurement on all qubits  → outputs ∈ [−1, +1]^N_QUBITS

Why we bypass TorchLayer:
  TorchLayer calls _to_qfunc_output_type which wraps results in pnp.tensor,
  calling .numpy() on a grad-tracked tensor → RuntimeError on Python 3.10 +
  older PennyLane. Solution: raw qnode with numpy I/O + custom autograd.Function.
"""

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method='parameter-shift')
def _qcircuit_raw(inputs, weights):
    qml.AngleEmbedding(inputs[:N_QUBITS], wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def quantum_circuit_np(inputs, weights):
    """Single-sample circuit. Returns np.ndarray (N_QUBITS,)."""
    return np.array(_qcircuit_raw(inputs, weights), dtype=np.float64)

def quantum_batch_forward(inputs_np, weights_np):
    """Batched forward pass. Returns np.ndarray (batch, N_QUBITS)."""
    return np.array([quantum_circuit_np(x, weights_np) for x in inputs_np],
                    dtype=np.float32)


class QuantumFunction(torch.autograd.Function):
    """
    Bridges PyTorch autograd ↔ PennyLane cleanly.
    forward : detach → numpy → run VQC → torch tensor
    backward: parameter-shift Jacobian → torch gradients
    """

    @staticmethod
    def forward(ctx, inputs, weights):
        inputs_np  = inputs.detach().cpu().numpy().astype(np.float64)
        weights_np = weights.detach().cpu().numpy().astype(np.float64)
        out_np = quantum_batch_forward(inputs_np, weights_np)
        ctx.save_for_backward(inputs, weights)
        ctx._inputs_np  = inputs_np
        ctx._weights_np = weights_np
        return torch.tensor(out_np, dtype=torch.float32,
                            requires_grad=inputs.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        inputs_np  = ctx._inputs_np
        weights_np = ctx._weights_np
        grad_np    = grad_output.detach().cpu().numpy()

        # Weight gradients via parameter-shift
        w_grad_np = np.zeros_like(weights_np)
        flat = weights_np.flatten()
        for idx in range(len(flat)):
            wp = flat.copy(); wp[idx] += np.pi / 2
            wm = flat.copy(); wm[idx] -= np.pi / 2
            jac = (quantum_batch_forward(inputs_np, wp.reshape(weights_np.shape)) -
                   quantum_batch_forward(inputs_np, wm.reshape(weights_np.shape))) * 0.5
            w_grad_np.flat[idx] = np.sum(grad_np * jac)

        # Input gradients via parameter-shift
        i_grad_np = np.zeros_like(inputs_np)
        for b in range(inputs_np.shape[0]):
            for feat in range(N_QUBITS):
                xp = inputs_np[b].copy(); xp[feat] += np.pi / 2
                xm = inputs_np[b].copy(); xm[feat] -= np.pi / 2
                jac_i = (quantum_circuit_np(xp, weights_np) -
                         quantum_circuit_np(xm, weights_np)) * 0.5
                i_grad_np[b, feat] = np.dot(grad_np[b], jac_i)

        return (torch.tensor(i_grad_np, dtype=torch.float32),
                torch.tensor(w_grad_np, dtype=torch.float32))


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  HYBRID QUANTUM-CLASSICAL NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
"""
Input(8)
  → Linear(8→N_QUBITS) + Tanh + BatchNorm1d   [Classical pre-processing]
  → QuantumFunction (N_QUBITS-qubit VQC)       [Quantum layer]
  → Linear(N_QUBITS→64) + ReLU + Dropout(0.3) [Classical post-processing]
  → Linear(64→32) + ReLU
  → Linear(32→N_CLASSES)
  → CrossEntropyLoss (with class weights)
"""

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        init = (torch.rand(N_LAYERS, N_QUBITS, 3) - 0.5) * 0.2
        self.vqc_weights = nn.Parameter(init)

    def forward(self, x):
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
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.qnn(x)
        x = self.post(x)
        return x

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Training Hybrid QNN (VQC)...")
print("─" * 95)
print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  "
      f"{'Time/Ep':>8}  {'ETA':>10}")
print("─" * 95)

model     = HybridQNN()
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)  # weighted for URLLC recall
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

hist = {
    'train_loss': [], 'train_acc': [],
    'val_loss'  : [], 'val_acc'  : [],
    'lr'        : [], 'precision': [], 'recall': [], 'f1': [],
}

t0          = time.time()
epoch_times = []

def _fmt_time(s):
    s = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()

    # ── Training pass ──────────────────────────────────────────────────────────
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

    # ── Validation pass ────────────────────────────────────────────────────────
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

    hist['train_loss'].append(tr_loss)
    hist['train_acc'] .append(tr_acc)
    hist['val_loss']  .append(val_loss)
    hist['val_acc']   .append(val_acc)
    hist['lr']        .append(cur_lr)
    hist['precision'] .append(prec)
    hist['recall']    .append(rec)
    hist['f1']        .append(f1)

    epoch_sec = time.time() - t_ep
    epoch_times.append(epoch_sec)
    avg_sec   = sum(epoch_times) / len(epoch_times)
    eta_sec   = avg_sec * (EPOCHS - epoch)
    marker    = ' ◄ best' if val_acc == max(hist['val_acc']) else ''

    print(f"  {epoch:5d}  {tr_loss:10.4f}  {tr_acc*100:8.2f}%  "
          f"{val_loss:8.4f}  {val_acc*100:6.2f}%  {cur_lr:10.6f}  "
          f"{epoch_sec:6.1f}s/ep  ETA {_fmt_time(eta_sec)}{marker}")

elapsed = time.time() - t0
print("─" * 95)
print(f"  Training complete in {elapsed:.1f}s  (avg {elapsed/EPOCHS:.1f}s/epoch)")
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
#  8.  VISUALISATIONS  — 10 report-quality figures
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/7] Generating report-quality figures...")

def save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      Saved → {path}")

epochs_x  = list(range(1, EPOCHS + 1))
best_ep   = hist['val_acc'].index(max(hist['val_acc'])) + 1
y_te_bin  = label_binarize(y_te, classes=[0, 1, 2])
cm        = confusion_matrix(y_te, qml_preds)
cm_pct    = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
_lbl_sh   = ['Tput','PktSz','IAT\nmean','IAT\nstd','PktCnt','DL/UL','BR\nmean','BR\nstd']

# ── FIG 01 — Training Curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG2)
ax.plot(epochs_x, hist['train_loss'], color=C_EMBB,  lw=2, label='Train Loss')
ax.plot(epochs_x, hist['val_loss'],   color=C_URLLC, lw=2, label='Val Loss', ls='--')
ax.fill_between(epochs_x, hist['train_loss'], alpha=0.10, color=C_EMBB)
ax.fill_between(epochs_x, hist['val_loss'],   alpha=0.10, color=C_URLLC)
ax.set_title('Loss per Epoch', color=TC, fontsize=13, pad=10)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss (weighted)')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.6); ax.set_xlim(1, EPOCHS)

ax = axes[1]; ax.set_facecolor(BG2)
ax.plot(epochs_x, [v*100 for v in hist['train_acc']], color=C_EMBB,  lw=2, label='Train Acc')
ax.plot(epochs_x, [v*100 for v in hist['val_acc']],   color=C_URLLC, lw=2, label='Val Acc', ls='--')
ax.plot(epochs_x, [v*100 for v in hist['f1']],        color=C_MMTC,  lw=1.5, label='Val F1', ls=':')
ax.fill_between(epochs_x, [v*100 for v in hist['val_acc']], alpha=0.12, color=C_URLLC)
ax.axvline(best_ep, color=C_QML, lw=1.5, ls='--', alpha=0.8, label=f'Best Epoch ({best_ep})')
ax.set_title('Accuracy & F1 per Epoch', color=TC, fontsize=13, pad=10)
ax.set_xlabel('Epoch'); ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.6); ax.set_xlim(1, EPOCHS)

fig.suptitle('VQC Training Curves — Real 5G Wireshark Data',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '01_training_curves.png')

# ── FIG 02 — Confusion Matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6)); fig.patch.set_facecolor(BG)
for ax, data, fmt, title in [
    (axes[0], cm,     'd',   'Confusion Matrix — Raw Counts'),
    (axes[1], cm_pct, '.1f', 'Confusion Matrix — Normalised (%)'),
]:
    ax.set_facecolor(BG2)
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=NAMES, yticklabels=NAMES,
                ax=ax, linewidths=1, linecolor='#333',
                annot_kws={'size': 14, 'weight': 'bold'}, cbar_kws={'shrink': 0.8})
    ax.set_title(title, color=TC, fontsize=13, pad=10)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

fig.suptitle('QML Traffic Classifier — Confusion Matrices',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '02_confusion_matrix.png')

# ── FIG 03 — Precision / Recall / F1 per class per model ─────────────────────
models_dict  = {'Hybrid QNN\n(VQC)': qml_preds, 'Random\nForest': rf_preds,
                'SVM (RBF)': svm_preds, 'Gradient\nBoosting': gbm_preds}
model_colors = [C_QML, C_RF, C_SVM, C_GBM]

fig, axes = plt.subplots(1, 3, figsize=(18, 6)); fig.patch.set_facecolor(BG)
for col, (metric_name, metric_fn) in enumerate(
        zip(['Precision','Recall','F1 Score'], [precision_score, recall_score, f1_score])):
    ax = axes[col]; ax.set_facecolor(BG2)
    x_pos = np.arange(N_CLASSES); width = 0.18
    for j, (mname, mpreds) in enumerate(models_dict.items()):
        per_class = metric_fn(y_te, mpreds, average=None, zero_division=0)
        bars = ax.bar(x_pos + (j - 1.5) * width, per_class * 100, width,
                      label=mname.replace('\n',' '), color=model_colors[j],
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=6.5, color='white', fontweight='bold')
    ax.set_title(f'Per-Class {metric_name}', color=TC, fontsize=12, pad=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(NAMES)
    ax.set_ylabel(f'{metric_name} (%)'); ax.set_ylim(0, 115)
    ax.legend(facecolor=BG3, edgecolor='#444', fontsize=8)
    ax.grid(axis='y', color=GC, alpha=0.5)

fig.suptitle('Precision / Recall / F1 — Per Class & Per Model',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '03_precision_recall_f1.png')

# ── FIG 04 — ROC Curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6)); fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG2)
for cls, (cname, clr) in enumerate(zip(NAMES, CLRS)):
    fpr, tpr, _ = roc_curve(y_te_bin[:, cls], qml_proba[:, cls])
    ax.plot(fpr, tpr, color=clr, lw=2.5, label=f'{cname}  AUC={auc(fpr,tpr):.4f}')
ax.plot([0,1],[0,1], color='#555', lw=1.5, ls='--', label='Random')
ax.set_title('ROC Curves — Hybrid QNN (VQC)', color=TC, fontsize=13, pad=8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.5); ax.set_xlim(0,1); ax.set_ylim(0,1.02)

ax = axes[1]; ax.set_facecolor(BG2)
for (mname, proba), clr in zip(
        [('Hybrid QNN (VQC)', qml_proba), ('Random Forest', rf_proba),
         ('SVM (RBF)', svm_proba), ('Gradient Boost', gbm_proba)],
        [C_QML, C_RF, C_SVM, C_GBM]):
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_te_bin[:,c], proba[:,c])[0] for c in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:,c], proba[:,c])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= N_CLASSES
    ax.plot(all_fpr, mean_tpr, color=clr, lw=3,
            label=f'{mname} (AUC={auc(all_fpr, mean_tpr):.4f})')
ax.plot([0,1],[0,1],'--', color='#555', lw=1.5)
ax.set_title('ROC — All Models (macro avg)', color=TC, fontsize=13, pad=8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(facecolor=BG3, edgecolor='#444', fontsize=9)
ax.grid(True, color=GC, alpha=0.5); ax.set_xlim(0,1); ax.set_ylim(0,1.02)

fig.suptitle('ROC Curves & AUC Analysis',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '04_roc_curves.png')

# ── FIG 05 — Model Comparison (grouped bar + radar) ───────────────────────────
fig = plt.figure(figsize=(16, 7)); fig.patch.set_facecolor(BG)
metric_keys = ['accuracy','precision','recall','f1']
metric_disp = ['Accuracy','Precision','Recall','F1']

ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG2)
x = np.arange(len(metric_keys)); width = 0.2
for j, (r, clr) in enumerate(zip(results, [C_QML, C_RF, C_SVM, C_GBM])):
    vals   = [r[k] for k in metric_keys]
    offset = (j - 1.5) * width
    bars   = ax.bar(x + offset, vals, width, label=r['name'],
                    color=clr, edgecolor='white', linewidth=0.5, alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metric_disp); ax.set_ylim(0, 115)
ax.set_title('All Metrics — Model Comparison', color=TC, fontsize=13, pad=8)
ax.set_ylabel('Score (%)'); ax.legend(facecolor=BG3, edgecolor='#444', fontsize=9)
ax.grid(axis='y', color=GC, alpha=0.5)

ax_r = fig.add_subplot(1, 2, 2, polar=True); ax_r.set_facecolor(BG3)
N_cat  = len(metric_disp)
angles = [n / N_cat * 2 * np.pi for n in range(N_cat)] + [0]
ax_r.set_theta_offset(np.pi / 2); ax_r.set_theta_direction(-1)
ax_r.set_thetagrids(np.degrees(angles[:-1]), metric_disp, color=TX, fontsize=10)
ax_r.set_ylim(0, 105); ax_r.set_yticks([20,40,60,80,100])
ax_r.set_yticklabels(['20','40','60','80','100'], color='#888', fontsize=7)
ax_r.grid(color=GC, alpha=0.6)
for r, clr in zip(results, [C_QML, C_RF, C_SVM, C_GBM]):
    vals = [r[k] for k in metric_keys] + [r[metric_keys[0]]]
    ax_r.plot(angles, vals, 'o-', color=clr, lw=2.5, label=r['name'])
    ax_r.fill(angles, vals, alpha=0.08, color=clr)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
            facecolor=BG3, edgecolor='#444', fontsize=9)
ax_r.set_title('Radar Chart', color=TC, fontsize=13, pad=20)

fig.suptitle('Model Performance Comparison',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '05_model_comparison.png')

# ── FIG 06 — Feature Analysis ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6)); fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG2)
for cls, (nm, clr) in enumerate(zip(NAMES, CLRS)):
    m = df['traffic_class'] == cls
    ax.scatter(df.loc[m,'throughput_kbps'], df.loc[m,'inter_arrival_mean_us'],
               alpha=0.4, s=15, c=clr, label=nm)
ax.set_title('Throughput vs IAT — by 5G Slice', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Throughput (kbps)'); ax.set_ylabel('IAT Mean (µs)')
ax.legend(facecolor=BG3, edgecolor='#444'); ax.grid(True, color=GC, alpha=0.4)
ax.set_yscale('symlog', linthresh=10); ax.set_xscale('symlog', linthresh=100)

ax = fig.add_subplot(gs[1]); ax.set_facecolor(BG2)
sns.heatmap(df[feat_cols].corr(), ax=ax, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#222',
            xticklabels=_lbl_sh, yticklabels=_lbl_sh,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', color=TC, fontsize=12, pad=8)

ax = fig.add_subplot(gs[2]); ax.set_facecolor(BG2)
imp   = rf.feature_importances_; order = np.argsort(imp)
ax.barh(range(N_FEATURES), imp[order],
        color=plt.cm.plasma(np.linspace(0.15, 0.9, N_FEATURES)))
ax.set_yticks(range(N_FEATURES))
ax.set_yticklabels([_lbl_sh[i] for i in order], fontsize=8)
ax.set_title('Feature Importance (Random Forest)', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Importance Score'); ax.grid(axis='x', color=GC, alpha=0.5)

fig.suptitle('Feature Analysis — Kim & Choi 5G Dataset',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '06_feature_analysis.png')

# ── FIG 07 — Quantum Circuit Diagram (fully dynamic for any N_QUBITS) ─────────
_fig_h   = max(7, N_QUBITS * 1.2)
_x_wire  = 18.0
_x_enc   = 0.5
_gw      = 0.85
_gh      = min(0.44, 0.9 / max(N_QUBITS, 1))
_lyr_gap = (_x_wire - _x_enc - _gw - 2.2) / N_LAYERS
_x_meas  = _x_wire - 1.8
_base_qcols = [C_EMBB, C_URLLC, C_MMTC, C_QML, '#c084fc', '#34d399', '#f97316', '#38bdf8']
_qcols   = [_base_qcols[i % len(_base_qcols)] for i in range(N_QUBITS)]
_enc_lbl = ['Throughput','IAT mean','Pkt Size','DL/UL','Pkt Count','BR mean','IAT std','BR std']

fig, ax = plt.subplots(figsize=(_x_wire + 2, _fig_h))
fig.patch.set_facecolor(BG3); ax.set_facecolor(BG3); ax.axis('off')
ax.set_xlim(-0.8, _x_wire + 0.5); ax.set_ylim(-1.8, N_QUBITS + 0.6)
ax.set_title(
    f'Variational Quantum Circuit (VQC) — {N_QUBITS} Qubits, {N_LAYERS} Layers\n'
    f'AngleEmbedding (Ry) → StronglyEntanglingLayers × {N_LAYERS} → PauliZ',
    color=TC, fontsize=13, pad=14, fontweight='bold')

for i in range(N_QUBITS):
    y  = N_QUBITS - 1 - i
    clr = _qcols[i]
    lbl = _enc_lbl[i] if i < len(_enc_lbl) else f'f{i}'
    ax.plot([-0.1, _x_wire + 0.2], [y, y], color='#3a3a5a', lw=1.8, zorder=1)
    ax.text(-0.15, y, '|0⟩', color=clr, fontsize=10, ha='right', va='center', fontweight='bold')
    ax.text(_x_enc - 0.05, y + _gh*0.85, f'q{i}', color=clr, fontsize=7.5, ha='center', alpha=0.85)
    r = plt.Rectangle((_x_enc, y - _gh/2), _gw, _gh, fc='#102040', ec='#00e5ff', lw=1.8, zorder=2)
    ax.add_patch(r)
    ax.text(_x_enc + _gw/2, y, f'Ry\n{lbl}',
            color='white', fontsize=6.2, ha='center', va='center', zorder=3, fontweight='bold')
    for lyr in range(N_LAYERS):
        xg  = _x_enc + _gw + 0.3 + lyr * _lyr_gap
        xg2 = xg + _gw * 0.92
        r1 = plt.Rectangle((xg,  y - _gh/2), _gw*0.85, _gh, fc='#2a0e3f', ec='#ff6b6b', lw=1.3, zorder=2)
        ax.add_patch(r1)
        ax.text(xg + _gw*0.425, y, f'Rz\nL{lyr+1}', color='white', fontsize=6, ha='center', va='center', zorder=3)
        r2 = plt.Rectangle((xg2, y - _gh/2), _gw*0.85, _gh, fc='#0e2f1f', ec='#4ecdc4', lw=1.3, zorder=2)
        ax.add_patch(r2)
        ax.text(xg2 + _gw*0.425, y, f'Ry\nL{lyr+1}', color='white', fontsize=6, ha='center', va='center', zorder=3)
    rm = plt.Rectangle((_x_meas, y - _gh/2), _gw*1.1, _gh, fc='#3f2800', ec='#ffd93d', lw=1.8, zorder=2)
    ax.add_patch(rm)
    ax.text(_x_meas + _gw*0.55, y, '⟨Z⟩\nmeas', color='white', fontsize=7,
            ha='center', va='center', zorder=3, fontweight='bold')

for lyr in range(N_LAYERS):
    xc = _x_enc + _gw + 0.3 + lyr * _lyr_gap + _gw * 1.85
    kw = dict(arrowstyle='->', color='#ffd93d', lw=1.6)
    for i in range(N_QUBITS - 1):
        y1 = N_QUBITS - 1 - i; y2 = y1 - 1; gap = _gh / 2 + 0.05
        ax.annotate('', xy=(xc, y2 + gap), xytext=(xc, y1 - gap), arrowprops=dict(**kw))
        ax.plot(xc, y1 - gap, 'o', color='#ffd93d', ms=7, zorder=4)
    ax.annotate('', xy=(xc, N_QUBITS - 1 + _gh/2 + 0.05), xytext=(xc, 0 - _gh/2 - 0.05),
                arrowprops=dict(**kw, connectionstyle=f'arc3,rad={-0.35 - N_QUBITS*0.03}'))
    ax.plot(xc, 0 - _gh/2 - 0.05, 'o', color='#ffd93d', ms=7, zorder=4)

ax.text(_x_enc,       -1.35, '■ Ry Encoding',       color='#00e5ff', fontsize=8.5)
ax.text(_x_enc + 4.2, -1.35, '■ Rz+Ry Variational', color='#ff6b6b', fontsize=8.5)
ax.text(_x_enc + 9.0, -1.35, '⬤→ CNOT Ring',        color='#ffd93d', fontsize=8.5)
ax.text(_x_enc + 13., -1.35, '■ ⟨Z⟩ Measurement',   color='#ffd93d', fontsize=8.5)
save(fig, '07_quantum_circuit.png')

# ── FIG 08 — Per-class P/R heatmap ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5)); fig.patch.set_facecolor(BG)
for ax_idx, (metric_fn, mname) in enumerate(
        [(precision_score, 'Precision'), (recall_score, 'Recall')]):
    ax = axes[ax_idx]; ax.set_facecolor(BG2)
    tbl = np.array([
        metric_fn(y_te, mp, average=None, zero_division=0) * 100
        for mp in [qml_preds, rf_preds, svm_preds, gbm_preds]
    ])
    vmin = max(0, tbl.min() - 5)
    sns.heatmap(tbl, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=NAMES,
                yticklabels=['Hybrid QNN (VQC)','Random Forest','SVM (RBF)','Gradient Boost'],
                ax=ax, linewidths=0.5, linecolor='#222',
                annot_kws={'size': 12, 'weight': 'bold'},
                vmin=vmin, vmax=100, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Per-Class {mname} (%) — All Models', color=TC, fontsize=13, pad=8)
    ax.set_xlabel('Traffic Class (5G Slice)'); ax.set_ylabel('Model')

fig.suptitle('Precision & Recall Heatmaps',
             color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '08_per_class_metrics.png')

# ── FIG 09 — Training Dynamics ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor(BG)

ax = axes[0]; ax.set_facecolor(BG2)
ax.plot(epochs_x, hist['lr'], color=C_MMTC, lw=2.5)
ax.fill_between(epochs_x, hist['lr'], alpha=0.15, color=C_MMTC)
ax.set_title('Learning Rate — Cosine Annealing', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
ax.grid(True, color=GC, alpha=0.5); ax.set_xlim(1, EPOCHS)

ax = axes[1]; ax.set_facecolor(BG2)
ax.plot(epochs_x, [v*100 for v in hist['precision']], color=C_EMBB,  lw=2, label='Precision (macro)')
ax.plot(epochs_x, [v*100 for v in hist['recall']],    color=C_URLLC, lw=2, label='Recall (macro)')
ax.plot(epochs_x, [v*100 for v in hist['f1']],        color=C_MMTC,  lw=2, label='F1 (macro)')
ax.set_title('Precision / Recall / F1 per Epoch', color=TC, fontsize=12, pad=8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, edgecolor='#444')
ax.grid(True, color=GC, alpha=0.5); ax.set_xlim(1, EPOCHS)

fig.suptitle('Training Dynamics', color='white', fontsize=14, y=1.02, fontweight='bold')
save(fig, '09_training_dynamics.png')

# ── FIG 10 — Full Dashboard ───────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 20)); fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

def dash_ax(pos, title):
    a = fig.add_subplot(pos); a.set_facecolor(BG2)
    a.set_title(title, color=TC, fontsize=10, pad=6)
    a.grid(True, color=GC, alpha=0.4)
    return a

# Row 0 — training curves
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

# Row 1 — confusion + ROC + comparison
ax = fig.add_subplot(gs[1,0]); ax.set_facecolor(BG2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=NAMES, yticklabels=NAMES,
            ax=ax, linewidths=0.8, linecolor='#333', annot_kws={'size': 11, 'weight': 'bold'}, cbar=False)
ax.set_title('Confusion Matrix', color=TC, fontsize=10, pad=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

ax = fig.add_subplot(gs[1,1]); ax.set_facecolor(BG2)
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=NAMES, yticklabels=NAMES,
            ax=ax, linewidths=0.8, linecolor='#333', annot_kws={'size': 10, 'weight': 'bold'}, cbar=False)
ax.set_title('Confusion Matrix (%)', color=TC, fontsize=10, pad=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

ax = dash_ax(gs[1,2], 'ROC Curves (QML)')
for cls, (cname, clr) in enumerate(zip(NAMES, CLRS)):
    fpr, tpr, _ = roc_curve(y_te_bin[:,cls], qml_proba[:,cls])
    ax.plot(fpr, tpr, color=clr, lw=2, label=f'{cname} AUC={auc(fpr,tpr):.3f}')
ax.plot([0,1],[0,1],'--', color='#555', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(facecolor=BG3, fontsize=8)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

ax = dash_ax(gs[1,3], 'Model Comparison')
mk2 = ['accuracy','precision','recall','f1']
x2  = np.arange(len(mk2)); w2 = 0.18
for j, (r, clr) in enumerate(zip(results, [C_QML, C_RF, C_SVM, C_GBM])):
    ax.bar(x2 + (j-1.5)*w2, [r[k] for k in mk2], w2,
           label=r['name'].split('(')[0].strip(), color=clr, edgecolor='w', lw=0.4, alpha=0.9)
ax.set_xticks(x2); ax.set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel('Score (%)')
ax.legend(facecolor=BG3, fontsize=7, ncol=2)

# Row 2 — feature plots + architecture
ax = dash_ax(gs[2,0], 'Throughput vs IAT — by Slice')
for cls, (nm, clr) in enumerate(zip(NAMES, CLRS)):
    m = df['traffic_class'] == cls
    ax.scatter(df.loc[m,'throughput_kbps'], df.loc[m,'inter_arrival_mean_us'],
               alpha=0.35, s=12, c=clr, label=nm)
ax.set_xlabel('Throughput (kbps)'); ax.set_ylabel('IAT Mean (µs)')
ax.legend(facecolor=BG3, fontsize=8)
ax.set_yscale('symlog', linthresh=10); ax.set_xscale('symlog', linthresh=100)

ax = dash_ax(gs[2,1], 'Avg Packet Size Distribution')
for cls, (nm, clr) in enumerate(zip(NAMES, CLRS)):
    m = df['traffic_class'] == cls
    ax.hist(df.loc[m,'avg_pkt_size_bytes'], bins=25, alpha=0.6,
            color=clr, label=nm, edgecolor='w', lw=0.2)
ax.set_xlabel('Avg Packet Size (bytes)'); ax.set_ylabel('Count')
ax.legend(facecolor=BG3, fontsize=8)

ax = dash_ax(gs[2,2], 'Feature Importance (RF)')
imp2  = rf.feature_importances_; order2 = np.argsort(imp2)
ax.barh(range(N_FEATURES), imp2[order2],
        color=plt.cm.plasma(np.linspace(0.15, 0.9, N_FEATURES)))
ax.set_yticks(range(N_FEATURES))
ax.set_yticklabels([_lbl_sh[i] for i in order2], fontsize=7)
ax.set_xlabel('Importance')

ax = fig.add_subplot(gs[2,3]); ax.set_facecolor(BG3); ax.axis('off')
ax.set_title('Hybrid QNN Architecture', color=TC, fontsize=10, pad=6)
arch = [
    ('INPUT',   '8 Wireshark Features',           '#264653'),
    ('Pre-Net', f'Linear(8→{N_QUBITS})+Tanh+BN',  '#2a9d8f'),
    ('VQC',     f'{N_QUBITS}q × {N_LAYERS}L VQC', '#e9c46a'),
    ('Post-Net', 'Linear(N→64→32→3)',              '#e76f51'),
    ('OUTPUT',  'eMBB | URLLC | mMTC',             '#264653'),
]
for k, (n, d, c) in enumerate(arch):
    yp = 0.88 - k * 0.19
    ax.add_patch(plt.Rectangle((0.05, yp-0.08), 0.9, 0.14, fc=c, ec='white', lw=1,
                                transform=ax.transAxes, zorder=2, alpha=0.9))
    ax.text(0.5, yp, f'{n}: {d}', color='white', fontsize=7.5,
            ha='center', va='center', transform=ax.transAxes, fontweight='bold', zorder=3)
    if k < len(arch) - 1:
        ax.annotate('', xy=(0.5, yp-0.09), xytext=(0.5, yp-0.02),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.8))

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
print(f"  Dataset       : {len(df)} samples  ({_mode})")
print(f"  Applications  : GeForce_Now, KT_GameBox, Battleground, Teamfight_Tactics,")
print(f"                  AfreecaTV, Naver_NOW, YouTube_Live, Netflix, Amazon_Prime,")
print(f"                  YouTube, Zoom, MS_Teams, Google_Meet, Roblox, Zepeto")
print(f"  Features      : {N_FEATURES}  (throughput, pkt_size, IAT_mean, IAT_std,")
print(f"                    pkt_count, DL/UL_ratio, bitrate_mean, bitrate_std)")
print(f"  Architecture  : Hybrid QNN  (Classical Pre → VQC → Classical Post)")
print(f"  Quantum Part  : {N_QUBITS}-qubit VQC | {N_LAYERS} StronglyEntangling layers")
print(f"                  State space: 2^{N_QUBITS} = {2**N_QUBITS} dimensions")
print(f"  Optimizer     : Adam + Cosine Annealing LR  ({LR} → 1e-4)")
print(f"  Class weights : eMBB={CLASS_WEIGHTS[0]:.1f}  URLLC={CLASS_WEIGHTS[1]:.1f}  "
      f"mMTC={CLASS_WEIGHTS[2]:.1f}  (URLLC upweighted for recall)")
print(f"  Epochs        : {EPOCHS}  (best at epoch {best_ep})")
print()
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("  " + "─" * 60)
for r in results:
    tag = "  ◄ QML" if r['name'].startswith('Hybrid') else ""
    print(f"  {r['name']:<22} {r['accuracy']:>8.2f}%  {r['precision']:>8.2f}%  "
          f"{r['recall']:>7.2f}%  {r['f1']:>7.2f}%{tag}")
print()
print("  OUTPUT FILES:")
for fname in [
    '01_training_curves.png       — Loss & Accuracy all epochs',
    '02_confusion_matrix.png      — Raw + Normalised confusion matrices',
    '03_precision_recall_f1.png   — Per-class P/R/F1 all models',
    '04_roc_curves.png            — ROC + AUC all classes & models',
    '05_model_comparison.png      — Grouped bar + Radar chart',
    '06_feature_analysis.png      — Scatter + Correlation + Importance',
    '07_quantum_circuit.png       — Full VQC gate diagram',
    '08_per_class_metrics.png     — P/R heatmaps all models',
    '09_training_dynamics.png     — LR schedule + P/R/F1 curves',
    '10_full_dashboard.png        — Combined report dashboard',
]:
    print(f"    {fname}")
print(HEADER)
print("  ✅  COMPLETE — ready to use in your research report!")
print(HEADER)