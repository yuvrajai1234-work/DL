import os, copy, random, warnings, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings("ignore")
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_DIR = r"c:\Users\yuvra\Downloads\archive"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Train")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "Test")
OUT       = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

ALIAS = {
    "freshapples":"freshapples","rottenapples":"rottenapples",
    "freshbanana":"freshbanana","rottenbanana":"rottenbanana",
    "freshoranges":"freshoranges","rottenoranges":"rottenoranges",
    "freshcucumber":"freshcucumber","rottencucumber":"rottencucumber",
    "freshokra":"freshokra","rottenokra":"rottenokra",
    "freshtomato":"freshtomato","rottentomato":"rottentomato",
    "freshtamto":"freshtomato","rottentamto":"rottentomato",
    "freshpatato":None,"rottenpatato":None,
    "freshbittergroud":None,"rottenbittergroud":None,
    "freshcapsicum":None,"rottencapsicum":None,
    "freshpotato":None,"rottenpotato":None,
}

CLS = [
    "freshapples","rottenapples","freshbanana","rottenbanana",
    "freshoranges","rottenoranges","freshcucumber","rottencucumber",
    "freshokra","rottenokra","freshtomato","rottentomato",
]
DL = [
    "Fresh Apple","Stale Apple","Fresh Banana","Stale Banana",
    "Fresh Orange","Stale Orange","Fresh Cucumber","Stale Cucumber",
    "Fresh Okra","Stale Okra","Fresh Tomato","Stale Tomato",
]
C2I = {c:i for i,c in enumerate(CLS)}

IMG=64; BS=128; EP=10; LR=1e-3; MX=1000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Dataset ──────────────────────────────────────────────────────────────────
class DS(torch.utils.data.Dataset):
    def __init__(self, root, mx=None):
        self.s = []
        for folder in sorted(os.listdir(root)):
            can = ALIAS.get(folder.lower())
            if can is None: continue
            lab = C2I[can]
            fp  = os.path.join(root, folder)
            files = [f for f in os.listdir(fp) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
            if mx:
                random.shuffle(files)
                files = files[:mx]
            for f in files:
                self.s.append((os.path.join(fp, f), lab))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]
        img = Image.open(p).convert("RGB")
        return tf(img), l

print("Loading data...", flush=True)
tr_ds = DS(TRAIN_DIR, MX)
te_ds = DS(TEST_DIR)
labs  = [s[1] for s in tr_ds.s]
idx   = list(range(len(tr_ds)))
ti, vi = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=labs)
tr_ld = DataLoader(Subset(tr_ds, ti), batch_size=BS, shuffle=True,  num_workers=0)
vl_ld = DataLoader(Subset(tr_ds, vi), batch_size=BS, shuffle=False, num_workers=0)
te_ld = DataLoader(te_ds,             batch_size=BS, shuffle=False, num_workers=0)
print(f"Train:{len(ti)}  Val:{len(vi)}  Test:{len(te_ds)}  Device:{DEV}", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(True), nn.MaxPool2d(2, 2))
    def forward(self, x): return self.b(x)

class BaselineCNN(nn.Module):
    """
    3-block CNN | Input 64x64x3 | Output 12 logits
    Trained via backpropagation (Adam + CrossEntropyLoss)
    """
    def __init__(self, n=12):
        super().__init__()
        self.feat = nn.Sequential(ConvBlock(3,32), ConvBlock(32,64), ConvBlock(64,128))
        self.cls  = nn.Sequential(
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(128*2*2, 256), nn.ReLU(True), nn.Linear(256, n))
    def forward(self, x): return self.cls(self.feat(x))

model = BaselineCNN().to(DEV)
crit  = nn.CrossEntropyLoss(label_smoothing=0.05)
opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
H     = {"tl":[], "vl":[], "ta":[], "va":[]}
bv    = 0.0; bw = None

# ── Training loop (backpropagation) ──────────────────────────────────────────
print(f"\nTraining Baseline CNN — 12 classes, {EP} epochs", flush=True)
print("-"*60, flush=True)
for ep in range(1, EP+1):
    model.train(); tl,tc,tt = 0.,0,0
    for x, y in tr_ld:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad()
        o    = model(x)
        loss = crit(o, y)
        loss.backward()                              # backprop
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()*x.size(0)
        tc += (o.argmax(1)==y).sum().item()
        tt += x.size(0)
    sched.step()
    model.eval(); vl,vc,vt = 0.,0,0
    with torch.no_grad():
        for x, y in vl_ld:
            x, y = x.to(DEV), y.to(DEV)
            o    = model(x)
            loss = crit(o, y)
            vl += loss.item()*x.size(0)
            vc += (o.argmax(1)==y).sum().item()
            vt += x.size(0)
    ta, va = 100*tc/tt, 100*vc/vt
    H["tl"].append(tl/tt); H["vl"].append(vl/vt)
    H["ta"].append(ta);    H["va"].append(va)
    if va > bv: bv = va; bw = copy.deepcopy(model.state_dict())
    print(f"  Ep{ep:>2}/{EP}  TrAcc={ta:.2f}%  VaAcc={va:.2f}%  TrLoss={tl/tt:.4f}  VaLoss={vl/vt:.4f}", flush=True)

# ── Test evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(bw)
model.eval(); ap, al = [], []
with torch.no_grad():
    for x, y in te_ld:
        ap.extend(model(x.to(DEV)).argmax(1).cpu().numpy())
        al.extend(y.numpy())
te_acc = 100*accuracy_score(al, ap)
te_f1  = f1_score(al, ap, average="macro")
print(f"\nTest Acc = {te_acc:.2f}%   F1 = {te_f1:.4f}", flush=True)

ER = range(1, EP+1)
DK = "#0F1117"; DA = "#1A1D2E"

# ── Plot 1: Accuracy Curve ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(DK); ax.set_facecolor(DA)

ax.plot(ER, H["ta"], color="#4FC3F7", lw=2.5, marker="o", ms=6, label="Train Accuracy")
ax.plot(ER, H["va"], color="#FFB74D", lw=2.5, marker="s", ms=6, label="Val Accuracy")
ax.fill_between(ER, H["ta"], H["va"], alpha=0.15, color="#7C4DFF")

be      = int(np.argmax(H["va"])) + 1
best_v  = max(H["va"])
ax.axvline(be, color="#FF6B6B", lw=1.5, ls="--", alpha=0.9, label=f"Best Val  Ep{be}  ({best_v:.2f}%)")

for i, (ta, va) in enumerate(zip(H["ta"], H["va"])):
    ax.annotate(f"{ta:.1f}%", xy=(i+1, ta), xytext=(0, 9),
                textcoords="offset points", ha="center", color="#4FC3F7", fontsize=8, fontweight="bold")
    ax.annotate(f"{va:.1f}%", xy=(i+1, va), xytext=(0, -15),
                textcoords="offset points", ha="center", color="#FFB74D", fontsize=8, fontweight="bold")

ax.set_xlabel("Epoch", color="white", fontsize=12)
ax.set_ylabel("Accuracy (%)", color="white", fontsize=12)
ax.set_title("Training vs Validation Accuracy  |  Baseline CNN  |  12-Class (6 Produce)",
             color="white", fontsize=14, pad=14)
ax.set_xticks(list(ER)); ax.tick_params(colors="white")
ax.spines[:].set_color("#333654"); ax.grid(True, ls="--", alpha=0.25, color="#555")
ax.legend(facecolor=DA, edgecolor="#555", labelcolor="white", fontsize=10)
ax.set_ylim(50, 102)

stats_text = (
    f"Final Train : {H['ta'][-1]:.2f}%\n"
    f"Best   Val  : {best_v:.2f}%\n"
    f"Test   Acc  : {te_acc:.2f}%\n"
    f"F1 Score    : {te_f1:.4f}"
)
ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, color="white",
        fontsize=9.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A2D3E", edgecolor="#4FC3F7", alpha=0.9))

fig.tight_layout()
fig.savefig(os.path.join(OUT, "baseline_accuracy.png"), dpi=150, bbox_inches="tight", facecolor=DK)
plt.close()
print("Saved: baseline_accuracy.png", flush=True)

# ── Plot 2: Loss Curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(DK); ax.set_facecolor(DA)

ax.plot(ER, H["tl"], color="#4FC3F7", lw=2.5, marker="o", ms=6, label="Train Loss")
ax.plot(ER, H["vl"], color="#FFB74D", lw=2.5, marker="s", ms=6, label="Val Loss")
ax.fill_between(ER, H["tl"], H["vl"], alpha=0.15, color="#F06292")

bl      = int(np.argmin(H["vl"])) + 1
best_vl = min(H["vl"])
ax.axvline(bl, color="#FF6B6B", lw=1.5, ls="--", alpha=0.9, label=f"Best Val  Ep{bl}  ({best_vl:.4f})")

for i, (tl_v, vl_v) in enumerate(zip(H["tl"], H["vl"])):
    ax.annotate(f"{tl_v:.3f}", xy=(i+1, tl_v), xytext=(0, -16),
                textcoords="offset points", ha="center", color="#4FC3F7", fontsize=8, fontweight="bold")
    ax.annotate(f"{vl_v:.3f}", xy=(i+1, vl_v), xytext=(0, 9),
                textcoords="offset points", ha="center", color="#FFB74D", fontsize=8, fontweight="bold")

ax.set_xlabel("Epoch", color="white", fontsize=12)
ax.set_ylabel("Loss", color="white", fontsize=12)
ax.set_title("Training vs Validation Loss  |  Baseline CNN  |  12-Class (6 Produce)",
             color="white", fontsize=14, pad=14)
ax.set_xticks(list(ER)); ax.tick_params(colors="white")
ax.spines[:].set_color("#333654"); ax.grid(True, ls="--", alpha=0.25, color="#555")
ax.legend(facecolor=DA, edgecolor="#555", labelcolor="white", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "baseline_loss.png"), dpi=150, bbox_inches="tight", facecolor=DK)
plt.close()
print("Saved: baseline_loss.png", flush=True)

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
cm   = confusion_matrix(al, ap)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.patch.set_facecolor(DK)
fig.suptitle("Confusion Matrix  |  Baseline CNN  |  12-Class  |  Test Set",
             color="white", fontsize=17, y=1.01)

for ax2, data, fmt, cmap, title in zip(
    axes,
    [cm, cm_n],
    ["d", ".2f"],
    ["YlOrRd", "Blues"],
    ["Absolute Counts", "Normalized (Row %)"]
):
    ax2.set_facecolor(DA)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=DL, yticklabels=DL,
                linewidths=0.5, linecolor="#222", ax=ax2,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax2.set_title(title, color="white", fontsize=14, pad=10)
    ax2.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax2.set_ylabel("True Label",      color="white", fontsize=11)
    ax2.tick_params(colors="white", labelsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "baseline_confusion.png"), dpi=150, bbox_inches="tight", facecolor=DK)
plt.close()
print("Saved: baseline_confusion.png", flush=True)

# ── Classification Report ─────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  CLASSIFICATION REPORT  —  Baseline CNN", flush=True)
print("="*65, flush=True)
print(classification_report(al, ap, target_names=DL, digits=4), flush=True)
print("ALL DONE — outputs saved to:", OUT, flush=True)
