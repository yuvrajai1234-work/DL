import os, sys, copy, random, warnings, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'Train')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'Test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLDER_ALIAS = {
    'freshapples':'freshapples','rottenapples':'rottenapples',
    'freshbanana':'freshbanana','rottenbanana':'rottenbanana',
    'freshoranges':'freshoranges','rottenoranges':'rottenoranges',
    'freshcucumber':'freshcucumber','rottencucumber':'rottencucumber',
    'freshokra':'freshokra','rottenokra':'rottenokra',
    'freshtomato':'freshtomato','rottentomato':'rottentomato',
    'freshtamto':'freshtomato','rottentamto':'rottentomato',
    'freshpatato':None,'rottenpatato':None,
    'freshbittergroud':None,'rottenbittergroud':None,
    'freshcapsicum':None,'rottencapsicum':None,
    'freshpotato':None,'rottenpotato':None,
}
CLASS_NAMES = [
    'freshapples','rottenapples','freshbanana','rottenbanana',
    'freshoranges','rottenoranges','freshcucumber','rottencucumber',
    'freshokra','rottenokra','freshtomato','rottentomato',
]
DISPLAY_LABELS = [
    'Fresh Apple','Stale Apple','Fresh Banana','Stale Banana',
    'Fresh Orange','Stale Orange','Fresh Cucumber','Stale Cucumber',
    'Fresh Okra','Stale Okra','Fresh Tomato','Stale Tomato',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES  = 12
IMG_SIZE     = 64
BATCH_SIZE   = 128
EPOCHS       = 10
LR           = 1e-3
MAX_PER_CLASS= 1000
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

base_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

aug_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class InMemAugDataset(Dataset):
    def __init__(self, pil_images, labels, transform):
        self.pil_images = pil_images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.pil_images)
    def __getitem__(self, idx):
        img = self.pil_images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, d2=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(True)
        ]
        if d2 > 0:
            layers.append(nn.Dropout2d(d2))
        layers.append(nn.MaxPool2d(2, 2))
        self.b = nn.Sequential(*layers)
    def forward(self, x):
        return self.b(x)

class BaselineCNN(nn.Module):
    def __init__(self, n=NUM_CLASSES, use_drop=False, dr=0.4):
        super().__init__()
        d = dr if use_drop else 0.0
        self.feat = nn.Sequential(
            ConvBlock(3, 32, d * 0.2),
            ConvBlock(32, 64, d * 0.2),
            ConvBlock(64, 128, d * 0.3)
        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(d) if use_drop else nn.Identity(),
            nn.Linear(256, n)
        )
    def forward(self, x):
        return self.cls(self.feat(x))

class DeeperCNN(nn.Module):
    def __init__(self, n=NUM_CLASSES, dr=0.35):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64, dr * 0.25),
            ConvBlock(64, 128, dr * 0.3),
            ConvBlock(128, 256, dr * 0.35)
        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dr),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(dr * 0.5),
            nn.Linear(128, n)
        )
    def forward(self, x):
        return self.cls(self.feat(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, tr_ld, vl_ld, name='', epochs=EPOCHS):
    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_va, best_w = 0.0, None

    print(f'\n[{name}] Starting training ({epochs} epochs)...', flush=True)
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labs in tr_ld:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labs).sum().item()
            tt += imgs.size(0)
        sched.step()

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in vl_ld:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs)
                loss = crit(out, labs)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == labs).sum().item()
                vt += imgs.size(0)

        ta, va = 100.0 * tc / tt, 100.0 * vc / vt
        hist['train_loss'].append(round(tl / tt, 4))
        hist['val_loss'].append(round(vl / vt, 4))
        hist['train_acc'].append(round(ta, 2))
        hist['val_acc'].append(round(va, 2))
        if va >= best_va:
            best_va = va
            best_w = copy.deepcopy(model.state_dict())
        print(f'  Ep{ep:>2}/{epochs} TrAcc={ta:.2f}% VaAcc={va:.2f}% TrLoss={tl/tt:.4f} VaLoss={vl/vt:.4f}', flush=True)

    elapsed = time.time() - t_start
    print(f'  Completed {name} in {elapsed:.1f}s | Best Val Acc: {best_va:.2f}%', flush=True)
    model.load_state_dict(best_w)
    return model, hist

def evaluate(model, loader):
    model.eval()
    ap, al, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            ap.extend(preds)
            al.extend(labs.numpy())
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    acc = 100.0 * accuracy_score(al, ap)
    f1 = f1_score(al, ap, average='macro')
    return acc, f1, np.array(ap), np.array(al), all_probs

def main():
    print(f'Device:{DEVICE} ImgSize:{IMG_SIZE} Epochs:{EPOCHS} MaxPerClass:{MAX_PER_CLASS}', flush=True)

    # 1. Load and cache training data
    print('Loading and pre-caching dataset into RAM...', flush=True)
    t0 = time.time()
    train_samples = []
    for folder in sorted(os.listdir(TRAIN_DIR)):
        canonical = FOLDER_ALIAS.get(folder.lower())
        if canonical is None: continue
        label = CLASS_TO_IDX[canonical]
        fp = os.path.join(TRAIN_DIR, folder)
        files = [f for f in sorted(os.listdir(fp)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
            rng = random.Random(SEED + label)
            rng.shuffle(files)
            files = files[:MAX_PER_CLASS]
        for f in files:
            train_samples.append((os.path.join(fp, f), label))

    train_pil_images = []
    for p, l in train_samples:
        img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        train_pil_images.append(img)
    train_labels = [s[1] for s in train_samples]

    tr_idx, vl_idx = train_test_split(
        list(range(len(train_samples))),
        test_size=0.2,
        random_state=SEED,
        stratify=train_labels
    )
    print(f'Train samples:{len(tr_idx)} Val samples:{len(vl_idx)} (Cached in {time.time()-t0:.1f}s)', flush=True)

    # Precompute validation tensors (never augmented)
    vl_tensors = torch.stack([transforms.ToTensor()(train_pil_images[i]) for i in vl_idx])
    vl_tensors = transforms.Normalize(MEAN, STD)(vl_tensors)
    vl_labels = torch.tensor([train_labels[i] for i in vl_idx], dtype=torch.long)
    vl_ld = DataLoader(TensorDataset(vl_tensors, vl_labels), batch_size=BATCH_SIZE, shuffle=False)

    # Precompute non-augmented train tensors for fast loader
    tr_base_tensors = torch.stack([transforms.ToTensor()(train_pil_images[i]) for i in tr_idx])
    tr_base_tensors = transforms.Normalize(MEAN, STD)(tr_base_tensors)
    tr_base_labels = torch.tensor([train_labels[i] for i in tr_idx], dtype=torch.long)
    tr_base_ds = TensorDataset(tr_base_tensors, tr_base_labels)

    # In-memory augmented train dataset
    tr_aug_pil = [train_pil_images[i] for i in tr_idx]
    tr_aug_labels = [train_labels[i] for i in tr_idx]
    tr_aug_ds = InMemAugDataset(tr_aug_pil, tr_aug_labels, aug_tf)

    # 2. Load and pre-cache test dataset
    t0 = time.time()
    test_samples = []
    for folder in sorted(os.listdir(TEST_DIR)):
        canonical = FOLDER_ALIAS.get(folder.lower())
        if canonical is None: continue
        label = CLASS_TO_IDX[canonical]
        fp = os.path.join(TEST_DIR, folder)
        files = [f for f in sorted(os.listdir(fp)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        for f in files:
            test_samples.append((os.path.join(fp, f), label))

    test_pil_images = []
    for p, l in test_samples:
        img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        test_pil_images.append(img)
    test_labels = torch.tensor([s[1] for s in test_samples], dtype=torch.long)
    test_tensors = torch.stack([transforms.ToTensor()(img) for img in test_pil_images])
    test_tensors = transforms.Normalize(MEAN, STD)(test_tensors)
    te_ld = DataLoader(TensorDataset(test_tensors, test_labels), batch_size=BATCH_SIZE * 2, shuffle=False)
    print(f'Test samples:{len(test_samples)} (Cached in {time.time()-t0:.1f}s)', flush=True)

    def get_loaders(use_aug):
        if use_aug:
            tr_ld = DataLoader(tr_aug_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        else:
            tr_ld = DataLoader(tr_base_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        return tr_ld, vl_ld, te_ld

    # 3. Define the 5 ablation configurations
    ablations = [
        ('Baseline CNN',        BaselineCNN(use_drop=False), False),
        ('CNN + Dropout',       BaselineCNN(use_drop=True),  False),
        ('CNN + Augmentation',  BaselineCNN(use_drop=False), True),
        ('CNN + Aug + Drop',    BaselineCNN(use_drop=True),  True),
        ('Deeper CNN (Final)',  DeeperCNN(),                 True),
    ]

    results = []
    histories = {}
    best_models = {}
    final_preds, final_labels, final_probs = None, None, None
    final_history = None

    for name, model, aug_flag in ablations:
        n_params = count_parameters(model)
        tr, vl, te = get_loaders(aug_flag)
        m, h = train_model(model, tr, vl, name, epochs=EPOCHS)
        best_models[name] = m
        tr_eval_ld = DataLoader(tr_base_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
        ta, tf1, _, _, _ = evaluate(m, tr_eval_ld)
        va, vf1, _, _, _ = evaluate(m, vl)
        tea, tef, p, l, probs = evaluate(m, te)
        res = {
            'name': name,
            'params': n_params,
            'tr_acc': round(ta, 2),
            'va_acc': round(va, 2),
            'te_acc': round(tea, 2),
            'te_f1':  round(tef, 4),
            'best_va_epoch': int(np.argmax(h['val_acc'])) + 1,
            'best_va_acc': max(h['val_acc'])
        }
        results.append(res)
        histories[name] = h
        print(f'  >> Result: {name:<22} Tr={ta:.2f}% Va={va:.2f}% Te={tea:.2f}% F1={tef:.4f} (Params:{n_params:,})', flush=True)

        if 'Baseline' in name and 'Dropout' not in name and 'Aug' not in name:
            torch.save(m.state_dict(), os.path.join(OUTPUT_DIR, 'best_baseline_cnn.pt'))
        if 'Final' in name:
            final_history = h
            final_preds = p
            final_labels = l
            final_probs = probs
            torch.save(m.state_dict(), os.path.join(OUTPUT_DIR, 'best_deeper_cnn.pt'))

    print('\n' + '=' * 70, flush=True)
    print('                      ABLATION STUDY SUMMARY                      ', flush=True)
    print('=' * 70, flush=True)
    print(f"{'Experiment':<22} {'Params':<10} {'Train Acc':<11} {'Val Acc':<10} {'Test Acc':<10} {'Macro F1':<8}", flush=True)
    print('-' * 70, flush=True)
    for r in results:
        print(f"{r['name']:<22} {r['params']:<10,} {r['tr_acc']:>7.2f}%   {r['va_acc']:>7.2f}%  {r['te_acc']:>7.2f}%  {r['te_f1']:>7.4f}", flush=True)
    print('=' * 70, flush=True)

    # 4. Generate Visualizations
    ER = range(1, EPOCHS + 1)
    DK = '#0F1117'
    DA = '#1A1D2E'
    PALETTE = ['#4FC3F7', '#FFB74D', '#81C784', '#F06292', '#CE93D8']

    # ── Plot 1: Accuracy Curve (Final Model) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    ta_c = final_history['train_acc']; va_c = final_history['val_acc']
    ax.plot(ER, ta_c, color='#4FC3F7', lw=2.5, marker='o', ms=5, label='Train Accuracy')
    ax.plot(ER, va_c, color='#FFB74D', lw=2.5, marker='s', ms=5, label='Val Accuracy')
    ax.fill_between(ER, ta_c, va_c, alpha=0.15, color='#7C4DFF')
    be = int(np.argmax(va_c)) + 1; bv = max(va_c)
    ax.axvline(be, color='#FF6B6B', lw=1.5, ls='--', alpha=0.9, label=f'Best Val Ep{be} ({bv:.2f}%)')
    ax.annotate(f'Best\n{bv:.2f}%', xy=(be, bv), xytext=(min(be + 0.5, EPOCHS - 0.5), bv - 5),
                color='white', fontsize=9, arrowprops=dict(arrowstyle='->', color='white', lw=1))
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=12)
    ax.set_title('Training vs Validation Accuracy — Deeper CNN (Final)', color='white', fontsize=14, pad=12)
    ax.set_xticks(list(ER))
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333654')
    ax.grid(True, ls='--', alpha=0.25, color='#555')
    ax.legend(facecolor=DA, edgecolor='#555', labelcolor='white', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curve.png'), dpi=150, bbox_inches='tight', facecolor=DK)
    plt.close()
    print('Saved: outputs/accuracy_curve.png', flush=True)

    # ── Plot 2: Loss Curve (Final Model) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    tl_c = final_history['train_loss']; vl_c = final_history['val_loss']
    ax.plot(ER, tl_c, color='#4FC3F7', lw=2.5, marker='o', ms=5, label='Train Loss')
    ax.plot(ER, vl_c, color='#FFB74D', lw=2.5, marker='s', ms=5, label='Val Loss')
    ax.fill_between(ER, tl_c, vl_c, alpha=0.15, color='#F06292')
    bl = int(np.argmin(vl_c)) + 1; bvl = min(vl_c)
    ax.axvline(bl, color='#FF6B6B', lw=1.5, ls='--', alpha=0.9, label=f'Best Val Ep{bl} ({bvl:.4f})')
    ax.annotate(f'Best\n{bvl:.4f}', xy=(bl, bvl), xytext=(min(bl + 0.5, EPOCHS - 0.5), bvl + 0.05),
                color='white', fontsize=9, arrowprops=dict(arrowstyle='->', color='white', lw=1))
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    ax.set_title('Training vs Validation Loss — Deeper CNN (Final)', color='white', fontsize=14, pad=12)
    ax.set_xticks(list(ER))
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333654')
    ax.grid(True, ls='--', alpha=0.25, color='#555')
    ax.legend(facecolor=DA, edgecolor='#555', labelcolor='white', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'), dpi=150, bbox_inches='tight', facecolor=DK)
    plt.close()
    print('Saved: outputs/loss_curve.png', flush=True)

    # ── Plot 3: Ablation Study Curves (Val Acc & Val Loss across 5 configs) ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DK)
    for ax in axes:
        ax.set_facecolor(DA)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333654')
        ax.grid(True, ls='--', alpha=0.25, color='#555')
        ax.set_xticks(list(ER))

    for i, (name, _, _) in enumerate(ablations):
        h = histories[name]
        axes[0].plot(ER, h['val_acc'], color=PALETTE[i], lw=2.2, marker='o', ms=4, label=f"{name} ({max(h['val_acc']):.1f}%)")
        axes[1].plot(ER, h['val_loss'], color=PALETTE[i], lw=2.2, marker='s', ms=4, label=f"{name} ({min(h['val_loss']):.3f})")

    axes[0].set_title('Ablation Study — Validation Accuracy Across Epochs', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Epoch', color='white', fontsize=11)
    axes[0].set_ylabel('Accuracy (%)', color='white', fontsize=11)
    axes[0].legend(facecolor=DA, edgecolor='#555', labelcolor='white', fontsize=8.5, loc='lower right')

    axes[1].set_title('Ablation Study — Validation Loss Across Epochs', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Epoch', color='white', fontsize=11)
    axes[1].set_ylabel('Cross-Entropy Loss', color='white', fontsize=11)
    axes[1].legend(facecolor=DA, edgecolor='#555', labelcolor='white', fontsize=8.5, loc='upper right')

    fig.suptitle('Ablation Study — 5 Model Configurations Compared', color='white', fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'ablation_study.png'), dpi=150, bbox_inches='tight', facecolor=DK)
    plt.close()
    print('Saved: outputs/ablation_study.png', flush=True)

    # ── Plot 4: Confusion Matrix (Test Set, Deeper CNN Final) ─────────────────
    cm = confusion_matrix(final_labels, final_preds)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor(DK)
    fig.suptitle('Confusion Matrix — Deeper CNN (Final, Test Set)', color='white', fontsize=16, y=1.01)

    for ax, data, fmt, cmap, title in zip(
        axes,
        [cm, cm_n],
        ['d', '.2f'],
        ['YlOrRd', 'Blues'],
        ['Absolute Counts', 'Normalized (Row %)']
    ):
        ax.set_facecolor(DA)
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
            linewidths=0.5, linecolor='#222', ax=ax,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8.5, 'weight': 'bold'}
        )
        ax.set_title(title, color='white', fontsize=13, pad=8)
        ax.set_xlabel('Predicted Label', color='white', fontsize=11)
        ax.set_ylabel('True Label', color='white', fontsize=11)
        ax.tick_params(colors='white', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight', facecolor=DK)
    plt.close()
    print('Saved: outputs/confusion_matrix.png', flush=True)

    # ── Plot 5: Ablation Bar Chart (Test Acc & Macro F1) ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    names = [r['name'] for r in results]
    tes   = [r['te_acc'] for r in results]
    f1s   = [r['te_f1'] * 100 for r in results]
    x = np.arange(len(names)); w = 0.35
    b1 = ax.bar(x - w/2, tes, w, color='#4FC3F7', alpha=0.9, label='Test Accuracy (%)')
    b2 = ax.bar(x + w/2, f1s, w, color='#FFB74D', alpha=0.9, label='Macro F1 × 100')

    for b in b1:
        h = b.get_height()
        ax.annotate(f'{h:.2f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', color='white', fontsize=8.5, fontweight='bold')
    for b in b2:
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', color='white', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, color='white', fontsize=9.5, rotation=15, ha='right')
    ax.set_ylabel('Score', color='white', fontsize=12)
    ax.set_title('Ablation Study — Test Accuracy & Macro F1-Score Comparison', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333654')
    ax.grid(axis='y', ls='--', alpha=0.25, color='#555')
    ax.set_ylim(min(min(tes), min(f1s)) - 6, 102)
    ax.legend(facecolor=DA, edgecolor='#555', labelcolor='white', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'ablation_bar.png'), dpi=150, bbox_inches='tight', facecolor=DK)
    plt.close()
    print('Saved: outputs/ablation_bar.png', flush=True)

    # ── Plot 6: Error Analysis — Visual Sample Grid of Misclassified Images ──
    wrong_indices = np.where(final_preds != final_labels)[0]
    print(f'Total Test Misclassifications: {len(wrong_indices)} / {len(final_labels)} ({100*len(wrong_indices)/len(final_labels):.2f}%)', flush=True)

    if len(wrong_indices) > 0:
        # Sample 12 representative errors
        rng = np.random.default_rng(SEED)
        sample_size = min(12, len(wrong_indices))
        chosen_indices = rng.choice(wrong_indices, size=sample_size, replace=False)

        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        fig.patch.set_facecolor(DK)
        fig.suptitle('Error Analysis — Test Set Misclassification Examples (Deeper CNN Final)', color='white', fontsize=15, y=0.99)

        for ax, idx in zip(axes.flat, chosen_indices):
            ax.set_facecolor(DA)
            pil_img = test_pil_images[idx]
            true_label = DISPLAY_LABELS[final_labels[idx]]
            pred_label = DISPLAY_LABELS[final_preds[idx]]
            conf = final_probs[idx, final_preds[idx]] * 100.0

            ax.imshow(pil_img)
            ax.set_xticks([]); ax.set_yticks([])
            border_color = '#FF5252' if final_labels[idx] // 2 == final_preds[idx] // 2 else '#FFB74D'
            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(2.5)

            title_color = '#FF8A80' if 'Stale' in pred_label else '#80D8FF'
            err_type = "Fresh/Stale Confusion" if final_labels[idx] // 2 == final_preds[idx] // 2 else "Cross-Produce Error"
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)\n[{err_type}]", color='white', fontsize=8.5, pad=6)

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'misclassified_samples.png'), dpi=150, bbox_inches='tight', facecolor=DK)
        plt.close()
        print('Saved: outputs/misclassified_samples.png', flush=True)

    # 5. Extract Detailed Error Analysis & Classification Report
    report_dict = classification_report(final_labels, final_preds, target_names=DISPLAY_LABELS, digits=4, output_dict=True)
    report_text = classification_report(final_labels, final_preds, target_names=DISPLAY_LABELS, digits=4)

    # Compute top confusion pairs
    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                is_same_produce = (i // 2 == j // 2)
                confusion_pairs.append({
                    'true_class': DISPLAY_LABELS[i],
                    'pred_class': DISPLAY_LABELS[j],
                    'count': int(cm[i, j]),
                    'error_type': 'Fresh/Stale Confusion' if is_same_produce else 'Cross-Produce Confusion'
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    # Save comprehensive results JSON
    dump_data = {
        'ablation_results': results,
        'histories': histories,
        'final_classification_report': report_dict,
        'top_confusion_pairs': confusion_pairs[:15],
        'total_test_samples': len(final_labels),
        'total_errors': int(len(wrong_indices)),
        'error_rate_percent': round(100.0 * len(wrong_indices) / len(final_labels), 2)
    }
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(dump_data, f, indent=2)
    print('Saved: outputs/ablation_results.json', flush=True)

    print('\n=== FINAL MODEL CLASSIFICATION REPORT ===', flush=True)
    print(report_text, flush=True)
    print('All training, evaluation, and plot generations completed successfully!')

if __name__ == '__main__':
    main()
