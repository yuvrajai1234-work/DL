import os, sys, copy, random, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter

warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_DIR   = r'c:\Users\yuvra\Downloads\archive'
TRAIN_DIR  = os.path.join(BASE_DIR,'dataset','Train')
TEST_DIR   = os.path.join(BASE_DIR,'dataset','Test')
OUTPUT_DIR = os.path.join(BASE_DIR,'outputs')
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
CLASS_NAMES=[
    'freshapples','rottenapples','freshbanana','rottenbanana',
    'freshoranges','rottenoranges','freshcucumber','rottencucumber',
    'freshokra','rottenokra','freshtomato','rottentomato',
]
DISPLAY_LABELS=[
    'Fresh Apple','Stale Apple','Fresh Banana','Stale Banana',
    'Fresh Orange','Stale Orange','Fresh Cucumber','Stale Cucumber',
    'Fresh Okra','Stale Okra','Fresh Tomato','Stale Tomato',
]
CLASS_TO_IDX={c:i for i,c in enumerate(CLASS_NAMES)}
NUM_CLASSES=12; IMG_SIZE=64; BATCH_SIZE=128; EPOCHS=10; LR=1e-3; MAX_PER_CLASS=1000
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device:{DEVICE} ImgSize:{IMG_SIZE} Epochs:{EPOCHS} MaxPerClass:{MAX_PER_CLASS}',flush=True)

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
base_tf=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
aug_tf=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),transforms.ColorJitter(0.3,0.3,0.2,0.1),
    transforms.ToTensor(),transforms.Normalize(MEAN,STD),
])

class ProduceDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None,max_per=None):
        self.samples=[]; self.transform=transform; per_cls={}
        for folder in sorted(os.listdir(root)):
            canonical=FOLDER_ALIAS.get(folder.lower())
            if canonical is None: continue
            label=CLASS_TO_IDX[canonical]
            fp=os.path.join(root,folder)
            files=[f for f in os.listdir(fp) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
            if max_per: random.shuffle(files); files=files[:max_per]
            for f in files: self.samples.append((os.path.join(fp,f),label))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        p,l=self.samples[idx]; img=Image.open(p).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,l

print('Loading data...',flush=True)
full_base=ProduceDataset(TRAIN_DIR,base_tf,MAX_PER_CLASS)
full_aug =ProduceDataset(TRAIN_DIR,aug_tf, MAX_PER_CLASS)
test_ds  =ProduceDataset(TEST_DIR, base_tf)
labels_all=[s[1] for s in full_base.samples]
idx_all=list(range(len(full_base)))
tr_idx,vl_idx=train_test_split(idx_all,test_size=0.2,random_state=SEED,stratify=labels_all)
print(f'Train:{len(tr_idx)} Val:{len(vl_idx)} Test:{len(test_ds)}',flush=True)

def get_loaders(aug=False):
    tr=DataLoader(Subset(full_aug if aug else full_base,tr_idx),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    vl=DataLoader(Subset(full_base,vl_idx),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    te=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    return tr,vl,te

class ConvBlock(nn.Module):
    def __init__(self,ic,oc,d2=0.0):
        super().__init__()
        l=[nn.Conv2d(ic,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(True)]
        if d2>0: l.append(nn.Dropout2d(d2))
        l.append(nn.MaxPool2d(2,2)); self.b=nn.Sequential(*l)
    def forward(self,x): return self.b(x)

class BaselineCNN(nn.Module):
    def __init__(self,n=NUM_CLASSES,use_drop=False,dr=0.4):
        super().__init__()
        d=dr if use_drop else 0.0
        self.feat=nn.Sequential(ConvBlock(3,32,d*0.2),ConvBlock(32,64,d*0.2),ConvBlock(64,128,d*0.3))
        self.cls=nn.Sequential(
            nn.AdaptiveAvgPool2d(2),nn.Flatten(),
            nn.Linear(128*2*2,256),nn.ReLU(True),
            nn.Dropout(d) if use_drop else nn.Identity(),nn.Linear(256,n))
    def forward(self,x): return self.cls(self.feat(x))

class DeeperCNN(nn.Module):
    def __init__(self,n=NUM_CLASSES,dr=0.35):
        super().__init__()
        self.feat=nn.Sequential(
            ConvBlock(3,32),ConvBlock(32,64,dr*0.25),
            ConvBlock(64,128,dr*0.3),ConvBlock(128,256,dr*0.35))
        self.cls=nn.Sequential(
            nn.AdaptiveAvgPool2d(2),nn.Flatten(),
            nn.Linear(256*2*2,512),nn.BatchNorm1d(512),nn.ReLU(True),nn.Dropout(dr),
            nn.Linear(512,128),nn.ReLU(True),nn.Dropout(dr*0.5),nn.Linear(128,n))
    def forward(self,x): return self.cls(self.feat(x))

def train_model(model,tr_ld,vl_ld,name=''):
    model=model.to(DEVICE)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    hist={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_va,best_w=0.0,None
    print(f'\n[{name}] Starting training...',flush=True)
    for ep in range(1,EPOCHS+1):
        model.train(); tl,tc,tt=0.0,0,0
        for imgs,labs in tr_ld:
            imgs,labs=imgs.to(DEVICE),labs.to(DEVICE)
            opt.zero_grad(); out=model(imgs); loss=crit(out,labs)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item()*imgs.size(0); tc+=(out.argmax(1)==labs).sum().item(); tt+=imgs.size(0)
        sched.step()
        model.eval(); vl,vc,vt=0.0,0,0
        with torch.no_grad():
            for imgs,labs in vl_ld:
                imgs,labs=imgs.to(DEVICE),labs.to(DEVICE)
                out=model(imgs); loss=crit(out,labs)
                vl+=loss.item()*imgs.size(0); vc+=(out.argmax(1)==labs).sum().item(); vt+=imgs.size(0)
        ta,va=100*tc/tt,100*vc/vt
        hist['train_loss'].append(tl/tt); hist['val_loss'].append(vl/vt)
        hist['train_acc'].append(ta); hist['val_acc'].append(va)
        if va>best_va: best_va=va; best_w=copy.deepcopy(model.state_dict())
        print(f'  Ep{ep:>2}/{EPOCHS} TrAcc={ta:.2f}% VaAcc={va:.2f}% TrLoss={tl/tt:.4f} VaLoss={vl/vt:.4f}',flush=True)
    model.load_state_dict(best_w); return model,hist

def evaluate(model,loader):
    model.eval(); ap,al=[],[]
    with torch.no_grad():
        for imgs,labs in loader:
            ap.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy()); al.extend(labs.numpy())
    return 100*accuracy_score(al,ap),f1_score(al,ap,average='macro'),np.array(ap),np.array(al)

ablations=[
    ('Baseline CNN',      BaselineCNN(use_drop=False),False),
    ('CNN + Dropout',     BaselineCNN(use_drop=True), False),
    ('CNN + Augmentation',BaselineCNN(use_drop=False),True),
    ('CNN + Aug + Drop',  BaselineCNN(use_drop=True), True),
    ('Deeper CNN (Final)',DeeperCNN(),                True),
]
results=[]; histories={}; fp2=None; fl2=None; final_history=None

for name,model,aug in ablations:
    tr,vl,te=get_loaders(aug)
    m,h=train_model(model,tr,vl,name)
    ta,_,_,_=evaluate(m,tr); va,_,_,_=evaluate(m,vl); tea,tef,p,l=evaluate(m,te)
    results.append({'name':name,'tr':ta,'va':va,'te':tea,'f1':tef})
    histories[name]=h
    print(f'  >> {name}: Tr={ta:.2f}% Va={va:.2f}% Te={tea:.2f}% F1={tef:.4f}',flush=True)
    if 'Final' in name: final_history=h; fp2=p; fl2=l

print('\n=== RESULTS ===',flush=True)
for r in results:
    print(f"  {r['name']:<28} Tr={r['tr']:.2f}% Va={r['va']:.2f}% Te={r['te']:.2f}% F1={r['f1']:.4f}",flush=True)

ER=range(1,EPOCHS+1)
DK='#0F1117'; DA='#1A1D2E'; PALETTE=['#4FC3F7','#FFB74D','#81C784','#F06292','#CE93D8']

# ── Plot 1: Accuracy Curve (Final Model) ────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
ta_c=final_history['train_acc']; va_c=final_history['val_acc']
ax.plot(ER,ta_c,color='#4FC3F7',lw=2.5,marker='o',ms=4,label='Train Accuracy')
ax.plot(ER,va_c,color='#FFB74D',lw=2.5,marker='s',ms=4,label='Val Accuracy')
ax.fill_between(ER,ta_c,va_c,alpha=0.15,color='#7C4DFF')
be=int(np.argmax(va_c))+1; bv=max(va_c)
ax.axvline(be,color='#FF6B6B',lw=1.5,ls='--',alpha=0.8,label=f'Best Val Ep{be}')
ax.annotate(f'Best\n{bv:.2f}%',xy=(be,bv),xytext=(min(be+0.5,EPOCHS-0.5),bv-5),color='white',fontsize=9,arrowprops=dict(arrowstyle='->',color='white',lw=1))
ax.set_xlabel('Epoch',color='white',fontsize=12); ax.set_ylabel('Accuracy (%)',color='white',fontsize=12)
ax.set_title('Training vs Validation Accuracy — Deeper CNN (Final)',color='white',fontsize=14,pad=12)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333654'); ax.grid(True,ls='--',alpha=0.25,color='#555')
ax.legend(facecolor=DA,edgecolor='#555',labelcolor='white',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'accuracy_curve.png'),dpi=150,bbox_inches='tight',facecolor=DK); plt.close()
print('Saved: accuracy_curve.png',flush=True)

# ── Plot 2: Loss Curve (Final Model) ────────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
tl_c=final_history['train_loss']; vl_c=final_history['val_loss']
ax.plot(ER,tl_c,color='#4FC3F7',lw=2.5,marker='o',ms=4,label='Train Loss')
ax.plot(ER,vl_c,color='#FFB74D',lw=2.5,marker='s',ms=4,label='Val Loss')
ax.fill_between(ER,tl_c,vl_c,alpha=0.15,color='#F06292')
bl=int(np.argmin(vl_c))+1; bvl=min(vl_c)
ax.axvline(bl,color='#FF6B6B',lw=1.5,ls='--',alpha=0.8,label=f'Best Val Ep{bl}')
ax.annotate(f'Best\n{bvl:.4f}',xy=(bl,bvl),xytext=(min(bl+0.5,EPOCHS-0.5),bvl+0.03),color='white',fontsize=9,arrowprops=dict(arrowstyle='->',color='white',lw=1))
ax.set_xlabel('Epoch',color='white',fontsize=12); ax.set_ylabel('Loss',color='white',fontsize=12)
ax.set_title('Training vs Validation Loss — Deeper CNN (Final)',color='white',fontsize=14,pad=12)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333654'); ax.grid(True,ls='--',alpha=0.25,color='#555')
ax.legend(facecolor=DA,edgecolor='#555',labelcolor='white',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'loss_curve.png'),dpi=150,bbox_inches='tight',facecolor=DK); plt.close()
print('Saved: loss_curve.png',flush=True)

# ── Plot 3: Ablation curves ──────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(16,6)); fig.patch.set_facecolor(DK)
for ax in axes:
    ax.set_facecolor(DA); ax.tick_params(colors='white'); ax.spines[:].set_color('#333654'); ax.grid(True,ls='--',alpha=0.25,color='#555')
for i,(name,_,_) in enumerate(ablations):
    h=histories[name]
    axes[0].plot(ER,h['val_acc'], color=PALETTE[i],lw=2,label=name)
    axes[1].plot(ER,h['val_loss'],color=PALETTE[i],lw=2,label=name)
for ax,title,ylabel in zip(axes,['Ablation – Val Accuracy','Ablation – Val Loss'],['Accuracy (%)','Loss']):
    ax.set_title(title,color='white',fontsize=13,pad=10); ax.set_xlabel('Epoch',color='white',fontsize=11); ax.set_ylabel(ylabel,color='white',fontsize=11)
    ax.legend(facecolor=DA,edgecolor='#555',labelcolor='white',fontsize=8,loc='lower right' if 'Acc' in title else 'upper right')
fig.suptitle('Ablation Study – 5 Configurations Compared',color='white',fontsize=15)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'ablation_study.png'),dpi=150,bbox_inches='tight',facecolor=DK); plt.close()
print('Saved: ablation_study.png',flush=True)

# ── Plot 4: Confusion Matrix ─────────────────────────────────────────────────
cm=confusion_matrix(fl2,fp2); cm_n=cm.astype(float)/cm.sum(axis=1,keepdims=True)
fig,axes=plt.subplots(1,2,figsize=(22,9)); fig.patch.set_facecolor(DK)
fig.suptitle('Confusion Matrix – Deeper CNN (Final, Test Set)',color='white',fontsize=16)
for ax,data,fmt,cmap,title in zip(axes,[cm,cm_n],['d','.2f'],['YlOrRd','Blues'],['Absolute Counts','Normalized (Row %)']):
    ax.set_facecolor(DA)
    sns.heatmap(data,annot=True,fmt=fmt,cmap=cmap,xticklabels=DISPLAY_LABELS,yticklabels=DISPLAY_LABELS,linewidths=0.5,linecolor='#222',ax=ax,cbar_kws={'shrink':0.8})
    ax.set_title(title,color='white',fontsize=13,pad=8); ax.set_xlabel('Predicted',color='white',fontsize=11); ax.set_ylabel('True',color='white',fontsize=11)
    ax.tick_params(colors='white',labelsize=8); plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'confusion_matrix.png'),dpi=150,bbox_inches='tight',facecolor=DK); plt.close()
print('Saved: confusion_matrix.png',flush=True)

# ── Plot 5: Ablation Bar Chart ───────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
names=[r['name'] for r in results]; tes=[r['te'] for r in results]; f1s=[r['f1']*100 for r in results]
x=np.arange(len(names)); w=0.35
b1=ax.bar(x-w/2,tes,w,color='#4FC3F7',alpha=0.85,label='Test Acc (%)'); b2=ax.bar(x+w/2,f1s,w,color='#FFB74D',alpha=0.85,label='F1×100')
for b in b1: h=b.get_height(); ax.annotate(f'{h:.1f}%',xy=(b.get_x()+b.get_width()/2,h),xytext=(0,4),textcoords='offset points',ha='center',color='white',fontsize=8)
for b in b2: h=b.get_height(); ax.annotate(f'{h:.1f}',xy=(b.get_x()+b.get_width()/2,h),xytext=(0,4),textcoords='offset points',ha='center',color='white',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names,color='white',fontsize=9,rotation=15,ha='right')
ax.set_ylabel('Score',color='white',fontsize=12); ax.set_title('Ablation – Test Accuracy & F1 Score',color='white',fontsize=13)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333654'); ax.grid(axis='y',ls='--',alpha=0.25,color='#555')
ax.legend(facecolor=DA,edgecolor='#555',labelcolor='white',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR,'ablation_bar.png'),dpi=150,bbox_inches='tight',facecolor=DK); plt.close()
print('Saved: ablation_bar.png',flush=True)

print('\n=== CLASSIFICATION REPORT ===',flush=True)
print(classification_report(fl2,fp2,target_names=DISPLAY_LABELS,digits=4),flush=True)
print('ALL DONE. Output dir:',OUTPUT_DIR,flush=True)
