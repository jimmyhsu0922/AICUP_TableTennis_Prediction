import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import math

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

FEATURES = ["sex", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId", "strikeId", "strikeNumber"]
PAD_TOKEN = 0

class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            alpha_w = self.alpha[targets]
            loss = alpha_w * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiTaskLSTMTransformer(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=32, hidden=256, num_layers=1, nhead=8, dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        input_dim = len(num_tokens_per_feature) * emb_dim
        
        # Unidirectional LSTM for sequential modeling
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.pos_encoder = PositionalEncoding(hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden*2,
                                                   dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        
        # Multi-task heads
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head  = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        
        # 1. Process features using LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        
        # 2. Generate Causal Mask to prevent looking ahead into future timesteps
        sz = X.size(1)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=X.device), diagonal=1)
        
        # 3. Transformer Encoder processing with temporal masking
        o = self.pos_encoder(o)
        o = self.transformer(o, mask=mask)
        o = self.ln(o)
        o = self.drop(o)
        
        # Global pooling for rally-level binary prediction
        pad_mask = (X[:,:,0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = pad_mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * pad_mask).sum(dim=1) / denom
        
        return self.act_head(o), self.pt_head(o), self.rly_head(mean_hidden).squeeze(1)

def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out

def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out

def pad2d_cap(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m); out[:T]=a[:T]; return out, T

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(" Loading and preprocessing data...")
    
    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])
    
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)
    
    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}
    def encode_frame(df):
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)
    
    # Target shift: predict the next step target based on current feature input
    X_list, yA_list, yP_list, yR_list, L_list, match_list = [], [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: continue
        X = encode_frame(g)[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)
        X_list.append(X); yA_list.append(yA); yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(X))
        match_list.append(g["match"].iloc[0])
        
    MAXLEN = max(L_list)
    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)
    groups = np.array(match_list)
    
    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)
    
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setup complete. Device: {device}")
    
    # 5-Fold GroupKFold cross-validation split by match
    gkf = GroupKFold(n_splits=5)
    models = []
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, yR_all, groups=groups)):
        print(f"\n=== Start Training Fold {fold+1}/5 ===")
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
        yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
        yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
        L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]
        
        # Calculate inverse frequency class weights for Balanced Focal Loss
        act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
        pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
        act_w = torch.tensor(1.0 / np.sqrt(act_counts), dtype=torch.float32)
        act_w = (act_w * (n_act / act_w.sum())).to(device)
        pt_w  = torch.tensor(1.0 / np.sqrt(pt_counts), dtype=torch.float32)
        pt_w  = (pt_w * (n_pt / pt_w.sum())).to(device)
        
        train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
        val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2, 128), shuffle=False)
        
        model = MultiTaskLSTMTransformer(num_tokens_per_feature, n_act, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
        
        focal_action = BalancedFocalLoss(alpha=act_w, gamma=2.0, ignore_index=-1)
        focal_point  = BalancedFocalLoss(alpha=pt_w, gamma=2.0, ignore_index=-1)
        bce_rally    = nn.BCEWithLogitsLoss()
        
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        
        best_final_score = -1.0
        fold_ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold}.pth")
        
        for ep in range(1, args.epochs+1):
            model.train(); run_loss=0.0
            for Xb,yAb,yPb,yRb,Lb in train_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                opt.zero_grad(); la,lp,lr = model(Xb,Lb)
                
                loss = 1.0 * focal_action(la.view(-1,la.size(-1)), yAb.view(-1)) + \
                       1.0 * focal_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + \
                       0.1 * bce_rally(lr,yRb)
                       
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                run_loss += loss.item()*Xb.size(0)
                
            scheduler.step()
            
            # Validation phase
            model.eval(); val_loss=0.0
            allA,allAp,allP,allPp,allR,allRp=[],[],[],[],[],[]
            with torch.no_grad():
                for Xb,yAb,yPb,yRb,Lb in val_loader:
                    Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                    la,lp,lr = model(Xb,Lb)
                    
                    loss = 1.0 * focal_action(la.view(-1,la.size(-1)), yAb.view(-1)) + \
                           1.0 * focal_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + \
                           0.1 * bce_rally(lr,yRb)
                    val_loss += loss.item()*Xb.size(0)
                    
                    allR += yRb.detach().cpu().tolist()
                    allRp += torch.sigmoid(lr).detach().cpu().tolist()
                    
                    # Evaluate predictions on the last unpadded timestep of each sequence
                    for i in range(Xb.size(0)):
                        last_idx = Lb[i].item() - 1
                        if last_idx >= 0:
                            true_a = yAb[i, last_idx].item()
                            true_p = yPb[i, last_idx].item()
                            if true_a != -1:
                                allA.append(true_a)
                                allAp.append(la[i, last_idx].argmax(-1).item())
                            if true_p != -1:
                                allP.append(true_p)
                                allPp.append(lp[i, last_idx].argmax(-1).item())
                                
            tr_loss = run_loss/len(train_loader.dataset); va_loss=val_loss/len(val_loader.dataset)
            try:
                f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
                f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
                auc = roc_auc_score(allR, allRp) if len(set(allR))>1 else 0.5
            except Exception: f1A,f1P,auc=0.0,0.0,0.5
            
            final = 0.4*f1A + 0.4*f1P + 0.2*auc
            if final > best_final_score:
                best_final_score = final
                torch.save(model.state_dict(), fold_ckpt_path)
                
            print(f"[Fold {fold+1} Epoch {ep}/{args.epochs}] val_loss={va_loss:.4f} F1_A={f1A:.4f} F1_P={f1P:.4f} AUC={auc:.4f} Final={final:.4f} (Best={best_final_score:.4f})")
            
        best_model = MultiTaskLSTMTransformer(num_tokens_per_feature, n_act, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
        best_model.load_state_dict(torch.load(fold_ckpt_path))
        best_model.eval()
        models.append(best_model)

    print("\n=== Inference: Ensemble Out-of-Fold Prediction ===")
    pred_rows = []
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            if len(g) == 0: continue
            Xg = encode_frame(g); Xp, T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            last_t = L_t.item() - 1
            
            sum_la_prob = np.zeros(n_act)
            sum_lp_prob = np.zeros(n_pt)
            sum_lr_prob = 0.0
            
            # Blend raw probability distributions across folds
            for model in models:
                la, lp, lr = model(X_t, L_t)
                sum_la_prob += torch.softmax(la[0, last_t], dim=-1).cpu().numpy()
                sum_lp_prob += torch.softmax(lp[0, last_t], dim=-1).cpu().numpy()
                sum_lr_prob += torch.sigmoid(lr).item()
                
            avg_la_prob = sum_la_prob / len(models)
            avg_lp_prob = sum_lp_prob / len(models)
            avg_lr_prob = sum_lr_prob / len(models)
            
            action_pred = int(act_classes[np.argmax(avg_la_prob)])
            point_pred = int(pt_classes[np.argmax(avg_lp_prob)])
            
            pred_rows.append({
                "rally_uid": int(rid),
                "actionId": action_pred,
                "pointId": point_pred,
                "serverGetPoint": float(avg_lr_prob)
            })

    print("\n=== Saving Predictions ===")
    output_df = pd.DataFrame(pred_rows)
    output_df = output_df[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out_path = args.out
    output_df.to_csv(out_path, index=False)
    print(f"\Completed! Predictions saved to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/TableTennis_ML/data/train.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/TableTennis_ML/data/test_new.csv")
    ap.add_argument("--out", default="/content/drive/MyDrive/TableTennis_ML/submissions/submission_hybrid_v4_2.csv")
    ap.add_argument("--checkpoint_dir", default="/content/checkpoints")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args([])
    main(args)