"""
train.py — Training entry point for POLY-SIM multimodal speaker identification.

All configuration is read from the .env file in the project root.
Edit .env to change models, hyperparameters, or paths — no CLI flags needed.

    python scripts/train.py

Loss weights
────────────
  L_total = L_CE  +  LAMBDA_CON * L_SupCon  +  LAMBDA_ORTH * L_Ortho

Checkpoints are saved to <OUTPUT_DIR>/best.pt and last.pt.
"""

from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# Allow running from repo root: python scripts/train.py
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MAVCelebDataset
from losses import OrthogonalityLoss, SupConLoss
from model import AudioEncoder, FaceEncoder, FusionModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Config — loaded entirely from .env
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> types.SimpleNamespace:
    """Parse .env and return a SimpleNamespace with typed config values."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        raise FileNotFoundError(
            f".env file not found at {env_file}. "
            "Copy .env.example to .env and set your values."
        )

    env: dict[str, str] = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()

    def _get(key: str, default: str) -> str:
        return env.get(key, os.environ.get(key, default))

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"

    return types.SimpleNamespace(
        # Data
        data_root           = _get("DATA_ROOT",              "Data"),
        train_split_csv     = _get("TRAIN_SPLIT_CSV",        "").strip(),
        val_split_csv       = _get("VAL_SPLIT_CSV",          "").strip(),
        k_faces             = int(_get("K_FACES",            "4")),
        max_audio_sec       = float(_get("MAX_AUDIO_SEC",    "6.0")),
        # Model
        face_encoder        = _get("FACE_ENCODER",           "vit_base"),
        audio_encoder       = _get("AUDIO_ENCODER",          "wavlm_base"),
        embed_dim           = int(_get("EMBED_DIM",          "256")),
        mask_prob           = float(_get("MASK_PROB",        "0.3")),
        freeze_encoders_epochs = int(_get("FREEZE_ENCODERS_EPOCHS", "5")),
        # Training
        epochs              = int(_get("EPOCHS",             "50")),
        batch_size          = int(_get("BATCH_SIZE",         "32")),
        lr                  = float(_get("LR",               "1e-4")),
        weight_decay        = float(_get("WEIGHT_DECAY",     "1e-4")),
        lambda_con          = float(_get("LAMBDA_CON",       "0.5")),
        lambda_orth         = float(_get("LAMBDA_ORTH",      "0.1")),
        num_workers         = int(_get("NUM_WORKERS",        "4")),
        # Runtime
        device              = _get("DEVICE",                 auto_device),
        seed                = int(_get("SEED",               "42")),
        # Checkpoints
        output_dir          = _get("OUTPUT_DIR",             "checkpoints"),
        checkpoint_interval = int(_get("CHECKPOINT_INTERVAL", "10")),
        resume_from         = _get("RESUME_FROM",             ""),
        # Memory optimisation
        use_amp             = _get("USE_AMP",           "true").lower()  == "true",
        grad_checkpoint     = _get("GRAD_CHECKPOINT",   "true").lower()  == "true",
        grad_accum_steps    = int(_get("GRAD_ACCUM_STEPS", "1")),
        cpu_offload_optim   = _get("CPU_OFFLOAD_OPTIM", "false").lower() == "true",
        # Validation & logging
        val_split           = float(_get("VAL_SPLIT",   "0.1")),
        tensorboard_dir     = _get("TENSORBOARD_DIR",   "runs"),
        tb_log_every        = int(_get("TB_LOG_EVERY",  "10")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_encoders_frozen(model: FusionModel, frozen: bool) -> None:
    for p in model.face_encoder.backbone.parameters():
        p.requires_grad = not frozen
    for p in model.audio_encoder.backbone.parameters():
        p.requires_grad = not frozen


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_acc: float,
    cfg: types.SimpleNamespace,
) -> None:
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc":  best_acc,
            "cfg":       vars(cfg),
        },
        path,
    )


def make_face_transform() -> transforms.Compose:
    """Standard ImageNet transform used by all HuggingFace vision backbones."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_val_face_transform() -> transforms.Compose:
    """Deterministic transform for validation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CPU optimizer state offload
# ─────────────────────────────────────────────────────────────────────────────

def _offload_optim_state_to_cpu(optimizer: torch.optim.Optimizer) -> None:
    """Move Adam m/v state tensors from GPU to CPU RAM after the update step.
    Frees ~2× model-size GPU memory at the cost of H2D/D2H overhead.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                for k, v in optimizer.state[p].items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        optimizer.state[p][k] = v.cpu()


def _restore_optim_state_to_gpu(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move Adam m/v state tensors back to GPU before the weight-update step."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                for k, v in optimizer.state[p].items():
                    if isinstance(v, torch.Tensor) and not v.is_cuda:
                        optimizer.state[p][k] = v.to(device, non_blocking=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: FusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_fn: nn.CrossEntropyLoss,
    con_fn: SupConLoss,
    orth_fn: OrthogonalityLoss,
    cfg: types.SimpleNamespace,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    tb_writer: SummaryWriter | None = None,
) -> dict[str, float]:
    model.train()

    total_loss = total_ce = total_con = total_orth = 0.0
    correct = total = 0
    grad_accum = max(1, cfg.grad_accum_steps)

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:3d}/{cfg.epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    optimizer.zero_grad()

    for step, (face_frames, waveforms, labels) in enumerate(bar, 1):
        face_frames = face_frames.to(device, non_blocking=True)
        waveforms   = waveforms.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            embeddings, logits = model(face_frames, waveforms)

            l_ce   = ce_fn(logits, labels)
            l_con  = con_fn(embeddings, labels)
            l_orth = orth_fn(embeddings, labels)

            loss = (l_ce + cfg.lambda_con * l_con + cfg.lambda_orth * l_orth) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_update_step = (step % grad_accum == 0) or (step == len(loader))
        if is_update_step:
            if cfg.cpu_offload_optim and device.type == "cuda":
                _restore_optim_state_to_gpu(optimizer, device)
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if cfg.cpu_offload_optim and device.type == "cuda":
                _offload_optim_state_to_cpu(optimizer)

        total_loss  += loss.item() * grad_accum
        total_ce    += l_ce.item()
        total_con   += l_con.item()
        total_orth  += l_orth.item()
        batch_correct = (logits.argmax(1) == labels).sum().item()
        correct     += batch_correct
        total       += labels.shape[0]

        # Update bar postfix with running averages
        bar.set_postfix(
            loss=f"{total_loss/step:.4f}",
            ce=f"{total_ce/step:.4f}",
            con=f"{total_con/step:.4f}",
            acc=f"{correct/total*100:.1f}%",
        )

        if tb_writer is not None and cfg.tb_log_every > 0:
            if step % cfg.tb_log_every == 0 or step == 1 or step == len(loader):
                global_step = (epoch - 1) * len(loader) + step
                tb_writer.add_scalar("BatchLoss/train", loss.item() * grad_accum, global_step)
                tb_writer.add_scalar("BatchCE/train", l_ce.item(), global_step)
                tb_writer.add_scalar("BatchConLoss/train", l_con.item(), global_step)
                tb_writer.add_scalar("BatchOrthLoss/train", l_orth.item(), global_step)
                tb_writer.add_scalar("BatchAccuracy/train", batch_correct / labels.shape[0] * 100, global_step)
                tb_writer.flush()

    bar.close()
    n = len(loader)
    return {
        "loss":  total_loss  / n,
        "ce":    total_ce    / n,
        "con":   total_con   / n,
        "orth":  total_orth  / n,
        "acc":   correct / total * 100,
    }


@torch.no_grad()
def eval_one_epoch(
    model: FusionModel,
    loader: DataLoader,
    ce_fn: nn.CrossEntropyLoss,
    con_fn: SupConLoss,
    orth_fn: OrthogonalityLoss,
    cfg: types.SimpleNamespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = total_ce = total_con = total_orth = 0.0
    correct = total = 0

    for face_frames, waveforms, labels in loader:
        face_frames = face_frames.to(device, non_blocking=True)
        waveforms   = waveforms.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        embeddings, logits = model(face_frames, waveforms)

        l_ce   = ce_fn(logits, labels)
        l_con  = con_fn(embeddings, labels)
        l_orth = orth_fn(embeddings, labels)
        loss   = l_ce + cfg.lambda_con * l_con + cfg.lambda_orth * l_orth

        total_loss  += loss.item()
        total_ce    += l_ce.item()
        total_con   += l_con.item()
        total_orth  += l_orth.item()
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += labels.shape[0]

    n = max(len(loader), 1)
    return {
        "loss":  total_loss  / n,
        "ce":    total_ce    / n,
        "con":   total_con   / n,
        "orth":  total_orth  / n,
        "acc":   correct / total * 100 if total > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Startup banner ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(" POLY-SIM  |  Multimodal Speaker Identification")
    print("═" * 60)
    print(f"  Resume from    : {cfg.resume_from or 'scratch'}")
    print(f"  Ckpt interval  : every {cfg.checkpoint_interval} epochs")
    print(f"  Face encoder   : {cfg.face_encoder}")
    print(f"  Audio encoder  : {cfg.audio_encoder}")
    print(f"  Embed dim      : {cfg.embed_dim}")
    print(f"  Mask prob      : {cfg.mask_prob}  (modality dropout)")
    print(f"  Epochs         : {cfg.epochs}  |  Batch size: {cfg.batch_size}")
    print(f"  LR             : {cfg.lr}  |  WD: {cfg.weight_decay}")
    print(f"  λ_con={cfg.lambda_con}  λ_orth={cfg.lambda_orth}")
    print(f"  Device         : {cfg.device}")
    print(f"  Train split CSV: {cfg.train_split_csv or 'off'}")
    print(f"  Val split CSV  : {cfg.val_split_csv or 'off'}")
    print(f"  Output dir     : {cfg.output_dir}")
    amp_label = "fp16" if cfg.use_amp and device.type == "cuda" else "off"
    print(
        f"  AMP: {amp_label}  |  Grad ckpt: {cfg.grad_checkpoint}  "
        f"|  Accum steps: {cfg.grad_accum_steps}  "
        f"|  CPU optim offload: {cfg.cpu_offload_optim}"
    )
    print(f"  Val split      : {cfg.val_split * 100:.0f}%  |  TensorBoard: {cfg.tensorboard_dir}")
    print("═" * 60 + "\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_tf = make_face_transform()
    val_tf   = make_val_face_transform()
    print("Loading dataset …")
    if cfg.train_split_csv:
        dataset = MAVCelebDataset(
            data_root=cfg.data_root,
            k_faces=cfg.k_faces,
            max_audio_sec=cfg.max_audio_sec,
            face_transform=train_tf,
            manifest_csv=cfg.train_split_csv,
        )
        train_weights = dataset.get_sample_weights()
        print(f"  Train (CSV): {len(dataset):,} samples · {dataset.num_speakers} speakers")
        if cfg.val_split_csv:
            val_dataset = MAVCelebDataset(
                data_root=cfg.data_root,
                k_faces=cfg.k_faces,
                max_audio_sec=cfg.max_audio_sec,
                face_transform=val_tf,
                manifest_csv=cfg.val_split_csv,
            )
            print(f"  Val   (CSV): {len(val_dataset):,} samples")
        else:
            val_dataset = None
            print("  No validation split CSV provided (VAL_SPLIT ignored)")
    else:
        full_dataset = MAVCelebDataset(
            data_root=cfg.data_root,
            k_faces=cfg.k_faces,
            max_audio_sec=cfg.max_audio_sec,
            face_transform=train_tf,
        )
        print(f"  {len(full_dataset):,} samples · {full_dataset.num_speakers} speakers")

        # ── Train / val split ─────────────────────────────────────────────────
        n_val = int(len(full_dataset) * cfg.val_split) if cfg.val_split > 0 else 0
        if n_val > 0:
            all_indices = list(range(len(full_dataset)))
            rng_split = random.Random(cfg.seed)
            rng_split.shuffle(all_indices)
            val_indices   = all_indices[:n_val]
            train_indices = all_indices[n_val:]

            # Build a separate val dataset (shadow copy with deterministic transform)
            val_dataset = MAVCelebDataset.__new__(MAVCelebDataset)
            val_dataset.data_root         = full_dataset.data_root
            val_dataset.k_faces           = full_dataset.k_faces
            val_dataset.max_audio_samples = full_dataset.max_audio_samples
            val_dataset.sample_rate       = full_dataset.sample_rate
            val_dataset.face_transform    = val_tf
            val_dataset.speakers          = full_dataset.speakers
            val_dataset.speaker_to_idx    = full_dataset.speaker_to_idx
            val_dataset._speaker_counts   = full_dataset._speaker_counts
            val_dataset.samples           = [full_dataset.samples[i] for i in val_indices]

            # Collect weights before trimming samples list
            all_weights   = full_dataset.get_sample_weights()
            train_weights = [all_weights[i] for i in train_indices]
            full_dataset.samples = [full_dataset.samples[i] for i in train_indices]
            dataset = full_dataset
            print(f"  Train: {len(train_indices):,} · Val: {n_val:,}")
        else:
            val_dataset   = None
            train_weights = full_dataset.get_sample_weights()
            dataset       = full_dataset
            print("  No validation split (VAL_SPLIT=0)")

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,   # avoids single-sample batches at epoch end
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
            drop_last=False,
        )
        if val_dataset is not None else None
    )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_writer = None
    if SummaryWriter is not None:
        tb_dir = Path(cfg.tensorboard_dir)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  TensorBoard logs → {tb_dir}")
        print(f"    Run: tensorboard --logdir {cfg.tensorboard_dir}")
    else:
        print("  TensorBoard: not available (pip install tensorboard)")

    # ── Model ─────────────────────────────────────────────────────────────────
    face_enc  = FaceEncoder(backbone=cfg.face_encoder,  embed_dim=cfg.embed_dim)
    audio_enc = AudioEncoder(backbone=cfg.audio_encoder, embed_dim=cfg.embed_dim)
    model = FusionModel(
        face_encoder=face_enc,
        audio_encoder=audio_enc,
        num_speakers=dataset.num_speakers,
        embed_dim=cfg.embed_dim,
        mask_prob=cfg.mask_prob,
    ).to(device)

    if cfg.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("  Gradient checkpointing : enabled")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    ckpt = None
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).parent.parent / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"  Resuming from  : {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}  (best acc so far: {best_acc:.2f}%)")
        print()

    # Freeze backbones only if we are still inside warmup after resuming.
    in_warmup = cfg.freeze_encoders_epochs > 0 and start_epoch <= cfg.freeze_encoders_epochs
    if in_warmup:
        set_encoders_frozen(model, frozen=True)
        print(f"  Backbones frozen for first {cfg.freeze_encoders_epochs} warmup epochs.")
    else:
        set_encoders_frozen(model, frozen=False)
        if cfg.freeze_encoders_epochs > 0 and start_epoch > 1:
            print("  Resuming after warmup; backbones left unfrozen.")

    print(f"  Trainable parameters : {count_trainable(model):,}")
    print(f"  Total parameters     : {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    if in_warmup:
        optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_lr = cfg.lr
        scheduler_t_max = cfg.epochs
    else:
        optimizer_params = model.parameters()
        optimizer_lr = cfg.lr if start_epoch == 1 else cfg.lr * 0.1
        scheduler_t_max = max(1, cfg.epochs - cfg.freeze_encoders_epochs)

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=optimizer_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp and device.type == "cuda" else None

    if ckpt is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError as exc:
            print(
                "  Warning: checkpoint optimizer state did not match the current "
                f"training phase ({exc}). Continuing with a fresh optimizer."
            )

    # ── Loss functions ────────────────────────────────────────────────────────
    ce_fn   = nn.CrossEntropyLoss().to(device)
    con_fn  = SupConLoss(temperature=0.07).to(device)
    orth_fn = OrthogonalityLoss().to(device)

    # ── Training ──────────────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(start_epoch, cfg.epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:

        # Unfreeze encoders after warmup
        if epoch == cfg.freeze_encoders_epochs + 1:
            set_encoders_frozen(model, frozen=False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr * 0.1,
                weight_decay=cfg.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - epoch
            )
            # scaler = torch.cuda.amp.GradScaler() if cfg.use_amp and device.type == "cuda" else None
            scaler = torch.amp.GradScaler('cuda') if cfg.use_amp and device.type == "cuda" else None
            tqdm.write(f"\n  ► Epoch {epoch}: backbones unfrozen — LR reset to {cfg.lr * 0.1:.2e}")

        metrics = train_one_epoch(
            model, loader, optimizer, ce_fn, con_fn, orth_fn, cfg, device, epoch,
            scaler=scaler, tb_writer=tb_writer,
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics: dict[str, float] | None = None
        if val_loader is not None:
            val_metrics = eval_one_epoch(
                model, val_loader, ce_fn, con_fn, orth_fn, cfg, device
            )

        # ── TensorBoard logging ───────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train",     metrics["loss"], epoch)
            tb_writer.add_scalar("CE/train",       metrics["ce"],   epoch)
            tb_writer.add_scalar("ConLoss/train",  metrics["con"],  epoch)
            tb_writer.add_scalar("OrthLoss/train", metrics["orth"], epoch)
            tb_writer.add_scalar("Accuracy/train", metrics["acc"],  epoch)
            if val_metrics is not None:
                tb_writer.add_scalar("Loss/val",     val_metrics["loss"], epoch)
                tb_writer.add_scalar("CE/val",       val_metrics["ce"],   epoch)
                tb_writer.add_scalar("ConLoss/val",  val_metrics["con"],  epoch)
                tb_writer.add_scalar("OrthLoss/val", val_metrics["orth"], epoch)
                tb_writer.add_scalar("Accuracy/val", val_metrics["acc"],  epoch)
            tb_writer.add_scalar("LearningRate", lr_now, epoch)

        compare_acc = val_metrics["acc"] if val_metrics is not None else metrics["acc"]
        is_best = compare_acc > best_acc
        marker  = " ★ best" if is_best else ""
        val_line = ""
        if val_metrics is not None:
            val_line = (
                f"  | val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['acc']:.2f}%"
            )
        tqdm.write(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={metrics['loss']:.4f}  ce={metrics['ce']:.4f}  "
            f"con={metrics['con']:.4f}  orth={metrics['orth']:.4f}  "
            f"acc={metrics['acc']:.2f}%  lr={lr_now:.2e}"
            f"{val_line}{marker}"
        )
        epoch_bar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            acc=f"{metrics['acc']:.2f}%",
            val=f"{val_metrics['acc']:.2f}%" if val_metrics else "—",
            best=f"{best_acc:.2f}%",
        )

        # Save best checkpoint
        if is_best:
            best_acc = compare_acc
            save_checkpoint(
                output_dir / "best.pt",
                epoch, model, optimizer, scheduler, best_acc, cfg,
            )
            tqdm.write(f"  ✔ best checkpoint saved → {output_dir}/best.pt")

        # Periodic checkpoint every N epochs
        if cfg.checkpoint_interval > 0 and epoch % cfg.checkpoint_interval == 0:
            periodic_path = output_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                periodic_path,
                epoch, model, optimizer, scheduler, best_acc, cfg,
            )
            tqdm.write(f"  ✔ periodic checkpoint saved → {periodic_path.name}")

    # Save final checkpoint
    save_checkpoint(
        output_dir / "last.pt",
        cfg.epochs, model, optimizer, scheduler, best_acc, cfg,
    )
    if tb_writer is not None:
        tb_writer.close()

    print("\n" + "═" * 60)
    print(f"  Training complete")
    print(f"  Best {'val' if val_loader else 'train'} accuracy : {best_acc:.2f}%")
    print(f"  Checkpoints    : {output_dir}/best.pt  |  last.pt")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
