import torch
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange
import numpy as np

# =========================
# 3.10 加Noise utilities
# =========================
def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.01, enable: bool = False):
    """
    x: [B, C, T] or [B, T, C]
    sigma: 高斯噪声强度
    enable: 是否启用噪声
    """
    if (not enable) or sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise


def add_channel_dropout(
    x: torch.Tensor,
    drop_prob: float = 0.1,
    enable: bool = False,
    channel_dim: int = 1,
):
    """
    对整个通道置零，模拟坏导联/电极失活。

    x: [B, C, T] 或 [B, T, C]
    drop_prob: 通道失活概率
    enable: 是否启用
    channel_dim:
        - 如果 x 是 [B, C, T]，就传 1
        - 如果 x 是 [B, T, C]，就传 2
    """
    if (not enable) or drop_prob <= 0:
        return x

    shape = list(x.shape)
    mask_shape = [1] * len(shape)
    mask_shape[0] = shape[0]
    mask_shape[channel_dim] = shape[channel_dim]

    mask = (torch.rand(mask_shape, device=x.device) > drop_prob).float()
    return x * mask


def apply_noise(
    x: torch.Tensor,
    noise_type: str = None,
    noise_level: float = 0.0,
    enable_noise: bool = False,
    channel_dim: int = 1,
):
    """
    统一噪声入口。
    noise_type:
        - None
        - "gaussian"
        - "channel_dropout"
    """
    if (not enable_noise) or noise_type is None or noise_level <= 0:
        return x

    if noise_type == "gaussian":
        return add_gaussian_noise(x, sigma=noise_level, enable=True)

    if noise_type == "channel_dropout":
        return add_channel_dropout(
            x,
            drop_prob=noise_level,
            enable=True,
            channel_dim=channel_dim,
        )

    raise ValueError(f"Unsupported noise_type: {noise_type}")
# =========================
class EarlyStopping(object):
    """Stop training when loss does not decrease"""

    def __init__(self, patience: int, path_to_save: str):
        self._min_loss = float("inf")
        self._patience = patience
        self._path = path_to_save
        self.__check_point = None
        self.__counter = 0

    def should_stop(self, loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop and save the check point if needed.

        :param loss: Current validation loss.
        :param model: Model to save (it will compare the model with prior saved model and save if better).
        :param epoch: current epoch (will be used as check point if needed).
        :return: True if training should stop, False otherwise.
        """
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            self.__check_point = epoch
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter == self._patience:
                return True
        return False

    def load(self, weights_only=True):
        """Load best model weights"""
        return torch.load(self._path, weights_only=weights_only)

    @property
    def check_point(self):
        """Return check point index

        :return: check point index
        """
        if self.__check_point is None:
            raise ValueError("No check point is saved!")
        return self.__check_point

    @property
    def best_loss(self):
        return self._min_loss


class WarmupScheduler(object):
    """Warmup learning rate and dynamically adjusts learning rate based on validation loss.

    When the loss increases, the learning rate will be divided by decay_factor.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        min_lr: float = 1e-6,
        warmup_steps: int = 10,
        decay_factor: float = 0.1,
    ):
        """Initialize Warmup Scheduler.

        :param optimizer: Optimizer for training.
        :param lr: Learning rate.
        :param min_lr: Minimum learning rate.
        :param warmup_steps: Number of warmup steps.
        :param decay_factor: Factor to multiply learning rate when loss increases.
        """
        self.optimizer = optimizer
        self.initial_lr = lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor

        # If user set warmup_steps=0, then set warmup_steps=1 to avoid ZeroDivisionError.
        self.warmup_steps = max(warmup_steps, 1)

        assert (
            self.initial_lr >= self.min_lr
        ), f"Learning rate must be greater than min_lr({self.min_lr})"
        assert 0 < self.decay_factor < 1, "Decay factor must be less than 1.0."

        self.global_step = 1
        self.best_loss = float("inf")

        # Initialize learning rates
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr * (self.global_step / self.warmup_steps)

    def step(self, loss: float):
        """Update learning rate based on current loss.

        :param loss: Current validation loss.
        """
        self.global_step += 1

        if self.global_step <= self.warmup_steps:
            # Linear warmup
            warmup_lr = self.initial_lr * (self.global_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            # Check if loss increased
            if loss > self.best_loss:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group["lr"] * self.decay_factor, self.min_lr)
                    param_group["lr"] = new_lr
            self.best_loss = min(self.best_loss, loss)

    def get_lr(self):
        """Return current learning rates."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


# def validate(model, device, criterion, val_loader):
#     model.eval()
#     val_loss = 0

#     with torch.no_grad():
#         for data, label in val_loader:
#             data = data.to(device)
#             label = label.to(device)
#             output = model(data)

#             batch_loss = criterion(output, label)
#             val_loss += batch_loss.item()

#         return val_loss / len(val_loader)

# def validate(model, device, criterion, val_loader):  #3.9 validate函数替换版，加对比 替换上面那版
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for data, label in val_loader:
#             data = data.to(device)
#             label = label.to(device)

#             output = model(data)
#             batch_loss = criterion(output, label)
#             val_loss += batch_loss.item()

#             pred = torch.argmax(output, dim=1)
#             correct += (pred == label).sum().item()
#             total += label.size(0)

#     val_loss /= len(val_loader)
#     val_acc = correct / total if total > 0 else 0.0
#     return val_loss, val_acc

#3.10 加噪声功能
def validate(
    model,
    device,
    criterion,
    val_loader,
    enable_noise: bool = False,
    noise_type: str = None,
    noise_level: float = 0.0,
    channel_dim: int = 1,
):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            # ===== 在这里统一加噪声 =====
            data = apply_noise(
                data,
                noise_type=noise_type,
                noise_level=noise_level,
                enable_noise=enable_noise,
                channel_dim=channel_dim,
            )

            output = model(data)
            batch_loss = criterion(output, label)
            val_loss += batch_loss.item()

            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc

# def _train(
#     model: torch.nn.Module,
#     device: torch.device,
#     model_path: str,
#     optimizer: torch.optim.Optimizer,
#     criterion: torch.nn.Module,
#     epochs: int,
#     train_loader: torch.utils.data.DataLoader,
#     val_loader: torch.utils.data.DataLoader,
#     gradient_step: int = 1,
#     patience: int = 0,
#     enable_fp16: bool = False,
#     scheduler=None,
# ):
#     if scheduler is not None:
#         assert callable(
#             getattr(scheduler, "step", None)
#         ), "Scheduler must have a step() method."

#     epoch_trange = trange(1, epochs + 1)
#     early_stopper = EarlyStopping(patience, model_path)
#     scaler = GradScaler(device=str(device), enabled=enable_fp16)

#     model.to(device)
#     criterion.to(device)

#     model.zero_grad()

#     for epoch_idx in epoch_trange:
#         model.train()
#         train_loss = 0
#         for batch_id, (data, label) in enumerate(train_loader, start=1):
#             data = data.to(device)
#             label = label.to(device)

#             with autocast(
#                 device_type=str(device), enabled=enable_fp16, dtype=torch.float16
#             ):
#                 output = model(data)
#                 batch_loss = criterion(output, label)

#             train_loss += batch_loss.item()

#             # Scale loss to prevent under/overflow
#             scaler.scale(batch_loss / gradient_step).backward()

#             # Gradient Accumulation
#             if batch_id % gradient_step == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#         # Validate Training Epoch
#         train_loss /= len(train_loader)
#         val_loss = validate(model, torch.device("cuda"), criterion, val_loader)
#         tqdm.write(
#             f"Epoch {epoch_idx}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
#         )

#         # Early stopping
#         if early_stopper.should_stop(val_loss, model, epoch_idx):
#             break

#         # Learning Rate Scheduling
#         if scheduler is not None:
#             if isinstance(scheduler, WarmupScheduler):
#                 scheduler.step(val_loss)
#             else:
#                 scheduler.step(epoch_idx)

#     return early_stopper.check_point, early_stopper.best_loss

#3.9_train函数替换版，加对比，原版在上面
def _train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler=None,
    # ===== 3.10新增：训练阶段噪声 =====
    enable_train_noise: bool = False,
    train_noise_type: str = None,
    train_noise_level: float = 0.0,

    # ===== 3.10新增：验证阶段噪声 =====
    enable_val_noise: bool = False,
    val_noise_type: str = None,
    val_noise_level: float = 0.0,

    # 3.10新增：数据形状 [B, C, T] 时用 1
    channel_dim: int = 1,

):
    if scheduler is not None:
        assert callable(
            getattr(scheduler, "step", None)
        ), "Scheduler must have a step() method."

    epoch_trange = trange(1, epochs + 1)
    early_stopper = EarlyStopping(patience, model_path)
    scaler = GradScaler(device=str(device), enabled=enable_fp16)

    model.to(device)
    criterion.to(device)

    model.zero_grad()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch_idx in epoch_trange:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_id, (data, label) in enumerate(train_loader, start=1):
            data = data.to(device)
            label = label.to(device)

            # ===== 训练阶段是否加噪声 =====
            data = apply_noise(
                data,
                noise_type=train_noise_type,
                noise_level=train_noise_level,
                enable_noise=enable_train_noise,
                channel_dim=channel_dim,
            )


            with autocast(
                device_type=str(device), enabled=enable_fp16, dtype=torch.float16
            ):
                output = model(data)
                batch_loss = criterion(output, label)

            train_loss += batch_loss.item()

            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            scaler.scale(batch_loss / gradient_step).backward()

            if batch_id % gradient_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 处理最后一个不足 gradient_step 的累积梯度
        if len(train_loader) % gradient_step != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc = validate(
            model,
            device,
            criterion,
            val_loader,
            enable_noise=enable_val_noise,   #3.10加噪声
            noise_type=val_noise_type,       #3.10加噪声
            noise_level=val_noise_level,     #3.10加噪声
            channel_dim=channel_dim,         #3.10加噪声
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)

        tqdm.write(
            f"Epoch {epoch_idx:03d} | "
            f"Train-Loss: {train_loss:.5f} | Train-Acc: {train_acc:.4f} | "
            f"Val-Loss: {val_loss:.5f} | Val-Acc: {val_acc:.4f}"
        )

        if early_stopper.should_stop(val_loss, model, epoch_idx):
            break

        if scheduler is not None:
            if isinstance(scheduler, WarmupScheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch_idx)

    result = {
        "best_epoch": early_stopper.check_point,
        "best_val_loss": early_stopper.best_loss,
        "best_val_acc": best_val_acc,
        "history": history,
    }

    return result

# def train(
#     model: torch.nn.Module,
#     device: torch.device,
#     model_path: str,
#     optimizer: torch.optim.Optimizer,
#     criterion: torch.nn.Module,
#     epochs: int,
#     train_loader: torch.utils.data.DataLoader,
#     val_loader: torch.utils.data.DataLoader,
#     gradient_step: int = 1,
#     patience: int = 0,
#     enable_fp16: bool = False,
#     scheduler=None,

# ):
#     """Train the model and return the best check point.

#     Batch accumulation, Early stopping, Warmup scheduler,
#     and Learning rate scheduler are included.

#     :param model: Model to train.
#     :param model_path: Path to save the best model.
#     :param device: Torch device (cpu or cuda).
#     :param optimizer: Optimizer for training.
#     :param criterion: Loss function.
#     :param epochs: Maximum number of epochs.
#     :param train_loader: Training data loader.
#     :param val_loader: Validation data loader.
#     :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
#     :param patience: Number of epochs to wait before early stopping (default: 0).
#     :param enable_fp16: Enable FP16 precision training (default: False).
#     :param scheduler: Learning rate scheduler (default: None).
#     """
#     if enable_fp16:
#         assert torch.amp.autocast_mode.is_autocast_available(
#             str(device)
#         ), "Unable to use autocast on current device."

#     check_point, _ = _train(
#         model,
#         device,
#         model_path,
#         optimizer,
#         criterion,
#         epochs,
#         train_loader,
#         val_loader,
#         gradient_step,
#         patience,
#         enable_fp16,
#         scheduler,
#     )
#     return check_point

#3.10 train加噪声控制，原版在上
def train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler=None,
    enable_train_noise: bool = False,
    train_noise_type: str = None,
    train_noise_level: float = 0.0,
    enable_val_noise: bool = False,
    val_noise_type: str = None,
    val_noise_level: float = 0.0,
    channel_dim: int = 1,
):
    if enable_fp16:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device."

    result = _train(
        model,
        device,
        model_path,
        optimizer,
        criterion,
        epochs,
        train_loader,
        val_loader,
        gradient_step,
        patience,
        enable_fp16,
        scheduler,
        enable_train_noise,
        train_noise_type,
        train_noise_level,
        enable_val_noise,
        val_noise_type,
        val_noise_level,
        channel_dim,
    )
    return result

# def train_with_kfold(
#     k_folds: int,
#     model_class: torch.nn,
#     device: torch.device,
#     model_path: str,
#     optimizer_class: torch.optim,
#     criterion: torch.nn.Module,
#     epochs: int,
#     train_dataset: torch.utils.data.Dataset,
#     batch: int,
#     model_params: dict = None,
#     optimizer_params: dict = None,
#     gradient_step: int = 1,
#     patience: int = 0,
#     enable_fp16: bool = False,
#     scheduler_class=None,
#     scheduler_params: dict = None,
# ):
#     """Train the model and return the best check point.

#     Batch accumulation, Early stopping, Warmup scheduler,
#     and Learning rate scheduler are included.

#     :param k_folds: Number of folds for K-fold cross validation.
#     :param model_class: Model class to train.
#     :param model_params: Parameters for 'model_class'.
#     :param model_path: Path to save the best model.
#     :param device: Torch device (cpu or cuda).
#     :param optimizer_class: Optimizer class for training.
#     :param optimizer_params: Parameters for 'optimizer_class'. model.parameters will be called automatically.
#     :param criterion: Loss function.
#     :param epochs: Maximum number of epochs.
#     :param train_dataset: Training data set.
#     :param batch: Batch size for training.
#     :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
#     :param patience: Number of epochs to wait before early stopping (default: 0).
#     :param enable_fp16: Enable FP16 precision training (default: False).
#     :param scheduler_class: Learning rate scheduler class. optimizer will be called automatically. (default: None).
#     :param scheduler_params: Parameters for 'scheduler_class'.  (default: None).
#     """
#     if enable_fp16:
#         assert torch.amp.autocast_mode.is_autocast_available(
#             str(device)
#         ), "Unable to use autocast on current device."

#     kf = KFold(n_splits=k_folds, shuffle=True)
#     best_fold = 0
#     best_check_point = 0
#     best_val_loss = float("inf")
#     model_name, ext = model_path.rsplit(".", 1)

#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
#         torch.cuda.empty_cache()
#         print(f"\n===== Fold {fold} =====")

#         train_dataset_fold = Subset(train_dataset, train_idx)
#         val_dataset_fold = Subset(train_dataset, val_idx)
#         train_loader = DataLoader(train_dataset_fold, batch_size=batch, shuffle=True)
#         val_loader = DataLoader(val_dataset_fold, batch_size=batch, shuffle=False)
#         model_path = f"{model_name}_{fold}.{ext}"

#         if model_params is None:
#             model_params = {}
#         if optimizer_params is None:
#             optimizer_params = {}

#         model = model_class(**model_params)
#         optimizer = optimizer_class(model.parameters(), **optimizer_params)

#         if scheduler_class is not None:
#             if scheduler_params is None:
#                 scheduler_params = {}
#             scheduler = scheduler_class(optimizer, **scheduler_params)
#         else:
#             scheduler = None

#         check_point, val_loss = _train(
#             model,
#             device,
#             model_path,
#             optimizer,
#             criterion,
#             epochs,
#             train_loader,
#             val_loader,
#             gradient_step,
#             patience,
#             enable_fp16,
#             scheduler,
#         )

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_check_point = check_point
#             best_fold = fold

#     best_model_path = f"{model_name}_{best_fold}.{ext}"
#     return best_check_point, best_model_path


#3.9改 每个folds打印结果 原版在上
def train_with_kfold(
    k_folds: int,
    model_class: torch.nn,
    device: torch.device,
    model_path: str,
    optimizer_class: torch.optim,
    criterion: torch.nn.Module,
    epochs: int,
    train_dataset: torch.utils.data.Dataset,
    batch: int,
    model_params: dict = None,
    optimizer_params: dict = None,
    gradient_step: int = 1,
    patience: int = 0,
    enable_fp16: bool = False,
    scheduler_class=None,
    scheduler_params: dict = None,
    
    enable_train_noise: bool = False,    #3.10加噪声
    train_noise_type: str = None,#3.10加噪声
    train_noise_level: float = 0.0,#3.10加噪声

    enable_val_noise: bool = False,#3.10加噪声
    val_noise_type: str = None,#3.10加噪声
    val_noise_level: float = 0.0,#3.10加噪声

    channel_dim: int = 1,#3.10加噪声
):
    if enable_fp16:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device."

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_fold = 0
    best_result = None
    best_val_loss = float("inf")
    model_name, ext = model_path.rsplit(".", 1)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        torch.cuda.empty_cache()
        print(f"\n===== Fold {fold} =====")

        train_dataset_fold = Subset(train_dataset, train_idx)
        val_dataset_fold = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_dataset_fold, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch, shuffle=False)
        fold_model_path = f"{model_name}_{fold}.{ext}"

        current_model_params = {} if model_params is None else model_params.copy()
        current_optimizer_params = {} if optimizer_params is None else optimizer_params.copy()

        model = model_class(**current_model_params)
        optimizer = optimizer_class(model.parameters(), **current_optimizer_params)

        if scheduler_class is not None:
            current_scheduler_params = {} if scheduler_params is None else scheduler_params.copy()
            scheduler = scheduler_class(optimizer, **current_scheduler_params)
        else:
            scheduler = None

        result = _train(
            model,
            device,
            fold_model_path,
            optimizer,
            criterion,
            epochs,
            train_loader,
            val_loader,
            gradient_step,
            patience,
            enable_fp16,
            scheduler,
            enable_train_noise,#3.10加噪声
            train_noise_type,#3.10加噪声
            train_noise_level,#3.10加噪声
            enable_val_noise,#3.10加噪声
            val_noise_type,#3.10加噪声
            val_noise_level,#3.10加噪声
            channel_dim,#3.10加噪声
        )

        fold_info = {
            "fold": fold,
            "best_epoch": result["best_epoch"],
            "val_loss": result["best_val_loss"],
            "val_acc": result["best_val_acc"],
            "model_path": fold_model_path,
            "history": result["history"],
        }
        fold_results.append(fold_info)

        print(
            f"Fold {fold} | "
            f"best_epoch={fold_info['best_epoch']} | "
            f"val_loss={fold_info['val_loss']:.6f} | "
            f"val_acc={fold_info['val_acc']:.4f}"
        )

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_result = result
            best_fold = fold

    best_model_path = f"{model_name}_{best_fold}.{ext}"

    val_losses = [r["val_loss"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]

    print("\n===== Cross-validation summary =====")
    for r in fold_results:
        print(
            f"Fold {r['fold']}: "
            f"best_epoch={r['best_epoch']}, "
            f"val_loss={r['val_loss']:.6f}, "
            f"val_acc={r['val_acc']:.4f}"
        )

    print(f"Mean val_loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
    print(f"Mean val_acc : {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Best fold: {best_fold}")

    return best_result, best_model_path, fold_results