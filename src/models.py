"""The three forecasting models, plus a benchmark.

Design contract
---------------
Every model exposes the same three methods so the experiment runner is model
agnostic and the timing statistics are measured identically:

    fit(train, val)                 -> None
    predict_rolling(history, test)  -> np.ndarray of len(test)
    describe()                      -> dict of the hyper-parameters actually used

``predict_rolling`` implements genuine **one-step-ahead** forecasting: to
predict x(t+1) the model is given the *observed* history up to and including
t. Predictions are never fed back as inputs, so errors do not compound - this
matches the task definition and is what makes the three models comparable.

Model choice
------------
The three models are deliberately from different families so the comparison is
informative rather than three flavours of the same inductive bias:

* **SARIMA** - linear, explicitly seasonal, no learned representation. It is
  the reference point the telecom-forecasting literature has used for two
  decades, and it directly encodes the daily period the ACF exposes.
* **LSTM** - recurrent, gated, learns nonlinear state from a window of lags.
  Capable of representing the burst/decay behaviour a linear model cannot.
* **TCN** - dilated causal convolutions. Same receptive field as the LSTM but a
  fundamentally different way of reaching back in time (parallel, fixed
  receptive field, no recurrent state), which is the point of including it.

A seasonal-naive benchmark is also reported. On strongly periodic series it is
a surprisingly hard baseline, and a model that cannot beat it has not earned
its complexity.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

try:  # torch is only needed for the two neural models
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Input representation
# --------------------------------------------------------------------------
@dataclass
class LogStandardScaler:
    """log1p followed by z-scoring, with statistics fitted on training data only.

    Internet activity is strongly right-skewed and non-negative, with occasional
    spikes an order of magnitude above the median. Training a squared-error
    model on raw values lets those spikes dominate the gradient; log1p
    compresses them while remaining exactly invertible (expm1) and defined at
    zero, which matters because low-traffic areas contain many exact zeros.

    Fitting the mean/std on the training split only is what keeps the test week
    genuinely held out - scaling on the full series would leak its statistics.
    """

    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, values: np.ndarray) -> "LogStandardScaler":
        z = np.log1p(np.asarray(values, dtype=np.float64))
        self.mean_ = float(z.mean())
        self.std_ = float(z.std()) or 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.log1p(np.asarray(values, dtype=np.float64)) - self.mean_) / self.std_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        out = np.expm1(np.asarray(values, dtype=np.float64) * self.std_ + self.mean_)
        return np.clip(out, 0.0, None)  # traffic cannot be negative


def make_windows(values: np.ndarray, seq_len: int, horizon: int = 1):
    """Sliding windows: X[i] = values[i:i+seq_len], y[i] = values[i+seq_len+horizon-1]."""
    values = np.asarray(values, dtype=np.float32)
    n = len(values) - seq_len - horizon + 1
    if n <= 0:
        raise ValueError(f"series too short ({len(values)}) for seq_len={seq_len}")
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    X = values[idx]
    y = values[np.arange(n) + seq_len + horizon - 1]
    return X, y


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # pragma: no cover
        pass
    if TORCH_AVAILABLE:
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps and mps.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif info["mps_available"]:
            # Apple Silicon does not expose a device name; the platform string
            # already identifies the chip, and RAM is unified with the CPU.
            info["gpu"] = "Apple Silicon GPU (MPS)"
    return info


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
class SeasonalNaive:
    """x_hat(t+1) = x(t+1-144): yesterday's value at the same time of day."""

    name = "SeasonalNaive"

    def __init__(self, period: int = C.SLOTS_PER_DAY):
        self.period = period
        self.train_seconds = 0.0

    def fit(self, train: pd.Series, val: pd.Series) -> "SeasonalNaive":
        self.train_seconds = 0.0
        return self

    def predict_rolling(self, history: pd.Series, test: pd.Series) -> np.ndarray:
        full = np.concatenate([history.to_numpy(), test.to_numpy()])
        start = len(history)
        return np.array([full[start + i - self.period] for i in range(len(test))])

    def describe(self) -> dict:
        return {"model": self.name, "period": self.period}


# --------------------------------------------------------------------------
# Model 1: SARIMA
# --------------------------------------------------------------------------
class SarimaForecaster:
    """SARIMA(p,d,q)(P,D,Q)_s with s = 144 (one day).

    A full seasonal ARMA at s=144 is not tractable with statsmodels' state-space
    representation (the state vector grows with s, and each iteration of the
    Kalman filter becomes O(s^2)). The standard practical route, and the one
    taken here, is to let the seasonal part be a pure seasonal *difference*,
    ``seasonal_order=(0, 1, 0, 144)``, and let a low-order non-seasonal ARMA
    model what remains. This removes the daily cycle exactly while keeping the
    model estimable in seconds rather than hours.

    Rolling forecasts use ``append(..., refit=False)``: the fitted parameters are
    held fixed and the Kalman filter is advanced through the test week one
    observation at a time. Each prediction therefore conditions on the true
    history and on parameters estimated only from the training data.
    """

    name = "SARIMA"

    def __init__(self, order=(2, 0, 1), seasonal_order=(0, 1, 0, C.SLOTS_PER_DAY),
                 use_log: bool = True, trend: str | None = None):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.use_log = use_log
        self.trend = trend
        self.res_ = None
        self.train_seconds = 0.0
        self.aic_ = None

    def _fwd(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(x, 0, None)) if self.use_log else np.asarray(x, float)

    def _inv(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(x), 0, None) if self.use_log else np.clip(x, 0, None)

    def fit(self, train: pd.Series, val: pd.Series) -> "SarimaForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Validation data is legitimate fitting material for SARIMA: it has no
        # early stopping, so it is trained on train+val to match the amount of
        # history the neural models effectively see.
        series = pd.concat([train, val])
        y = self._fwd(series.to_numpy())

        # The seasonal difference is applied here rather than inside SARIMAX.
        # Passing seasonal_order=(0,1,0,144) to the state-space implementation
        # forces a state vector of length ~s, making the Kalman filter both slow
        # and memory-hungry (it OOMs on a 4 GB machine). Differencing by hand is
        # numerically identical for D=1 with no seasonal AR/MA terms, and the
        # inverse is a simple additive carry-back, so nothing is approximated.
        self.s_ = int(self.seasonal_order[3])
        self.D_ = int(self.seasonal_order[1])
        z = self._seasonal_diff(y)

        t0 = time.perf_counter()
        model = SARIMAX(
            z,
            order=self.order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.res_ = model.fit(disp=False)
        self.train_seconds = time.perf_counter() - t0
        self.aic_ = float(self.res_.aic)
        return self

    def _seasonal_diff(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y, dtype=float)
        for _ in range(self.D_):
            z = z[self.s_:] - z[:-self.s_]
        return z

    def predict_rolling(self, history: pd.Series, test: pd.Series) -> np.ndarray:
        if self.res_ is None:
            raise RuntimeError("call fit() first")
        if self.D_ != 1:
            raise NotImplementedError("rolling inversion implemented for D=1 only")

        s = self.s_
        y_hist = self._fwd(history.to_numpy())
        y_test = self._fwd(test.to_numpy())
        full = np.concatenate([y_hist, y_test])
        start = len(y_hist)

        # z(t) = y(t) - y(t-s). Every term on the right is an observation that is
        # already available when predicting step t, so this stays a genuine
        # one-step-ahead setup.
        lagged = full[start - s: start + len(y_test) - s]
        z_test = y_test - lagged

        # Advance the filter through the test period with parameters frozen and
        # read off the one-step-ahead predictions it produced along the way.
        extended = self.res_.append(z_test, refit=False)
        z_pred = np.asarray(
            extended.get_prediction(start=extended.nobs - len(z_test),
                                    end=extended.nobs - 1).predicted_mean
        )
        return self._inv(z_pred + lagged)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "log_transform": self.use_log,
            "aic": self.aic_,
            "n_params": int(len(self.res_.params)) if self.res_ is not None else None,
        }


# --------------------------------------------------------------------------
# Neural models
# --------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):                 # x: (B, L)
            out, _ = self.lstm(x.unsqueeze(-1))   # (B, L, H)
            return self.head(out[:, -1]).squeeze(-1)  # (B,)

    class _Chomp(nn.Module):
        """Trim the right padding so the convolution stays causal."""

        def __init__(self, size: int):
            super().__init__()
            self.size = size

        def forward(self, x):
            return x[:, :, : -self.size] if self.size > 0 else x

    class _TCNBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float):
            super().__init__()
            pad = (k - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
                _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
                _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            )
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
            self.relu = nn.ReLU()

        def forward(self, x):
            res = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + res)

    class _TCNNet(nn.Module):
        def __init__(self, channels: int, levels: int, kernel: int, dropout: float):
            super().__init__()
            blocks, c_in = [], 1
            for i in range(levels):
                blocks.append(_TCNBlock(c_in, channels, kernel, 2 ** i, dropout))
                c_in = channels
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Linear(channels, 1)

        def forward(self, x):                    # x: (B, L)
            h = self.tcn(x.unsqueeze(1))         # (B, C, L)
            return self.head(h[:, :, -1]).squeeze(-1)


@dataclass
class NeuralConfig:
    seq_len: int = C.SEQ_LEN
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = C.RANDOM_SEED
    # LSTM
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    # TCN
    channels: int = 32
    levels: int = 5          # receptive field = 1 + 2*(k-1)*(2^levels - 1)
    kernel: int = 3
    extra: dict = field(default_factory=dict)


def _default_device() -> "torch.device":
    """Pick the best available backend.

    Apple Silicon exposes its GPU through the MPS backend, not CUDA, so a plain
    ``cuda.is_available()`` check silently falls back to CPU on a Mac.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _NeuralForecaster:
    """Shared training / rolling-prediction logic for the LSTM and the TCN."""

    name = "Neural"

    def __init__(self, cfg: NeuralConfig | None = None, device: str | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the neural models; pip install torch")
        self.cfg = cfg or NeuralConfig()
        self.device = torch.device(device) if device else _default_device()
        self.scaler = LogStandardScaler()
        self.net: "nn.Module | None" = None
        self.train_seconds = 0.0
        self.history_: list[dict] = []
        self.best_val_ = np.inf
        self.epochs_run_ = 0

    def _build(self) -> "nn.Module":  # pragma: no cover - overridden
        raise NotImplementedError

    def _loaders(self, train: pd.Series, val: pd.Series):
        L = self.cfg.seq_len
        self.scaler.fit(train.to_numpy())

        Xtr, ytr = make_windows(self.scaler.transform(train.to_numpy()), L)
        # Validation windows need the tail of the training series as context, so
        # the last L points of train are prepended. This is context, not label
        # leakage: no validation target ever appears in a training window.
        val_ctx = np.concatenate([train.to_numpy()[-L:], val.to_numpy()])
        Xva, yva = make_windows(self.scaler.transform(val_ctx), L)

        to = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
        tr = DataLoader(TensorDataset(to(Xtr), to(ytr)), batch_size=self.cfg.batch_size,
                        shuffle=True, drop_last=False)
        va = DataLoader(TensorDataset(to(Xva), to(yva)), batch_size=512, shuffle=False)
        return tr, va

    def fit(self, train: pd.Series, val: pd.Series):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        tr_loader, va_loader = self._loaders(train, val)
        self.net = self._build().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = nn.MSELoss()

        best_state, bad_epochs = None, 0
        t0 = time.perf_counter()
        for epoch in range(1, self.cfg.epochs + 1):
            self.epochs_run_ = epoch
            self.net.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                # Gated RNNs still see occasional large gradients on spiky
                # traffic; clipping keeps training stable without changing the
                # objective.
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)

            self.net.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    va_loss += loss_fn(self.net(xb), yb).item() * len(xb)
            va_loss /= len(va_loader.dataset)
            sched.step(va_loss)
            self.history_.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

            if va_loss < self.best_val_ - 1e-6:
                self.best_val_, bad_epochs = va_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        self.train_seconds = time.perf_counter() - t0
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_rolling(self, history: pd.Series, test: pd.Series) -> np.ndarray:
        L = self.cfg.seq_len
        full = np.concatenate([history.to_numpy()[-L:], test.to_numpy()])
        scaled = self.scaler.transform(full)
        # Window i ends at the last observation before test step i, so every
        # prediction uses only genuinely past values.
        idx = np.arange(L)[None, :] + np.arange(len(test))[:, None]

        self.net.eval()
        outs = []
        with torch.no_grad():
            X = torch.tensor(scaled[idx], dtype=torch.float32, device=self.device)
            for i in range(0, len(X), 512):
                outs.append(self.net(X[i:i + 512]).cpu().numpy())
        return self.scaler.inverse_transform(np.concatenate(outs))

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) if self.net else 0

    def describe(self) -> dict:
        cfg = dict(vars(self.cfg))
        cfg.pop("extra", None)
        return {
            "model": self.name,
            "device": str(self.device),
            "n_parameters": self.n_parameters(),
            "epochs_run": self.epochs_run_,
            "best_val_mse_scaled": float(self.best_val_),
            **cfg,
        }


class LSTMForecaster(_NeuralForecaster):
    name = "LSTM"

    def _build(self):
        return _LSTMNet(self.cfg.hidden, self.cfg.layers, self.cfg.dropout)

    def describe(self) -> dict:
        d = super().describe()
        for k in ("channels", "levels", "kernel"):
            d.pop(k, None)
        return d


class TCNForecaster(_NeuralForecaster):
    name = "TCN"

    def _build(self):
        return _TCNNet(self.cfg.channels, self.cfg.levels, self.cfg.kernel, self.cfg.dropout)

    @property
    def receptive_field(self) -> int:
        # Two dilated convs per block, dilations 1, 2, 4, ... 2^(levels-1)
        return 1 + 2 * (self.cfg.kernel - 1) * (2 ** self.cfg.levels - 1)

    def describe(self) -> dict:
        d = super().describe()
        for k in ("hidden", "layers"):
            d.pop(k, None)
        d["receptive_field"] = self.receptive_field
        return d


MODEL_REGISTRY = {
    "seasonal_naive": SeasonalNaive,
    "sarima": SarimaForecaster,
    "lstm": LSTMForecaster,
    "tcn": TCNForecaster,
}
