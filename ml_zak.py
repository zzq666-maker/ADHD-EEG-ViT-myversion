import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

HAS_XGBOOST = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGBOOST = False


# =========================
# 1. 参数
# =========================
DATA_DIR = "/home/zhangzhouqi/ADHD-EEG-ViT/ADHD-EEG-ViT-myversion/data_raw"

# 新输出文件夹名字
OUTPUT_DIR = "/home/zhangzhouqi/ADHD-EEG-ViT/ADHD-EEG-ViT-myversion/ml_results"

FS = 128

# 独立测试集比例
TEST_RATIO = 0.20

# 在 train+val 内部再切验证集，最终总体验证集约占 20%
VAL_RATIO_WITHIN_TRAINVAL = 0.25

# 重复实验次数
N_REPEATS = 20

WINDOW_SEC = 4
STEP_SEC = 2

TOP_K_OPTIONS = [40, 80, 120]

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"
]

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "P7", "P8", "Pz"],
    "occipital": ["O1", "O2"],
    "temporal": ["T7", "T8"],
}

ASYM_PAIRS = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("O1", "O2"),
    ("T7", "T8"),
]

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. 频域函数
# =========================
def bandpower(freqs, psd, band):
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def extract_window_features(signal, fs=128):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)

    nperseg = min(256, len(signal))
    if nperseg < 32:
        return None

    noverlap = nperseg // 2

    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density"
    )

    total_mask = np.logical_and(freqs >= 0.5, freqs <= 45)
    total_power = np.trapz(psd[total_mask], freqs[total_mask]) if np.any(total_mask) else 1e-12
    total_power = max(total_power, 1e-12)

    feats = {}
    band_vals = {}

    for band_name, band_range in BANDS.items():
        bp = bandpower(freqs, psd, band_range)
        band_vals[band_name] = bp
        feats[f"{band_name}_abs"] = bp
        feats[f"{band_name}_rel"] = bp / total_power

    feats["theta_beta_ratio"] = band_vals["theta"] / max(band_vals["beta"], 1e-12)
    feats["theta_alpha_ratio"] = band_vals["theta"] / max(band_vals["alpha"], 1e-12)
    feats["beta_alpha_ratio"] = band_vals["beta"] / max(band_vals["alpha"], 1e-12)

    # PSD bins: 2~30Hz，每2Hz一个
    for hz in range(2, 31, 2):
        idx = np.argmin(np.abs(freqs - hz))
        feats[f"psd_{hz}hz"] = psd[idx]

    return feats


def sliding_windows(signal, fs=128, window_sec=4, step_sec=2):
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)

    if len(signal) < win_len:
        return []

    windows = []
    for start in range(0, len(signal) - win_len + 1, step_len):
        end = start + win_len
        windows.append(signal[start:end])
    return windows


def summarize_feature_list(feature_list, prefix):
    if len(feature_list) == 0:
        return {}

    keys = list(feature_list[0].keys())
    out = {}

    for k in keys:
        vals = np.array([d[k] for d in feature_list], dtype=np.float64)
        out[f"{prefix}_{k}_mean"] = float(np.mean(vals))
        out[f"{prefix}_{k}_std"] = float(np.std(vals))
        out[f"{prefix}_{k}_max"] = float(np.max(vals))
        out[f"{prefix}_{k}_min"] = float(np.min(vals))

    return out


def extract_subject_features(eeg, subject_id):
    if eeg.ndim != 2:
        raise ValueError(f"{subject_id} 不是二维矩阵，shape={eeg.shape}")

    n_time, n_ch = eeg.shape
    row = {
        "subject": subject_id,
        "n_timepoints": n_time,
        "n_channels": n_ch,
    }

    channel_summary_cache = {}

    for ch in range(n_ch):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch+1}"
        signal = eeg[:, ch]

        win_signals = sliding_windows(signal, fs=FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC)
        win_feats = []

        for ws in win_signals:
            feats = extract_window_features(ws, fs=FS)
            if feats is not None:
                win_feats.append(feats)

        if len(win_feats) == 0:
            continue

        summary = summarize_feature_list(win_feats, ch_name)
        row.update(summary)
        channel_summary_cache[ch_name] = summary

    if len(channel_summary_cache) == 0:
        raise ValueError(f"{subject_id} 没有提取到有效窗口特征")

    core_metrics = [
        "theta_rel_mean",
        "alpha_rel_mean",
        "beta_rel_mean",
        "theta_beta_ratio_mean",
        "theta_alpha_ratio_mean",
        "beta_alpha_ratio_mean",
    ]

    for region_name, region_channels in REGIONS.items():
        valid_channels = [ch for ch in region_channels if ch in channel_summary_cache]
        if len(valid_channels) == 0:
            continue

        for metric in core_metrics:
            vals = []
            for ch in valid_channels:
                key = f"{ch}_{metric}"
                if key in row:
                    vals.append(row[key])
            if len(vals) > 0:
                row[f"{region_name}_{metric}"] = float(np.mean(vals))

    diff_metrics = [
        "theta_rel_mean",
        "alpha_rel_mean",
        "beta_rel_mean",
        "theta_beta_ratio_mean",
    ]

    for left, right in ASYM_PAIRS:
        for metric in diff_metrics:
            lk = f"{left}_{metric}"
            rk = f"{right}_{metric}"
            if lk in row and rk in row:
                row[f"{left}_{right}_{metric}_diff"] = row[lk] - row[rk]

    return row


def evaluate_binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = None
    return metrics


def get_positive_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    return None


# =========================
# 3. 读取数据并提特征
# =========================
rows = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if not file.lower().endswith(".mat"):
            continue

        root_lower = root.lower()
        if "adhd_part" in root_lower:
            label = 1
            label_name = "ADHD"
        elif "control_part" in root_lower:
            label = 0
            label_name = "Control"
        else:
            continue

        path = os.path.join(root, file)

        try:
            mat = loadmat(path)
            valid_keys = [k for k in mat.keys() if not k.startswith("__")]
            if len(valid_keys) == 0:
                continue

            eeg = mat[valid_keys[0]]
            subject_id = os.path.splitext(file)[0]

            row = extract_subject_features(eeg, subject_id)
            row["label"] = label
            row["label_name"] = label_name
            row["file_path"] = path
            rows.append(row)

        except Exception as e:
            print(f"[跳过] {file}: {e}")

if len(rows) == 0:
    raise ValueError("没有读取到有效样本，请检查路径。")

df = pd.DataFrame(rows).fillna(0.0)
df.to_csv(os.path.join(OUTPUT_DIR, "all_subject_features.csv"), index=False, encoding="utf-8-sig")

print("总样本数:", len(df))
print(df["label_name"].value_counts())


# =========================
# 4. 定义模型搜索空间
# =========================
def get_models_and_params(feature_dim):
    k_options = [k for k in TOP_K_OPTIONS if k <= feature_dim]
    if len(k_options) == 0:
        k_options = [min(20, feature_dim)]

    configs = {
        "svm_rbf": (
            SVC(probability=True, random_state=42),
            {
                "selector__k": k_options,
                "clf__C": [0.5, 1, 2, 5, 10],
                "clf__gamma": ["scale", 0.01, 0.001],
                "clf__kernel": ["rbf"],
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "selector__k": k_options,
                "clf__n_estimators": [300, 500],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_split": [2, 4],
                "clf__min_samples_leaf": [1, 2],
            }
        ),
    }

    if HAS_XGBOOST:
        configs["xgboost"] = (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),
            {
                "selector__k": k_options,
                "clf__n_estimators": [200, 300],
                "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            }
        )
    return configs


# =========================
# 5. 重复随机实验：train / val / test
# =========================
all_model_results = []
selected_repeat_results = []
best_global = None

# 去掉标签列和明显会引入数据集偏差的元信息列
exclude_cols = ["subject", "label", "label_name", "file_path", "n_timepoints", "n_channels"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

for repeat_idx in range(N_REPEATS):
    seed = 42 + repeat_idx

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=seed,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_RATIO_WITHIN_TRAINVAL,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[feature_cols].fillna(0.0)
    X_val = val_df[feature_cols].fillna(0.0)
    X_test = test_df[feature_cols].fillna(0.0)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    models_and_params = get_models_and_params(X_train.shape[1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    repeat_best = None

    train_df.to_csv(os.path.join(OUTPUT_DIR, f"train_features_repeat_{repeat_idx+1}.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUTPUT_DIR, f"val_features_repeat_{repeat_idx+1}.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, f"test_features_repeat_{repeat_idx+1}.csv"), index=False, encoding="utf-8-sig")

    for model_name, (clf, param_grid) in models_and_params.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("clf", clf),
        ])

        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        y_val_prob = get_positive_scores(best_model, X_val)
        y_test_prob = get_positive_scores(best_model, X_test)

        val_metrics = evaluate_binary_metrics(y_val, y_val_pred, y_val_prob)
        test_metrics = evaluate_binary_metrics(y_test, y_test_pred, y_test_prob)

        result = {
            "repeat": repeat_idx + 1,
            "seed": seed,
            "model_name": model_name,
            "cv_best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "train_subjects": train_df["subject"].tolist(),
            "val_subjects": val_df["subject"].tolist(),
            "test_subjects": test_df["subject"].tolist(),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_specificity": val_metrics["specificity"],
            "val_auc": val_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_specificity": test_metrics["specificity"],
            "test_auc": test_metrics["auc"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_tp": test_metrics["tp"],
        }
        all_model_results.append(result)

        if repeat_best is None or val_metrics["accuracy"] > repeat_best["val_accuracy"]:
            repeat_best = {
                **result,
                "best_estimator": best_model,
            }

    selected_result = {
        k: v
        for k, v in repeat_best.items()
        if k != "best_estimator"
    }
    selected_repeat_results.append(selected_result)

    if best_global is None or repeat_best["val_accuracy"] > best_global["val_accuracy"]:
        best_global = repeat_best

    print(
        f"重复 {repeat_idx+1}/{N_REPEATS} 完成，"
        f"验证集选中模型: {repeat_best['model_name']} | "
        f"val_acc={repeat_best['val_accuracy']:.4f} | "
        f"test_acc={repeat_best['test_accuracy']:.4f}"
    )


# =========================
# 6. 保存结果
# =========================
all_results_df = pd.DataFrame(all_model_results)
selected_df = pd.DataFrame(selected_repeat_results)

all_results_df.to_csv(os.path.join(OUTPUT_DIR, "all_model_results.csv"), index=False, encoding="utf-8-sig")
selected_df.to_csv(os.path.join(OUTPUT_DIR, "selected_model_results.csv"), index=False, encoding="utf-8-sig")

summary_df = all_results_df.groupby("model_name").agg(
    mean_val_accuracy=("val_accuracy", "mean"),
    std_val_accuracy=("val_accuracy", "std"),
    mean_test_accuracy=("test_accuracy", "mean"),
    std_test_accuracy=("test_accuracy", "std"),
    mean_test_f1=("test_f1", "mean"),
    std_test_f1=("test_f1", "std"),
    mean_test_recall=("test_recall", "mean"),
    std_test_recall=("test_recall", "std"),
    mean_test_precision=("test_precision", "mean"),
    std_test_precision=("test_precision", "std"),
    mean_test_specificity=("test_specificity", "mean"),
    std_test_specificity=("test_specificity", "std"),
    mean_test_auc=("test_auc", "mean"),
    std_test_auc=("test_auc", "std"),
    mean_cv_score=("cv_best_score", "mean"),
).reset_index()

selected_summary = {
    "repeats": int(len(selected_df)),
    "selection_rule": "Within each repeat, choose the model with the highest validation accuracy and report its independent test performance.",
    "selected_model_counts": selected_df["model_name"].value_counts().to_dict(),
    "selected_model_mean_test_accuracy": float(selected_df["test_accuracy"].mean()),
    "selected_model_std_test_accuracy": float(selected_df["test_accuracy"].std()),
    "selected_model_mean_test_f1": float(selected_df["test_f1"].mean()),
    "selected_model_std_test_f1": float(selected_df["test_f1"].std()),
    "selected_model_mean_test_recall": float(selected_df["test_recall"].mean()),
    "selected_model_std_test_recall": float(selected_df["test_recall"].std()),
    "selected_model_mean_test_precision": float(selected_df["test_precision"].mean()),
    "selected_model_std_test_precision": float(selected_df["test_precision"].std()),
    "selected_model_mean_test_specificity": float(selected_df["test_specificity"].mean()),
    "selected_model_std_test_specificity": float(selected_df["test_specificity"].std()),
    "selected_model_mean_test_auc": float(selected_df["test_auc"].dropna().mean()) if selected_df["test_auc"].notna().any() else None,
    "selected_model_std_test_auc": float(selected_df["test_auc"].dropna().std()) if selected_df["test_auc"].notna().any() else None,
}

summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False, encoding="utf-8-sig")

joblib.dump(best_global["best_estimator"], os.path.join(OUTPUT_DIR, "best_model.pkl"))

with open(os.path.join(OUTPUT_DIR, "best_model_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"best_repeat: {best_global['repeat']}\n")
    f.write(f"best_seed: {best_global['seed']}\n")
    f.write(f"best_model_name: {best_global['model_name']}\n")
    f.write(f"best_val_accuracy: {best_global['val_accuracy']:.6f}\n")
    f.write(f"best_test_accuracy: {best_global['test_accuracy']:.6f}\n")
    f.write(f"best_cv_score: {best_global['cv_best_score']:.6f}\n")
    f.write(f"best_params: {best_global['best_params']}\n")

with open(os.path.join(OUTPUT_DIR, "selected_model_summary.json"), "w", encoding="utf-8") as f:
    json.dump(selected_summary, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "all_model_results": all_model_results,
            "selected_model_results": selected_repeat_results,
            "selected_model_summary": selected_summary,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n===== 全部完成 =====")
print("各模型在独立测试集上的平均结果：")
print(summary_df)
print("\n按验证集选模后的稳定泛化性能：")
print(json.dumps(selected_summary, ensure_ascii=False, indent=2))
print("\n单次最佳验证结果：")
print(best_global["model_name"], best_global["val_accuracy"], best_global["test_accuracy"])
print("结果目录：", OUTPUT_DIR)
