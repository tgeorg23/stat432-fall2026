from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "saheart.csv"
SEED = 2026
TEST_SIZE = 0.25
PREDICTORS = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "famhist",
    "typea",
    "obesity",
    "alcohol",
    "age",
]


def make_model(kernel: str, C: float, gamma: Union[str, float] = "scale"):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma),
    )


def write_skill_appendix() -> None:
    skill_path = HERE.parent.parent / ".github" / "skills" / "svm_fitting" / "SKILL.md"
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    appendix = [r"\begingroup\scriptsize\ttfamily\raggedright"]
    for line in skill_path.read_text().splitlines():
        if line:
            appendix.append(line.translate(str.maketrans(replacements)) + r"\par")
        else:
            appendix.append(r"\par\smallskip")
    appendix.append(r"\endgroup")
    (HERE / "svm-skill-appendix.tex").write_text("\n".join(appendix) + "\n")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    write_skill_appendix()
    df["famhist"] = df["famhist"].map({"Present": 1, "Absent": 0})
    if df[PREDICTORS + ["chd"]].isna().any().any():
        raise ValueError("Analysis variables contain missing values.")
    if not set(df["chd"].unique()).issubset({0, 1}):
        raise ValueError("The response must contain only binary labels 0 and 1.")

    X = df[PREDICTORS]
    y = df["chd"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train))

    candidates = [
        ("linear", C, "scale")
        for C in (0.1, 1, 10)
    ] + [
        ("rbf", C, gamma)
        for C in (0.1, 1, 10)
        for gamma in ("scale", 0.01, 0.1, 1)
    ]

    cv_rows = []
    for candidate_number, (kernel, C, gamma) in enumerate(candidates, start=1):
        model = make_model(kernel, C, gamma)
        fold_errors = 1 - cross_val_score(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring="accuracy",
        )
        cv_rows.append(
            {
                "candidate": candidate_number,
                "model": kernel,
                "C": C,
                "gamma": gamma,
                "mean_cv_error": fold_errors.mean(),
                "se_cv_error": fold_errors.std(ddof=1) / np.sqrt(len(fold_errors)),
            }
        )

    cv_results = pd.DataFrame(cv_rows)
    best = cv_results.loc[cv_results["mean_cv_error"].idxmin()]
    threshold = best["mean_cv_error"] + best["se_cv_error"]
    eligible = cv_results[cv_results["mean_cv_error"] <= threshold].copy()
    eligible["kernel_preference"] = eligible["model"].map(
        {"linear": 0, "rbf": 1}
    )
    selected = eligible.sort_values(
        ["kernel_preference", "C", "mean_cv_error"]
    ).iloc[0]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline_errors = 1 - cross_val_score(
        baseline,
        X_train,
        y_train,
        cv=folds,
        scoring="accuracy",
    )
    baseline_cv = {
        "candidate": "B",
        "model": "majority",
        "C": np.nan,
        "gamma": "",
        "mean_cv_error": baseline_errors.mean(),
        "se_cv_error": baseline_errors.std(ddof=1) / np.sqrt(len(baseline_errors)),
    }
    cv_results = pd.concat(
        [cv_results, pd.DataFrame([baseline_cv])], ignore_index=True
    )
    cv_results.to_csv(HERE / "svm_cv_results.csv", index=False)

    best_linear = cv_results[cv_results["model"] == "linear"].sort_values(
        "mean_cv_error"
    ).iloc[0]
    best_rbf = cv_results[cv_results["model"] == "rbf"].sort_values(
        "mean_cv_error"
    ).iloc[0]
    test_models = [
        ("Selected SVM", make_model(selected["model"], selected["C"], selected["gamma"])),
        ("Best linear SVM", make_model("linear", best_linear["C"])),
        ("Best RBF SVM", make_model("rbf", best_rbf["C"], best_rbf["gamma"])),
        ("Majority baseline", DummyClassifier(strategy="most_frequent")),
    ]

    test_rows = []
    confusion = None
    for label, model in test_models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        errors = predictions != y_test
        error_rate = errors.mean()
        test_rows.append(
            {
                "model": label,
                "test_error": error_rate,
                "test_se": np.sqrt(error_rate * (1 - error_rate) / len(y_test)),
            }
        )
        if label == "Selected SVM":
            confusion = pd.crosstab(
                y_test.rename("observed"),
                pd.Series(predictions, index=y_test.index, name="predicted"),
            ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(HERE / "svm_test_results.csv", index=False)

    plot_data = cv_results[cv_results["model"] != "majority"].copy()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    colors = {"linear": "#2166ac", "rbf": "#b2182b"}
    for kernel in ("linear", "rbf"):
        subset = plot_data[plot_data["model"] == kernel]
        ax.errorbar(
            subset["candidate"].astype(int),
            subset["mean_cv_error"],
            yerr=subset["se_cv_error"],
            fmt="o",
            capsize=2,
            color=colors[kernel],
            label=kernel.upper(),
        )

    selected_mask = (plot_data["model"] == selected["model"]) & (
        plot_data["C"] == selected["C"]
    ) & (plot_data["gamma"].astype(str) == str(selected["gamma"]))
    chosen = plot_data.loc[selected_mask].iloc[0]
    ax.scatter(
        [int(chosen["candidate"])],
        [chosen["mean_cv_error"]],
        marker="*",
        s=150,
        color="black",
        zorder=5,
        label="Selected",
    )
    ax.set_ylabel("Five-fold CV misclassification rate")
    ax.set_xlabel("Candidate number (settings listed in the table)")
    ax.set_xticks(plot_data["candidate"].astype(int))
    ax.set_xticklabels(plot_data["candidate"].astype(int))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "svm-cv-errors.pdf")
    plt.close(fig)

    print(f"Rows: {len(df)}; predictors: {len(PREDICTORS)}")
    print(f"Missing values in analysis columns: {int(df[PREDICTORS + ['chd']].isna().sum().sum())}")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print(f"Train/test sizes: {len(y_train)}/{len(y_test)}; seed: {SEED}")
    print("\nCV results:")
    print(cv_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nLowest CV error: {best['model']} C={best['C']:g} "
        f"gamma={best['gamma']}: {best['mean_cv_error']:.4f} "
        f"(SE {best['se_cv_error']:.4f})"
    )
    print(
        f"One-SE threshold: {threshold:.4f}; selected: {selected['model']} "
        f"C={selected['C']:g} gamma={selected['gamma']}"
    )
    print("\nTest results:")
    print(test_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSelected-model confusion matrix (rows observed 0,1; columns predicted 0,1):")
    print(confusion)


if __name__ == "__main__":
    main()
