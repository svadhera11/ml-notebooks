import datetime
import os
import pickle
import time

import gower
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import umap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("TkAgg")

for dirname, _, filenames in os.walk("dataset/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

formatted_string = datetime.datetime.fromtimestamp(time.time()).strftime(
    "%Y-%m-%d_%H:%M:%S"
)
print(f"TIMESTAMP FOR RUN: {formatted_string}")

df_train = pd.read_csv("dataset/train.csv", index_col="id")

df_train

df_train.info()

df_train.describe()

cat_vars = list(df_train.keys()[df_train.dtypes == "object"])
real_vars = list(df_train.keys()[df_train.dtypes != "object"])

real_vars

cat_vars

for var in cat_vars:
    print()
    print(var, " ---TAKES VALUES-->", df_train[var].unique())
    print("VALUE COUNTS")
    print(df_train[var].value_counts())
    print("--------------")

for var in real_vars:
    print()
    print(f"VARIABLE NAME: {var}")
    print(df_train[var].describe())
    print("--------------")

for var in real_vars:
    plt.figure(figsize=(10, 5))
    plt.hist(df_train[var], bins=20, density=True, log=False)
    plt.title(f"HISTOGRAM OF VARIABLE: {var}")
    plt.tight_layout()
    plt.show()

for var in cat_vars:
    plt.figure(figsize=(10, 5))
    plt.hist(df_train[var], bins=20, density=True, log=False)
    plt.title(f"HISTOGRAM OF VARIABLE: {var}")
    plt.tight_layout()
    plt.show()

# Removing the target variable after some initial exploration.
real_vars = real_vars[:-1]
cat_vars = cat_vars

# Split into train/val sets. Test set is given without any associated labels.

x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("loan_status", axis=1),
    df_train["loan_status"],
    test_size=0.3,
    random_state=42,
    stratify=df_train["loan_status"],
)

# Performining standardization and one-hot encoding.

real_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline([("one_hot_encoder", OneHotEncoder())])

preprocessor = ColumnTransformer(
    [
        ("real_pipeline", real_pipeline, real_vars),
        ("cat_pipeline", cat_pipeline, cat_vars),
    ]
)

preprocessor.fit(x_train)

x_train = preprocessor.transform(x_train).astype(np.float64)
x_val = preprocessor.transform(x_val).astype(np.float64)
y_train = y_train.to_numpy(dtype=np.float64)
y_val = y_val.to_numpy(dtype=np.float64)

# Dimensionality reduction and plotting using Gower Distance for mixed data.
# Sample 2,000 random points from the training data (df_train).
indices = np.random.choice(
    df_train.drop("loan_status", axis=1).shape[0], size=(2_000,), replace=False
)
x_samples = df_train.drop("loan_status", axis=1).iloc[indices]
y_samples = df_train["loan_status"].iloc[indices]

sample_gower_matrix = gower.gower_matrix(x_samples)

umap_results = umap.UMAP(
    metric="precomputed", n_neighbors=500, min_dist=0.5
).fit_transform(sample_gower_matrix)

plt.scatter(x=umap_results[:, 0], y=umap_results[:, 1], c=y_samples, alpha=0.5)
plt.xlabel("UMAP COMPONENT 1")
plt.ylabel("UMAP COMPONENT 2")
plt.title("UMAP REPRESENTATION OF `train.csv` (1000 POINTS)")
plt.tight_layout()
plt.show()


# Goal: Trying different classification models: Random Forest, XGBoost, LightGBM, XGBoost
# HYPERPARAMETER TUNING: Optuna


def RandomForestModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    """
    Find the Random Forest Model with the best parameters.
    Return the fit model, and the best parameter(s).
    """

    def objective(trial, x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
        """
        objective function for optuna
        """
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 1, 100)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        class_weight = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        )
        ccp_alpha = trial.suggest_float("ccp_alpha", 1e-9, 1.0, log=True)
        OptunaRFModel = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            n_jobs=-1,
            random_state=42,
        )
        # use stratified k-fold cross-validation
        # to compute the score to be maximized.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            OptunaRFModel, x_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return cv_scores.mean()

    # create a study and optimize
    study = optuna.create_study(direction="maximize")  # Maximize ROC-AUC
    study.optimize(objective, n_trials=500)  # run 500 trials

    best_params = study.best_params

    # this is the Random Forest Model with the best parameters.
    RFModel = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=best_params["class_weight"],
        ccp_alpha=best_params["ccp_alpha"],
        n_jobs=-1,
        random_state=42,
    )

    RFModel.fit(x_train, y_train)

    return RFModel, best_params


def LoadRandomForestModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    if os.path.exists("RFModel_data.pkl"):
        print("Loading Model and Parameters from `RFModel_data.pkl`...")
        with open("RFModel_data.pkl", "rb") as f:
            data = pickle.load(f)
        RFModel = data["RFModel"]
        best_params = data["best_params"]
        print("Loading Complete!")
    else:
        print("RUNNING OPTUNA...THIS MIGHT TAKE A WHILE")
        RFModel, best_params = RandomForestModel(
            x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
        )
        print("Done! Saving Model and Parameters to 'RFModel_data.pkl'...")
        with open("RFModel_data.pkl", "wb") as f:
            pickle.dump({"RFModel": RFModel, "best_params": best_params}, f)
        print("Saving Complete!")

    print(
        f"TRAINING SET ROC-AUC: {roc_auc_score(y_train, RFModel.predict_proba(x_train)[:, 1]):.5f}"
    )
    print(
        f"VALIDATION SET ROC-AUC: {roc_auc_score(y_val, RFModel.predict_proba(x_val)[:, 1]):.5f}"
    )

    return RFModel, best_params


RFModel, best_params = LoadRandomForestModel(
    x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
)


def create_submission_rf(
    df_test=None, preprocessor=preprocessor, RFModel=RFModel, timestamp=formatted_string
):

    if df_test is None:
        print("Loading test data...")
        df_test = pd.read_csv("dataset/test.csv")
        print("Done!")
    x_test = preprocessor.transform(df_test).astype(np.float64)
    y_pred = RFModel.predict_proba(x_test)[:, 1]

    df_test["loan_status"] = y_pred

    df_submit = df_test[["id", "loan_status"]]
    df_submit.to_csv(
        f"submission_random_forest_{timestamp}.csv", header=True, index=None
    )

    print(f"`submission_random_forest_{timestamp}.csv` saved!")
    return None


create_submission_rf(df_test=None, preprocessor=preprocessor, RFModel=RFModel)


def XGBoostModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    """
    Find the XGBoost Model with the best parameters.
    Use Optuna and GPU.
    """

    def objective(trial):
        params = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "gpu_hist",
            "sampling_method": "gradient_based",
            "random_state": 42,
            "device": "cuda",
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        }
        XGBModel = xgb.XGBClassifier(**params, use_label_encoder=False)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            XGBModel, x_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)
    best_params = study.best_params

    XGBModel = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        tree_method="gpu_hist",
        device="cuda",
        verbosity=0,
        objective="binary:logistic",
        eval_metric="auc",
        sampling_method="gradient_based",
        random_state=42,
    )

    XGBModel.fit(x_train, y_train)

    return XGBModel, best_params


def LoadXGBoostModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    if os.path.exists("XGBModel_data.pkl"):
        print("Loading Model and Parameters from `XGBModel_data.pkl`...")
        with open("XGBModel_data.pkl", "rb") as f:
            data = pickle.load(f)
        XGBModel = data["XGBModel"]
        best_params = data["best_params"]
        print("Loading Complete!")
    else:
        print("RUNNING OPTUNA...THIS MIGHT TAKE A WHILE")
        XGBModel, best_params = XGBoostModel(
            x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
        )
        print("Done! Saving Model and Parameters to 'XGBModel_data.pkl'...")
        with open("XGBModel_data.pkl", "wb") as f:
            pickle.dump({"XGBModel": XGBModel, "best_params": best_params}, f)
        print("Saving Complete!")

    print(
        f"TRAINING SET ROC-AUC: {roc_auc_score(y_train, XGBModel.predict_proba(x_train)[:, 1]):.5f}"
    )
    print(
        f"VALIDATION SET ROC-AUC: {roc_auc_score(y_val, XGBModel.predict_proba(x_val)[:, 1]):.5f}"
    )

    return XGBModel, best_params


XGBModel, best_params = LoadXGBoostModel(
    x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
)


def create_submission_xgb(
    df_test=None,
    preprocessor=preprocessor,
    XGBModel=XGBModel,
    timestamp=formatted_string,
):

    if df_test is None:
        print("Loading test data...")
        df_test = pd.read_csv("dataset/test.csv")
        print("Done!")
    x_test = preprocessor.transform(df_test).astype(np.float64)
    y_pred = XGBModel.predict_proba(x_test)[:, 1]

    df_test["loan_status"] = y_pred

    df_submit = df_test[["id", "loan_status"]]
    df_submit.to_csv(f"submission_xgboost_{timestamp}.csv", header=True, index=None)

    print(f"`submission_xgboost_{timestamp}.csv` saved!")
    return None


create_submission_xgb(df_test=None, preprocessor=preprocessor, XGBModel=XGBModel)


def LightGBMModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    """
    Find the LightGBM model with the best parameters.
    Use Optuna and GPU.
    """

    def objective(trial):
        params = {
            "verbosity": -1,
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "dart",
            "device": "cuda",
            "random_state": 42,
            "bagging_freq": 1,
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_iterations": trial.suggest_int("num_iterations", 100, 1000),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", 1e-8, 10.0, log=True
            ),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        }

        LGBModel = lgb.LGBMClassifier(**params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            LGBModel, x_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )

        return cv_scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)
    best_params = study.best_params

    LGBModel = lgb.LGBMClassifier(
        **best_params,
        verbosity=-1,
        objective="binary",
        metric="auc",
        boosting_type="dart",
        device="cuda",
        random_state=42,
        bagging_freq=1,
    )
    LGBModel.fit(x_train, y_train)
    return LGBModel, best_params


def LoadLGBModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    if os.path.exists("LGBModel_data.pkl"):
        print("Loading Model and Parameters from `LGBModel_data.pkl`...")
        with open("LGBModel_data.pkl", "rb") as f:
            data = pickle.load(f)
        LGBModel = data["LGBModel"]
        best_params = data["best_params"]
        print("Loading Complete!")
    else:
        print("RUNNING OPTUNA...THIS MIGHT TAKE A WHILE")
        LGBModel, best_params = LightGBMModel(
            x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
        )
        print("Done! Saving Model and Parameters to 'LGBModel_data.pkl'...")
        with open("LGBModel_data.pkl", "wb") as f:
            pickle.dump({"LGBModel": LGBModel, "best_params": best_params}, f)
        print("Saving Complete!")

    print(
        f"TRAINING SET ROC-AUC: {roc_auc_score(y_train, LGBModel.predict_proba(x_train)[:, 1]):.5f}"
    )
    print(
        f"VALIDATION SET ROC-AUC: {roc_auc_score(y_val, LGBModel.predict_proba(x_val)[:, 1]):.5f}"
    )

    return LGBModel, best_params


LGBModel, best_params = LoadLGBModel(
    x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
)


def create_submission_lgb(
    df_test=None,
    preprocessor=preprocessor,
    LGBModel=LGBModel,
    timestamp=formatted_string,
):

    if df_test is None:
        print("Loading test data...")
        df_test = pd.read_csv("dataset/test.csv")
        print("Done!")
    x_test = preprocessor.transform(df_test).astype(np.float64)
    y_pred = LGBModel.predict_proba(x_test)[:, 1]

    df_test["loan_status"] = y_pred

    df_submit = df_test[["id", "loan_status"]]
    df_submit.to_csv(f"submission_lgbm_{timestamp}.csv", header=True, index=None)

    print(f"`submission_lgbm_{timestamp}.csv` saved!")
    return None


create_submission_lgb(df_test=None, preprocessor=preprocessor, LGBModel=LGBModel)


def CatBoostModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    """
    Find and use the CatBoost Model with the best parameters.
    Use Optuna and CPU.
    """

    def objective(trial):
        params = {
            "verbose": 0,
            "used_ram_limit": "10gb",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced"]
            ),
            "depth": trial.suggest_int("depth", 4, 15),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bernoulli", "MVS", "Bayesian"]
            ),
        }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            )
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        CatModel = CatBoostClassifier(**params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            CatModel, x_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)
    best_params = study.best_params

    CatModel = CatBoostClassifier(
        **best_params,
        verbose=0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        used_ram_limit="10gb",
    )

    CatModel.fit(x_train, y_train)

    return CatModel, best_params


def LoadCatBoostModel(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val):
    if os.path.exists("CatModel_data.pkl"):
        print("Loading Model and Parameters from `CatModel_data.pkl`...")
        with open("CatModel_data.pkl", "rb") as f:
            data = pickle.load(f)
        CatModel = data["CatModel"]
        best_params = data["best_params"]
        print("Loading Complete!")
    else:
        print("RUNNING OPTUNA...THIS MIGHT TAKE A WHILE")
        CatModel, best_params = CatBoostModel(
            x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
        )
        print("Done! Saving Model and Parameters to 'CatModel_data.pkl'...")
        with open("CatModel_data.pkl", "wb") as f:
            pickle.dump({"CatModel": CatModel, "best_params": best_params}, f)
        print("Saving Complete!")

    print(
        f"TRAINING SET ROC-AUC: {roc_auc_score(y_train, CatModel.predict_proba(x_train)[:, 1]):.5f}"
    )
    print(
        f"VALIDATION SET ROC-AUC: {roc_auc_score(y_val, CatModel.predict_proba(x_val)[:, 1]):.5f}"
    )

    return CatModel, best_params


CatModel, best_params = LoadCatBoostModel(
    x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val
)


def create_submission_cb(
    df_test=None,
    preprocessor=preprocessor,
    CatModel=CatModel,
    timestamp=formatted_string,
):

    if df_test is None:
        print("Loading test data...")
        df_test = pd.read_csv("dataset/test.csv")
        print("Done!")
    x_test = preprocessor.transform(df_test).astype(np.float64)
    y_pred = CatModel.predict_proba(x_test)[:, 1]

    df_test["loan_status"] = y_pred

    df_submit = df_test[["id", "loan_status"]]
    df_submit.to_csv(f"submission_catboost_{timestamp}.csv", header=True, index=None)

    print(f"`submission_catboost_{timestamp}.csv` saved!")
    return None


create_submission_cb(
    df_test=None,
    preprocessor=preprocessor,
    CatModel=CatModel,
    timestamp=formatted_string,
)

model_list = [XGBModel, LGBModel, CatModel]


def evaluate_models(model_list, x_train, y_train, x_val, y_val, x_test):

    train_probs = np.zeros((x_train.shape[0], len(model_list)))
    val_probs = np.zeros((x_val.shape[0], len(model_list)))
    test_probs = np.zeros((x_test.shape[0], len(model_list)))

    for i, model in enumerate(model_list):

        train_preds = model.predict_proba(x_train)[:, 1]
        val_preds = model.predict_proba(x_val)[:, 1]
        test_preds = model.predict_proba(x_test)[:, 1]

        train_probs[:, i] = train_preds
        val_probs[:, i] = val_preds
        test_probs[:, i] = test_preds

    blended_train_preds = np.mean(train_probs, axis=1)
    blended_val_preds = np.mean(val_probs, axis=1)

    roc_auc_train_score = roc_auc_score(y_train, blended_train_preds)
    roc_auc_val_score = roc_auc_score(y_val, blended_val_preds)

    # Compute average predicted probabilities for the test set
    avg_test_probs = np.mean(test_probs, axis=1)

    return roc_auc_train_score, roc_auc_val_score, avg_test_probs


def create_submission_blend(
    df_test=None,
    preprocessor=preprocessor,
    model_list=model_list,
    timestamp=formatted_string,
):

    if df_test is None:
        print("Loading test data...")
        df_test = pd.read_csv("dataset/test.csv")
        print("Done!")
    x_test = preprocessor.transform(df_test).astype(np.float64)
    _1, _2, y_pred = evaluate_models(model_list, x_train, y_train, x_val, y_val, x_test)

    print(f"TRAINING SET ROC-AUC: {_1:.5f}")
    print(f"VALIDATION SET ROC-AUC: {_2:.5f}")

    df_test["loan_status"] = y_pred

    df_submit = df_test[["id", "loan_status"]]
    df_submit.to_csv(f"submission_blend_{timestamp}.csv", header=True, index=None)

    print(f"`submission_blend_{timestamp}.csv` saved!")
    return None


create_submission_blend(
    df_test=None,
    preprocessor=preprocessor,
    model_list=model_list,
    timestamp=formatted_string,
)
