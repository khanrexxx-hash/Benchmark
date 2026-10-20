#!/usr/bin/env python3
"""
dx_pipeline.py  -  ONE-FILE reproduction of the whole study
===========================================================
Missingness-shortcut audit + cost-aware sequential diagnostic testing with
risk-controlled (Learn-then-Test) stopping on four UCI clinical datasets.

USAGE (run from the folder that contains this file)
  python dx_pipeline.py prepare            # unzip the 4 dataset zips into ./data
  python dx_pipeline.py audit              # Part 1 audit            (~2.5 min)
  python dx_pipeline.py seq Hepatitis 4    # Part 2, one dataset, 4 CV repeats
  python dx_pipeline.py seq CKD 4
  python dx_pipeline.py seq Mammography 4
  python dx_pipeline.py seq Cervical 4
  python dx_pipeline.py report             # tables (results/) + figures 2-6
  python dx_pipeline.py flowchart          # figure 1
  python dx_pipeline.py fetch              # download 4 public datasets (PBC, Wisconsin biopsy, Cleveland heart, Pima)
  python dx_pipeline.py seq PBC 4          # also: seq Biopsy 4, seq Heart 4, seq Pima 4
  python dx_pipeline.py scenarios          # cost-scenario robustness runs (uniform / lab_heavy / invert / invasive_30)
  python dx_pipeline.py transport          # hepatitis -> PBC cross-cohort transport test of the certificate
  python dx_pipeline.py custom my.csv cfg.json 4   # YOUR OWN hospital data + YOUR OWN costs (see README)
  python dx_pipeline.py all                # everything above (~100 min, 1 CPU)
  python dx_pipeline.py seq Hepatitis 1 1  # quick smoke test (1 repeat, 1 split)

Expected data layout in ./data (created by `prepare`):
  hepatitis.data, costs/*, ckd/chronic_kidney_disease_full.arff,
  mammographic_masses.data, risk_factors_cervical_cancer.csv
Place mammographic_mass.zip, cervical_cancer_risk_factors.zip, ckd.zip and
hepatitis.zip next to this file (or in ./data) before running `prepare`.
"""
from __future__ import annotations
import sys, json, time, pickle, warnings, zipfile, copy, urllib.request
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RES = OUT = ROOT / "results"
FIG = ROOT / "figures"
for _d in (DATA_DIR, RES, FIG):
    _d.mkdir(exist_ok=True)



# ============================================================================
# PART 0: DATA LOADERS, ACQUISITION UNITS, COST MODELS
# ============================================================================
"""
Dataset loaders + clinical "acquisition unit" and cost definitions.

Every dataset is returned as a `Dataset` with
  X        : DataFrame of numeric features (NaN = value not recorded)
  y        : 0/1 numpy array (1 = disease / adverse outcome = "positive")
  units    : dict  unit_name -> list of column names acquired together
  cost     : dict  unit_name -> (full_cost, discounted_cost)
  group    : dict  unit_name -> group id (units in the same group share a fixed
             collection cost, e.g. venipuncture) or None
  excluded : dict  column -> reason it is NOT used as a predictor

Cost provenance
  Hepatitis : real Ontario Health Insurance fee-schedule costs (CAD) from
              Turney (1995), files costs/hepatitis.{cost,expense,group,delay}
  Others    : NO published per-test prices ship with these datasets.  Costs are
              ILLUSTRATIVE tiers (clinical history 1, urine dipstick 2, urine
              microscopy 3, blood assay 7.27 = Turney's price of a comparable
              chemistry assay, imaging/colposcopy tiers higher) and the shared
              venipuncture discount (2.10) is borrowed from Turney.  Conclusions
              are stress-tested with random cost perturbation
              (see run_sequential.py).  Replace with a local fee schedule.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

VENI = 2.10  # shared blood-draw cost (Turney 1995)


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    units: dict
    cost: dict
    group: dict
    excluded: dict = field(default_factory=dict)
    pos_label: str = "positive"
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.y)


def _rd(p):
    return open(p, encoding="latin-1").read()


# --------------------------------------------------------------------------
# Hepatitis  (155 pts, 19 predictors, class DIE=1 / LIVE=2 in raw file)
# --------------------------------------------------------------------------
def load_hepatitis() -> Dataset:
    names = ["class", "age", "sex", "steroid", "antiviral", "fatigue", "malaise",
             "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
             "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
             "protime", "histology"]
    d = pd.read_csv(DATA_DIR / "hepatitis.data", header=None, names=names, na_values="?")
    y = (d["class"] == 1).astype(int).to_numpy()          # 1 = DIE (adverse outcome)
    X = d.drop(columns="class").astype(float)
    cont = ("age", "bilirubin", "alk_phosphate", "sgot", "albumin", "protime")
    for c in X.columns:                                   # 1/2 coding -> 0/1
        if c not in cont:
            X[c] = X[c] - 1
    expense = {"bilirubin": (7.27, 5.17), "alk_phosphate": (7.27, 5.17),
               "sgot": (7.27, 5.17), "albumin": (7.27, 5.17), "protime": (8.30, 6.20)}
    units, cost, group = {}, {}, {}
    for c in X.columns:
        units[c] = [c]
        cost[c] = expense.get(c, (1.0, 1.0))
        group[c] = "A" if c in expense else None
    return Dataset("Hepatitis", X, y, units, cost, group, pos_label="death",
                   extra={"delayed": set(expense)})


# --------------------------------------------------------------------------
# Chronic kidney disease (400 pts; 250 CKD / 150 not-CKD)
# --------------------------------------------------------------------------
def _parse_ckd():
    txt = _rd(DATA_DIR / "ckd" / "chronic_kidney_disease_full.arff")
    cols = [l.split()[1].strip("'") for l in txt.splitlines()
            if l.lower().startswith("@attribute")]
    body = [l for l in txt[txt.lower().find("@data") + 5:].strip().splitlines() if l.strip()]
    rows = []
    for l in body:
        f = [v.strip() for v in l.strip().split(",")]
        if len(f) > len(cols):
            # 3 malformed rows: trailing comma (x2) or one duplicated delimiter (x1).
            # Empty strings are never legitimate values ("?" is the missing marker).
            f = [v for v in f if v != ""]
        assert len(f) == len(cols), (len(f), l)
        rows.append(f)
    return pd.DataFrame(rows, columns=cols).replace({"?": np.nan})


def load_ckd() -> Dataset:
    d = _parse_ckd()
    y = (d["class"] == "ckd").astype(int).to_numpy()
    X = d.drop(columns="class").copy()
    binmap = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
              "yes": 1, "no": 0, "good": 0, "poor": 1}
    for c in X.columns:
        X[c] = X[c].map(lambda v: binmap.get(v, v) if isinstance(v, str) else v)
        X[c] = pd.to_numeric(X[c], errors="coerce")
    units = {
        "history": ["age", "htn", "dm", "cad", "appet", "pe", "ane"],
        "bp": ["bp"],
        "urine_dipstick": ["sg", "al", "su"],
        "urine_microscopy": ["rbc", "pc", "pcc", "ba"],
        "glucose": ["bgr"],
        "urea": ["bu"],
        "creatinine": ["sc"],
        "electrolytes": ["sod", "pot"],
        "haemoglobin": ["hemo"],
        "pcv": ["pcv"],
        "wbc_count": ["wbcc"],
        "rbc_count": ["rbcc"],
    }
    base = {"history": 1.0, "bp": 1.0, "urine_dipstick": 2.0, "urine_microscopy": 3.0}
    cost, group = {}, {}
    for u in units:
        if u in base:
            cost[u] = (base[u], base[u]); group[u] = None
        else:  # blood assays: 7.27 full, minus shared venipuncture when blood already drawn
            cost[u] = (7.27, 7.27 - VENI); group[u] = "blood"
    return Dataset("CKD", X, y, units, cost, group, pos_label="CKD")


# --------------------------------------------------------------------------
# Mammographic mass (961 masses; severity 1 = malignant)
# --------------------------------------------------------------------------
def load_mammo() -> Dataset:
    d = pd.read_csv(DATA_DIR / "mammographic_masses.data", header=None,
                    names=["birads", "age", "shape", "margin", "density", "severity"],
                    na_values="?").astype(float)
    y = d["severity"].astype(int).to_numpy()
    X = d[["age", "shape", "margin", "density"]].copy()
    units = {c: [c] for c in X.columns}
    cost = {"age": (1.0, 1.0), "shape": (5.0, 5.0), "margin": (5.0, 5.0), "density": (5.0, 5.0)}
    group = {c: None for c in units}
    return Dataset("Mammography", X, y, units, cost, group, pos_label="malignant",
                   excluded={"birads": "radiologist's overall assessment; UCI flags it "
                             "non-predictive -> used only as a benchmark"},
                   extra={"birads": d["birads"].to_numpy()})


# --------------------------------------------------------------------------
# Cervical cancer risk factors (858 women; target = Biopsy)
# --------------------------------------------------------------------------
def load_cervical() -> Dataset:
    d = pd.read_csv(DATA_DIR / "risk_factors_cervical_cancer.csv", na_values="?")
    y = d["Biopsy"].astype(int).to_numpy()
    std_cols = [c for c in d.columns if c.startswith("STDs")]
    units = {
        "demographics": ["Age", "Number of sexual partners", "First sexual intercourse",
                         "Num of pregnancies"],
        "smoking": ["Smokes", "Smokes (years)", "Smokes (packs/year)"],
        "hormonal_contraception": ["Hormonal Contraceptives", "Hormonal Contraceptives (years)"],
        "iud": ["IUD", "IUD (years)"],
        "std_panel": std_cols,
        "cytology": ["Citology"],
        "colposcopy_hinselmann": ["Hinselmann"],
        "schiller": ["Schiller"],
    }
    cost = {"demographics": (1.0, 1.0), "smoking": (1.0, 1.0),
            "hormonal_contraception": (1.0, 1.0), "iud": (1.0, 1.0),
            "std_panel": (15.0, 15.0), "cytology": (30.0, 30.0),
            "colposcopy_hinselmann": (70.0, 70.0), "schiller": (75.0, 75.0)}
    group = {u: None for u in units}
    used = sum(units.values(), [])
    X = d[used].astype(float)
    excl = {c: "prior-diagnosis field recorded in the same work-up (leakage risk)"
            for c in ["Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]}
    excl["Biopsy"] = "target"
    return Dataset("Cervical", X, y, units, cost, group, excluded=excl, pos_label="biopsy+",
                   extra={"raw": d})



EXT_URLS = {"pbc.csv": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv",
            "biopsy.csv": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/biopsy.csv",
            "pima.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "heart.csv": "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"}


def fetch_external():
    (DATA_DIR / "external").mkdir(exist_ok=True)
    for f, u in EXT_URLS.items():
        p = DATA_DIR / "external" / f
        if not p.exists():
            urllib.request.urlretrieve(u, p)
        print("ok", p)


# --------------------------------------------------------------------------
# EXTERNAL replication dataset 1: Mayo Clinic primary biliary cirrhosis (PBC)
# outcome = death within 4 years (patients censored / transplanted before 4 y are excluded).
# Tests priced with Turney's real hepatitis prices where the same assay exists; other assays ASSUMED.
# --------------------------------------------------------------------------
def load_pbc() -> Dataset:
    d = pd.read_csv(DATA_DIR / "external" / "pbc.csv")
    h = 4 * 365.25
    pos = (d["status"] == 2) & (d["time"] <= h)
    d = d[pos | (d["time"] > h)].reset_index(drop=True)
    y = ((d["status"] == 2) & (d["time"] <= h)).astype(int).to_numpy()
    X = pd.DataFrame({
        "age": d["age"], "sex": (d["sex"] == "f").astype(float), "ascites": d["ascites"],
        "hepato": d["hepato"], "spiders": d["spiders"], "edema": d["edema"],
        "bilirubin": d["bili"], "cholesterol": d["chol"], "albumin": d["albumin"],
        "copper": d["copper"], "alk_phosphate": d["alk.phos"], "sgot": d["ast"],
        "triglycerides": d["trig"], "platelets": d["platelet"], "protime": d["protime"],
        "stage": d["stage"]}).astype(float)
    units = {c: [c] for c in X.columns}
    cost, group = {}, {}
    for c in X.columns:
        if c in ("bilirubin", "alk_phosphate", "sgot", "albumin"):
            cost[c], group[c] = (7.27, 5.17), "A"
        elif c == "protime":
            cost[c], group[c] = (8.30, 6.20), "A"
        elif c in ("cholesterol", "triglycerides", "platelets"):
            cost[c], group[c] = (7.27, 5.17), "A"          # assumed = same assay price
        elif c == "copper":
            cost[c], group[c] = (14.54, 14.54), None       # assumed (24 h urine copper)
        else:
            cost[c], group[c] = (1.0, 1.0), None           # history / exam / histology (Turney convention)
    return Dataset("PBC", X, y, units, cost, group, pos_label="death<=4y",
                   extra={"delayed": {"bilirubin", "alk_phosphate", "sgot", "albumin", "protime",
                                      "cholesterol", "triglycerides", "platelets", "copper"}})


# --------------------------------------------------------------------------
# EXTERNAL replication dataset 2: Wisconsin breast-cytology biopsy (MASS::biopsy), one row per patient ID
# --------------------------------------------------------------------------
def load_biopsy() -> Dataset:
    d = pd.read_csv(DATA_DIR / "external" / "biopsy.csv").drop_duplicates("ID").reset_index(drop=True)
    names = ["clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", "marginal_adhesion",
             "epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses"]
    X = d[[f"V{i}" for i in range(1, 10)]].astype(float)
    X.columns = names
    y = (d["class"] == "malignant").astype(int).to_numpy()
    units = {c: [c] for c in names}
    cost = {c: (3.0, 3.0) for c in names}                  # ASSUMED reader-scoring cost per attribute
    return Dataset("Biopsy", X, y, units, cost, {c: None for c in names}, pos_label="malignant")


# --------------------------------------------------------------------------
# YOUR OWN DATA: CSV + JSON config  (see README)
#   {"name": "MyHospital", "target": "outcome", "positive": 1,
#    "units": {"history": ["age","sex"], "creatinine": ["creat"]},
#    "costs": {"history": [1.0, 1.0], "creatinine": [4.5, 3.0]},     # [full, discounted]
#    "groups": {"creatinine": "blood"}}                               # optional shared-collection groups
# --------------------------------------------------------------------------
def load_custom(csv_path, cfg_path) -> Dataset:
    cfg = json.load(open(cfg_path))
    d = pd.read_csv(csv_path, na_values=["?", "NA", ""])
    y = (d[cfg["target"]] == cfg.get("positive", 1)).astype(int).to_numpy()
    cols = sum(cfg["units"].values(), [])
    X = d[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            codes, _ = pd.factorize(X[c]); X[c] = np.where(X[c].isna(), np.nan, codes)
    X = X.astype(float)
    cost = {u: tuple(cfg["costs"][u]) for u in cfg["units"]}
    group = {u: cfg.get("groups", {}).get(u) for u in cfg["units"]}
    return Dataset(cfg.get("name", "Custom"), X, y, cfg["units"], cost, group)


# --------------------------------------------------------------------------
# COST SCENARIOS (robustness of conclusions to the cost model)
#   uniform     : every unit costs 1, no discounts
#   lab_heavy   : units whose full price >= 7 cost 3x more
#   invert      : price ranks reversed (cheap tests become expensive and vice-versa), no discounts
#   invasive_30 : histology / stage (liver biopsy) priced at 30 instead of Turney's 1.00
# --------------------------------------------------------------------------
SCENARIOS = ["uniform", "lab_heavy", "invert", "invasive_30"]


def apply_cost_scenario(ds, scen):
    if scen == "primary":
        return ds
    if scen not in SCENARIOS:
        raise ValueError(f"unknown cost scenario {scen!r}; choose from primary, {SCENARIOS}")
    new = copy.copy(ds)
    cost, group = dict(ds.cost), dict(ds.group)
    if scen == "uniform":
        cost = {u: (1.0, 1.0) for u in cost}; group = {u: None for u in cost}
    elif scen == "lab_heavy":
        cost = {u: ((3 * f, 3 * dsc) if f >= 7 else (f, dsc)) for u, (f, dsc) in cost.items()}
    elif scen == "invert":
        order = sorted(cost, key=lambda u: cost[u][0])
        prices = [cost[u][0] for u in order][::-1]
        cost = {u: (p, p) for u, p in zip(order, prices)}; group = {u: None for u in cost}
    elif scen == "invasive_30":
        hit = [u for u in cost if u in ("histology", "stage")]
        if not hit:
            return None
        for u in hit:
            cost[u] = (30.0, 30.0)
    new.cost, new.group = cost, group
    return new



# --------------------------------------------------------------------------
# ADDITIONAL DATASET (REAL COSTS) : Pima Indians diabetes (768 women).  Prices are Turney's published
# Ontario fee-schedule values (JAIR 1995, Table 21): glucose tolerance 17.61 (15.51 if blood already drawn),
# insulin 22.78 (20.68), all other tests 1.00.  Zeros in glucose / BP / skin / insulin / BMI are physiologically
# impossible and are treated as missing (insulin is missing in ~49% of women).
# --------------------------------------------------------------------------
def load_pima() -> Dataset:
    d = pd.read_csv(DATA_DIR / "external" / "pima.csv", header=None,
                    names=["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "class"])
    y = d["class"].astype(int).to_numpy()
    X = d.drop(columns="class").astype(float)
    for c in ["glucose", "bp", "skin", "insulin", "bmi"]:
        X.loc[X[c] == 0, c] = np.nan
    units = {c: [c] for c in X.columns}
    cost = {c: (1.0, 1.0) for c in X.columns}
    group = {c: None for c in X.columns}
    cost["glucose"], cost["insulin"] = (17.61, 15.51), (22.78, 20.68)
    group["glucose"] = group["insulin"] = "A"
    return Dataset("Pima", X, y, units, cost, group, pos_label="diabetes",
                   extra={"delayed": {"glucose", "insulin"}})


# --------------------------------------------------------------------------
# ADDITIONAL DATASET (REAL COSTS) : Cleveland heart disease (303 patients, 13 tests).  Per-test prices are Turney's
# Ontario fee-schedule values (as bundled in the R package FFTrees): age/sex/cp/trestbps 1.00, cholesterol 7.27,
# fasting glucose 5.20, resting ECG 15.50, max heart rate 102.90, exercise angina / ST depression / slope 87.30,
# fluoroscopy (ca) 100.90, thallium scan (thal) 102.90.  Blood tests share a 2.10 collection cost (stated by
# Turney); we conservatively apply NO discount for the exercise / imaging groups (their membership is not stated
# in the sources we could verify).  The GitHub copy of the data has the target coded 1 = healthy; the loader
# checks the direction from the data (disease = lower max heart rate, higher ST depression) and the
# missing-value placeholders (ca = 4, thal = 0) are set to NaN.
# --------------------------------------------------------------------------
def load_heart() -> Dataset:
    d = pd.read_csv(DATA_DIR / "external" / "heart.csv", encoding="utf-8-sig")
    X = d.drop(columns="target").astype(float)
    X.loc[X["ca"] == 4, "ca"] = np.nan
    X.loc[X["thal"] == 0, "thal"] = np.nan
    t = d["target"].to_numpy()
    sick = int(d.loc[d["target"] == 1, "oldpeak"].mean() > d.loc[d["target"] == 0, "oldpeak"].mean())
    y = (t == sick).astype(int)
    assert d.loc[y == 1, "thalach"].mean() < d.loc[y == 0, "thalach"].mean(), "target direction check failed"
    price = {"age": 1.0, "sex": 1.0, "cp": 1.0, "trestbps": 1.0, "chol": 7.27, "fbs": 5.20, "restecg": 15.50,
             "thalach": 102.90, "exang": 87.30, "oldpeak": 87.30, "slope": 87.30, "ca": 100.90, "thal": 102.90}
    units = {c: [c] for c in X.columns}
    cost = {c: (price[c], price[c]) for c in X.columns}
    group = {c: None for c in X.columns}
    for c in ("chol", "fbs"):
        cost[c], group[c] = (price[c], price[c] - 2.10), "A"
    return Dataset("Heart", X, y, units, cost, group, pos_label="heart disease")


LOADERS = {"Hepatitis": load_hepatitis, "CKD": load_ckd,
           "Mammography": load_mammo, "Cervical": load_cervical}
EXTERNAL_LOADERS = {"PBC": load_pbc, "Biopsy": load_biopsy, "Heart": load_heart, "Pima": load_pima}


def load(name: str) -> Dataset:
    return {**LOADERS, **EXTERNAL_LOADERS}[name]()

# ============================================================================
# MASKED-TRAINING CLASSIFIER
# ============================================================================
"""
Models that can predict from an ARBITRARY subset of acquisition units.

MaskedHGB  ("deployment-faithful" model)
    1. natural missing values in the training fold are replaced by stochastic
       (posterior-sampled) multiple imputation fitted on the FEATURES ONLY, so the
       historical, label-correlated missingness pattern is erased;
    2. each of `n_aug` copies of the training data is then masked at the
       acquisition-unit level with a copy-specific random rate
       (missing-completely-at-random by construction);
    3. one histogram gradient-boosting classifier is fitted on the stacked copies.
    At inference an un-acquired unit is simply NaN.  Because the training masks are
    MCAR, the classifier cannot exploit *which* tests were historically ordered; it
    only uses *what* was measured.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

warnings.filterwarnings("ignore")


def make_hgb(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.06, max_depth=3, min_samples_leaf=8,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def unit_index(ds):
    """unit name -> integer column positions in ds.X"""
    cols = list(ds.X.columns)
    return {u: [cols.index(c) for c in cs] for u, cs in ds.units.items()}


def snap_to_observed(Xc, Xobs):
    """round imputed values of discrete columns back onto the observed grid"""
    Xc = Xc.copy()
    for j in range(Xc.shape[1]):
        vals = np.unique(Xobs[~np.isnan(Xobs[:, j]), j])
        if 1 < len(vals) <= 8:
            idx = np.abs(Xc[:, [j]] - vals[None, :]).argmin(1)
            Xc[:, j] = vals[idx]
    return Xc


class MultiImputer:
    """posterior-sampling iterative imputer; each .draw() returns a fresh imputation"""

    def __init__(self, seed=0):
        self.imp = IterativeImputer(sample_posterior=True, max_iter=6, random_state=seed,
                                    keep_empty_features=True, skip_complete=True)

    def fit(self, X):
        self.Xobs = np.asarray(X, float)
        self.imp.fit(self.Xobs)
        return self

    def draw(self):
        return snap_to_observed(self.imp.transform(self.Xobs), self.Xobs)


class MaskedHGB:
    def __init__(self, ds, n_aug=8, seed=0, q_max=0.95, ref_k=1):
        self.ds, self.n_aug, self.seed, self.q_max = ds, n_aug, seed, q_max
        self.uidx = unit_index(ds)
        self.unit_names = list(ds.units)

    # -- training ---------------------------------------------------------
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.mi = MultiImputer(self.seed).fit(X)
        copies, ys = [], []
        for r in range(self.n_aug):
            Xc = self.mi.draw()
            if r == 0:
                self.Xc_ref = Xc.copy()                 # complete reference (kNN sampling)
            q = rng.uniform(0, self.q_max, size=(len(Xc), 1))
            for u in self.unit_names:
                m = rng.random(len(Xc)) < q[:, 0]
                for j in self.uidx[u]:
                    Xc[m, j] = np.nan
            copies.append(Xc); ys.append(y)
        self.clf = make_hgb(self.seed).fit(np.vstack(copies), np.concatenate(ys))
        # scale for neighbour search
        self.mu = np.nanmean(self.Xc_ref, 0)
        self.sd = np.nanstd(self.Xc_ref, 0) + 1e-9
        self.y_ref = y
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X, float))[:, 1]


def apply_unit_mask(X, uidx, keep_units):
    """return copy of X with all units not in keep_units set to NaN"""
    X = np.asarray(X, float).copy()
    for u, js in uidx.items():
        if u not in keep_units:
            X[:, js] = np.nan
    return X

# ============================================================================
# ACQUISITION POLICIES AND LEARN-THEN-TEST STOPPING
# ============================================================================
"""
Cost-aware sequential diagnostic testing with risk-controlled stopping.

Policies (all share the same masked-training classifier and the same stopping rule):
  adaptive : next unit = argmax  MI(Y; X_c | x_observed) / marginal_cost(c)
             MI is estimated by sampling X_c from the K nearest training patients
             (in the currently observed coordinates) and pushing each sampled value
             through the classifier.
  static   : one global order, learned greedily on the fit set with the same
             MI/cost criterion averaged over patients (population-level analogue).
  cost     : cheapest unit first (common clinical practice: history -> dipstick -> blood).
  random   : random order per patient (sanity floor).
  full     : acquire everything, then classify (no sequencing).

Stopping rule (identical for every policy): stop at the first step whose posterior
confidence max(p, 1-p) >= tau.  tau is chosen on a held-out calibration set by
Learn-then-Test fixed-sequence testing so that the *silent error rate*
    R(tau) = P( patient is accepted AND misclassified )
satisfies  P_cal( R(tau_hat) <= alpha ) >= 1 - delta   (exchangeable patients, policy and
classifier fixed before calibration).  Patients never reaching confidence are DEFERRED
(escalated to the full panel / specialist) and are not counted as silent errors.
"""
import numpy as np
from scipy.stats import binom

EPS = 1e-6


def H(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class Traj:
    """one patient's acquisition trajectory"""
    __slots__ = ("units", "p", "cum", "cumR")

    def __init__(self, units, p, cum, cumR):
        self.units, self.p, self.cum, self.cumR = units, np.asarray(p), np.asarray(cum), cumR


class EG2Tree:
    """Published cost-sensitive test-selection baseline: EG2 decision tree (Nunez, 1991; used as a
    benchmark by Turney, 1995).  Attribute (unit) criterion  ICF = (2^dI - 1) / (C + 1)^w  with w = 1.
    A patient's test order is the sequence of units met along the tree path (missing value -> larger child);
    units not reached are appended in the static learned order so that every policy yields a full trajectory."""

    def __init__(self, uidx, unit_cost, w=1.0, min_leaf=8, max_depth=8):
        self.uidx, self.cost, self.w, self.min_leaf, self.max_depth = uidx, unit_cost, w, min_leaf, max_depth

    @staticmethod
    def _ent(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.root = self._grow(np.arange(len(self.y)), set(), 0)
        return self

    def _best(self, idx, acquired):
        yn, n = self.y[idx], len(idx)
        base = self._ent(yn.mean()); best = None
        for u, cols in self.uidx.items():
            cu = 0.0 if u in acquired else self.cost[u]
            for j in cols:
                v = self.X[idx, j]; o = np.argsort(v, kind="stable"); vs, ys = v[o], yn[o]
                cs = np.cumsum(ys)
                pos = np.flatnonzero(vs[:-1] != vs[1:])
                pos = pos[(pos + 1 >= self.min_leaf) & (n - (pos + 1) >= self.min_leaf)]
                if len(pos) == 0:
                    continue
                if len(pos) > 25:
                    pos = pos[np.linspace(0, len(pos) - 1, 25).astype(int)]
                nl = pos + 1
                gain = base - (nl / n) * self._ent(cs[pos] / nl) - ((n - nl) / n) * self._ent((cs[-1] - cs[pos]) / (n - nl))
                k = int(np.argmax(gain))
                icf = (2 ** gain[k] - 1) / (cu + 1) ** self.w
                if best is None or icf > best[0]:
                    best = (icf, gain[k], u, j, (vs[pos[k]] + vs[pos[k] + 1]) / 2)
        return best

    def _grow(self, idx, acquired, depth):
        node = {"n": len(idx), "p": float(self.y[idx].mean())}
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf or node["p"] in (0.0, 1.0):
            return node
        b = self._best(idx, acquired)
        if b is None or b[1] <= 1e-6:
            return node
        _, _, u, j, thr = b
        left, right = idx[self.X[idx, j] <= thr], idx[self.X[idx, j] > thr]
        acq = acquired | {u}
        node.update(unit=u, col=j, thr=thr, left=self._grow(left, acq, depth + 1),
                    right=self._grow(right, acq, depth + 1))
        return node

    def order(self, x, fallback):
        node, seq = self.root, []
        while "unit" in node:
            if node["unit"] not in seq:
                seq.append(node["unit"])
            v = x[node["col"]]
            if np.isnan(v):
                node = node["left"] if node["left"]["n"] >= node["right"]["n"] else node["right"]
            else:
                node = node["left"] if v <= node["thr"] else node["right"]
        return seq + [u for u in fallback if u not in seq]


class Sequencer:
    def __init__(self, ds, model, K=25, seed=0, n_pert=25):
        self.ds, self.m, self.K = ds, model, K
        self.rng = np.random.default_rng(seed)
        self.uidx = unit_index(ds)
        self.units = list(ds.units)
        self.ref = model.Xc_ref
        self.z = (self.ref - model.mu) / model.sd
        self.prior_nb = self.rng.choice(len(self.ref), min(K * 2, len(self.ref)), replace=False)
        # perturbed cost vectors (evaluation-time sensitivity analysis)
        prng = np.random.default_rng(12345)
        self.mult = np.exp(prng.uniform(np.log(0.25), np.log(4.0), size=(n_pert, len(self.units))))
        self.n_pert = n_pert

    # -- costs ---------------------------------------------------------------
    def mcost(self, u, paid):
        full, disc = self.ds.cost[u]
        g = self.ds.group[u]
        return disc if (g is not None and g in paid) else full

    def mcostR(self, u, paid):
        i = self.units.index(u)
        return self.mcost(u, paid) * self.mult[:, i]

    def _acquirable(self, x, u):
        return not np.all(np.isnan(x[self.uidx[u]]))

    # -- MI/cost choice for a given partial state ------------------------------
    def _mi_all(self, state, remaining, nb=None):
        """MI estimate of every remaining unit given the partially observed `state`."""
        if nb is None:
            obs = np.where(~np.isnan(state))[0]
            if len(obs) == 0:
                nb = self.prior_nb
            else:
                zx = (state[obs] - self.m.mu[obs]) / self.m.sd[obs]
                d = ((self.z[:, obs] - zx) ** 2).sum(1)
                nb = np.argpartition(d, self.K)[: self.K] if len(d) > self.K else np.arange(len(d))
        refnb = self.ref[nb]
        k = len(nb)
        batch = np.repeat(state[None, :], len(remaining) * k, axis=0)
        for i, c in enumerate(remaining):
            cols = self.uidx[c]
            batch[i * k:(i + 1) * k][:, cols] = refnb[:, cols]
        P = self.m.predict_proba(batch).reshape(len(remaining), k)
        return H(P.mean(1)) - H(P).mean(1)

    def _choose(self, state, remaining, paid):
        mi = self._mi_all(state, remaining)
        mc = np.array([self.mcost(c, paid) for c in remaining])
        return remaining[int(np.argmax(mi / mc))]

    # -- trajectories -------------------------------------------------------------
    def trajectory(self, x, policy, order=None):
        x = np.asarray(x, float)
        state = np.full_like(x, np.nan)
        remaining = list(self.units)
        if policy == "random":
            order = list(self.rng.permutation(self.units))
        if policy == "eg2":
            order = self.eg2.order(x, self.static_ref)
        acq, states, paid = [], [state.copy()], set()
        inc, incR = [0.0], [np.zeros(self.n_pert)]
        while remaining:
            if policy == "adaptive":
                p_now = None
                c = self._choose(state, remaining, paid)
            else:
                c = next(u for u in order if u in remaining)
            remaining.remove(c)
            if not self._acquirable(x, c):
                continue                           # record has no value -> no cost, no info
            cols = self.uidx[c]
            inc.append(self.mcost(c, paid)); incR.append(self.mcostR(c, paid))
            g = self.ds.group[c]
            if g is not None:
                paid.add(g)
            state[cols] = x[cols]
            acq.append(c); states.append(state.copy())
        p = self.m.predict_proba(np.vstack(states))
        return Traj(acq, p, np.cumsum(inc), np.cumsum(np.vstack(incR), axis=0).T)

    # -- global static order ---------------------------------------------------------
    def static_order(self, Xfit, n_rec=60):
        idx = self.rng.choice(len(Xfit), min(n_rec, len(Xfit)), replace=False)
        order, paid = [], set()
        remaining = list(self.units)
        while remaining:
            score = np.zeros(len(remaining))
            for i in idx:
                state = np.full(Xfit.shape[1], np.nan)
                for u in order:
                    state[self.uidx[u]] = Xfit[i, self.uidx[u]]
                mi = self._mi_all(state, remaining)
                mc = np.array([self.mcost(c, paid) for c in remaining])
                score += mi / mc
            c = remaining[int(np.argmax(score))]
            order.append(c); remaining.remove(c)
            g = self.ds.group[c]
            if g is not None:
                paid.add(g)
        return order

    def fit_eg2(self, static_order):
        self.static_ref = list(static_order)
        self.eg2 = EG2Tree(self.uidx, {u: self.ds.cost[u][0] for u in self.units}).fit(self.m.Xc_ref, self.m.y_ref)

    def cost_order(self):
        return sorted(self.units, key=lambda u: self.ds.cost[u][0])


# ------------------------------------------------------------------------------------
# stopping rule evaluation + Learn-then-Test calibration
# ------------------------------------------------------------------------------------
def stop_eval(trajs, y, tau, only_final=False):
    """apply threshold tau; return dict of per-patient arrays"""
    n = len(trajs)
    acc = np.zeros(n, bool); err = np.zeros(n, bool); k = np.zeros(n, int)
    cost = np.zeros(n); costR = None
    pred_fin = np.zeros(n, int)
    R = trajs[0].cumR.shape[0]
    costR = np.zeros((n, R))
    for i, t in enumerate(trajs):
        conf = np.maximum(t.p, 1 - t.p)
        ok = conf >= tau
        if only_final:
            ok[:-1] = False
        hit = np.flatnonzero(ok)
        if len(hit):
            s = hit[0]; acc[i] = True
            err[i] = (t.p[s] >= 0.5) != bool(y[i])
        else:
            s = len(t.p) - 1                       # deferred: full panel already paid
        k[i] = s; cost[i] = t.cum[s]; costR[i] = t.cumR[:, s]
        pred_fin[i] = t.p[-1] >= 0.5
    return dict(acc=acc, err=err, silent=acc & err, k=k, cost=cost, costR=costR,
                pred_final=pred_fin)


TAU_GRID = np.array(sorted(set(list(np.round(np.arange(0.995, 0.495, -0.005), 3))), reverse=True))


def ltt_tau(trajs_cal, y_cal, alpha, delta=0.10, only_final=False, mode="joint"):
    """fixed-sequence LTT over a descending (conservative -> liberal) threshold grid.
    returns (tau_hat, certified_flag). If even the strictest threshold fails -> inf (defer all)."""
    n = len(trajs_cal); y_cal = np.asarray(y_cal)
    tau_hat, ok_any = np.inf, False
    for tau in TAU_GRID:
        sil = stop_eval(trajs_cal, y_cal, tau, only_final)["silent"]
        if mode == "joint":                       # R = P(accepted & wrong)
            pval = binom.cdf(sil.sum(), n, alpha); thr = delta     # valid p for H0: R > alpha
        else:                                     # class-conditional: R1 = P(silent|Y=1), R0 = P(silent|Y=0), both <= alpha
            n1 = int((y_cal == 1).sum()); n0 = n - n1
            p1 = binom.cdf(sil[y_cal == 1].sum(), n1, alpha) if n1 else 1.0
            p0 = binom.cdf(sil[y_cal == 0].sum(), n0, alpha) if n0 else 1.0
            pval = max(p1, p0); thr = delta / 2                    # Bonferroni over the two classes
        if pval <= thr:
            tau_hat, ok_any = tau, True
        else:
            break
    return tau_hat, ok_any

# ============================================================================
# PART 1: BENCHMARK AUDIT
# ============================================================================
"""
Part 1 - Benchmark audit.
For each dataset quantify
  (a) how well the PATTERN of missingness alone predicts the label (mask-only AUC);
  (b) how conventional missing-data handling compares with MCAR-masked training
      under natural test conditions and under an MCAR "deployment shift";
  (c) label circularity: how few acquisition units reach AUC >= 0.99 (greedy);
  (d) leakage candidates (cervical Dx*, mammography BI-RADS).
Outputs -> results/audit_*.csv
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

N_SPLITS, N_REPEATS, SHIFT_Q = 5, 3, 0.30


def mcar_shift(X, uidx, q, rng):
    X = X.copy()
    for u, js in uidx.items():
        m = rng.random(len(X)) < q
        for j in js:
            X[m, j] = np.nan
    return X


def metrics(y, p):
    return roc_auc_score(y, p), accuracy_score(y, p >= 0.5)


def fit_naive(kind, Xtr, ytr, seed):
    if kind == "native":                       # NaN-aware GBM sees missingness directly
        clf = make_hgb(seed).fit(Xtr, ytr)
        return lambda X: clf.predict_proba(X)[:, 1]
    imp = SimpleImputer(strategy="median", add_indicator=(kind == "impute+indicator"),
                        keep_empty_features=True).fit(Xtr)
    clf = make_hgb(seed).fit(imp.transform(Xtr), ytr)
    return lambda X: clf.predict_proba(imp.transform(X))[:, 1]


def audit_models(ds):
    X, y = ds.X.to_numpy(float), ds.y
    uidx = unit_index(ds)
    rows = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=7)
    for f, (tr, te) in enumerate(rskf.split(X, y)):
        rng = np.random.default_rng(1000 + f)
        Xte_shift = mcar_shift(X[te], uidx, SHIFT_Q, rng)
        preds = {k: fit_naive(k, X[tr], y[tr], f) for k in ("native", "impute", "impute+indicator")}
        mh = MaskedHGB(ds, n_aug=8, seed=f).fit(X[tr], y[tr])
        preds["masked-training"] = mh.predict_proba
        for name, fn in preds.items():
            for cond, Xt in (("natural", X[te]), ("MCAR-shift", Xte_shift)):
                auc, acc = metrics(y[te], fn(Xt))
                rows.append(dict(dataset=ds.name, model=name, test=cond, fold=f, auc=auc, acc=acc))
    return pd.DataFrame(rows)


def mask_only(ds):
    X, y = ds.X.to_numpy(float), ds.y
    M = np.isnan(X).astype(float)
    M = M[:, M.std(0) > 0]
    rows = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=7)
    for f, (tr, te) in enumerate(rskf.split(M, y)):
        lr = LogisticRegression(max_iter=2000, C=0.5).fit(M[tr], y[tr])
        gb = make_hgb(f).fit(M[tr], y[tr])
        rows.append(dict(dataset=ds.name, fold=f,
                         auc_lr=roc_auc_score(y[te], lr.predict_proba(M[te])[:, 1]),
                         auc_gb=roc_auc_score(y[te], gb.predict_proba(M[te])[:, 1])))
    return pd.DataFrame(rows)


def unit_cv_auc(ds, units, seed=0):
    cols = sum([ds.units[u] for u in units], [])
    X = ds.X[cols].to_numpy(float)
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ds.y):
        clf = make_hgb(seed).fit(X[tr], ds.y[tr])
        aucs.append(roc_auc_score(ds.y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def circularity(ds, target=0.99):
    """single-unit AUCs and greedy forward selection until CV-AUC >= target"""
    # NOTE: single-unit models here use native-NaN GBM, hence may partly use the
    # missingness of that unit; we therefore also report the same on an
    # imputed copy (see 'imputed' column) to separate signal from artefact.
    singles = {u: unit_cv_auc(ds, [u]) for u in ds.units}
    chosen, path, remaining = [], [], list(ds.units)
    while remaining:
        best = max(remaining, key=lambda u: unit_cv_auc(ds, chosen + [u]))
        chosen.append(best); remaining.remove(best)
        a = unit_cv_auc(ds, chosen)
        path.append(dict(dataset=ds.name, k=len(chosen), added=best, auc=a))
        if a >= target or len(chosen) >= 6:
            break
    return pd.DataFrame([dict(dataset=ds.name, unit=u, auc=a) for u, a in singles.items()]), \
        pd.DataFrame(path)


def audit_main():
    t0 = time.time()
    allm, allmask, sing, paths = [], [], [], []
    for name in list(LOADERS) + list(EXTERNAL_LOADERS):
        ds = load(name)
        print(f"[{name}] n={ds.n} prevalence={ds.y.mean():.3f}", flush=True)
        allm.append(audit_models(ds)); allmask.append(mask_only(ds))
        s, p = circularity(ds); sing.append(s); paths.append(p)
        print(f"   done {time.time()-t0:.0f}s", flush=True)
    pd.concat(allm).to_csv(OUT / "audit_models_folds.csv", index=False)
    pd.concat(allmask).to_csv(OUT / "audit_maskonly_folds.csv", index=False)
    pd.concat(sing).to_csv(OUT / "audit_single_units.csv", index=False)
    pd.concat(paths).to_csv(OUT / "audit_greedy_path.csv", index=False)

    # leakage checks -------------------------------------------------------
    leak = []
    c = load("Cervical").extra["raw"]
    for col in ["Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]:
        a = roc_auc_score(c["Biopsy"], c[col].fillna(0))
        leak.append(dict(dataset="Cervical", variable=col, single_variable_auc=a))
    for col in ["Hinselmann", "Schiller", "Citology"]:
        leak.append(dict(dataset="Cervical", variable=col,
                         single_variable_auc=roc_auc_score(c["Biopsy"], c[col].fillna(0))))
    m = load("Mammography")
    ok = ~np.isnan(m.extra["birads"])
    leak.append(dict(dataset="Mammography", variable="BI-RADS",
                     single_variable_auc=roc_auc_score(m.y[ok], m.extra["birads"][ok])))
    pd.DataFrame(leak).to_csv(OUT / "audit_leakage.csv", index=False)
    print("audit complete", f"{time.time()-t0:.0f}s")



# ============================================================================
# PART 2: SEQUENTIAL EXPERIMENTS
# ============================================================================
"""
Part 2 - Cost-aware sequential testing with risk-controlled stopping.

usage:  python src/run_sequential.py DATASET [n_repeats] [max_splits]
Outputs results/seq_<dataset>_{records,tau,sens}.csv
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
DELTA = 0.10
POLICIES = ["adaptive", "static", "eg2", "cost", "random"]
BETAS = [0.05, 0.10, 0.15, 0.20, 0.30]      # class-conditional silent-error levels
FIXED_TAUS = [0.90, 0.95]
FIT_FRAC = 0.60            # of each training fold; remaining 40% = calibration set


def run(name, n_repeats=3, max_splits=None, n_splits=5, ds=None, scenario="primary"):
    ds = ds if ds is not None else load(name)
    ds = apply_cost_scenario(ds, scenario)
    if ds is None:
        print(f"[{name}] scenario {scenario} not applicable"); return
    tag = "" if scenario == "primary" else f"_{scenario}"
    X, y = ds.X.to_numpy(float), ds.y
    complete = ~np.isnan(X).any(1)
    delayed = ds.extra.get("delayed", set())
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2024)
    rec_rows, tau_rows, sens_rows, dump = [], [], [], []
    t0 = time.time()
    for s, (tr, te) in enumerate(rskf.split(X, y)):
        if max_splits and s >= max_splits:
            break
        fit, cal = train_test_split(tr, test_size=1 - FIT_FRAC, stratify=y[tr], random_state=s)
        model = MaskedHGB(ds, n_aug=8, seed=s).fit(X[fit], y[fit])
        sq = Sequencer(ds, model, K=25, seed=s)
        orders = {"static": sq.static_order(X[fit], n_rec=40), "cost": sq.cost_order()}
        sq.fit_eg2(orders["static"]); orders["eg2"] = None
        traj = {"cal": {}, "te": {}}
        for pol in POLICIES:
            for part, idx in (("cal", cal), ("te", te)):
                traj[part][pol] = [sq.trajectory(X[i], pol, orders.get(pol)) for i in idx]
        # 'full' panel baseline reuses static trajectories but may only stop at the end
        traj["cal"]["full"], traj["te"]["full"] = traj["cal"]["static"], traj["te"]["static"]

        for pol in (POLICIES if scenario == "primary" else []):
            for j, i in enumerate(te):
                tj = traj["te"][pol][j]
                dump.append((s, pol, int(i), int(y[i]), tj.cum, tj.p, list(tj.units)))

        def record(pol, alpha, tau, cert, ev, src=None, mode="joint"):
            src = src or pol
            for j, i in enumerate(te):
                tj = traj["te"][src][j]
                rec_rows.append(dict(
                    dataset=name, split=s, patient=int(i), policy=pol, alpha=alpha, tau=tau, mode=mode,
                    certified=cert, accepted=bool(ev["acc"][j]), err=bool(ev["err"][j]),
                    silent=bool(ev["silent"][j]), cost=float(ev["cost"][j]), k=int(ev["k"][j]),
                    n_delayed=int(sum(u in delayed for u in tj.units[:ev["k"][j]])),
                    acq="|".join(tj.units[:ev["k"][j]]),
                    y=int(y[i]), pred_final=int(ev["pred_final"][j]), complete=bool(complete[i]),
                    full_cost=float(tj.cum[-1])))
            for r in (range(sq.n_pert) if mode == "joint" else []):
                sens_rows.append(dict(dataset=name, split=s, policy=pol, alpha=alpha, r=r,
                                      mean_cost=float(ev["costR"][:, r].mean()),
                                      full_cost=float(np.mean([t.cumR[r, -1] for t in traj["te"][src]]))))

        for pol in POLICIES + ["full"]:
            only_final = pol == "full"
            for a in ALPHAS:
                tau, cert = ltt_tau(traj["cal"][pol], y[cal], a, DELTA, only_final)
                ev = stop_eval(traj["te"][pol], y[te], tau, only_final)
                record(pol, a, tau, cert, ev)
                tau_rows.append(dict(dataset=name, split=s, policy=pol, alpha=a, tau=tau,
                                     certified=cert, n_cal=len(cal)))
        if scenario == "primary":
            for ft in FIXED_TAUS:                          # un-calibrated reference rule
                ev = stop_eval(traj["te"]["adaptive"], y[te], ft)
                record("adaptive_fixed_tau", ft, ft, True, ev, src="adaptive")
            for pol in POLICIES + ["full"]:                # class-conditional certification (per-class silent error <= beta)
                only_final = pol == "full"
                for b in BETAS:
                    tau, cert = ltt_tau(traj["cal"][pol], y[cal], b, DELTA, only_final, mode="class")
                    ev = stop_eval(traj["te"][pol], y[te], tau, only_final)
                    record(pol, b, tau, cert, ev, mode="class")
        print(f"[{name}{tag}] split {s+1}  elapsed {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rec_rows).to_csv(OUT / f"seq_{name}{tag}_records.csv", index=False)
    pd.DataFrame(tau_rows).to_csv(OUT / f"seq_{name}{tag}_tau.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUT / f"seq_{name}{tag}_sens.csv", index=False)
    if scenario == "primary":
        pickle.dump(dump, open(OUT / f"traj_{name}.pkl", "wb"))



# ============================================================================
# TABLES AND FIGURES
# ============================================================================
"""
Aggregate results -> tables (results/*.csv, results/headline.json) and figures (figures/*.png).
Primary alpha per dataset was fixed BEFORE inspecting full results:
    Hepatitis 0.10, CKD 0.02, Mammography 0.10, Cervical 0.02
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binom

DATASETS = ["Hepatitis", "CKD", "Mammography", "Cervical"]
EXTERNAL = ["PBC", "Biopsy", "Heart", "Pima"]
PRIMARY = {"Hepatitis": 0.10, "CKD": 0.02, "Mammography": 0.10, "Cervical": 0.02,
           "PBC": 0.10, "Biopsy": 0.05, "Heart": 0.10, "Pima": 0.10}   # alphas fixed before running those datasets
POL = ["adaptive", "static", "eg2", "cost", "random", "full"]
LAB = {"adaptive": "Adaptive (info. gain / cost)", "static": "Static learned order", "eg2": "EG2 tree (Nunez 1991)",
       "cost": "Cheapest-first", "random": "Random order", "full": "Full panel"}
COL = {"adaptive": "#c0392b", "static": "#2471a3", "eg2": "#1e8449", "cost": "#7d6608", "random": "#7f8c8d", "full": "#000000"}
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def boot_mean(v, n=2000):
    v = np.asarray(v, float)
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


def boot_ratio(num, den, n=2000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = RNG.integers(0, len(num), size=(n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), *np.percentile(r, [2.5, 97.5])


def per_patient(df, cols):
    """average over repeated-CV repetitions -> one value per patient"""
    return df.groupby("patient")[cols].mean()


def summarise(name):
    rec = pd.read_csv(RES / f"seq_{name}_records.csv")
    if "mode" in rec:
        rec = rec[rec["mode"] == "joint"]
    tau = pd.read_csv(RES / f"seq_{name}_tau.csv")
    rows = []
    for (pol, a), g in rec[rec.policy.isin(POL)].groupby(["policy", "alpha"]):
        pp = per_patient(g, ["silent", "accepted", "cost", "full_cost", "k"])
        cm, clo, chi = boot_ratio(pp.cost, pp.full_cost)
        sm, slo, shi = boot_mean(pp.silent)
        am, _, _ = boot_mean(pp.accepted)
        acc_g = g[g.accepted]
        spl = g.groupby("split").silent.mean()
        ss = g.groupby("split").silent.agg(["sum", "count"])
        viol = np.mean([binom.sf(e - 1, c, a) < 0.05 for e, c in zip(ss["sum"], ss["count"])])
        t = tau[(tau.policy == pol) & (tau.alpha == a)]
        defg = g[~g.accepted]
        rows.append(dict(
            dataset=name, policy=pol, alpha=a,
            cost_pct=100 * cm, cost_lo=100 * clo, cost_hi=100 * chi,
            silent=100 * sm, silent_lo=100 * slo, silent_hi=100 * shi,
            accepted_pct=100 * am,
            silent_given_accepted=100 * acc_g.silent.mean() if len(acc_g) else np.nan,
            mean_units_accepted=acc_g.k.mean() if len(acc_g) else np.nan,
            certified_pct=100 * t.certified.mean(),
            split_exceed_pct=100 * (spl > a + 1e-12).mean(),
            split_violation_pct=100 * viol,
            deferred_acc=100 * (defg.pred_final == defg.y).mean() if len(defg) else np.nan,
            n_delayed=g.n_delayed.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"summary_{name}.csv", index=False)
    return rec, out


def paired_vs(rec, name, a):
    """paired split-level comparison of mean cost, adaptive vs each other policy at alpha a"""
    g = rec[(rec.alpha == a) & rec.policy.isin(POL)]
    sc = g.groupby(["split", "policy"]).cost.mean().unstack()
    res = {}
    for other in POL[1:]:
        d = sc["adaptive"] - sc[other]
        try:
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except ValueError:
            p = 1.0
        res[other] = dict(median_diff=float(d.median()), mean_diff=float(d.mean()),
                          adaptive_cheaper_pct=float(100 * (d < 0).mean()), wilcoxon_p=float(p))
    return res


# ---------------------------------------------------------------- budget curves
def fast_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def budget_curves(name, n_budget=18, n_boot=120):
    dump = pickle.load(open(RES / f"traj_{name}.pkl", "rb"))
    df = pd.DataFrame(dump, columns=["split", "policy", "patient", "y", "cum", "p", "units"])
    fullc = np.array([c[-1] for c in df.cum])
    grid = np.linspace(0, np.quantile(fullc, 0.95), n_budget)
    out, store = [], {}
    for pol in POL[:-1]:
        d = df[df.policy == pol]
        P = np.zeros((len(d), n_budget))
        for i, (cum, p) in enumerate(zip(d.cum, d.p)):
            for b, B in enumerate(grid):
                P[i, b] = p[np.searchsorted(cum, B + 1e-9, side="right") - 1]
        store[pol] = (d.patient.to_numpy(), d.y.to_numpy(), P)
        for b, B in enumerate(grid):
            out.append(dict(dataset=name, policy=pol, budget=B, auc=fast_auc(d.y, P[:, b])))
    out = pd.DataFrame(out)
    aubc = {}
    pats = np.unique(store["adaptive"][0])
    for _ in range(n_boot):
        samp = RNG.choice(pats, len(pats))
        cnt = pd.Series(samp).value_counts()
        for pol, (pt, yy, P) in store.items():
            w = cnt.reindex(pt).fillna(0).to_numpy().astype(int)
            idx = np.repeat(np.arange(len(pt)), w)
            a = np.nanmean([fast_auc(yy[idx], P[idx, b]) for b in range(1, n_budget)])
            aubc.setdefault(pol, []).append(a)
    aubc = pd.DataFrame(aubc)
    summ = {}
    for pol in aubc.columns:
        summ[pol] = dict(aubc=float(aubc[pol].mean()), lo=float(aubc[pol].quantile(.025)),
                         hi=float(aubc[pol].quantile(.975)))
        if pol != "adaptive":
            dlt = aubc["adaptive"] - aubc[pol]
            summ[pol].update(diff_vs_adaptive=float(dlt.mean()), diff_lo=float(dlt.quantile(.025)),
                             diff_hi=float(dlt.quantile(.975)))
    out.to_csv(RES / f"budget_{name}.csv", index=False)
    return out, summ, df


# ------------------------------------------------------------------- figures
ALLD = DATASETS + EXTERNAL


def fig_audit():
    m = pd.read_csv(RES / "audit_models_folds.csv")
    mo = pd.read_csv(RES / "audit_maskonly_folds.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    a = ax[0]
    g = mo.groupby("dataset").auc_lr.agg(["mean", "std"]).reindex(ALLD)
    a.bar(range(len(ALLD)), g["mean"], yerr=g["std"], color="#c0392b", capsize=3)
    a.axhline(0.5, ls="--", c="k", lw=0.8)
    a.set_xticks(range(len(ALLD))); a.set_xticklabels(ALLD, rotation=30, ha="right")
    a.set_ylim(0.3, 1.0); a.set_ylabel("AUC (missingness pattern only)")
    a.set_title("(a) Label information in the missingness mask", fontsize=9)
    models = ["impute", "impute+indicator", "native", "masked-training"]
    cols = ["#aab7b8", "#5d6d7e", "#2471a3", "#c0392b"]
    for k, cond in enumerate(["natural", "MCAR-shift"]):
        a = ax[k + 1]
        for j, mod in enumerate(models):
            v = m[(m.model == mod) & (m.test == cond)].groupby("dataset").auc.agg(["mean", "std"]).reindex(ALLD)
            a.bar(np.arange(len(ALLD)) + j * 0.2 - 0.3, v["mean"], 0.2, yerr=v["std"], color=cols[j],
                  capsize=1.5, label=mod)
        a.set_xticks(range(len(ALLD))); a.set_xticklabels(ALLD, rotation=30, ha="right")
        a.set_ylim(0.5, 1.02); a.set_ylabel("Test AUC")
        a.set_title(f"({'bc'[k]}) {'Natural missingness' if k == 0 else '30% extra MCAR unit masking'}", fontsize=9)
    ax[2].legend(fontsize=7, loc="lower left", frameon=False)
    plt.tight_layout(); plt.savefig(FIG / "fig1_audit.png"); plt.close()


def fig_frontier(summ):
    names = [n for n in DATASETS + EXTERNAL if n in summ]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.0))
    for a, name in zip(axs.ravel(), names):
        s = summ[name]
        for pol in POL:
            d = s[s.policy == pol].sort_values("alpha")
            a.plot(d.alpha * 100, d.cost_pct, "-o", ms=2.5, lw=1.4, c=COL[pol], label=LAB[pol])
        a.axvline(PRIMARY[name] * 100, ls=":", c="grey", lw=0.8)
        a.set_title(name, fontsize=9); a.set_xlabel("Certified silent-error level \u03b1 (%)", fontsize=8)
        a.set_ylim(0, 105); a.tick_params(labelsize=7)
    for r in range(2):
        axs[r, 0].set_ylabel("Mean cost (% of full panel)", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(FIG / "fig2_cost_frontier.png"); plt.close()


def fig_budget(curves):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 6.2))
    for a, name in zip(ax.ravel(), DATASETS):
        d = curves[name]
        for pol in POL[:-1]:
            g = d[d.policy == pol]
            a.plot(g.budget, g.auc, lw=1.8, c=COL[pol], label=LAB[pol])
        a.set_title(name, fontsize=10); a.set_xlabel("Test budget per patient (cost units)")
        a.set_ylabel("AUC at budget")
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "fig3_budget_auc.png"); plt.close()


def fig_validity(summ):
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    cols = dict(zip(DATASETS, ["#c0392b", "#2471a3", "#1e8449", "#7d3c98"]))
    for k, pol in enumerate(["adaptive", "full"]):
        a = ax[k]
        for name in DATASETS:
            d = summ[name]; d = d[d.policy == pol].sort_values("alpha")
            a.errorbar(d.alpha * 100, d.silent, yerr=[d.silent - d.silent_lo, d.silent_hi - d.silent],
                       fmt="-o", ms=3, c=cols[name], label=name, lw=1.2, capsize=2)
        a.plot([0, 20], [0, 20], "k--", lw=0.8)
        a.set_xlabel("Target level α (%)"); a.set_ylabel("Observed silent-error rate on test (%)")
        a.set_title(f"({'ab'[k]}) {LAB[pol]}", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(FIG / "fig4_validity.png"); plt.close()


def fig_protocol(rec):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    out = {}
    for a, name in zip(ax, DATASETS):
        g = rec[name]
        g = g[(g.policy == "adaptive") & (g.alpha == PRIMARY[name])]
        n = len(g)
        freq = {}
        for s in g.acq.fillna(""):
            for u in s.split("|"):
                if u:
                    freq[u] = freq.get(u, 0) + 1
        fr = pd.Series(freq).sort_values(ascending=True) / n * 100
        out[name] = fr.sort_values(ascending=False).round(1).to_dict()
        top = fr.tail(10)
        a.barh(top.index, top.values, color="#c0392b")
        a.set_xlim(0, 100); a.set_xlabel("Patients in whom unit was ordered (%)")
        a.set_title(f"{name} (α={PRIMARY[name]*100:g}%)", fontsize=9)
        a.tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig5_protocol.png"); plt.close()
    return out


def extra_tables(avail):
    """class-conditional certification, cost-scenario robustness, external-dataset summary"""
    rows = []
    for name in avail:
        r = pd.read_csv(RES / f"seq_{name}_records.csv")
        if "mode" not in r:
            continue
        for (pol, b), g in r[r["mode"] == "class"].groupby(["policy", "alpha"]):
            rows.append(dict(dataset=name, policy=pol, beta=b,
                             certified_pct=100 * g.groupby("split").certified.first().mean(),
                             accepted_pct=100 * g.accepted.mean(), cost_pct=100 * g.cost.sum() / g.full_cost.sum(),
                             miss_rate_positives_pct=100 * g[g.y == 1].silent.mean(),
                             false_alarm_rate_negatives_pct=100 * g[g.y == 0].silent.mean()))
    pd.DataFrame(rows).to_csv(RES / "classcond_summary.csv", index=False)
    srows = []
    for name in avail:
        for scen in ["primary"] + SCENARIOS:
            f = RES / (f"seq_{name}_records.csv" if scen == "primary" else f"seq_{name}_{scen}_records.csv")
            if not f.exists():
                continue
            r = pd.read_csv(f)
            if "mode" in r:
                r = r[r["mode"] == "joint"]
            g = r[(r.alpha == PRIMARY[name]) & r.policy.isin(POL)]
            for pol, gp in g.groupby("policy"):
                srows.append(dict(dataset=name, scenario=scen, policy=pol, alpha=PRIMARY[name],
                                  cost_pct=100 * gp.cost.sum() / gp.full_cost.sum(), silent_pct=100 * gp.silent.mean()))
    pd.DataFrame(srows).to_csv(RES / "scenario_summary.csv", index=False)



def meta_analysis(avail):
    """Random-effects (DerSimonian-Laird) pooling across datasets of paired per-patient cost differences
    (percentage points of the full-panel cost; negative = first policy cheaper) at each dataset's primary alpha."""
    pairs = [("adaptive", "static"), ("adaptive", "eg2"), ("static", "eg2"), ("adaptive", "cost"),
             ("static", "cost"), ("eg2", "cost"), ("adaptive", "random"), ("static", "random"), ("eg2", "random"),
             ("adaptive", "full"), ("static", "full"), ("eg2", "full")]
    recs = {}
    for name in avail:
        r = pd.read_csv(RES / f"seq_{name}_records.csv")
        recs[name] = r[(r["mode"] == "joint") & (r.alpha == PRIMARY[name])]
    rows = []
    for A, B in pairs:
        eff, se = [], []
        for name in avail:
            r = recs[name]
            a = r[r.policy == A].groupby("patient").cost.mean(); b = r[r.policy == B].groupby("patient").cost.mean()
            sc = r[r.policy == A].groupby("patient").full_cost.mean().mean()
            d = (a - b).dropna()
            eff.append(100 * d.mean() / sc); se.append(max(100 * d.std(ddof=1) / np.sqrt(len(d)) / sc, 1e-6))
        eff, se = np.array(eff), np.array(se)
        w = 1 / se ** 2; fe = (w * eff).sum() / w.sum(); Q = (w * (eff - fe) ** 2).sum(); k = len(eff)
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (se ** 2 + tau2); mu = (wr * eff).sum() / wr.sum(); sm = np.sqrt(1 / wr.sum())
        rows.append(dict(first=A, second=B, k_datasets=k, pooled_diff_pp=mu, ci_lo=mu - 1.96 * sm, ci_hi=mu + 1.96 * sm,
                         tau2=tau2, I2_pct=(max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0),
                         first_cheaper_in=int((eff < 0).sum()), per_dataset="; ".join(f"{n}:{e:.1f}" for n, e in zip(avail, eff))))
    pd.DataFrame(rows).to_csv(RES / "meta_analysis.csv", index=False)
    return pd.DataFrame(rows)


def report_main():
    summ, rec, curves, bsum, paired, extra = {}, {}, {}, {}, {}, {}
    avail = [d for d in DATASETS + EXTERNAL if (RES / f"seq_{d}_records.csv").exists()]
    for name in avail:
        rec[name], summ[name] = summarise(name)
        curves[name], bsum[name], _ = budget_curves(name)
        paired[name] = paired_vs(rec[name], name, PRIMARY[name])
        # ---- secondary analyses at primary alpha
        r = rec[name]; a = PRIMARY[name]
        ad = r[(r.policy == "adaptive") & (r.alpha == a)]
        fu = r[(r.policy == "full") & (r.alpha == a)]
        cc = {}
        for lab, g in (("adaptive", ad), ("full", fu)):
            gc = g[g.complete]
            cc[lab] = dict(n=int(gc.patient.nunique()), cost_pct=float(100 * gc.cost.sum() / gc.full_cost.sum()),
                           silent=float(100 * gc.silent.mean()))
        depth = {}
        acc = ad[ad.accepted]
        for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 99)):
            s = acc[(acc.k >= lo) & (acc.k <= hi)]
            depth[f"{lo}-{hi}"] = dict(n=int(len(s)), error_pct=float(100 * s.err.mean()) if len(s) else None)
        pos = ad[ad.y == 1]; neg = ad[ad.y == 0]
        extra[name] = dict(
            complete_case=cc, depth=depth,
            silent_fn_among_pos_pct=float(100 * (pos.silent).mean()),
            silent_fp_among_neg_pct=float(100 * (neg.silent).mean()),
            deferred_pct=float(100 * (~ad.accepted).mean()),
            deferred_pos_share=float(100 * ad[~ad.accepted].y.mean()) if (~ad.accepted).any() else None,
            prevalence=float(100 * ad.y.mean()))
        # sensitivity to mis-specified costs
        sn = pd.read_csv(RES / f"seq_{name}_sens.csv")
        sn = sn[sn.alpha == a]
        piv = sn.pivot_table(index=["split", "r"], columns="policy", values="mean_cost")
        fullc = sn[sn.policy == "full"].set_index(["split", "r"]).full_cost
        ratio = (piv["adaptive"] / fullc)
        extra[name]["cost_perturb"] = dict(
            adaptive_pct_median=float(100 * ratio.median()),
            adaptive_pct_p2_5=float(100 * ratio.quantile(.025)), adaptive_pct_p97_5=float(100 * ratio.quantile(.975)),
            adaptive_cheaper_than_cost_order_pct=float(100 * (piv["adaptive"] < piv["cost"]).mean()),
            adaptive_cheaper_than_static_pct=float(100 * (piv["adaptive"] < piv["static"]).mean()),
            adaptive_cheaper_than_full_pct=float(100 * (piv["adaptive"] < fullc).mean()))
    if "Mammography" in avail:                      # BI-RADS benchmark
        ds = load("Mammography"); bi = ds.extra["birads"]
        r = rec["Mammography"]
        rows = []
        for a in (0.05, 0.10, 0.15, 0.20):
            g = r[(r.policy == "adaptive") & (r.alpha == a)]
            sens = 1 - g[g.y == 1].silent.mean()
            pred_acc_benign = g.accepted & ((g.y ^ g.err.astype(int)) == 0)
            spec = pred_acc_benign[g.y == 0].mean()
            rows.append(dict(rule=f"Adaptive, alpha={a}", sensitivity=100 * sens, biopsies_avoided_pct=100 * spec,
                             cost_pct=100 * g.cost.sum() / g.full_cost.sum()))
        pts = np.arange(ds.n); ok = ~np.isnan(bi)
        for t in (3, 4, 5):
            pred = bi[ok] >= t; yy = ds.y[ok]
            rows.append(dict(rule=f"BI-RADS >= {t}", sensitivity=100 * pred[yy == 1].mean(),
                             biopsies_avoided_pct=100 * (~pred)[yy == 0].mean(), cost_pct=np.nan))
        pd.DataFrame(rows).to_csv(RES / "mammo_birads_comparison.csv", index=False)
    fig_audit()
    if all(d in avail for d in DATASETS):
        fig_frontier(summ); fig_budget(curves); fig_validity(summ)
        extra["protocol_freq"] = fig_protocol(rec)
    extra_tables(avail)
    meta_analysis(avail)
    json.dump(dict(paired=paired, budget=bsum, extra=extra), open(RES / "headline.json", "w"), indent=1)
    print("report written for", avail)



# ============================================================================
# CROSS-COHORT TRANSPORT TEST: hepatitis (UCI) -> PBC (Mayo)
# ============================================================================
def transport_hep_pbc(n_rep=12, alphas=(0.10, 0.20)):
    """Train + calibrate the whole procedure on hepatitis (shared units only) and apply the frozen
    certificate to PBC.  Compares silent error on held-out hepatitis (exchangeable) vs PBC (shifted)."""
    hep, pbc = load_hepatitis(), load_pbc()
    shared = ["age", "sex", "ascites", "spiders", "liver_big", "bilirubin", "alk_phosphate", "sgot", "albumin"]
    pmap = {"liver_big": "hepato"}
    H = Dataset("HepShared", hep.X[shared].copy(), hep.y, {c: [c] for c in shared},
                {c: hep.cost[c] for c in shared}, {c: hep.group[c] for c in shared})
    PX = pd.DataFrame({c: pbc.X[pmap.get(c, c)] for c in shared})
    XH, XP, yH, yP = H.X.to_numpy(float), PX.to_numpy(float), hep.y, pbc.y
    rows = []
    t0 = time.time()
    for r in range(n_rep):
        tr, te = train_test_split(np.arange(len(yH)), test_size=0.25, stratify=yH, random_state=r)
        fit, cal = train_test_split(tr, test_size=0.4, stratify=yH[tr], random_state=r)
        model = MaskedHGB(H, n_aug=8, seed=r).fit(XH[fit], yH[fit])
        sq = Sequencer(H, model, K=25, seed=r)
        order = sq.static_order(XH[fit], n_rec=40)
        T = {}
        for pol in ("static", "adaptive"):
            o = order if pol == "static" else None
            T[pol] = dict(cal=[sq.trajectory(XH[i], pol, o) for i in cal],
                          hep=[sq.trajectory(XH[i], pol, o) for i in te],
                          pbc=[sq.trajectory(XP[i], pol, o) for i in range(len(yP))])
        T["full"] = T["static"]
        for pol in ("static", "adaptive", "full"):
            only_final = pol == "full"
            for a in alphas:
                tau, cert = ltt_tau(T[pol]["cal"], yH[cal], a, DELTA, only_final)
                for dom, y_, key in (("hepatitis_heldout", yH[te], "hep"), ("PBC", yP, "pbc")):
                    ev = stop_eval(T[pol][key], y_, tau, only_final)
                    pf = np.array([t.p[-1] for t in T[pol][key]])
                    rows.append(dict(rep=r, policy=pol, alpha=a, domain=dom, certified=cert,
                                     accepted=ev["acc"].mean(), silent=ev["silent"].mean(),
                                     cost_pct=100 * ev["cost"].sum() / np.sum([t.cum[-1] for t in T[pol][key]]),
                                     auc_full=roc_auc_score(y_, pf), prevalence=y_.mean()))
        print(f"[transport] rep {r+1}/{n_rep}  {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "transport_hep_pbc.csv", index=False)
    print(out.groupby(["policy", "alpha", "domain"])[["accepted", "silent", "cost_pct", "auc_full"]].mean().round(3))


# ============================================================================
# FLOWCHART (FIGURE 1)
# ============================================================================
def make_flowchart():
    fig, ax = plt.subplots(figsize=(10, 4.4)); ax.set_xlim(0, 103); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f3f4", fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="#222", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p, q, text=None, tx=0, ty=1.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1, color="#222"))
        if text: ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=7, ha="center")

    def line(xs, ys): ax.plot(xs, ys, color="#222", lw=1, solid_capstyle="butt")

    box(2, 44, 22, 10, "Four public datasets\nHepatitis (n=155)\nCKD (n=400)\nMammography (n=961)\nCervical (n=858)")
    box(30, 44, 26, 10, "Part 1 - Benchmark audit\nmissingness-mask AUC; single-unit\nAUC and circularity; leakage\ncandidates; shift robustness")
    box(64, 44, 34, 10, "Masked-training classifier\nmultiple imputation (features only)\n+ random unit-level masks (MCAR)\n-> boosting model for any subset")
    arrow((24, 49), (30, 49)); arrow((56, 49), (64, 49))

    box(64, 29, 34, 10, "Part 2 - Acquire next unit chosen by policy\nAdaptive | Static order | Cheapest-first | Random\n(full panel = acquire all)\ngroup discounts; unavailable value = no information")
    arrow((81, 44), (81, 39))
    box(64, 15, 34, 9, "Update posterior p\nConfidence max(p, 1-p) >= tau ?", fc="#fdfefe")
    arrow((81, 29), (81, 24), "", 0, 0); ax.text(83, 26.2, "acquire, update", fontsize=6.8)
    line([98, 101.5], [19.5, 19.5]); line([101.5, 101.5], [19.5, 34]); arrow((101.5, 34), (98, 34))
    ax.text(100.8, 26.5, "no", fontsize=7, rotation=90, ha="right")

    box(28, 13, 28, 13, "Calibration set (40% of training fold)\nLearn-then-Test, fixed sequence:\nleast conservative tau with\nP(silent error <= alpha) >= 1 - delta\n(delta = 0.10, exact binomial p-values)", fc="#eaf2f8")
    arrow((56, 19.5), (64, 19.5)); ax.text(60, 21.0, "tau-hat", fontsize=7, ha="center")
    box(2, 13, 20, 13, "Fit set (60%)\nclassifier +\nstatic order\ntrained here\n(test fold untouched)", fc="#f9f9f9")
    arrow((22, 19.5), (28, 19.5))

    box(64, 3, 16, 8, "ACCEPT: stop and\nclassify (a wrong\ncall = silent error)", fc="#e8f6ef", fs=7)
    box(84, 3, 14, 8, "DEFER: all units\ntried -> full panel /\nspecialist review", fc="#fdebd0", fs=7)
    arrow((75, 15), (73, 11)); ax.text(77.2, 12.6, "yes", fontsize=7)
    arrow((91, 15), (91, 11)); ax.text(93.5, 12.6, "never reached", fontsize=6.6)

    box(2, 2, 50, 8, "Evaluation: 5-fold CV x 4 repeats; patient-level bootstrap;\ncost as % of full panel, silent-error rate, AUC-vs-budget (AUBC),\ncost-perturbation and complete-case sensitivity", fc="#f9f9f9", fs=7)
    arrow((64, 7), (52, 7)); line([91, 91], [3, 0.8]); line([91, 27], [0.8, 0.8]); arrow((27, 0.8), (27, 2))
    plt.tight_layout(); plt.savefig(FIG / "fig0_flowchart.png", dpi=300); print("ok")


# ============================================================================
# COMMAND LINE
# ============================================================================
def prepare():
    zips = ["mammographic_mass.zip", "cervical_cancer_risk_factors.zip", "ckd.zip", "hepatitis.zip"]
    for z in zips:
        for base in (ROOT, DATA_DIR):
            p = base / z
            if p.exists():
                zipfile.ZipFile(p).extractall(DATA_DIR)
                print("extracted", p)
                break
        else:
            print("NOT FOUND:", z)
    print("data folder contents:", sorted(x.name for x in DATA_DIR.iterdir()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "prepare":
        prepare()
    elif cmd == "audit":
        audit_main()
    elif cmd == "fetch":
        fetch_external()
    elif cmd == "seq":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4,
            int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None,
            scenario=sys.argv[5] if len(sys.argv) > 5 else "primary")
    elif cmd == "scenarios":
        for d in ["Hepatitis", "CKD", "Mammography", "Cervical"]:
            for sc in SCENARIOS:
                run(d, 1, None, scenario=sc)
    elif cmd == "transport":
        transport_hep_pbc()
    elif cmd == "custom":                      # python dx_pipeline.py custom my.csv cfg.json [repeats]
        ds = load_custom(sys.argv[2], sys.argv[3])
        run(ds.name, int(sys.argv[4]) if len(sys.argv) > 4 else 4, ds=ds)
        PRIMARY[ds.name] = float(sys.argv[5]) if len(sys.argv) > 5 else 0.10
        rec, sm = summarise(ds.name)
        print(sm[["policy", "alpha", "cost_pct", "silent", "accepted_pct", "certified_pct"]].round(2).to_string(index=False))
        print("full table: results/summary_%s.csv" % ds.name)
    elif cmd == "report":
        report_main()
    elif cmd == "flowchart":
        make_flowchart()
    elif cmd == "all":
        prepare(); fetch_external(); audit_main()
        for d in ["Mammography", "Hepatitis", "CKD", "Cervical", "PBC", "Biopsy", "Heart", "Pima"]:
            run(d, 4)
        for d in ["Hepatitis", "CKD", "Mammography", "Cervical"]:
            for sc in SCENARIOS:
                run(d, 1, None, scenario=sc)
        transport_hep_pbc(); report_main(); make_flowchart()
    else:
        print(__doc__)
