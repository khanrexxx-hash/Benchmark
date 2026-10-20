# Shortcut-aware benchmark for cost-efficient sequential diagnostic testing

One file, `dx_pipeline.py`, reproduces every number, table and figure in the manuscript, including the analyses added
in revision. This is a benchmarking and methodology study on retrospective public data. All costs are modelled, and it
makes no claim of clinical effectiveness or safety.

## 1. Install
    python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt                        # exact versions used; others may change the third decimal

## 2. Data
Put the four UCI archives next to `dx_pipeline.py`: `hepatitis.zip`, `ckd.zip`, `mammographic_mass.zip`,
`cervical_cancer_risk_factors.zip` (UCI Machine Learning Repository: Hepatitis, Chronic Kidney Disease, Mammographic
Mass, Cervical cancer (Risk Factors)). PBC, Biopsy, Heart and Pima are downloaded by `fetch` (URLs in the code).
Check file integrity after `prepare` and `fetch` with `sha256sum -c data_checksums_sha256.txt`.
No patient data are stored in this repository.

## 3. Run everything
    python dx_pipeline.py all          

Or step by step:

| Command | Produces | Manuscript |
|---|---|---|
| `prepare`, `fetch` | `data/` | Section 2 |
| `audit` | `results/audit_*.csv`, `figures/fig1_audit.png` (via `report`) | Table 2, Fig. 2 |
| `seq <Dataset> 4` (8 datasets) | `results/seq_<Dataset>_*` | Tables 3, 7; Fig. 3 |
| `scenarios` | `results/seq_<Dataset>_<scenario>_*` | Section 4.3 |
| `transport` | `results/transport_hep_pbc.csv` | Section 4.5 |
| `report` | `results/summary_*.csv`, `meta_analysis.csv`, `scenario_summary.csv`, `classcond_summary.csv`, `headline.json`, `figures/fig1-5` | Tables 3, 7; Figs. 2-3; Sections 4.2-4.5 |
| `revision` | `results/revision/*.csv`, `results/revision/tables/` | Tables 4, 5, 6, A1; Section 4.2 |
| `labelx CKD 4`, `labelx Cervical 4` | `results/seq_*_noLabelTests_*` | Section 4.5 |
| `labelx_summary` | `results/revision/label_exclusion_summary.csv`, `meta_reduced_CKD_Cervical.csv` | Section 4.5 |
| `costmap`, `checksums` | `cost_unit_map.csv`, `data_checksums_sha256.txt` | Section 2, Data availability |
| `seq Hepatitis 1 1` | quick smoke test (about 2 minutes) | |

## 4. Your own cohort and fee schedule
    python dx_pipeline.py custom my.csv config.json 4 0.10
See `custom_config_example.json` (target column, acquisition units, [full, discounted] cost per unit, optional
shared-collection groups).

## 5. Contents of this repository
- `results/`: summary-level outputs (tables, split-level thresholds and cost-perturbation results, meta-analysis,
  audit). Per-patient test-fold records (`seq_*_records.csv`, `traj_*.pkl`, about 340 MB) are not stored here;
  `dx_pipeline.py` regenerates them, and they are archived with the versioned release.
- `results/revision/`: analyses added in revision (dependence-corrected comparisons with Holm adjustment, held-out risk
  and class-specific errors, certification counts, meta-analysis sensitivity, label-exclusion runs);
  `results/revision/tables/` holds formatted Tables 4, 5, 6 and A1.
- `figures/`: figures produced by `report` and `flowchart` (`fig1_audit` = Fig. 2, `fig2_cost_frontier` = Fig. 3).
  `Fig1_framework_manuscript.png` is the designed framework figure used as Fig. 1 in the manuscript; `fig0_flowchart`
  is a simpler code-generated version.
- `cost_unit_map.csv`: every acquisition unit, its variables, full and discounted price, shared-collection group, price
  provenance (historical Turney 1995 or assumed) and missingness, plus every excluded variable and the reason.

## 6. Notes
- Seeds are fixed (CV 2024; audit 7; fit/calibration split and model seed = split index; cost perturbation 12345;
  revision bootstrap 20261020).
- Primary alpha per dataset was fixed before inspecting results: Hepatitis .10, CKD .02, Mammography .10, Cervical .02,
  PBC .10, Biopsy .05, Heart .10, Pima .10.
