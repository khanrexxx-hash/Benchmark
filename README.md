# Shortcut-aware benchmark for cost-efficient sequential diagnostic testing

ONE file, `dx_pipeline.py`, reproduces every number, table and figure in the manuscript.
It is a benchmarking / methodology study on retrospective public data. It makes no claim of clinical effectiveness.

## Datasets (8)
Hepatitis, CKD, Mammographic masses, Cervical cancer (your four UCI zips) + Mayo PBC, Wisconsin breast cytology,
Cleveland heart disease, Pima diabetes (downloaded by `fetch`; copies are in data/external/).
Published test prices (Turney 1995, Ontario fee schedule): Hepatitis, Heart, Pima. All other prices are ASSUMED tiers
(see the data section of dx_pipeline.py) and are stress-tested by cost scenarios.

## 1. Install
    python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## 2. Put in ONE folder
dx_pipeline.py + mammographic_mass.zip, cervical_cancer_risk_factors.zip, ckd.zip, hepatitis.zip

## 3. Run
    python dx_pipeline.py all          # everything (about 100 minutes on one CPU)

Step by step:
    python dx_pipeline.py prepare                  # unzip the four UCI archives into ./data
    python dx_pipeline.py fetch                    # download PBC, biopsy, heart, Pima into ./data/external
    python dx_pipeline.py audit                    # missingness / shift audit (8 datasets, ~6 min)
    python dx_pipeline.py seq Hepatitis 4          # main experiment, 4 CV repeats; same for CKD, Mammography,
    python dx_pipeline.py seq Cervical 4           #   Cervical, PBC, Biopsy, Heart, Pima (3-6 min each)
    python dx_pipeline.py scenarios                # cost scenarios on the 4 main datasets (~12 min)
    python dx_pipeline.py transport                # hepatitis -> PBC transport test (~8 min)
    python dx_pipeline.py report                   # tables (results/), figures (figures/), meta-analysis
    python dx_pipeline.py flowchart                # framework diagram
    python dx_pipeline.py seq Hepatitis 1 1        # 10-second smoke test

## 4. YOUR OWN cohort / YOUR OWN fee schedule
    python dx_pipeline.py custom my.csv config.json 4 0.10
config.json (see custom_config_example.json): target column + positive label, acquisition units (columns acquired together),
[full, discounted] cost per unit, optional shared-collection groups. Output: results/summary_<name>.csv.

## 5. Outputs
results/: summary_*.csv, meta_analysis.csv (random-effects pooling), scenario_summary.csv, classcond_summary.csv,
          transport_hep_pbc.csv, audit_*.csv, headline.json
figures/: fig0_flowchart (paper Fig. 1), fig1_audit (Fig. 2), fig2_cost_frontier (Fig. 3); fig3-fig5 extra

## 6. Notes
- Primary alpha per dataset was fixed before inspecting results: Hepatitis .10, CKD .02, Mammography .10, Cervical .02,
  PBC .10, Biopsy .05, Heart .10, Pima .10. For Cervical, alpha >= 10% is vacuous (prevalence 6.4%).
- Heart data: GitHub copy of the Cleveland file; the target coding is verified from the data; ca = 4 and thal = 0 are set
  to missing. Only the 2.10 blood-collection discount is applied to the heart prices (exercise/imaging group discounts
  could not be verified, so none are applied: conservative).
- Class-conditional certification needs roughly 90 positives (beta = 0.10).
- Seeds are fixed. Library versions in requirements.txt are those used; others may change the third decimal.
