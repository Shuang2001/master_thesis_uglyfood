# uglyfood-thesis

## Overview

This repository contains the data, scripts, and outputs for a master's thesis investigating how anthropomorphic framing affects purchase intention toward cosmetically imperfect ("ugly") produce among Chinese consumers.

The project consists of two components:

- **NLP-driven systematic literature review**:  
  84 peer-reviewed papers (2017–2026) were processed using sentence-transformer embeddings (all-MiniLM-L6-v2), UMAP dimensionality reduction, and BERTopic topic modelling. This pipeline maps the thematic structure of the ugly food research field and identifies research gaps at the corpus level.

- **Between-subjects experiment (N = 200)**:  
  Participants viewed either an anthropomorphic or a descriptive label applied to imperfect apples. The design tests whether sympathy mediates the effect of framing on purchase intention (PROCESS Model 4), and whether food waste awareness moderates this mediation along the sympathy-to-intention path (PROCESS Model 14).

---

## Repository Structure

The `scripts/` directory is organised by task:

- **NLP pipeline scripts**:  
  Handle corpus ingestion, abstract embedding, UMAP projection, and BERTopic modelling. These scripts also generate bibliometric outputs, including keyword frequency, co-occurrence networks, and geographic distribution figures.

- **Experiment analysis scripts**:  
  Cover data cleaning and exclusion (completion-time filtering, attention checks), scale reliability testing, and inferential analysis using `pingouin` and `statsmodels`. Moderated mediation is implemented via a PROCESS-equivalent bootstrap procedure.

- **Visualisation scripts**:  
  Generate all figures using `matplotlib` with a fixed colour palette.

- **scripts/office/**:  
  Utilities for unpacking, editing, and repacking `.docx` files used in thesis document production.


## Repository structure

### `data/`
Contains the datasets used in the project.

- `data/raw/`
  - raw exported bibliometric data
  - geographic base files used for map visualization

- `data/processed/`
  - cleaned datasets
  - labeled samples
  - subset files
  - BERTopic outputs
  - keyword co-occurrence tables
  - final analysis datasets

### `scripts/`
Contains all Python scripts used for data cleaning, analysis, modeling, and figure generation.

- `scripts/NLP/Data_Cleaning/`
  - data cleaning
  - subset construction
  - labeling sample generation
  - model training
  - bibliometric and topic analyses

- `scripts/NLP/Data_Visualization/`
  - scripts for generating figures for the literature review section

- `scripts/Experiment/`
  - scripts for generating figures for the experimental section

### `outputs/`
Contains generated outputs.

- `outputs/figures/literature_review/`
  - figures for the literature review and bibliometric/NLP analysis

- `outputs/figures/experiment/`
  - figures for the experimental study

- `outputs/tables/`
  - output tables used in analysis

### `manuscript/`
Contains the thesis manuscript file.

## Main data files

Examples of key processed files include:

- `df_clean.csv`: cleaned bibliometric dataset
- `df_clean_en.csv`: cleaned English version of the dataset
- `consumer_ugly_subset.csv`: subset related to consumer and ugly food topics
- `consumer_ugly_core_subset.csv`: core subset used for more focused analysis
- `core_sample_for_labeling.csv`: labeled sample for model-related screening tasks
- `core_ranked_by_model.csv`: model-ranked records
- `final_dataset.csv`: final processed dataset used in later analyses
- `bertopic_doc_topics.csv`: document-topic assignments from BERTopic
- `bertopic_topic_info.csv`: BERTopic topic summary
- `bertopic_topic_info_final.csv`: refined BERTopic topic summary

## Main scripts

Examples of important scripts include:

### NLP / data preparation and analysis
- `clean_scopus.py`
- `filter_consumer_subset.py`
- `make_label_sample.py`
- `make_core_label_sample.py`
- `rename_columns.py`
- `train_model.py`
- `analysis_trend.py`
- `analysis_keywords.py`
- `analysis_cooccurrence.py`
- `analysis_bertopic.py`
- `analysis_journal.py`
- `analysis_theory_gap.py`
- `analysis_network_metrics.py`

### NLP / visualization
- `fig1_prisma.py`
- `fig2_trend.py`
- `fig3_global_map_v2.py`
- `fig4_pie_charts.py`
- `fig5_journals.py`
- `fig6_keywords.py`
- `fig7_cooccurrence.py`
- `fig8_bertopic.py`
- `fig9_theory_gap.py`
- `figure_style.py`

### Experiment / visualization
- `fig51_demographic.py`
- `fig52_dataquality.py`
- `fig53_reliability.py`
- `fig54_items.py`
- `fig55_manip.py`
- `fig56_h1_pi.py`
- `fig57_sympathy.py`
- `fig58_mediation.py`
- `fig59_items_ef.py`
- `fig510_momediation.py`
- `fig511_summary.py`
- `fig512_fullmodel.py`
- `fig513_jn.py`

## Workflow overview

The project consists of two main parts.

### Part 1: Literature review and NLP-based analysis
1. Raw bibliometric data are stored in `data/raw/`
2. Data cleaning and preparation are conducted in `scripts/NLP/Data_Cleaning/`
3. Topic modeling, keyword analysis, co-occurrence analysis, and related analyses are run through the NLP scripts
4. Visual outputs are generated through `scripts/NLP/Data_Visualization/`
5. Final figures are saved in `outputs/figures/literature_review/`

### Part 2: Experimental study
1. Experimental results are stored in the project outputs/tables or related processed files
2. Experimental figures are generated using scripts in `scripts/Experiment/`
3. Final figures are saved in `outputs/figures/experiment/`

## Notes

- Some raw data may be subject to database access restrictions and may not be freely redistributable.
- Geographic shapefiles in `data/raw/geodata/` are used only for visualization purposes.
- This repository is intended to document the thesis workflow, analysis process, and figure generation pipeline.

## Author

Shuang Wu
Master Thesis Project
