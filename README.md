# SHARE

Code, prompts, and result files for

**When Context Overrides Content: Measuring Cultural Cue Sensitivity in Vision-Language Models**
Javad Ismayilzada and Jong C. Park, School of Computing, KAIST.
Findings of the Association for Computational Linguistics: EACL 2027.

Dataset: https://huggingface.co/datasets/fooddataset/SHARE

Four food clusters (`pilaf_cl`, `mantou_cl`, `samosa_cl`, `meatball_cl`) are
evaluated across four models (`gemini`, `gpt41`, `qwen3vl`, `gemma3`) under
visual cues (flag, banknote, reservation sign) and textual cues.

## Contents

```
notes/        cluster definitions, country name aliases, excluded conditions
prompts/      prompt templates used for evaluation and validation
scripts/      pipeline (Python) and figure generation (R)
data/         CLIP embeddings and pipeline visualizations
results/      model outputs, reports, and figures
robustness/   bootstrap, exclusion-sensitivity and repeat-stability analyses
```

## Not in this repository

The source photographs and `results/analysis/all_results.csv` (159 MB) are
distributed with the dataset. Place the CSV at `results/analysis/all_results.csv`
and the images under `data/share_upload/` before running the analysis scripts.

## Credentials

Not included. Supply your own before running `evaluate.py` or the judging
scripts:

```
notes/gemini_api_key.txt      # evaluate.py --model gemini
notes/openai_api_key.txt      # evaluate.py --model gpt41
ANTHROPIC_API_KEY             # env var, for transparency_check.py
```

These paths are listed in `.gitignore`. `qwen3vl` and `gemma3` run locally via
`transformers`.

## Pipeline order

Image collection and cleaning:

```
scrape.py -> filter.py -> manual_filter.py -> curate.py -> sample.py
```

Image normalization and cue insertion:

```
normalize.py -> materialize_normalized.py -> extract_dish.py
build_neutral.py / build_flag.py / build_currency.py / build_sign.py
```

Evaluation and aggregation:

```
evaluate.py -> repair_truncated.py -> normalize_raw_strings.py
           -> apply_normalization_map.py -> build_all_results_csv.py
```

Analysis and figures:

```
cue_bar_csv.R          -> results/analysis/bars/
flip_bar_csv.R         -> results/analysis/lang_robust/
clash_matrix_csv.R     -> results/analysis/heatmaps/clash/
transparency_check.py -> transparency_bar.R -> results/analysis/transparency/
clip_viz.py            -> data/analysis/
```

The `*_report_csv.R` scripts write the matching `.txt` summaries.

`robustness/` holds the additional analyses reported in the paper: trial-level
and modal-cell bootstraps, sensitivity to the exclusion set, baseline
concentration, isolated cue validation, and a repeat-stability run against the
API models.

## Notes

- All paths are relative to the repository root. Run scripts from there.
- `evaluate.py` defaults to `--device cuda:1`; change as needed.
- `manual_filter.py` and `interactive_mask.py` are local Flask review tools
  (default `http://localhost:5000`).

## License

CC BY-NC 4.0, covering the material created by the authors: the compositing and
normalization pipeline, the inserted cues, the prompts, the annotations, and the
analysis code. It does not cover the underlying photographs. See `LICENSE`.

## Citation

```bibtex
@inproceedings{ismayilzada2027share,
  title     = {When Context Overrides Content: Measuring Cultural Cue
               Sensitivity in Vision-Language Models},
  author    = {Ismayilzada, Javad and Park, Jong C.},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
