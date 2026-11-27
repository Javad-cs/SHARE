#!/usr/bin/env Rscript
# cue_bar_report_csv.R
# ----------------
# Computes pooled target-country attribution rates and writes a structured report.
# All aggregations use sum(target-country attributions) /
# sum(successful responses).
#
# Output: results/analysis/bars/cue_bar_report.txt
#
# Usage:
#   Rscript scripts/cue_bar_report_csv.R
#   Rscript scripts/cue_bar_report_csv.R --model gemini
#   Rscript scripts/cue_bar_report_csv.R --cluster pilaf_cl

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
})

# ─────────────────────────────────────────────
# CLI ARGS
# ─────────────────────────────────────────────

args    <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) > 0 && length(args) >= i[1] + 1) args[i[1] + 1] else default
}

opt_model      <- get_arg("--model")
opt_cluster    <- get_arg("--cluster")
opt_results_csv <- get_arg("--results-csv",   "results/analysis/all_results.csv")
opt_output     <- get_arg("--output-dir",    "results/analysis/bars")
opt_exclusions <- get_arg("--exclusions",    "notes/cue_exclusions.json")
opt_aliases    <- get_arg("--aliases",       "notes/aliases.json")
opt_clusters   <- get_arg("--clusters-json", "notes/clusters.json")

ALL_MODELS   <- c("gemini", "gpt41", "qwen3vl", "gemma3")
ALL_CLUSTERS <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
VISUAL_CUES  <- c("flag", "currency", "sign")
TEXTUAL_CUES <- c("native", "endonym", "perspective")
ALL_CUES     <- c(VISUAL_CUES, TEXTUAL_CUES)

REPORT_CUE_ORDER <- c(
  "flag",
  "currency",
  "sign",
  "endonym",
  "perspective",
  "native"
)

models   <- if (!is.null(opt_model))   opt_model   else ALL_MODELS
clusters <- if (!is.null(opt_cluster)) opt_cluster else ALL_CLUSTERS

CUE_LABELS <- c(
  flag        = "Flag",
  currency    = "Banknote",
  sign        = "Reservation sign",
  native      = "Native-language prompt",
  endonym     = "Endonym prompt",
  perspective = "Perspective prompt"
)

CLUSTER_NAMES <- c(
  pilaf_cl = "Pilaf", mantou_cl = "Mantou",
  samosa_cl = "Samosa", meatball_cl = "Meatball"
)

MODEL_LABELS <- c(
  gemini = "Gemini", gpt41 = "GPT-4.1",
  qwen3vl = "Qwen3-VL", gemma3 = "Gemma3"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

build_lookup <- function(aliases_path) {
  raw <- fromJSON(aliases_path, simplifyVector = FALSE)
  lookup <- list()
  for (canonical in names(raw)) {
    if (startsWith(canonical, "_")) next

    # Ensure the canonical label itself is always recognized.
    lookup[[trimws(tolower(canonical))]] <- canonical

    for (variant in raw[[canonical]])
      lookup[[trimws(tolower(variant))]] <- canonical
  }
  lookup
}

normalize_country <- function(name, lookup) {
  if (is.null(name) || is.na(name) || nchar(trimws(name)) == 0) return("other")
  result <- lookup[[trimws(tolower(name))]]
  if (is.null(result)) "other" else result
}

# Convert a condition/path identifier such as "south_korea" or "korea"
# into the same canonical label used for normalized model predictions.
normalize_target_country <- function(country_id, lookup) {
  readable_name <- gsub("_", " ", country_id, fixed = TRUE)
  normalize_country(readable_name, lookup)
}

load_exclusions <- function(path, model, cluster_id) {
  if (!file.exists(path)) return(list())

  data <- fromJSON(path, simplifyVector = TRUE)
  excl <- data$exclusions

  if (is.null(excl) || nrow(excl) == 0) return(list())

  # Missing cluster_id means the exclusion is global.
  if (!"cluster_id" %in% names(excl)) {
    excl$cluster_id <- NA_character_
  }

  model_key   <- trimws(tolower(model))
  cluster_key <- trimws(tolower(cluster_id))

  entry_models <- trimws(tolower(as.character(excl$model)))
  entry_clusters <- trimws(tolower(as.character(excl$cluster_id)))
  entry_clusters[is.na(entry_clusters)] <- ""

  keep <- entry_models == model_key & (
    entry_clusters == "" |
    entry_clusters %in% c("*", "all") |
    entry_clusters == cluster_key
  )

  excl <- excl[keep, , drop = FALSE]

  if (nrow(excl) == 0) return(list())

  lapply(seq_len(nrow(excl)), function(i) {
    c(
      trimws(tolower(excl$cue[i])),
      trimws(tolower(excl$country[i]))
    )
  })
}

is_excluded <- function(cue, country, exclusions) {
  if (length(exclusions) == 0) return(FALSE)
  key <- c(trimws(tolower(cue)), trimws(tolower(country)))
  any(sapply(exclusions, function(e) identical(e, key)))
}

count_correct <- function(results_df, cluster_id, model,
                          visual_cue, visual_country,
                          textual_cue_type, textual_country,
                          target_country, lookup) {
  sub <- results_df[
    results_df$cluster_id == cluster_id &
    results_df$model == model &
    results_df$visual_cue == visual_cue &
    results_df$visual_country == visual_country &
    results_df$experiment == "country_classification" &
    results_df$textual_cue_type == textual_cue_type &
    results_df$textual_country == textual_country,
    , drop = FALSE
  ]

  if (nrow(sub) == 0) return(list(correct = 0L, total = 0L))

  successful <- sub$success %in% c(TRUE, "True", "TRUE", "true", 1, "1")
  sub <- sub[successful, , drop = FALSE]
  if (nrow(sub) == 0) return(list(correct = 0L, total = 0L))

  normalized <- vapply(
    sub$primary_country,
    function(x) normalize_country(x, lookup),
    character(1)
  )

  canonical_target <- normalize_target_country(target_country, lookup)

  if (canonical_target == "other") {
    stop(sprintf(
      "Target country is missing from aliases.json: %s",
      target_country
    ))
  }

  list(
    correct = as.integer(sum(normalized == canonical_target)),
    total   = as.integer(nrow(sub))
  )
}

# ─────────────────────────────────────────────
# COMPUTE SCORES
# returns data frame with columns:
#   model, cluster, cue, correct, total
# ─────────────────────────────────────────────

compute_all_scores <- function(lookup, cl_raw, results_df) {
  rows <- list()

  for (cluster_id in clusters) {
    cl_entry <- Filter(function(x) x$cluster_id == cluster_id, cl_raw$clusters)
    if (length(cl_entry) == 0) next
    countries <- tolower(gsub(" ", "_",
                              sapply(cl_entry[[1]]$countries,
                                     function(c) c$country)))

    canonical_countries <- vapply(
      countries,
      function(country) normalize_target_country(country, lookup),
      character(1)
    )

    unmapped <- countries[canonical_countries == "other"]
    if (length(unmapped) > 0) {
      stop(sprintf(
        "Cluster %s contains target countries missing from aliases.json: %s",
        cluster_id, paste(unmapped, collapse = ", ")
      ))
    }

    cat(sprintf(
      "  Target mappings for %s: %s\n",
      cluster_id,
      paste(sprintf("%s -> %s", countries, canonical_countries),
            collapse = ", ")
    ))

    for (model in models) {
      cat(sprintf("  Computing %s / %s ...\n", model, cluster_id))
      exclusions <- load_exclusions(opt_exclusions, model, cluster_id)

      # visual cues
      for (vcue in VISUAL_CUES) {
        total_correct <- 0L
        total_denom   <- 0L
        for (country in countries) {
          if (is_excluded(vcue, country, exclusions)) next
          res <- count_correct(
            results_df, cluster_id, model,
            vcue, country, "en", "none",
            country, lookup
          )
          total_correct <- total_correct + res$correct
          total_denom   <- total_denom   + res$total
        }
        rows[[length(rows) + 1]] <- data.frame(
          model     = model,
          cluster   = cluster_id,
          cue       = vcue,
          cue_type  = "visual",
          correct   = total_correct,
          total     = total_denom,
          stringsAsFactors = FALSE
        )
      }

      # textual cues
      for (tcue in TEXTUAL_CUES) {
        total_correct <- 0L
        total_denom   <- 0L
        for (country in countries) {
          if (is_excluded(tcue, country, exclusions)) next
          res <- count_correct(
            results_df, cluster_id, model,
            "neutral", "none", tcue, country,
            country, lookup
          )
          total_correct <- total_correct + res$correct
          total_denom   <- total_denom   + res$total
        }
        rows[[length(rows) + 1]] <- data.frame(
          model     = model,
          cluster   = cluster_id,
          cue       = tcue,
          cue_type  = "textual",
          correct   = total_correct,
          total     = total_denom,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  bind_rows(rows)
}

# ─────────────────────────────────────────────
# REPORT HELPERS
# ─────────────────────────────────────────────

SEP_HEAVY <- paste0(strrep("=", 60), "\n")
SEP_LIGHT <- paste0(strrep("-", 60), "\n")

w  <- function(...) cat(..., file = con)
wl <- function(x = "") w(x, "\n")
ws <- function() wl(SEP_LIGHT)
wh <- function() wl(SEP_HEAVY)

pct <- function(correct, total) {
  if (total == 0) "N/A" else sprintf("%.1f%%", 100 * correct / total)
}

agg_rate <- function(df, ...) {
  # aggregate using (sum correct) / (sum total)
  df %>%
    group_by(...) %>%
    summarise(correct = sum(correct),
              total   = sum(total),
              rate    = if (sum(total) > 0) 100 * sum(correct) / sum(total)
                        else NA_real_,
              .groups = "drop")
}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

cat("Loading aliases...\n")
lookup <- build_lookup(opt_aliases)

cat("Loading all_results.csv...\n")
if (!file.exists(opt_results_csv)) {
  stop(sprintf("Results CSV not found: %s", opt_results_csv))
}
results_df <- read.csv(
  opt_results_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8"
)

required_columns <- c(
  "cluster_id", "model", "visual_cue", "visual_country",
  "textual_cue_type", "textual_country", "experiment",
  "success", "primary_country"
)
missing_columns <- setdiff(required_columns, names(results_df))
if (length(missing_columns) > 0) {
  stop(sprintf(
    "Results CSV is missing required columns: %s",
    paste(missing_columns, collapse = ", ")
  ))
}

cl_raw <- fromJSON(opt_clusters, simplifyVector = FALSE)
dir.create(opt_output, showWarnings = FALSE, recursive = TRUE)

cat("Computing scores...\n")
df <- compute_all_scores(lookup, cl_raw, results_df)

report_path <- file.path(opt_output, "cue_bar_report.txt")
con <- file(report_path, open = "wt")

wh()
wl("  TARGET-COUNTRY ATTRIBUTION REPORT")
wl(paste0("  Models  : ", paste(models,   collapse = ", ")))
wl(paste0("  Clusters: ", paste(clusters, collapse = ", ")))
wl("  Rate    : target-country attributions / successful responses")
wl("  Visual  : cue-added image with English prompt")
wl("  Textual : neutral image with target textual prompt")
wl("  Excluded cue-country conditions are removed")
wh()

# ══════════════════════════════════════════════
# LEVEL 1 — GLOBAL
# ══════════════════════════════════════════════

wl()
wl("LEVEL 1 - GLOBAL POOLED TARGET-COUNTRY ATTRIBUTION RATE")
ws()
wl(sprintf(
  "  %-26s  %s",
  "CUE",
  "RATE (target attributions / successful responses)"
))
wl(paste0("  ", strrep("-", 40)))

global <- agg_rate(df, cue)
global$cue <- factor(global$cue, levels = REPORT_CUE_ORDER)
global <- global[order(global$cue), ]

for (i in seq_len(nrow(global))) {
  cue_name <- CUE_LABELS[as.character(global$cue[i])]
  wl(sprintf("  %-26s  %s  (%d/%d)",
             cue_name,
             pct(global$correct[i], global$total[i]),
             global$correct[i], global$total[i]))
}

# ══════════════════════════════════════════════
# LEVEL 2 — PER MODEL
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 2 — PER MODEL (pooled across clusters)")
ws()

per_model <- agg_rate(df, model, cue)

for (model in models) {
  wl(sprintf("  %s:", MODEL_LABELS[model]))
  sub <- per_model[per_model$model == model, ]
  sub$cue <- factor(sub$cue, levels = REPORT_CUE_ORDER)
  sub <- sub[order(sub$cue), ]
  for (i in seq_len(nrow(sub))) {
    cue_name <- CUE_LABELS[as.character(sub$cue[i])]
    wl(sprintf("    %-26s  %s  (%d/%d)",
               cue_name,
               pct(sub$correct[i], sub$total[i]),
               sub$correct[i], sub$total[i]))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 3 — PER CLUSTER
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 3 — PER CLUSTER (pooled across models)")
ws()

per_cluster <- agg_rate(df, cluster, cue)

for (cluster_id in clusters) {
  wl(sprintf("  %s:", CLUSTER_NAMES[cluster_id]))
  sub <- per_cluster[per_cluster$cluster == cluster_id, ]
  sub$cue <- factor(sub$cue, levels = REPORT_CUE_ORDER)
  sub <- sub[order(sub$cue), ]
  for (i in seq_len(nrow(sub))) {
    cue_name <- CUE_LABELS[as.character(sub$cue[i])]
    wl(sprintf("    %-26s  %s  (%d/%d)",
               cue_name,
               pct(sub$correct[i], sub$total[i]),
               sub$correct[i], sub$total[i]))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 4 — PER MODEL x CLUSTER
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 4 — PER MODEL x CLUSTER")
ws()

for (model in models) {
  wl(sprintf("  %s:", MODEL_LABELS[model]))
  for (cluster_id in clusters) {
    sub <- df[df$model == model & df$cluster == cluster_id, ]
    sub$cue <- factor(sub$cue, levels = REPORT_CUE_ORDER)
    sub <- sub[order(sub$cue), ]
    parts <- sapply(seq_len(nrow(sub)), function(i) {
      sprintf(
        "%s=%s (%d/%d)",
        CUE_LABELS[as.character(sub$cue[i])],
        pct(sub$correct[i], sub$total[i]),
        sub$correct[i],
        sub$total[i]
      )
    })
    wl(sprintf("    %-10s  %s",
               CLUSTER_NAMES[cluster_id],
               paste(parts, collapse = "  ")))
  }
  wl()
}

wh()
wl(paste0("  Report generated: ", Sys.time()))
wh()

close(con)
cat(sprintf("[OK] Report saved: %s\n", report_path))