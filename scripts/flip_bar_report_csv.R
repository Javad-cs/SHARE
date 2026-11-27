#!/usr/bin/env Rscript
# flip_bar_report_csv.R
# ---------------------
# Computes English-to-native prompt country-attribution flip rates
# by visual condition and writes a structured report.
#
# Flip = differing normalized country attributions between paired
#        successful English and native-prompt evaluations for the
#        same image and visual condition.
#
# Aggregations use sum(flips) / sum(valid paired evaluations).
#
# Output: results/analysis/lang_robust/flip_bar_report.txt
#
# Usage:
#   Rscript scripts/flip_bar_report_csv.R
#   Rscript scripts/flip_bar_report_csv.R --model gemini
#   Rscript scripts/flip_bar_report_csv.R --cluster pilaf_cl

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
opt_results_csv <- get_arg("--results-csv", "results/analysis/all_results.csv")
opt_output <- get_arg("--output-dir", "results/analysis/lang_robust")
opt_exclusions <- get_arg("--exclusions",    "notes/cue_exclusions.json")
opt_aliases    <- get_arg("--aliases",       "notes/aliases.json")
opt_clusters   <- get_arg("--clusters-json", "notes/clusters.json")

ALL_MODELS        <- c("gemini", "gpt41", "qwen3vl", "gemma3")
ALL_CLUSTERS      <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
VISUAL_CUES       <- c("flag", "currency", "sign")
VISUAL_CONDITIONS <- c("flag", "currency", "sign", "neutral")

models   <- if (!is.null(opt_model))   opt_model   else ALL_MODELS
clusters <- if (!is.null(opt_cluster)) opt_cluster else ALL_CLUSTERS

CONDITION_LABELS <- c(
  flag     = "Flag",
  currency = "Banknote",
  sign     = "Reservation sign",
  neutral  = "Neutral image"
)

CLUSTER_NAMES <- c(
  pilaf_cl    = "Pilaf",
  mantou_cl   = "Mantou",
  samosa_cl   = "Samosa",
  meatball_cl = "Meatball"
)

MODEL_LABELS <- c(
  gemini  = "Gemini",
  gpt41   = "GPT-4.1",
  qwen3vl = "Qwen3-VL",
  gemma3  = "Gemma3"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

build_lookup <- function(aliases_path) {
  raw <- fromJSON(aliases_path, simplifyVector = FALSE)
  lookup <- list()

  for (canonical in names(raw)) {
    if (startsWith(canonical, "_")) next

    # Ensure the canonical label itself is recognized.
    lookup[[trimws(tolower(canonical))]] <- canonical

    for (variant in raw[[canonical]]) {
      lookup[[trimws(tolower(variant))]] <- canonical
    }
  }

  lookup
}

normalize_country <- function(name, lookup) {
  if (is.null(name) || is.na(name) || nchar(trimws(name)) == 0) return("other")
  result <- lookup[[trimws(tolower(name))]]
  if (is.null(result)) "other" else result
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

read_prediction <- function(rows, image_name, lookup) {
  d <- rows[rows$image == image_name, , drop = FALSE]
  if (nrow(d) == 0) return(NULL)
  tryCatch({
    if (!isTRUE(d$success[1])) return(NULL)
    normalize_country(d$primary_country[1], lookup)
  }, error = function(e) NULL)
}

# ─────────────────────────────────────────────
# COMPUTE FLIP COUNTS
# returns data frame: model, cluster, condition, flips, total
# ─────────────────────────────────────────────

compute_all_flips <- function(lookup, cl_raw) {
  rows <- list()

  for (cluster_id in clusters) {
    cl_entry <- Filter(function(x) x$cluster_id == cluster_id, cl_raw$clusters)
    if (length(cl_entry) == 0) next
    countries <- tolower(gsub(" ", "_",
                              sapply(cl_entry[[1]]$countries,
                                     function(c) c$country)))

    for (model in models) {
      cat(sprintf("  Computing %s / %s ...\n", model, cluster_id))
      exclusions <- load_exclusions(opt_exclusions, model, cluster_id)

      # all visual conditions
      visual_conditions <- list(list(vcue = "neutral", vcountry = "none"))
      for (vcue in VISUAL_CUES)
        for (vcountry in countries)
          visual_conditions[[length(visual_conditions) + 1]] <-
            list(vcue = vcue, vcountry = vcountry)

      # accumulators per condition
      flips  <- c(neutral = 0L, flag = 0L, currency = 0L, sign = 0L)
      totals <- c(neutral = 0L, flag = 0L, currency = 0L, sign = 0L)

      for (vc in visual_conditions) {
        vcue     <- vc$vcue
        vcountry <- vc$vcountry
        seg      <- vcue

        if (vcountry != "none" && is_excluded(vcue, vcountry, exclusions)) next

        en_rows <- all_results[
          all_results$cluster_id == cluster_id &
          all_results$model == model &
          all_results$visual_cue == vcue &
          all_results$visual_country == vcountry &
          all_results$experiment == "country_classification" &
          all_results$textual_cue_type == "en" &
          all_results$textual_country == "none",
          ,
          drop = FALSE
        ]
        if (nrow(en_rows) == 0) next

        en_files <- en_rows$image
        if (length(en_files) == 0) next

        for (tcountry in countries) {
          if (is_excluded("native", tcountry, exclusions)) next

          nat_rows <- all_results[
            all_results$cluster_id == cluster_id &
            all_results$model == model &
            all_results$visual_cue == vcue &
            all_results$visual_country == vcountry &
            all_results$experiment == "country_classification" &
            all_results$textual_cue_type == "native" &
            all_results$textual_country == tcountry,
            ,
            drop = FALSE
          ]
          if (nrow(nat_rows) == 0) next

          for (img in en_files) {
            en_pred  <- read_prediction(en_rows,  img, lookup)
            nat_pred <- read_prediction(nat_rows, img, lookup)
            if (is.null(en_pred) || is.null(nat_pred)) next

            totals[seg] <- totals[seg] + 1L
            if (en_pred != nat_pred)
              flips[seg] <- flips[seg] + 1L
          }
        }
      }

      for (seg in VISUAL_CONDITIONS) {
        rows[[length(rows) + 1]] <- data.frame(
          model     = model,
          cluster   = cluster_id,
          condition = seg,
          flips     = as.integer(flips[seg]),
          total     = as.integer(totals[seg]),
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

pct <- function(flips, total) {
  if (total == 0) "N/A" else sprintf("%.1f%%", 100 * flips / total)
}

agg_rate <- function(df, ...) {
  df %>%
    group_by(...) %>%
    summarise(flips = sum(flips),
              total = sum(total),
              rate  = if (sum(total) > 0) 100 * sum(flips) / sum(total)
                      else NA_real_,
              .groups = "drop")
}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

cat("Loading aliases...\n")
lookup <- build_lookup(opt_aliases)
cl_raw <- fromJSON(opt_clusters, simplifyVector = FALSE)

cat("Loading all_results.csv...\n")
all_results <- read.csv(
  opt_results_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  encoding = "UTF-8"
)

# Match JSON boolean behaviour exactly.
all_results$success <- tolower(trimws(as.character(all_results$success))) == "true"

dir.create(opt_output, showWarnings = FALSE, recursive = TRUE)

cat("Computing flip rates...\n")
df <- compute_all_flips(lookup, cl_raw)

report_path <- file.path(opt_output, "flip_bar_report.txt")
con <- file(report_path, open = "wt")

wh()
wl("  ENGLISH-TO-NATIVE PROMPT COUNTRY-ATTRIBUTION FLIP REPORT")
wl(paste0("  Models  : ", paste(models,   collapse = ", ")))
wl(paste0("  Clusters: ", paste(clusters, collapse = ", ")))
wl("  Flip = normalized country attributions differ between English and native prompts")
wl("  Pair = same image and same visual condition")
wl("  Rate = flips / paired evaluations where both predictions succeeded")
wl("  Aggregation = pooled numerator and denominator counts")
wh()

# ══════════════════════════════════════════════
# LEVEL 1 - GLOBAL
# ══════════════════════════════════════════════

wl()
wl("LEVEL 1 - GLOBAL POOLED COUNTRY-ATTRIBUTION FLIP RATE")
ws()
wl(sprintf("  %-18s  %s", "VISUAL CONDITION", "FLIP RATE (flips/paired successful evaluations)"))
wl(paste0("  ", strrep("-", 45)))

global <- agg_rate(df, condition)
global$condition <- factor(global$condition, levels = VISUAL_CONDITIONS)
global <- global[order(global$condition), ]

for (i in seq_len(nrow(global))) {
  cond_name <- CONDITION_LABELS[global$condition[i]]
  wl(sprintf("  %-18s  %s  (%d/%d)",
             cond_name,
             pct(global$flips[i], global$total[i]),
             global$flips[i], global$total[i]))
}

# ══════════════════════════════════════════════
# LEVEL 2 - PER MODEL
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 2 - PER MODEL (pooled across clusters)")
ws()

per_model <- agg_rate(df, model, condition)

for (model in models) {
  wl(sprintf("  %s:", MODEL_LABELS[model]))
  sub <- per_model[per_model$model == model, ]
  sub$condition <- factor(sub$condition, levels = VISUAL_CONDITIONS)
  sub <- sub[order(sub$condition), ]
  for (i in seq_len(nrow(sub))) {
    cond_name <- CONDITION_LABELS[sub$condition[i]]
    wl(sprintf("    %-18s  %s  (%d/%d)",
               cond_name,
               pct(sub$flips[i], sub$total[i]),
               sub$flips[i], sub$total[i]))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 3 - PER CLUSTER
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 3 - PER CLUSTER (pooled across models)")
ws()

per_cluster <- agg_rate(df, cluster, condition)

for (cluster_id in clusters) {
  wl(sprintf("  %s:", CLUSTER_NAMES[cluster_id]))
  sub <- per_cluster[per_cluster$cluster == cluster_id, ]
  sub$condition <- factor(sub$condition, levels = VISUAL_CONDITIONS)
  sub <- sub[order(sub$condition), ]
  for (i in seq_len(nrow(sub))) {
    cond_name <- CONDITION_LABELS[sub$condition[i]]
    wl(sprintf("    %-18s  %s  (%d/%d)",
               cond_name,
               pct(sub$flips[i], sub$total[i]),
               sub$flips[i], sub$total[i]))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 4 - PER MODEL x CLUSTER
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 4 - PER MODEL x CLUSTER (rates and paired counts)")
ws()

for (model in models) {
  wl(sprintf("  %s:", MODEL_LABELS[model]))
  for (cluster_id in clusters) {
    sub <- df[df$model == model & df$cluster == cluster_id, ]
    sub$condition <- factor(sub$condition, levels = VISUAL_CONDITIONS)
    sub <- sub[order(sub$condition), ]
    parts <- sapply(seq_len(nrow(sub)), function(i) {
      sprintf(
        "%s=%s (%d/%d)",
        CONDITION_LABELS[sub$condition[i]],
        pct(sub$flips[i], sub$total[i]),
        sub$flips[i],
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