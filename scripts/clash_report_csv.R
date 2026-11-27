#!/usr/bin/env Rscript
# clash_report_csv.R
# --------------
# Generates a text report summarizing clash matrix statistics.
#
# Usage:
#   Rscript scripts/clash_report_csv.R
#   Rscript scripts/clash_report_csv.R --model gemini
#   Rscript scripts/clash_report_csv.R --cluster pilaf_cl
#   Rscript scripts/clash_report_csv.R --model gemini --cluster pilaf_cl
#
# Output: results/analysis/heatmaps/clash/clash_report.txt

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
opt_output     <- get_arg("--output-dir",    "results/analysis/heatmaps/clash")
opt_exclusions <- get_arg("--exclusions",    "notes/cue_exclusions.json")
opt_aliases    <- get_arg("--aliases",       "notes/aliases.json")
opt_clusters   <- get_arg("--clusters-json", "notes/clusters.json")

ALL_MODELS   <- c("gemini", "gpt41", "qwen3vl", "gemma3")
ALL_CLUSTERS <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
VISUAL_CUES  <- c("flag", "currency", "sign")
TEXTUAL_CUES <- c("native", "endonym", "perspective")

models   <- if (!is.null(opt_model))   opt_model   else ALL_MODELS
clusters <- if (!is.null(opt_cluster)) opt_cluster else ALL_CLUSTERS

REPORT_CATS <- c("visual", "text", "tie", "prior", "vague", "other")
NON_BLUE    <- c("text", "tie", "prior", "vague", "other")

CAT_NAMES <- c(
  visual = "Visual-cue country most frequent (blue)",
  text   = "Textual-cue country most frequent (green)",
  tie    = "Visual-textual tie (purple)",
  prior  = "Neutral-baseline country most frequent (yellow)",
  vague  = "Region-level response most frequent (orange)",
  other  = "Other country most frequent (red)"
)

COUNTRY_ABBR <- c(
  uzbekistan  = "UZ", afghanistan = "AF", azerbaijan = "AZ",
  tajikistan  = "TJ", iran        = "IR", india      = "IN",
  turkey      = "TR", spain       = "ES", korea      = "KR",
  kazakhstan  = "KZ"
)

CUE_ABB <- c(
  flag = "flg", currency = "cur", sign = "sgn",
  native = "nat", endonym = "end", perspective = "per"
)

# ─────────────────────────────────────────────
# SHARED HELPERS (duplicated from clash_matrix.R
# to keep this script self-contained)
# ─────────────────────────────────────────────

abbr_country <- function(country) {
  a <- COUNTRY_ABBR[country]
  if (is.na(a)) toupper(substr(country, 1, 2)) else unname(a)
}

build_lookup <- function(aliases_path) {
  raw    <- fromJSON(aliases_path, simplifyVector = FALSE)
  lookup <- list()
  for (canonical in names(raw)) {
    if (startsWith(canonical, "_")) next

    # Ensure the canonical label itself is always recognized.
    lookup[[trimws(tolower(canonical))]] <- canonical

    for (variant in raw[[canonical]]) {
      key           <- trimws(tolower(variant))
      lookup[[key]] <- canonical
    }
  }
  lookup
}

normalize_country <- function(name, lookup) {
  if (is.null(name) || is.na(name) || nchar(trimws(name)) == 0) return("other")
  key    <- trimws(tolower(name))
  result <- lookup[[key]]
  if (is.null(result)) "other" else result
}

# Convert a condition/path identifier such as "south_korea" or "korea"
# into the canonical label used for normalized model predictions.
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

load_all_results <- function(csv_path) {
  if (!file.exists(csv_path))
    stop(sprintf("Results CSV not found: %s", csv_path))

  data <- read.csv(
    csv_path,
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = "character",
    na.strings = character(0),
    fileEncoding = "UTF-8"
  )

  required <- c(
    "image", "cluster_id", "model", "visual_cue", "visual_country",
    "textual_cue_type", "textual_country", "experiment", "success",
    "primary_country"
  )
  missing <- setdiff(required, names(data))
  if (length(missing) > 0)
    stop(sprintf("Results CSV is missing columns: %s",
                 paste(missing, collapse = ", ")))

  data
}

load_baseline <- function(results_df, cluster_id, model, lookup) {
  rows <- results_df[
    results_df$cluster_id == cluster_id &
    results_df$model == model &
    results_df$visual_cue == "neutral" &
    results_df$visual_country == "none" &
    results_df$experiment == "country_classification" &
    results_df$textual_cue_type == "en" &
    results_df$textual_country == "none",
    , drop = FALSE
  ]
  if (nrow(rows) == 0) return(character(0))

  counts <- list()
  for (i in seq_len(nrow(rows))) {
    if (tolower(rows$success[i]) != "true") next
    norm <- normalize_country(rows$primary_country[i], lookup)
    if (norm != "region") {
      counts[[norm]] <- (if (is.null(counts[[norm]])) 0L else counts[[norm]]) + 1L
    }
  }
  if (length(counts) == 0) return(character(0))

  count_vector <- unlist(counts)
  top_count <- max(count_vector)

  # Keep all countries tied for the highest baseline count.
  names(count_vector[count_vector == top_count])
}

load_cell_preds <- function(results_df, cluster_id, model,
                            vcue, vcountry, tcue, tcountry, lookup) {
  rows <- results_df[
    results_df$cluster_id == cluster_id &
    results_df$model == model &
    results_df$visual_cue == vcue &
    results_df$visual_country == vcountry &
    results_df$experiment == "country_classification" &
    results_df$textual_cue_type == tcue &
    results_df$textual_country == tcountry,
    , drop = FALSE
  ]
  if (nrow(rows) == 0) return(character(0))

  preds <- character(0)
  for (i in seq_len(nrow(rows))) {
    if (tolower(rows$success[i]) != "true") next
    preds <- c(preds, normalize_country(rows$primary_country[i], lookup))
  }
  preds
}

classify_cell <- function(preds, vcountry, tcountry, prior, lookup) {
  if (length(preds) == 0) return(list(cat = "missing", intensity = 0.0))

  # Keep raw identifiers such as "korea" for row selection, but compare
  # predictions against canonical labels such as "south korea".
  visual_target  <- normalize_target_country(vcountry, lookup)
  textual_target <- normalize_target_country(tcountry, lookup)

  if (visual_target == "other") {
    stop(sprintf("Visual target is missing from aliases.json: %s", vcountry))
  }
  if (textual_target == "other") {
    stop(sprintf("Textual target is missing from aliases.json: %s", tcountry))
  }

  n_valid   <- length(preds)
  counts    <- table(preds)
  top_count <- max(counts)
  top_set   <- names(counts[counts == top_count])

  has_visual  <- visual_target %in% top_set
  has_textual <- textual_target %in% top_set
  has_prior <- length(prior) > 0 && any(prior %in% top_set)
  has_vague   <- "region" %in% top_set

  intensity <- top_count / n_valid

  if (has_visual && has_textual) {
    return(list(cat = "tie",    intensity = 2 * top_count / n_valid))
  } else if (has_visual) {
    return(list(cat = "visual", intensity = intensity))
  } else if (has_textual) {
    return(list(cat = "text",   intensity = intensity))
  } else if (has_prior) {
    return(list(cat = "prior",  intensity = intensity))
  } else if (has_vague) {
    return(list(cat = "vague",  intensity = intensity))
  } else {
    return(list(cat = "other",  intensity = intensity))
  }
}

# ─────────────────────────────────────────────
# BUILD FULL FLAT DATA FRAME
# ─────────────────────────────────────────────

build_all_cells <- function(lookup) {
  cl_raw <- fromJSON(opt_clusters, simplifyVector = FALSE)
  all_rows <- list()

  for (model in models) {
    for (cluster_id in clusters) {
      cat(sprintf("  Loading %s / %s ...\n", model, cluster_id))

      cl_entry <- Filter(function(x) x$cluster_id == cluster_id, cl_raw$clusters)
      if (length(cl_entry) == 0) next
      cl_entry  <- cl_entry[[1]]
      countries <- tolower(gsub(" ", "_",
                                sapply(cl_entry$countries, function(c) c$country)))

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
        "    Target mappings: %s\n",
        paste(sprintf("%s -> %s", countries, canonical_countries),
              collapse = ", ")
      ))

      exclusions <- load_exclusions(opt_exclusions, model, cluster_id)
      prior      <- load_baseline(results_df, cluster_id, model, lookup)

      for (vcue in VISUAL_CUES) {
        for (vcountry in countries) {
          for (tcue in TEXTUAL_CUES) {
            for (tcountry in countries) {

              if (vcountry == tcountry) next

              if (is_excluded(vcue, vcountry, exclusions) ||
                  is_excluded(tcue, tcountry, exclusions)) next

              preds  <- load_cell_preds(results_df, cluster_id, model,
                                        vcue, vcountry, tcue, tcountry, lookup)
              result <- classify_cell(preds, vcountry, tcountry, prior, lookup)

              if (result$cat %in% c("missing")) next

              all_rows[[length(all_rows) + 1]] <- data.frame(
                model      = model,
                cluster    = cluster_id,
                vcue       = vcue,
                vcountry   = vcountry,
                tcue       = tcue,
                tcountry   = tcountry,
                cat        = result$cat,
                intensity  = result$intensity,
                prior = if (length(prior) == 0) {
                  "unknown"
                } else {
                  paste(prior, collapse = " / ")
                },
                cell_label = paste0(
                  abbr_country(vcountry), "-", CUE_ABB[vcue],
                  " x ",
                  abbr_country(tcountry), "-", CUE_ABB[tcue]
                ),
                stringsAsFactors = FALSE
              )
            }
          }
        }
      }
    }
  }

  bind_rows(all_rows)
}

# ─────────────────────────────────────────────
# REPORT WRITING HELPERS
# ─────────────────────────────────────────────

SEP_HEAVY <- paste0(strrep("=", 60), "\n")
SEP_LIGHT <- paste0(strrep("-", 60), "\n")

w <- function(...) cat(..., file = con)
wl <- function(x = "") w(x, "\n")
ws <- function() wl(SEP_LIGHT)
wh <- function() wl(SEP_HEAVY)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

cat("Loading results CSV...\n")
results_df <- load_all_results(opt_results_csv)
cat(sprintf("Loaded %d rows.\n", nrow(results_df)))

cat("Loading aliases...\n")
lookup <- build_lookup(opt_aliases)

cat("Building cell data...\n")
df <- build_all_cells(lookup)

duplicate_cells <- df %>%
  count(model, cluster, vcue, vcountry, tcue, tcountry) %>%
  filter(n != 1)

if (nrow(duplicate_cells) > 0) {
  stop(sprintf(
    "Found %d duplicated clash cells.",
    nrow(duplicate_cells)
  ))
}

dir.create(opt_output, showWarnings = FALSE, recursive = TRUE)
report_path <- file.path(opt_output, "clash_report.txt")
con <- file(report_path, open = "wt")

# only count REPORT_CATS in totals
df_report <- df[df$cat %in% REPORT_CATS, ]
total     <- nrow(df_report)

# ══════════════════════════════════════════════
# LEVEL 1 - GLOBAL SUMMARY
# ══════════════════════════════════════════════

wh()
wl("  CLASH MATRIX REPORT")
wl(paste0("  Models  : ", paste(models, collapse = ", ")))
wl(paste0("  Clusters: ", paste(clusters, collapse = ", ")))
wl(paste0("  Total classified cells: ", total))
wl("  Neutral baseline: all countries tied for the highest neutral-English count are retained")
wh()
wl()
wl("LEVEL 1 - GLOBAL SUMMARY")
ws()

for (cat in REPORT_CATS) {
  n   <- sum(df_report$cat == cat)
  pct <- if (total > 0) sprintf("%.1f%%", 100 * n / total) else "N/A"
  wl(sprintf("  %-30s %5d  (%s)", CAT_NAMES[cat], n, pct))
}

# ══════════════════════════════════════════════
# LEVEL 2 - PER MODEL (all outcome categories)
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 2 - PER MODEL")
ws()

df_all <- df_report

for (cat in REPORT_CATS) {
  n_total <- sum(df_report$cat == cat)
  if (n_total == 0) next
  wl(sprintf("  %s  [total: %d]", CAT_NAMES[cat], n_total))
  for (model in models) {
    n <- sum(df_all$cat == cat & df_all$model == model)
    wl(sprintf("    %-12s %4d", model, n))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 3 - PER MODEL x CLUSTER (all outcome categories)
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 3 - PER MODEL x CLUSTER")
ws()

for (cat in REPORT_CATS) {
  n_total <- sum(df_report$cat == cat)
  if (n_total == 0) next
  wl(sprintf("  %s", CAT_NAMES[cat]))
  for (model in models) {
    parts <- c()
    for (cluster_id in clusters) {
      n <- sum(df_all$cat == cat &
               df_all$model == model &
               df_all$cluster == cluster_id)
      cl_short <- gsub("_cl$", "", cluster_id)
      parts <- c(parts, sprintf("%s=%d", cl_short, n))
    }
    wl(sprintf("    %-12s  %s", model, paste(parts, collapse="  ")))
  }
  wl()
}

# ══════════════════════════════════════════════
# LEVEL 3.5 - VISUAL CUE BREAKDOWN (for visual wins)
#             TEXTUAL CUE BREAKDOWN (for text wins)
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 3.5 - CUE BREAKDOWN WITHIN VISUAL- AND TEXTUAL-CUE OUTCOMES")
ws()

# visual wins broken down by vcue
wl("  Visual-cue-country outcomes by visual cue:")
for (vcue in c("flag", "currency", "sign")) {
  n_total <- sum(df_report$cat == "visual")
  n_cue   <- sum(df_report$cat == "visual" & df_report$vcue == vcue)
  pct     <- if (n_total > 0) sprintf("%.1f%%", 100 * n_cue / n_total) else "N/A"
  wl(sprintf("    %-12s %5d  (%s of all visual-cue-country outcomes)", vcue, n_cue, pct))
}

wl()

# text wins broken down by tcue
wl("  Textual-cue-country outcomes by textual cue:")
for (tcue in c("native", "endonym", "perspective")) {
  n_total <- sum(df_report$cat == "text")
  n_cue   <- sum(df_report$cat == "text" & df_report$tcue == tcue)
  pct     <- if (n_total > 0) sprintf("%.1f%%", 100 * n_cue / n_total) else "N/A"
  wl(sprintf("    %-12s %5d  (%s of all textual-cue-country outcomes)", tcue, n_cue, pct))
}

wl()

# per model breakdown
wl("  Visual-cue-country outcomes by cue - per model:")
for (model in models) {
  n_vis <- sum(df_report$cat == "visual" & df_report$model == model)
  parts <- c()
  for (vcue in c("flag", "currency", "sign")) {
    n <- sum(df_report$cat == "visual" & 
             df_report$model == model & 
             df_report$vcue == vcue)
    parts <- c(parts, sprintf("%s=%d(%.0f%%)", vcue, n, 100*n/max(n_vis,1)))
  }
  wl(sprintf("    %-12s  %s", model, paste(parts, collapse="  ")))
}

wl()

wl("  Textual-cue-country outcomes by cue - per model:")
for (model in models) {
  n_txt <- sum(df_report$cat == "text" & df_report$model == model)
  parts <- c()
  for (tcue in c("native", "endonym", "perspective")) {
    n <- sum(df_report$cat == "text" & 
             df_report$model == model & 
             df_report$tcue == tcue)
    parts <- c(parts, sprintf("%s=%d(%.0f%%)", tcue, n, 100*n/max(n_txt,1)))
  }
  wl(sprintf("    %-12s  %s", model, paste(parts, collapse="  ")))
}

# ══════════════════════════════════════════════
# LEVEL 3.6 - CUE WIN RATES (wins / opportunities)
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 3.6 - CUE OUTCOME RATES (outcome cells / classified cells)")
ws()

# For each visual cue: visual-cue-country outcome cells / classified cells.
wl("  Visual-cue-country outcome rates:")
for (vcue in c("flag", "currency", "sign")) {
  n_outcomes <- sum(
    df_report$cat == "visual" &
    df_report$vcue == vcue
  )

  n_total <- sum(df_report$vcue == vcue)

  pct <- if (n_total > 0) {
    sprintf("%.1f%%", 100 * n_outcomes / n_total)
  } else {
    "N/A"
  }

  wl(sprintf(
    "    %-12s  outcomes=%d  classified_cells=%d  rate=%s",
    vcue, n_outcomes, n_total, pct
  ))
}

wl()

# For each textual cue: textual-cue-country outcome cells / classified cells.
wl("  Textual-cue-country outcome rates:")
for (tcue in c("native", "endonym", "perspective")) {
  n_outcomes <- sum(
    df_report$cat == "text" &
    df_report$tcue == tcue
  )

  n_total <- sum(df_report$tcue == tcue)

  pct <- if (n_total > 0) {
    sprintf("%.1f%%", 100 * n_outcomes / n_total)
  } else {
    "N/A"
  }

  wl(sprintf(
    "    %-12s  outcomes=%d  classified_cells=%d  rate=%s",
    tcue, n_outcomes, n_total, pct
  ))
}

wl()

# per model breakdown
wl("  Visual-cue-country outcome rates - per model:")
for (model in models) {
  parts <- c()
  for (vcue in c("flag", "currency", "sign")) {
    n_wins  <- sum(df_report$cat == "visual" &
                   df_report$model == model &
                   df_report$vcue == vcue)
    n_total <- sum(df_report$model == model &
                   df_report$vcue == vcue &
                   !df_report$cat %in% c("diagonal", "excluded"))
    pct <- if (n_total > 0) sprintf("%.0f%%", 100 * n_wins / n_total) else "N/A"
    parts <- c(parts, sprintf("%s=%d/%d(%s)", vcue, n_wins, n_total, pct))
  }
  wl(sprintf("    %-12s  %s", model, paste(parts, collapse="  ")))
}

wl()

wl("  Textual-cue-country outcome rates - per model:")
for (model in models) {
  parts <- c()
  for (tcue in c("native", "endonym", "perspective")) {
    n_wins  <- sum(df_report$cat == "text" &
                   df_report$model == model &
                   df_report$tcue == tcue)
    n_total <- sum(df_report$model == model &
                   df_report$tcue == tcue &
                   !df_report$cat %in% c("diagonal", "excluded"))
    pct <- if (n_total > 0) sprintf("%.0f%%", 100 * n_wins / n_total) else "N/A"
    parts <- c(parts, sprintf("%s=%d/%d(%s)", tcue, n_wins, n_total, pct))
  }
  wl(sprintf("    %-12s  %s", model, paste(parts, collapse="  ")))
}

wl("  Visual-cue-country outcome rates - per cluster:")
for (cluster_id in clusters) {
  cl_short <- gsub("_cl$", "", cluster_id)
  parts <- c()
  for (vcue in c("flag", "currency", "sign")) {
    n_wins  <- sum(df_report$cat == "visual" &
                   df_report$cluster == cluster_id &
                   df_report$vcue == vcue)
    n_total <- sum(df_report$cluster == cluster_id &
                   df_report$vcue == vcue &
                   !df_report$cat %in% c("diagonal", "excluded"))
    pct <- if (n_total > 0) sprintf("%.0f%%", 100 * n_wins / n_total) else "N/A"
    parts <- c(parts, sprintf("%s=%d/%d(%s)", vcue, n_wins, n_total, pct))
  }
  wl(sprintf("    %-12s  %s", cl_short, paste(parts, collapse="  ")))
}

wl()

wl("  Textual-cue-country outcome rates - per cluster:")
for (cluster_id in clusters) {
  cl_short <- gsub("_cl$", "", cluster_id)
  parts <- c()
  for (tcue in c("native", "endonym", "perspective")) {
    n_wins  <- sum(df_report$cat == "text" &
                   df_report$cluster == cluster_id &
                   df_report$tcue == tcue)
    n_total <- sum(df_report$cluster == cluster_id &
                   df_report$tcue == tcue &
                   !df_report$cat %in% c("diagonal", "excluded"))
    pct <- if (n_total > 0) sprintf("%.0f%%", 100 * n_wins / n_total) else "N/A"
    parts <- c(parts, sprintf("%s=%d/%d(%s)", tcue, n_wins, n_total, pct))
  }
  wl(sprintf("    %-12s  %s", cl_short, paste(parts, collapse="  ")))
}

# ══════════════════════════════════════════════
# LEVEL 4 - ALL NON-BLUE CELLS (sorted by intensity)
# ══════════════════════════════════════════════

wl()
wh()
wl("LEVEL 4 - ALL NON-BLUE CELLS (sorted by outcome share descending)")
ws()
wl("  Outcome share = most-frequent response share;")
wl("  for visual-textual ties, it is the combined share of both tied countries.")
wl()

for (cat in NON_BLUE) {
  sub <- df_all[df_all$cat == cat, ]
  if (nrow(sub) == 0) next

  sub <- sub[order(-sub$intensity), ]

  wl(sprintf("  %s  [%d cells]", CAT_NAMES[cat], nrow(sub)))
  wl(sprintf("  %-10s  %-12s  %-22s  %s",
             "MODEL", "CLUSTER", "CELL", "OUTCOME SHARE"))
  wl(paste0("  ", strrep("-", 56)))

  for (i in seq_len(nrow(sub))) {
    row <- sub[i, ]
    cl_short <- gsub("_cl$", "", row$cluster)
    wl(sprintf("  %-10s  %-12s  %-22s  %s%%",
               row$model,
               cl_short,
               row$cell_label,
               round(row$intensity * 100)))
  }
  wl()
}

wh()
wl(paste0("  Report generated: ", Sys.time()))
wh()

close(con)
cat(sprintf("[OK] Report saved: %s\n", report_path))