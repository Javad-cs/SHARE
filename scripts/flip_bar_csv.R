#!/usr/bin/env Rscript
# flip_bar_csv.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(dplyr)
  library(patchwork)
})

args    <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) > 0 && length(args) >= i[1] + 1) args[i[1] + 1] else default
}

opt_model      <- get_arg("--model")
opt_cluster    <- get_arg("--cluster")
opt_results_csv <- get_arg("--results-csv", "results/analysis/all_results.csv")
opt_output     <- get_arg("--output-dir",    "results/analysis/lang_robust")
opt_exclusions <- get_arg("--exclusions",    "notes/cue_exclusions.json")
opt_aliases    <- get_arg("--aliases",       "notes/aliases.json")
opt_clusters   <- get_arg("--clusters-json", "notes/clusters.json")

ALL_MODELS        <- c("gemini", "gpt41", "qwen3vl", "gemma3")
ALL_CLUSTERS      <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
VISUAL_CUES       <- c("flag", "currency", "sign")
VISUAL_CONDITIONS <- c("neutral", "flag", "currency", "sign")

models   <- if (!is.null(opt_model))   opt_model   else ALL_MODELS
clusters <- if (!is.null(opt_cluster)) opt_cluster else ALL_CLUSTERS

CONDITION_COLORS <- c(
  "Neutral image"     = "#94a3b8",
  "Flag"              = "#2563eb",
  "Banknote"          = "#f59e0b",
  "Reservation sign"  = "#16a34a"
)

MODEL_LABELS <- c(
  gemini  = "Gemini",
  gpt41   = "GPT-4.1",
  qwen3vl = "Qwen3-VL",
  gemma3  = "Gemma3"
)

CLUSTER_NAMES <- c(
  pilaf_cl    = "Pilaf",
  mantou_cl   = "Mantou",
  samosa_cl   = "Samosa",
  meatball_cl = "Meatball"
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

compute_flip_rates <- function(model, cluster_id, countries, lookup, exclusions) {
  flips  <- c(neutral = 0L, flag = 0L, currency = 0L, sign = 0L)
  totals <- c(neutral = 0L, flag = 0L, currency = 0L, sign = 0L)

  visual_conditions <- list(list(vcue = "neutral", vcountry = "none"))
  for (vcue in VISUAL_CUES)
    for (vcountry in countries)
      visual_conditions[[length(visual_conditions) + 1]] <-
        list(vcue = vcue, vcountry = vcountry)

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
        if (en_pred != nat_pred) flips[seg] <- flips[seg] + 1L
      }
    }
  }

  list(flips = flips, totals = totals)
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

all_rows <- list()

for (cluster_id in clusters) {
  cl_entry <- Filter(function(x) x$cluster_id == cluster_id, cl_raw$clusters)
  if (length(cl_entry) == 0) next
  countries <- tolower(gsub(" ", "_",
                            sapply(cl_entry[[1]]$countries,
                                   function(c) c$country)))

  for (model in models) {
    cat(sprintf("  Computing %s / %s ...\n", model, cluster_id))
    exclusions <- load_exclusions(opt_exclusions, model, cluster_id)

    result <- tryCatch(
      compute_flip_rates(model, cluster_id, countries, lookup, exclusions),
      error = function(e) {
        cat(sprintf("  [ERROR] %s/%s: %s\n", model, cluster_id, conditionMessage(e)))
        NULL
      }
    )

    if (!is.null(result)) {
      flips  <- result$flips
      totals <- result$totals

      for (seg in VISUAL_CONDITIONS) {
        n_f <- as.integer(flips[seg])
        n_t <- as.integer(totals[seg])
        r   <- if (n_t > 0) 100.0 * n_f / n_t else 0.0
        cat(sprintf("    %s: %d/%d (%.1f%%)\n", seg, n_f, n_t, r))
        all_rows[[length(all_rows) + 1]] <- data.frame(
          cluster   = cluster_id,
          model     = model,
          condition = seg,
          rate      = r,
          n_flips   = n_f,
          n_valid   = n_t,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

all_df <- bind_rows(all_rows)

# aggregate across clusters per model x condition
all_df <- all_df %>%
  group_by(model, condition) %>%
  summarise(
    n_flips = sum(n_flips),
    n_valid = sum(n_valid),
    rate = if (sum(n_valid) > 0) {
      100 * sum(n_flips) / sum(n_valid)
    } else {
      NA_real_
    },
    .groups = "drop"
  )

all_df$model <- factor(all_df$model,
                       levels = models,
                       labels = MODEL_LABELS[models])

all_df$condition <- factor(
  all_df$condition,
  levels = c("flag", "currency", "sign", "neutral"),
  labels = c("Flag", "Banknote", "Reservation sign", "Neutral image")
)

max_rate <- max(all_df$rate, na.rm = TRUE) * 1.40

plot_list <- list()

for (mi in seq_along(models)) {
  model_label <- MODEL_LABELS[models[mi]]
  sub         <- all_df[all_df$model == model_label, ]

  p <- ggplot(sub, aes(x = condition, y = rate, fill = condition)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.2) +
    geom_text(
      aes(label = sprintf("%.1f%%\n(n=%d)", rate, n_valid)), hjust = -0.1, size = 3.0, 
      color = "#374151", fontface = "bold") +
    coord_flip(clip = "off") +
    scale_fill_manual(values = CONDITION_COLORS, guide = "none") +
    scale_y_continuous(
      limits = c(0, max_rate),
      labels = function(x) paste0(x, "%"),
      expand = c(0, 0)
    ) +
    labs(title = model_label, x = NULL, y = "Country-Attribution Flip Rate (%)") +
    theme_minimal(base_family = "sans") +
    theme(
      plot.title         = element_text(size = 11, face = "bold",
                                        hjust = 0.5, color = "#0f172a",
                                        margin = margin(b = 6, t = 4)),
      axis.text.y        = element_text(size = 10, color = "#1a202c",
                                        face = "bold"),
      axis.text.x        = element_text(size = 8, color = "#4a5568"),
      axis.title.x       = element_text(size = 8, color = "#4a5568",
                                        margin = margin(t = 4)),
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(color = "#f1f5f9", linewidth = 0.5),
      plot.background    = element_rect(fill = "white", color = NA),
      panel.background   = element_rect(fill = "white", color = NA),
      plot.margin        = margin(t = 6, r = 40, b = 6, l = 8)
    )

  plot_list[[mi]] <- p
}

combined <- wrap_plots(plot_list, nrow = 1, ncol = 4) +
  plot_annotation(
    title = "English-to-Native Prompt Country-Attribution Flip Rate",
    caption = paste0(
      "Flip rate = differing normalized country attributions / ",
      "paired successful English-native evaluations  |  ",
      "Rates pooled across clusters"
    ),
    theme = theme(
      plot.title = element_text(
        size = 13,
        face = "bold",
        hjust = 0.5,
        color = "#0f172a",
        margin = margin(b = 8, t = 8)
      ),
      plot.caption = element_text(
        size = 9,
        color = "#374151",
        hjust = 0.5,
        margin = margin(t = 6)
      ),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

base_path <- file.path(opt_output, "flip_bar")
ggsave(paste0(base_path, ".pdf"), plot = combined,
       width = 14, height = 5, units = "in", device = "pdf")
ggsave(paste0(base_path, ".png"), plot = combined,
       width = 14, height = 5, units = "in", dpi = 200, device = "png")
cat(sprintf("[OK] Saved: %s.pdf / .png\n", base_path))