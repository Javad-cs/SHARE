#!/usr/bin/env Rscript
# cue_bar_csv.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(dplyr)
  library(patchwork)
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
ALL_CUES     <- c("flag", "currency", "sign", "endonym", "perspective", "native")

models   <- if (!is.null(opt_model))   opt_model   else ALL_MODELS
clusters <- if (!is.null(opt_cluster)) opt_cluster else ALL_CLUSTERS


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CUE_COLORS <- c(
  "Flag"        = "#2563eb",
  "Banknote"    = "#964B00",
  "Reservation sign"        = "#d9c21d",
  "Native-language prompt"      = "#7c3aed",
  "Endonym prompt"     = "#16a34a",
  "Perspective prompt" = "#ea580c"
)

CUE_LABELS <- c(
  flag = "Flag", currency = "Banknote", sign = "Reservation sign",
  native = "Native-language prompt", endonym = "Endonym prompt", perspective = "Perspective prompt"
)

MODEL_LABELS <- c(
  gemini = "Gemini", gpt41 = "GPT-4.1",
  qwen3vl = "Qwen3-VL", gemma3 = "Gemma3"
)

CLUSTER_NAMES <- c(
  pilaf_cl = "Pilaf", mantou_cl = "Mantou",
  samosa_cl = "Samosa", meatball_cl = "Meatball"
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

  if (nrow(sub) == 0) {
    return(list(correct = 0L, total = 0L))
  }

  successful <- sub$success %in% c(TRUE, "True", "TRUE", "true", 1, "1")
  sub <- sub[successful, , drop = FALSE]
  if (nrow(sub) == 0) {
    return(list(correct = 0L, total = 0L))
  }

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

compute_scores <- function(model, cluster_id, countries, lookup, exclusions, results_df) {
  scores <- numeric(length(ALL_CUES))
  names(scores) <- ALL_CUES

  for (vcue in VISUAL_CUES) {
    total_correct <- 0L; total_denom <- 0L
    for (country in countries) {
      if (is_excluded(vcue, country, exclusions)) next
      res <- tryCatch(
        count_correct(results_df, cluster_id, model,
                      vcue, country, "en", "none",
                      country, lookup),
        error = function(e) {
          stop(sprintf(
            "Failed to calculate %s/%s/%s/%s: %s",
            model, cluster_id, vcue, country, conditionMessage(e)
          ))
        }
      )
      total_correct <- total_correct + res$correct
      total_denom   <- total_denom + res$total
    }
    scores[vcue] <- if (total_denom > 0) 100 * total_correct / total_denom else 0
  }

  for (tcue in TEXTUAL_CUES) {
    total_correct <- 0L; total_denom <- 0L
    for (country in countries) {
      if (is_excluded(tcue, country, exclusions)) next
      res <- tryCatch(
        count_correct(results_df, cluster_id, model,
                      "neutral", "none", tcue, country,
                      country, lookup),
        error = function(e) {
          stop(sprintf(
            "Failed to calculate %s/%s/%s/%s: %s",
            model, cluster_id, tcue, country, conditionMessage(e)
          ))
        }
      )
      total_correct <- total_correct + res$correct
      total_denom   <- total_denom + res$total
    }
    scores[tcue] <- if (total_denom > 0) 100 * total_correct / total_denom else 0
  }

  scores
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

all_rows <- list()

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
    scores <- tryCatch(
      compute_scores(model, cluster_id, countries, lookup, exclusions, results_df),
      error = function(e) {
        cat(sprintf("  [ERROR] %s/%s: %s\n", model, cluster_id, conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(scores)) {
      cat(sprintf("    %s\n",
                  paste(sprintf("%s=%.1f%%", ALL_CUES, scores), collapse = "  ")))
      for (cue in ALL_CUES) {
        all_rows[[length(all_rows) + 1]] <- data.frame(
          cluster     = cluster_id,
          model       = model,
          cue         = CUE_LABELS[cue],
          score       = scores[cue],
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

all_df <- bind_rows(all_rows)

# set factor levels for correct ordering
all_df$model <- factor(all_df$model,
                       levels = models,
                       labels = MODEL_LABELS[models])
all_df$cue   <- factor(all_df$cue,
                       levels = c("Flag", "Banknote", "Reservation sign",
                                  "Endonym prompt", "Perspective prompt", "Native-language prompt"))

# ─────────────────────────────────────────────
# BUILD PLOT PER CLUSTER
# ─────────────────────────────────────────────

plot_list <- list()

for (ci in seq_along(clusters)) {
  cluster_id   <- clusters[ci]
  cluster_name <- CLUSTER_NAMES[cluster_id]
  sub          <- all_df[all_df$cluster == cluster_id, ]

  if (nrow(sub) == 0) {
    plot_list[[ci]] <- ggplot() + theme_void() +
      annotate("text", x = 0.5, y = 0.5,
               label = "NO DATA", color = "#cc0000", size = 4)
    next
  }

  p <- ggplot(sub, aes(x = cue, y = score, fill = cue)) +
    geom_col(width = 0.7, color = "white", linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.1f%%", score)),
              vjust = -0.4, size = 2.4, color = "#374151") +
    facet_wrap(~ model, nrow = 1, strip.position = "bottom") +
    scale_fill_manual(values = CUE_COLORS, name = NULL) +
    scale_y_continuous(
      limits = c(0, 115),
      breaks = c(0, 25, 50, 75, 100),
      labels = c("0%", "25%", "50%", "75%", "100%"),
      expand = c(0, 0)
    ) +
    labs(title = cluster_name, x = NULL, y = "Target-Country Attribution Rate (%)") +
    theme_minimal(base_family = "sans") +
    theme(
      plot.title          = element_text(size = 11, face = "bold",
                                         hjust = 0.5, color = "#0f172a",
                                         margin = margin(b = 4, t = 4)),
      axis.text.x         = element_blank(),
      axis.ticks.x        = element_blank(),
      axis.text.y         = element_text(size = 8, color = "#4a5568"),
      axis.title.y        = element_text(size = 8, color = "#4a5568",
                                          margin = margin(r = 4)),
      panel.grid.major.x  = element_blank(),
      panel.grid.minor    = element_blank(),
      panel.grid.major.y  = element_line(color = "#f1f5f9", linewidth = 0.5),
      panel.spacing       = unit(0.8, "lines"),
      strip.text       = element_text(size = 9, face = "bold",
                                      color = "#1a202c",
                                      margin = margin(t = 6, b = 2)),
      strip.background = element_rect(fill = "white", color = NA),
      strip.placement  = "outside",
      plot.background     = element_rect(fill = "white", color = NA),
      panel.background    = element_rect(fill = "white", color = NA),
      plot.margin         = margin(t = 6, r = 8, b = 6, l = 8),
      legend.position = "bottom",
      legend.title        = element_blank(),
      legend.text         = element_text(size = 9, color = "#1a202c"),
      legend.key.size     = unit(0.45, "cm"),
      legend.spacing.x    = unit(0.3, "cm")
    )

  plot_list[[ci]] <- p
}

combined <- (
  wrap_plots(
    plot_list,
    nrow = 2,
    ncol = 2,
    guides = "collect"
  ) +
    plot_annotation(
      title = "Target-Country Attribution Rate by Cue and Model",
      theme = theme(
        plot.title = element_text(
          size = 14,
          face = "bold",
          hjust = 0.5,
          color = "#0f172a",
          margin = margin(b = 8, t = 8)
        ),
        plot.background = element_rect(
          fill = "white",
          color = NA
        )
      )
    )
) &
  theme(legend.position = "bottom")

base_path <- file.path(opt_output, "cue_bar")
ggsave(paste0(base_path, ".pdf"), plot = combined,
       width = 16, height = 12, units = "in", device = "pdf")
ggsave(paste0(base_path, ".png"), plot = combined,
       width = 16, height = 12, units = "in", dpi = 200, device = "png")
cat(sprintf("[OK] Saved: %s.pdf / .png\n", base_path))