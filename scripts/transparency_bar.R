#!/usr/bin/env Rscript
# transparency_bar.R
# ------------------
# Reads transparency_flips.csv and produces two figures:
#
# Figure A (main): 4 panels (one per model), 3 bars per panel (one per cue)
#                  aggregated across all clusters
#
# Figure B (appendix): 16 panels (4 clusters x 4 models), 3 bars per panel
#                      not aggregated
#
# Each bar = 100% height:
#   acknowledged (bottom), unclear (middle), omitted (top)
#
# Usage:
#   Rscript scripts/transparency_bar.R
#   Rscript scripts/transparency_bar.R \
#     --input results/analysis/transparency/transparency_flips.csv

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) > 0 && length(args) >= i[1] + 1) {
    args[i[1] + 1]
  } else {
    default
  }
}

opt_input <- get_arg(
  "--input",
  "results/analysis/transparency/transparency_flips.csv"
)
opt_output <- get_arg(
  "--output-dir",
  "results/analysis/transparency"
)

MODEL_LABELS <- c(
  gemini  = "Gemini",
  gpt41   = "GPT-4.1",
  qwen3vl = "Qwen3-VL",
  gemma3  = "Gemma3"
)

CUE_LABELS <- c(
  flag     = "Flag",
  currency = "Banknote",
  sign     = "Reservation sign"
)

CLUSTER_NAMES <- c(
  pilaf_cl    = "Pilaf",
  mantou_cl   = "Mantou",
  samosa_cl   = "Samosa",
  meatball_cl = "Meatball"
)

ALL_MODELS   <- c("gemini", "gpt41", "qwen3vl", "gemma3")
ALL_CLUSTERS <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
ALL_CUES     <- c("flag", "currency", "sign")
ALL_LABELS   <- c("acknowledged", "unclear", "omitted")

MIN_LABEL_PCT <- 3

LABEL_DISPLAY <- c(
  acknowledged = "Acknowledged",
  unclear      = "Unclear",
  omitted      = "Omitted"
)

COL_ACKNOWLEDGED <- "#16a34a"
COL_UNCLEAR      <- "#d97706"
COL_OMITTED      <- "#dc2626"

cat("Reading CSV...\n")

if (!file.exists(opt_input)) {
  stop(sprintf("Input file not found: %s", opt_input))
}

df <- read.csv(
  opt_input,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_columns <- c("model", "cluster", "vcue", "judge_label")
missing_columns <- setdiff(required_columns, names(df))

if (length(missing_columns) > 0) {
  stop(
    paste0(
      "Input CSV is missing required column(s): ",
      paste(missing_columns, collapse = ", ")
    )
  )
}

df$judge_label <- trimws(tolower(df$judge_label))

if (any(is.na(df$judge_label) | df$judge_label == "")) {
  stop(
    "The input CSV contains empty judge_label values. ",
    "Complete or rerun those judgments before plotting."
  )
}

if (any(df$judge_label == "error")) {
  n_errors <- sum(df$judge_label == "error")
  stop(
    sprintf("Found %d row(s) with judge_label = 'error'. ", n_errors),
    "Delete or rerun those error cases before plotting."
  )
}

unexpected_labels <- setdiff(unique(df$judge_label), ALL_LABELS)
if (length(unexpected_labels) > 0) {
  stop(
    paste0(
      "Unexpected judge_label value(s): ",
      paste(unexpected_labels, collapse = ", ")
    )
  )
}

unexpected_models <- setdiff(unique(df$model), ALL_MODELS)
unexpected_clusters <- setdiff(unique(df$cluster), ALL_CLUSTERS)
unexpected_cues <- setdiff(unique(df$vcue), ALL_CUES)

if (length(unexpected_models) > 0) {
  stop(paste0("Unexpected model value(s): ", paste(unexpected_models, collapse = ", ")))
}
if (length(unexpected_clusters) > 0) {
  stop(paste0("Unexpected cluster value(s): ", paste(unexpected_clusters, collapse = ", ")))
}
if (length(unexpected_cues) > 0) {
  stop(paste0("Unexpected vcue value(s): ", paste(unexpected_cues, collapse = ", ")))
}

cat(sprintf("Total flips: %d\n", nrow(df)))

label_totals <- df %>%
  count(judge_label, name = "n") %>%
  complete(judge_label = ALL_LABELS, fill = list(n = 0))

for (label in ALL_LABELS) {
  n_label <- label_totals$n[label_totals$judge_label == label]
  pct_label <- if (nrow(df) > 0) 100 * n_label / nrow(df) else 0
  cat(sprintf("%s: %d (%.1f%%)\n", LABEL_DISPLAY[[label]], n_label, pct_label))
}

summarise_judgments <- function(data, group_columns) {
  data %>%
    count(across(all_of(group_columns)), judge_label, name = "n") %>%
    complete(
      !!!syms(group_columns),
      judge_label = ALL_LABELS,
      fill = list(n = 0)
    ) %>%
    group_by(across(all_of(group_columns))) %>%
    mutate(
      total = sum(n),
      pct = if_else(total > 0, 100 * n / total, 0)
    ) %>%
    ungroup()
}

summary_a <- summarise_judgments(df, c("model", "vcue"))
summary_b <- summarise_judgments(df, c("cluster", "model", "vcue"))

make_panel <- function(data, title = "", show_legend = FALSE, show_y = TRUE) {
  long <- data %>%
    mutate(
      type = factor(
        judge_label,
        levels = ALL_LABELS,
        labels = LABEL_DISPLAY[ALL_LABELS]
      ),
      vcue = factor(
        vcue,
        levels = ALL_CUES,
        labels = CUE_LABELS[ALL_CUES]
      )
    )

  n_labels <- long %>% distinct(vcue, total)

  ggplot(long, aes(x = vcue, y = pct, fill = type)) +
    geom_col(
      width = 0.65,
      color = "white",
      linewidth = 0.3,
      position = position_stack(reverse = TRUE)
    ) +
    geom_text(
      data = long[long$pct >= MIN_LABEL_PCT, ],
      aes(label = sprintf("%.1f%%", pct)),
      position = position_stack(vjust = 0.5, reverse = TRUE),
      size = 3.0,
      color = "white",
      fontface = "bold"
    ) +
    geom_text(
      data = n_labels,
      aes(x = vcue, y = 102, label = paste0("n=", total)),
      inherit.aes = FALSE,
      size = 3.2,
      color = "#374151",
      vjust = 0
    ) +
    scale_fill_manual(
      values = c(
        "Acknowledged" = COL_ACKNOWLEDGED,
        "Unclear"      = COL_UNCLEAR,
        "Omitted"      = COL_OMITTED
      ),
      breaks = c("Acknowledged", "Unclear", "Omitted"),
      name = NULL,
      drop = FALSE
    ) +
    scale_y_continuous(
      limits = c(0, 112),
      breaks = c(0, 25, 50, 75, 100),
      labels = c("0%", "25%", "50%", "75%", "100%"),
      expand = c(0, 0)
    ) +
    labs(
      title = title,
      x = NULL,
      y = if (show_y) "Share of Attribution Flips (%)" else NULL
    ) +
    theme_minimal(base_family = "sans") +
    theme(
      plot.title = element_text(
        size = 10, face = "bold", hjust = 0.5,
        color = "#0f172a", margin = margin(b = 4, t = 2)
      ),
      axis.text.x = element_text(size = 9, color = "#1a202c", face = "bold"),
      axis.text.y = if (show_y) element_text(size = 8, color = "#4a5568") else element_blank(),
      axis.title.y = if (show_y) {
        element_text(size = 8, color = "#4a5568", margin = margin(r = 4))
      } else {
        element_blank()
      },
      axis.ticks.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#f1f5f9", linewidth = 0.5),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 4, r = 6, b = 4, l = 6),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = 9, color = "#1a202c"),
      legend.key.size = unit(0.4, "cm")
    )
}

cat("Building Figure A...\n")
plots_a <- list()

for (mi in seq_along(ALL_MODELS)) {
  model <- ALL_MODELS[mi]
  model_name <- MODEL_LABELS[model]
  sub <- summary_a[summary_a$model == model, ]

  plots_a[[mi]] <- make_panel(
    data = sub,
    title = model_name,
    show_legend = (mi == length(ALL_MODELS)),
    show_y = (mi == 1)
  )
}

fig_a <- wrap_plots(plots_a, nrow = 1, ncol = 4) +
  plot_annotation(
    title = "Visual-Cue Acknowledgment in Reasoning Among Country-Attribution Flips",
    caption = paste0(
      "Counts pooled across clusters; ",
      "n = judged country-attribution flips per model x cue."
    ),
    theme = theme(
      plot.title = element_text(
        size = 13, face = "bold", hjust = 0.5,
        color = "#0f172a", margin = margin(b = 6, t = 6)
      ),
      plot.caption = element_text(
        size = 8, color = "#0f172a", hjust = 0.5,
        margin = margin(t = 4)
      ),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

cat("Building Figure B...\n")
plots_b <- list()

for (ci in seq_along(ALL_CLUSTERS)) {
  cluster_id <- ALL_CLUSTERS[ci]
  cluster_name <- CLUSTER_NAMES[cluster_id]

  for (mi in seq_along(ALL_MODELS)) {
    model <- ALL_MODELS[mi]
    model_name <- MODEL_LABELS[model]
    sub <- summary_b[
      summary_b$cluster == cluster_id & summary_b$model == model,
    ]

    is_left <- (mi == 1)
    is_last <- (ci == length(ALL_CLUSTERS) && mi == length(ALL_MODELS))
    title <- if (ci == 1) model_name else ""

    p <- make_panel(
      data = sub,
      title = title,
      show_legend = is_last,
      show_y = is_left
    )

    if (is_left) {
      p <- p +
        labs(y = cluster_name) +
        theme(
          axis.title.y = element_text(
            size = 9, face = "bold", color = "#0f172a",
            angle = 90, margin = margin(r = 4)
          )
        )
    }

    plots_b[[length(plots_b) + 1]] <- p
  }
}

fig_b <- wrap_plots(
  plots_b,
  nrow = length(ALL_CLUSTERS),
  ncol = length(ALL_MODELS)
) +
  plot_annotation(
    title = "Visual-Cue Acknowledgment Among Country-Attribution Flips - By Cluster",
    caption = "n = judged country-attribution flips per cluster x model x cue.",
    theme = theme(
      plot.title = element_text(
        size = 13, face = "bold", hjust = 0.5,
        color = "#0f172a", margin = margin(b = 6, t = 6)
      ),
      plot.caption = element_text(
        size = 12, color = "#6b7280", hjust = 0.5,
        margin = margin(t = 4)
      ),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

dir.create(opt_output, showWarnings = FALSE, recursive = TRUE)

base_a <- file.path(opt_output, "transparency_bar_A")
ggsave(
  paste0(base_a, ".pdf"), plot = fig_a,
  width = 12, height = 5, units = "in", device = "pdf"
)
ggsave(
  paste0(base_a, ".png"), plot = fig_a,
  width = 12, height = 5, units = "in", dpi = 200, device = "png"
)
cat(sprintf("[OK] Figure A: %s.pdf / .png\n", base_a))

base_b <- file.path(opt_output, "transparency_bar_B")
ggsave(
  paste0(base_b, ".pdf"), plot = fig_b,
  width = 14, height = 17, units = "in", device = "pdf"
)
ggsave(
  paste0(base_b, ".png"), plot = fig_b,
  width = 14, height = 17, units = "in", dpi = 200, device = "png"
)
cat(sprintf("[OK] Figure B: %s.pdf / .png\n", base_b))
