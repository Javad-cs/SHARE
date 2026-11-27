#!/usr/bin/env Rscript
# clash_matrix_csv.R
# --------------
# Generates publication-ready clash matrix heatmaps.
# Output: one figure per cluster, each a 1x4 grid (one panel per model).
#
# Classification priority:
#   If visual AND textual both in top_countries → Purple (tie)
#   Elif visual in top_countries                → Blue
#   Elif textual in top_countries               → Green
#   Elif prior in top_countries                 → Yellow
#   Elif vague in top_countries                 → Orange
#   Else                                        → Red
#   No valid predictions                        → Gray
#
# Usage:
#   Rscript scripts/clash_matrix_csv.R
#   Rscript scripts/clash_matrix_csv.R --model gemini
#   Rscript scripts/clash_matrix_csv.R --cluster pilaf_cl

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

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

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

BASE_COLORS <- list(
  visual   = c(0.173, 0.380, 0.733),  # blue
  text     = c(0.106, 0.620, 0.322),  # green
  tie      = c(0.506, 0.149, 0.659),  # purple
  prior    = c(0.780, 0.737, 0.071),  # yellow
  vague    = c(0.867, 0.494, 0.071),  # orange
  other    = c(0.780, 0.180, 0.180),  # red
  missing  = c(0.88,  0.88,  0.88 ),  # gray
  diagonal = c(0.93,  0.93,  0.93 ),  # light gray
  excluded = c(0.97,  0.97,  0.97 )   # near white
)

CAT_ORDER <- c("visual", "text", "tie", "prior", "vague", "other")

CAT_LABELS <- c(
  visual   = "Visual-cue country",
  text     = "Textual-cue country",
  tie      = "Visual-textual tie",
  prior    = "Baseline country",
  vague    = "Region-level response",
  other    = "Other country"
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

abbr_country <- function(country) {
  a <- COUNTRY_ABBR[country]
  if (is.na(a)) toupper(substr(country, 1, 2)) else unname(a)
}

blend_color <- function(category, intensity) {
  base <- BASE_COLORS[[category]]
  if (is.null(base)) base <- BASE_COLORS$missing
  if (category %in% c("missing", "diagonal", "excluded"))
    return(rgb(base[1], base[2], base[3]))
  alpha <- max(0.12, min(1.0, intensity))
  mixed <- base * alpha + c(1, 1, 1) * (1 - alpha)
  rgb(mixed[1], mixed[2], mixed[3])
}

get_label_color <- function(hex_color) {
  r   <- strtoi(substr(hex_color, 2, 3), 16L) / 255
  g   <- strtoi(substr(hex_color, 4, 5), 16L) / 255
  b   <- strtoi(substr(hex_color, 6, 7), 16L) / 255
  if (0.299*r + 0.587*g + 0.114*b < 0.52) "#FFFFFF" else "#1a202c"
}

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

build_lookup <- function(aliases_path) {
  raw    <- fromJSON(aliases_path, simplifyVector = FALSE)
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
# into the canonical label used for normalized model predictions.
normalize_target_country <- function(country_id, lookup) {
  readable_name <- gsub("_", " ", country_id, fixed = TRUE)
  normalize_country(readable_name, lookup)
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

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
  if (nrow(rows) == 0) return(NULL)

  counts <- list()
  for (i in seq_len(nrow(rows))) {
    if (tolower(rows$success[i]) != "true") next
    norm <- normalize_country(rows$primary_country[i], lookup)
    if (norm != "region")
      counts[[norm]] <- (if (is.null(counts[[norm]])) 0L
                         else counts[[norm]]) + 1L
  }
  if (length(counts) == 0) return(NULL)

  count_vector <- unlist(counts)
  top_count <- max(count_vector)
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

# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

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
# BUILD DATA FRAME
# ─────────────────────────────────────────────

build_matrix_df <- function(model, cluster_id, countries, results_df,
                             exclusions_path, lookup) {
  exclusions <- load_exclusions(exclusions_path, model, cluster_id)
  prior      <- load_baseline(results_df, cluster_id, model, lookup)
  n          <- length(countries)
  rows       <- list()

  for (vi in seq_along(VISUAL_CUES)) {
    vcue <- VISUAL_CUES[vi]
    for (vci in seq_along(countries)) {
      vcountry <- countries[vci]
      for (ti in seq_along(TEXTUAL_CUES)) {
        tcue <- TEXTUAL_CUES[ti]
        for (tci in seq_along(countries)) {
          tcountry <- countries[tci]
          row_idx  <- (vi - 1) * n + vci
          col_idx  <- (ti - 1) * n + tci

          if (vcountry == tcountry) {
            rows[[length(rows) + 1]] <- data.frame(
              row_idx = row_idx, col_idx = col_idx,
              vcue = vcue, vcountry = vcountry,
              tcue = tcue, tcountry = tcountry,
              cat = "diagonal", intensity = 0.0,
              label = "", stringsAsFactors = FALSE)
            next
          }

          if (is_excluded(vcue, vcountry, exclusions) ||
              is_excluded(tcue, tcountry, exclusions)) {
            rows[[length(rows) + 1]] <- data.frame(
              row_idx = row_idx, col_idx = col_idx,
              vcue = vcue, vcountry = vcountry,
              tcue = tcue, tcountry = tcountry,
              cat = "excluded", intensity = 0.0,
              label = "x", stringsAsFactors = FALSE)
            next
          }

          preds  <- load_cell_preds(results_df, cluster_id, model,
                                    vcue, vcountry, tcue, tcountry, lookup)
          result <- classify_cell(preds, vcountry, tcountry, prior, lookup)

          lbl <- if (!result$cat %in%
                     c("diagonal", "excluded", "missing"))
                   paste0(round(result$intensity * 100), "%") else ""

          rows[[length(rows) + 1]] <- data.frame(
            row_idx = row_idx, col_idx = col_idx,
            vcue = vcue, vcountry = vcountry,
            tcue = tcue, tcountry = tcountry,
            cat = result$cat, intensity = result$intensity,
            label = lbl, stringsAsFactors = FALSE)
        }
      }
    }
  }

  df            <- bind_rows(rows)
  df$fill_color <- mapply(blend_color, df$cat, df$intensity)
  df$lbl_color  <- sapply(df$fill_color, get_label_color)
  df$prior <- if (length(prior) == 0) {
    "unknown"
  } else {
    paste(prior, collapse = " / ")
  }
  df$countries  <- list(countries)
  df
}

# ─────────────────────────────────────────────
# MAKE ONE PANEL
# ─────────────────────────────────────────────

make_panel <- function(df, model, cluster_id,
                       show_row_labels = TRUE,
                       show_col_labels = TRUE,
                       show_title      = TRUE,
                       show_legend     = FALSE) {

  countries <- df$countries[[1]]
  n         <- length(countries)
  nv        <- length(VISUAL_CUES)
  nt        <- length(TEXTUAL_CUES)
  prior     <- df$prior[1]

  df$px <- df$col_idx
  df$py <- (nv * n + 1) - df$row_idx

  max_px <- nt * n
  max_py <- nv * n

  row_lbl <- data.frame(
    py    = (nv * n + 1) - seq_len(nv * n),
    label = sapply(seq_len(nv * n), function(ri) {
      vi  <- ceiling(ri / n)
      vci <- ((ri - 1) %% n) + 1
      paste0(abbr_country(countries[vci]), "-", CUE_ABB[VISUAL_CUES[vi]])
    }),
    stringsAsFactors = FALSE
  )

  col_lbl <- data.frame(
    px    = seq_len(nt * n),
    label = sapply(seq_len(nt * n), function(ci) {
      ti  <- ceiling(ci / n)
      tci <- ((ci - 1) %% n) + 1
      paste0(abbr_country(countries[tci]), "-", CUE_ABB[TEXTUAL_CUES[ti]])
    }),
    stringsAsFactors = FALSE
  )

  # Fixed legend — always show all semantic categories
  present <- CAT_ORDER
  leg_labels <- CAT_LABELS[present]

  legend_cols <- 3
  legend_x <- (seq_len(legend_cols) - 1) * (max_px / legend_cols) + 0.8
  legend_y <- c(-1.7, -3.2)

  legend_df <- data.frame(
    lx = rep(legend_x, times = 2)[seq_along(present)],
    ly = rep(legend_y, each = legend_cols)[seq_along(present)],
    label = unname(leg_labels),
    color = sapply(
      present,
      function(category) blend_color(category, 0.85)
    ),
    stringsAsFactors = FALSE
  )

  left_margin <- if (show_row_labels) -5.0 else -0.5
  top_margin  <- if (show_col_labels)  2.0 else  0.6
  bot_margin  <- -4.0  # always same, legend drawn outside grid

  df_pct  <- df[df$label != "" & df$label != "x", ]
  df_excl <- df[df$label == "x", ]

  p <- ggplot(df, aes(x = px, y = py)) +

    geom_tile(aes(fill = fill_color), color = "white",
              linewidth = 0.3, width = 0.96, height = 0.96) +
    scale_fill_identity() +
    scale_color_identity() +

    geom_text(data = df_pct,
              aes(label = label, color = lbl_color),
              size = 2.2, fontface = "bold") +

    geom_text(data = df_excl,
              aes(label = "\u00d7"),
              color = "#aaaaaa", size = 2.5)

  if (show_row_labels) {
    p <- p + geom_text(
      data = row_lbl,
      aes(x = -0.1, y = py, label = label),
      inherit.aes = FALSE,
      hjust = 1, size = 3.2, color = "#1a202c", family = "mono"
    )
  }

  if (show_col_labels) {
    p <- p + geom_text(
      data = col_lbl,
      aes(x = px, y = max_py + 0.6, label = label),
      inherit.aes = FALSE,
      hjust = 0, vjust = 0, angle = 60,
      size = 3.2, color = "#1a202c", family = "mono"
    )
  }

  if (show_legend) {
    p <- p +
      geom_tile(
        data = legend_df,
        aes(x = lx, y = ly, fill = color),
        inherit.aes = FALSE,
        width = 0.65,
        height = 0.50
      ) +
      geom_text(
        data = legend_df,
        aes(
          x = lx + 0.50,
          y = ly,
          label = label
        ),
        inherit.aes = FALSE,
        hjust = 0,
        size = 4.0,
        color = "#1a202c"
      ) +
      annotate(
        "text",
        x = max_px / 2,
        y = -0.65,
        label = "Most frequent attribution",
        hjust = 0.5,
        vjust = 0.5,
        size = 4.2,
        fontface = "bold",
        color = "#374151"
      )
  }

  if (show_title) {
    p <- p + ggtitle(MODEL_LABELS[model])
  }

  p +
    coord_fixed(xlim = c(left_margin, max_px + 0.5),
                ylim = c(bot_margin,  max_py + top_margin),
                expand = FALSE) +
    theme_void(base_family = "sans") +
    theme(
      plot.title      = element_text(size = 10, face = "bold",
                                     hjust = 0.5, color = "#0f172a",
                                     margin = margin(b = 3, t = 6)),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin     = margin(t = 1, r = 1, b = 1, l = 1)
    )
}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

cat("Loading aliases...\n")
lookup <- build_lookup(opt_aliases)
cat("Loading all-results CSV...\n")
all_results <- load_all_results(opt_results_csv)
cl_raw <- fromJSON(opt_clusters, simplifyVector = FALSE)
dir.create(opt_output, showWarnings = FALSE, recursive = TRUE)

for (cluster_id in clusters) {
  cluster_name <- CLUSTER_NAMES[cluster_id]
  cat(sprintf("\nCluster: %s\n", cluster_name))

  cl_entry <- Filter(function(x) x$cluster_id == cluster_id, cl_raw$clusters)
  if (length(cl_entry) == 0) {
    cat(sprintf("  [SKIP] not found in clusters.json\n"))
    next
  }

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

  cat(sprintf("  Countries: %s\n", paste(countries, collapse = ", ")))
  cat(sprintf(
    "  Target mappings: %s\n",
    paste(sprintf("%s -> %s", countries, canonical_countries),
          collapse = ", ")
  ))
  cat(sprintf(
    "  Baseline country will be computed per model from neutral image + English prompt\n"
  ))

  plot_list <- list()

  for (mi in seq_along(models)) {
    model <- models[mi]
    cat(sprintf("  Building %s ...\n", model))

    df <- tryCatch(
      build_matrix_df(model, cluster_id, countries,
                      all_results, opt_exclusions, lookup),
      error = function(e) {
        cat(sprintf("  [ERROR] %s: %s\n", model, conditionMessage(e)))
        NULL
      }
    )

    if (!is.null(df)) {
      cat(sprintf("    Baseline country: %s\n", df$prior[1]))
    }

    if (is.null(df)) {
      plot_list[[mi]] <- ggplot() + theme_void() +
        annotate("text", x = 0.5, y = 0.5,
                 label = "NO DATA", color = "#cc0000", size = 3)
      next
    }

    is_left   <- (mi == 1)
    is_legend <- (mi == length(models))

    plot_list[[mi]] <- make_panel(
      df              = df,
      model           = model,
      cluster_id      = cluster_id,
      show_row_labels = is_left,
      show_col_labels = TRUE,
      show_title      = TRUE,
      show_legend     = is_legend
    )
  }

  combined <- wrap_plots(plot_list, nrow = 1, ncol = length(models)) +
    plot_annotation(
      title = paste0(
        "Country-Attribution Outcomes Under Visual-Textual Cue Clash: ",
        cluster_name
      ),
      theme = theme(
        plot.title      = element_text(size = 13, face = "bold",
                                       hjust = 0.5, color = "#0f172a",
                                       margin = margin(b = 6, t = 6)),
        plot.background = element_rect(fill = "white", color = NA)
      )
    )

  base_path <- file.path(opt_output, paste0("clash_", cluster_id))
  ggsave(paste0(base_path, ".pdf"), plot = combined,
         width = 4 * 9, height = 11, units = "in", device = "pdf")
  ggsave(paste0(base_path, ".png"), plot = combined,
         width = 4 * 9, height = 11, units = "in", dpi = 180, device = "png")
  cat(sprintf("  [OK] %s.pdf / .png\n", base_path))
}