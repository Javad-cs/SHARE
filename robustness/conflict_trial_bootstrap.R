#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

# ============================================================
# CLI
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) > 0 && length(args) >= i[1] + 1) {
    args[i[1] + 1]
  } else {
    default
  }
}

results_path   <- get_arg("--results", "results/analysis/all_results.csv")
aliases_path   <- get_arg("--aliases", "notes/aliases.json")
original_path  <- get_arg("--original-exclusions", "notes/cue_exclusions.json")
common_path    <- get_arg("--common-exclusions",
                          "robustness/common_valid_exclusions.json")
output_dir     <- get_arg("--output-dir",
                          "robustness/trial_bootstrap")

B    <- as.integer(get_arg("--bootstrap", "5000"))
SEED <- as.integer(get_arg("--seed", "4160"))

MODELS <- c("gemini", "gpt41", "qwen3vl", "gemma3")
CLUSTERS <- c("pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl")
VISUAL_CUES <- c("flag", "currency", "sign")
TEXTUAL_CUES <- c("native", "endonym", "perspective")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# HELPERS: EXACTLY MIRROR SUBMITTED NORMALIZATION
# ============================================================

build_lookup <- function(path) {
  raw <- fromJSON(path, simplifyVector = FALSE)
  lookup <- list()

  for (canonical in names(raw)) {
    if (startsWith(canonical, "_")) next

    lookup[[trimws(tolower(canonical))]] <- canonical

    for (variant in raw[[canonical]]) {
      lookup[[trimws(tolower(variant))]] <- canonical
    }
  }

  lookup
}

normalize_country <- function(name, lookup) {
  if (is.null(name) || is.na(name) ||
      nchar(trimws(name)) == 0) {
    return("other")
  }

  key <- trimws(tolower(name))
  result <- lookup[[key]]

  if (is.null(result)) "other" else result
}

normalize_target_country <- function(country_id, lookup) {
  readable <- gsub("_", " ", country_id, fixed = TRUE)
  normalize_country(readable, lookup)
}

read_exclusions <- function(path) {
  if (!file.exists(path)) {
    return(data.frame(
      model = character(),
      country = character(),
      cue = character(),
      cluster_id = character(),
      stringsAsFactors = FALSE
    ))
  }

  x <- fromJSON(path, simplifyVector = TRUE)$exclusions

  if (is.null(x) || nrow(x) == 0) {
    return(data.frame(
      model = character(),
      country = character(),
      cue = character(),
      cluster_id = character(),
      stringsAsFactors = FALSE
    ))
  }

  if (!"cluster_id" %in% names(x)) {
    x$cluster_id <- NA_character_
  }

  x$model <- trimws(tolower(as.character(x$model)))
  x$country <- trimws(tolower(as.character(x$country)))
  x$cue <- trimws(tolower(as.character(x$cue)))
  x$cluster_id <- trimws(tolower(as.character(x$cluster_id)))
  x$cluster_id[is.na(x$cluster_id)] <- ""

  x
}

excluded_cells <- function(cells, rules) {
  if (nrow(rules) == 0) {
    return(rep(FALSE, nrow(cells)))
  }

  global <- rules[
    rules$cluster_id %in% c("", "*", "all"),
    , drop = FALSE
  ]

  scoped <- rules[
    !(rules$cluster_id %in% c("", "*", "all")),
    , drop = FALSE
  ]

  global_keys <- paste(
    global$model,
    global$cue,
    global$country,
    sep = "|||"
  )

  scoped_keys <- paste(
    scoped$model,
    scoped$cluster_id,
    scoped$cue,
    scoped$country,
    sep = "|||"
  )

  visual_global <- paste(
    cells$model,
    cells$visual_cue,
    cells$visual_country,
    sep = "|||"
  )

  textual_global <- paste(
    cells$model,
    cells$textual_cue_type,
    cells$textual_country,
    sep = "|||"
  )

  visual_scoped <- paste(
    cells$model,
    cells$cluster_id,
    cells$visual_cue,
    cells$visual_country,
    sep = "|||"
  )

  textual_scoped <- paste(
    cells$model,
    cells$cluster_id,
    cells$textual_cue_type,
    cells$textual_country,
    sep = "|||"
  )

  (visual_global %in% global_keys) |
    (textual_global %in% global_keys) |
    (visual_scoped %in% scoped_keys) |
    (textual_scoped %in% scoped_keys)
}

make_cell_key <- function(df) {
  paste(
    df$model,
    df$cluster_id,
    df$visual_cue,
    df$visual_country,
    df$textual_cue_type,
    df$textual_country,
    sep = "|||"
  )
}

# ============================================================
# LOAD EXACT SUBMITTED DATA
# ============================================================

cat("Loading submitted results...\n")

d <- read.csv(
  results_path,
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = "character",
  na.strings = character(0),
  fileEncoding = "UTF-8"
)

if (nrow(d) != 231040) {
  stop(sprintf(
    "Expected exactly 231040 submitted rows; found %d.",
    nrow(d)
  ))
}

required <- c(
  "image",
  "cluster_id",
  "model",
  "visual_cue",
  "visual_country",
  "textual_cue_type",
  "textual_country",
  "experiment",
  "success",
  "primary_country"
)

missing_cols <- setdiff(required, names(d))

if (length(missing_cols) > 0) {
  stop(sprintf(
    "Missing required columns: %s",
    paste(missing_cols, collapse = ", ")
  ))
}

lookup <- build_lookup(aliases_path)

# ============================================================
# BUILD ALL THEORETICAL CONFLICT TRIALS
# ============================================================

conf <- d[
  d$experiment == "country_classification" &
    d$visual_cue %in% VISUAL_CUES &
    d$textual_cue_type %in% TEXTUAL_CUES,
  , drop = FALSE
]

conf$visual_target <- vapply(
  conf$visual_country,
  normalize_target_country,
  character(1),
  lookup = lookup
)

conf$textual_target <- vapply(
  conf$textual_country,
  normalize_target_country,
  character(1),
  lookup = lookup
)

# Conflict means visual and textual targets are different.
conf <- conf[
  conf$visual_target != conf$textual_target,
  , drop = FALSE
]

conf$cell_key <- make_cell_key(conf)

cell_cols <- c(
  "model",
  "cluster_id",
  "visual_cue",
  "visual_country",
  "textual_cue_type",
  "textual_country",
  "visual_target",
  "textual_target",
  "cell_key"
)

cells <- unique(conf[, cell_cols, drop = FALSE])

if (nrow(cells) != 4320) {
  stop(sprintf(
    "Expected 4320 theoretical conflict cells; found %d.",
    nrow(cells)
  ))
}

cell_sizes <- table(conf$cell_key)

if (length(cell_sizes) != 4320 || any(cell_sizes != 40)) {
  stop(
    "Expected every theoretical conflict cell to contain exactly 40 base-image rows."
  )
}

if (nrow(conf) != 4320 * 40) {
  stop(sprintf(
    "Expected %d conflict rows; found %d.",
    4320 * 40,
    nrow(conf)
  ))
}

# Each cell-image combination must occur exactly once.
cell_image_key <- paste(conf$cell_key, conf$image, sep = "|||IMAGE|||")

if (anyDuplicated(cell_image_key) != 0) {
  stop("Duplicate cell-image trial detected.")
}

# ============================================================
# VERIFY THE BASE-IMAGE RESAMPLING UNITS
# ============================================================

base_images <- unique(
  conf[, c("cluster_id", "image"), drop = FALSE]
)

if (nrow(base_images) != 160) {
  stop(sprintf(
    "Expected 160 unique cluster+image units; found %d.",
    nrow(base_images)
  ))
}

per_cluster_images <- table(base_images$cluster_id)

for (cluster in CLUSTERS) {
  n <- unname(per_cluster_images[cluster])

  if (is.na(n) || n != 40) {
    stop(sprintf(
      "Expected 40 base images in %s; found %s.",
      cluster,
      as.character(n)
    ))
  }
}

cat("Validated: 4320 cells x 40 images; 160 base-image units.\n")

# ============================================================
# NORMALIZE MODEL PREDICTIONS
# ============================================================

conf$success_bool <- tolower(conf$success) == "true"

conf$pred_norm <- NA_character_

successful_idx <- which(conf$success_bool)

conf$pred_norm[successful_idx] <- vapply(
  conf$primary_country[successful_idx],
  normalize_country,
  character(1),
  lookup = lookup
)

conf$trial_category <- NA_character_

conf$trial_category[
  conf$success_bool &
    conf$pred_norm == conf$visual_target
] <- "visual"

conf$trial_category[
  conf$success_bool &
    conf$pred_norm == conf$textual_target
] <- "text"

conf$trial_category[
  conf$success_bool &
    is.na(conf$trial_category)
] <- "other"

# ============================================================
# POLICY ELIGIBILITY
# ============================================================

original_rules <- read_exclusions(original_path)
common_rules <- read_exclusions(common_path)

cells$eligible_original <- !excluded_cells(cells, original_rules)
cells$eligible_common <- !excluded_cells(cells, common_rules)
cells$eligible_all <- TRUE

expected_cells <- c(
  original = 3385L,
  common = 2920L,
  all = 4320L
)

observed_cells <- c(
  original = sum(cells$eligible_original),
  common = sum(cells$eligible_common),
  all = sum(cells$eligible_all)
)

if (!identical(as.integer(observed_cells),
               as.integer(expected_cells))) {
  stop(sprintf(
    paste0(
      "Eligibility check failed. Expected [%s], observed [%s]."
    ),
    paste(expected_cells, collapse = ", "),
    paste(observed_cells, collapse = ", ")
  ))
}

cat(
  sprintf(
    "Eligibility validated: original=%d common=%d all=%d\n",
    observed_cells["original"],
    observed_cells["common"],
    observed_cells["all"]
  )
)

# ============================================================
# REPRODUCE SUBMITTED MODAL-CELL ANALYSIS
# BEFORE TRUSTING NEW STATISTICS
# ============================================================

baseline_rows <- d[
  d$experiment == "country_classification" &
    d$visual_cue == "neutral" &
    d$visual_country == "none" &
    d$textual_cue_type == "en" &
    d$textual_country == "none" &
    tolower(d$success) == "true",
  , drop = FALSE
]

baseline_rows$pred_norm <- vapply(
  baseline_rows$primary_country,
  normalize_country,
  character(1),
  lookup = lookup
)

baseline_sets <- list()

for (model in MODELS) {
  for (cluster in CLUSTERS) {
    x <- baseline_rows[
      baseline_rows$model == model &
        baseline_rows$cluster_id == cluster,
      "pred_norm"
    ]

    x <- x[x != "region"]

    counts <- table(x)
    top_n <- max(counts)

    baseline_sets[[
      paste(model, cluster, sep = "|||")
    ]] <- names(counts[counts == top_n])
  }
}

successful_conf <- conf[
  conf$success_bool,
  , drop = FALSE
]

cell_split <- split(
  successful_conf,
  successful_conf$cell_key
)

classify_modal_cell <- function(cell_row) {
  key <- cell_row$cell_key
  x <- cell_split[[key]]

  if (is.null(x) || nrow(x) == 0) {
    return(c(
      category = "missing",
      support = "0",
      margin = "0",
      n = "0"
    ))
  }

  counts <- sort(
    table(x$pred_norm),
    decreasing = TRUE
  )

  top_n <- as.integer(counts[1])
  n_valid <- sum(counts)

  top_set <- names(counts[counts == top_n])

  second_n <- if (length(counts) >= 2) {
    as.integer(counts[2])
  } else {
    0L
  }

  has_visual <- cell_row$visual_target %in% top_set
  has_text <- cell_row$textual_target %in% top_set

  prior <- baseline_sets[[
    paste(
      cell_row$model,
      cell_row$cluster_id,
      sep = "|||"
    )
  ]]

  has_prior <- any(prior %in% top_set)
  has_vague <- "region" %in% top_set

  category <- if (has_visual && has_text) {
    "tie"
  } else if (has_visual) {
    "visual"
  } else if (has_text) {
    "text"
  } else if (has_prior) {
    "prior"
  } else if (has_vague) {
    "vague"
  } else {
    "other"
  }

  c(
    category = category,
    support = as.character(top_n / n_valid),
    margin = as.character((top_n - second_n) / n_valid),
    n = as.character(n_valid)
  )
}

original_cells <- cells[
  cells$eligible_original,
  , drop = FALSE
]

modal_result <- t(vapply(
  seq_len(nrow(original_cells)),
  function(i) classify_modal_cell(original_cells[i, ]),
  character(4)
))

modal_counts <- table(
  factor(
    modal_result[, "category"],
    levels = c(
      "visual",
      "text",
      "tie",
      "prior",
      "vague",
      "other"
    )
  )
)

expected_modal <- c(
  visual = 2215L,
  text = 628L,
  tie = 19L,
  prior = 347L,
  vague = 5L,
  other = 171L
)

if (!identical(
  as.integer(modal_counts),
  as.integer(expected_modal)
)) {
  cat("Expected:\n")
  print(expected_modal)
  cat("Observed:\n")
  print(modal_counts)
  stop(
    "FAILED to reproduce submitted modal-cell classification."
  )
}

cat(
  "Exact submitted modal-cell counts reproduced: ",
  "2215 / 628 / 19 / 347 / 5 / 171\n",
  sep = ""
)

# ============================================================
# COMPUTE CELL-LEVEL SUPPORT STATISTICS
# ============================================================

analyze_cells <- function(policy_name, eligibility_column) {
  selected <- cells[
    cells[[eligibility_column]],
    , drop = FALSE
  ]

  rows <- vector("list", nrow(selected))

  for (i in seq_len(nrow(selected))) {
    cell <- selected[i, ]
    x <- cell_split[[cell$cell_key]]

    if (is.null(x) || nrow(x) == 0) {
      next
    }

    counts <- sort(
      table(x$pred_norm),
      decreasing = TRUE
    )

    n_valid <- sum(counts)

    visual_n <- sum(
      x$pred_norm == cell$visual_target
    )

    text_n <- sum(
      x$pred_norm == cell$textual_target
    )

    top_n <- as.integer(counts[1])

    second_n <- if (length(counts) >= 2) {
      as.integer(counts[2])
    } else {
      0L
    }

    modal <- classify_modal_cell(cell)

    rows[[i]] <- data.frame(
      policy = policy_name,
      cell_key = cell$cell_key,
      model = cell$model,
      cluster_id = cell$cluster_id,
      visual_cue = cell$visual_cue,
      textual_cue_type = cell$textual_cue_type,
      n_success = n_valid,
      visual_n = visual_n,
      text_n = text_n,
      visual_share = visual_n / n_valid,
      text_share = text_n / n_valid,
      modal_category = modal["category"],
      modal_support = top_n / n_valid,
      modal_margin = (top_n - second_n) / n_valid,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, rows)
}

cell_stats <- rbind(
  analyze_cells("original", "eligible_original"),
  analyze_cells("common", "eligible_common"),
  analyze_cells("all", "eligible_all")
)

write.csv(
  cell_stats,
  file.path(output_dir, "cell_support_stats.csv"),
  row.names = FALSE
)

# ============================================================
# TRIAL-LEVEL SUMMARY
# ============================================================

trial_summary_rows <- list()
row_i <- 1L

for (policy in c("original", "common", "all")) {
  eligible_col <- paste0("eligible_", policy)

  valid_keys <- cells$cell_key[
    cells[[eligible_col]]
  ]

  x <- conf[
    conf$success_bool &
      conf$cell_key %in% valid_keys,
    , drop = FALSE
  ]

  groups <- c("overall", MODELS)

  for (group in groups) {
    y <- if (group == "overall") {
      x
    } else {
      x[x$model == group, , drop = FALSE]
    }

    n <- nrow(y)
    visual_n <- sum(y$trial_category == "visual")
    text_n <- sum(y$trial_category == "text")
    other_n <- sum(y$trial_category == "other")

    trial_summary_rows[[row_i]] <- data.frame(
      policy = policy,
      group = group,
      n_successful_trials = n,
      visual_n = visual_n,
      textual_n = text_n,
      other_n = other_n,
      visual_rate = visual_n / n,
      textual_rate = text_n / n,
      other_rate = other_n / n,
      visual_minus_text = (visual_n - text_n) / n,
      stringsAsFactors = FALSE
    )

    row_i <- row_i + 1L
  }
}

trial_summary <- do.call(
  rbind,
  trial_summary_rows
)

write.csv(
  trial_summary,
  file.path(output_dir, "trial_level_summary.csv"),
  row.names = FALSE
)

# ============================================================
# MEAN CELL-LEVEL TARGET SHARES
#
# This directly avoids treating 21/40 and 40/40 as equivalent:
# one contributes 0.525 visual share, the other contributes 1.0.
# ============================================================

cell_share_rows <- list()
row_i <- 1L

for (policy in c("original", "common", "all")) {
  for (group in c("overall", MODELS)) {
    x <- cell_stats[
      cell_stats$policy == policy,
      , drop = FALSE
    ]

    if (group != "overall") {
      x <- x[x$model == group, , drop = FALSE]
    }

    visual_cells <- x[
      x$modal_category == "visual",
      , drop = FALSE
    ]

    text_cells <- x[
      x$modal_category == "text",
      , drop = FALSE
    ]

    cell_share_rows[[row_i]] <- data.frame(
      policy = policy,
      group = group,
      n_cells = nrow(x),

      mean_visual_target_share =
        mean(x$visual_share),

      mean_textual_target_share =
        mean(x$text_share),

      median_modal_support_visual_cells =
        if (nrow(visual_cells) > 0)
          median(visual_cells$modal_support)
        else NA_real_,

      median_modal_support_text_cells =
        if (nrow(text_cells) > 0)
          median(text_cells$modal_support)
        else NA_real_,

      median_modal_margin_visual_cells =
        if (nrow(visual_cells) > 0)
          median(visual_cells$modal_margin)
        else NA_real_,

      median_modal_margin_text_cells =
        if (nrow(text_cells) > 0)
          median(text_cells$modal_margin)
        else NA_real_,

      stringsAsFactors = FALSE
    )

    row_i <- row_i + 1L
  }
}

cell_share_summary <- do.call(
  rbind,
  cell_share_rows
)

write.csv(
  cell_share_summary,
  file.path(output_dir, "cell_share_summary.csv"),
  row.names = FALSE
)

# ============================================================
# STRATIFIED BASE-IMAGE CLUSTER BOOTSTRAP
#
# Resampling unit = base image.
# Sampling is performed WITHIN each food cluster.
#
# Each resampled image carries ALL its eligible repeated
# model/cue observations together.
# ============================================================

cat(
  sprintf(
    "Running %d base-image bootstrap replicates (seed=%d)...\n",
    B,
    SEED
  )
)

set.seed(SEED)

# Fixed image ordering within each food cluster.
image_lists <- lapply(
  CLUSTERS,
  function(cluster) {
    sort(unique(
      base_images$image[
        base_images$cluster_id == cluster
      ]
    ))
  }
)

names(image_lists) <- CLUSTERS

# Bootstrap weights: 40 original images x B replicates.
# Each column says how many times each image was sampled.
bootstrap_weights <- list()

for (cluster in CLUSTERS) {
  n_img <- length(image_lists[[cluster]])

  if (n_img != 40) {
    stop(
      sprintf(
        "%s does not contain exactly 40 images.",
        cluster
      )
    )
  }

  W <- matrix(
    0L,
    nrow = n_img,
    ncol = B
  )

  for (b in seq_len(B)) {
    draws <- sample.int(
      n_img,
      size = n_img,
      replace = TRUE
    )

    W[, b] <- tabulate(
      draws,
      nbins = n_img
    )
  }

  bootstrap_weights[[cluster]] <- W
}

make_image_matrix <- function(
  conflict_rows,
  cluster,
  group
) {
  images <- image_lists[[cluster]]

  M <- matrix(
    0,
    nrow = length(images),
    ncol = 3,
    dimnames = list(
      images,
      c("total", "visual", "text")
    )
  )

  x <- conflict_rows[
    conflict_rows$cluster_id == cluster,
    , drop = FALSE
  ]

  if (group != "overall") {
    x <- x[
      x$model == group,
      , drop = FALSE
    ]
  }

  for (i in seq_along(images)) {
    y <- x[
      x$image == images[i],
      , drop = FALSE
    ]

    M[i, "total"] <- nrow(y)
    M[i, "visual"] <-
      sum(y$trial_category == "visual")
    M[i, "text"] <-
      sum(y$trial_category == "text")
  }

  M
}

bootstrap_draw_rows <- list()
bootstrap_ci_rows <- list()

draw_row <- 1L
ci_row <- 1L

for (policy in c("original", "common", "all")) {
  eligible_col <- paste0(
    "eligible_",
    policy
  )

  valid_keys <- cells$cell_key[
    cells[[eligible_col]]
  ]

  x <- conf[
    conf$success_bool &
      conf$cell_key %in% valid_keys,
    , drop = FALSE
  ]

  for (group in c("overall", MODELS)) {
    totals <- matrix(
      0,
      nrow = 3,
      ncol = B,
      dimnames = list(
        c("total", "visual", "text"),
        NULL
      )
    )

    for (cluster in CLUSTERS) {
      M <- make_image_matrix(
        x,
        cluster,
        group
      )

      # 3 x B matrix:
      totals <- totals +
        t(M) %*% bootstrap_weights[[cluster]]
    }

    visual_rate <-
      totals["visual", ] / totals["total", ]

    textual_rate <-
      totals["text", ] / totals["total", ]

    gap <- visual_rate - textual_rate

    point <- trial_summary[
      trial_summary$policy == policy &
        trial_summary$group == group,
      , drop = FALSE
    ]

    bootstrap_ci_rows[[ci_row]] <- data.frame(
      policy = policy,
      group = group,
      bootstrap_replicates = B,

      visual_rate =
        point$visual_rate,

      visual_ci_low =
        unname(quantile(
          visual_rate,
          0.025,
          type = 7
        )),

      visual_ci_high =
        unname(quantile(
          visual_rate,
          0.975,
          type = 7
        )),

      textual_rate =
        point$textual_rate,

      textual_ci_low =
        unname(quantile(
          textual_rate,
          0.025,
          type = 7
        )),

      textual_ci_high =
        unname(quantile(
          textual_rate,
          0.975,
          type = 7
        )),

      visual_minus_text =
        point$visual_minus_text,

      gap_ci_low =
        unname(quantile(
          gap,
          0.025,
          type = 7
        )),

      gap_ci_high =
        unname(quantile(
          gap,
          0.975,
          type = 7
        )),

      stringsAsFactors = FALSE
    )

    ci_row <- ci_row + 1L

    bootstrap_draw_rows[[draw_row]] <- data.frame(
      policy = policy,
      group = group,
      replicate = seq_len(B),
      visual_rate = visual_rate,
      textual_rate = textual_rate,
      visual_minus_text = gap,
      stringsAsFactors = FALSE
    )

    draw_row <- draw_row + 1L
  }
}

bootstrap_ci <- do.call(
  rbind,
  bootstrap_ci_rows
)

bootstrap_draws <- do.call(
  rbind,
  bootstrap_draw_rows
)

write.csv(
  bootstrap_ci,
  file.path(output_dir, "bootstrap_ci.csv"),
  row.names = FALSE
)

write.csv(
  bootstrap_draws,
  file.path(output_dir, "bootstrap_draws.csv"),
  row.names = FALSE
)

# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_path <- file.path(
  output_dir,
  "trial_bootstrap_report.txt"
)

sink(report_path)

cat("============================================================\n")
cat("CONFLICT TRIAL-LEVEL + BASE-IMAGE BOOTSTRAP ANALYSIS\n")
cat("============================================================\n\n")

cat(sprintf("Submitted rows:        %d\n", nrow(d)))
cat(sprintf("Conflict cells:        %d\n", nrow(cells)))
cat(sprintf("Base images:           %d\n", nrow(base_images)))
cat(sprintf("Bootstrap replicates:  %d\n", B))
cat(sprintf("Bootstrap seed:        %d\n\n", SEED))

cat(
  "Bootstrap design: 40 base images are resampled WITH REPLACEMENT\n",
  "within each of the four food clusters. All repeated model/cue\n",
  "observations attached to a sampled image are retained together.\n\n",
  sep = ""
)

cat("SUBMITTED MODAL ANALYSIS REPRODUCTION\n")
cat("-------------------------------------\n")
print(modal_counts)
cat("\n")

for (policy in c("original", "common", "all")) {
  cat("============================================================\n")
  cat(sprintf("POLICY: %s\n", toupper(policy)))
  cat("============================================================\n\n")

  cat("Trial-level estimates:\n")
  print(
    trial_summary[
      trial_summary$policy == policy,
      ,
      drop = FALSE
    ],
    row.names = FALSE
  )

  cat("\nCell-level target-share/support summary:\n")
  print(
    cell_share_summary[
      cell_share_summary$policy == policy,
      ,
      drop = FALSE
    ],
    row.names = FALSE
  )

  cat("\nBase-image bootstrap 95% intervals:\n")
  print(
    bootstrap_ci[
      bootstrap_ci$policy == policy,
      ,
      drop = FALSE
    ],
    row.names = FALSE
  )

  cat("\n")
}

sink()

cat("\n[DONE]\n")
cat(sprintf("Report: %s\n", report_path))
cat(sprintf(
  "Trial summary: %s\n",
  file.path(output_dir, "trial_level_summary.csv")
))
cat(sprintf(
  "Cell support: %s\n",
  file.path(output_dir, "cell_share_summary.csv")
))
cat(sprintf(
  "Bootstrap CIs: %s\n",
  file.path(output_dir, "bootstrap_ci.csv")
))
cat(sprintf(
  "Raw bootstrap draws: %s\n",
  file.path(output_dir, "bootstrap_draws.csv")
))
