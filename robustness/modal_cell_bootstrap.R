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

results_path <- get_arg(
  "--results",
  "results/analysis/all_results.csv"
)

aliases_path <- get_arg(
  "--aliases",
  "notes/aliases.json"
)

original_exclusions_path <- get_arg(
  "--original-exclusions",
  "notes/cue_exclusions.json"
)

common_exclusions_path <- get_arg(
  "--common-exclusions",
  "robustness/common_valid_exclusions.json"
)

output_dir <- get_arg(
  "--output-dir",
  "robustness/modal_bootstrap"
)

B <- as.integer(
  get_arg("--bootstrap", "5000")
)

SEED <- as.integer(
  get_arg("--seed", "4160")
)

MODELS <- c(
  "gemini",
  "gpt41",
  "qwen3vl",
  "gemma3"
)

CLUSTERS <- c(
  "pilaf_cl",
  "mantou_cl",
  "samosa_cl",
  "meatball_cl"
)

VISUAL_CUES <- c(
  "flag",
  "currency",
  "sign"
)

TEXTUAL_CUES <- c(
  "native",
  "endonym",
  "perspective"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

build_lookup <- function(path) {
  raw <- fromJSON(
    path,
    simplifyVector = FALSE
  )

  lookup <- list()

  for (canonical in names(raw)) {
    if (startsWith(canonical, "_")) {
      next
    }

    lookup[[
      trimws(tolower(canonical))
    ]] <- canonical

    for (variant in raw[[canonical]]) {
      lookup[[
        trimws(tolower(variant))
      ]] <- canonical
    }
  }

  lookup
}

normalize_country <- function(name, lookup) {
  if (
    is.null(name) ||
    is.na(name) ||
    nchar(trimws(name)) == 0
  ) {
    return("other")
  }

  key <- trimws(tolower(name))
  result <- lookup[[key]]

  if (is.null(result)) {
    "other"
  } else {
    result
  }
}

normalize_target_country <- function(
  country_id,
  lookup
) {
  readable <- gsub(
    "_",
    " ",
    country_id,
    fixed = TRUE
  )

  normalize_country(
    readable,
    lookup
  )
}

# ============================================================
# EXCLUSION HANDLING
# ============================================================

read_exclusions <- function(path) {
  x <- fromJSON(
    path,
    simplifyVector = TRUE
  )$exclusions

  if (
    is.null(x) ||
    nrow(x) == 0
  ) {
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

  x$model <- trimws(
    tolower(as.character(x$model))
  )

  x$country <- trimws(
    tolower(as.character(x$country))
  )

  x$cue <- trimws(
    tolower(as.character(x$cue))
  )

  x$cluster_id <- trimws(
    tolower(as.character(x$cluster_id))
  )

  x$cluster_id[
    is.na(x$cluster_id)
  ] <- ""

  x
}

excluded_cells <- function(cells, rules) {
  if (nrow(rules) == 0) {
    return(rep(FALSE, nrow(cells)))
  }

  global <- rules[
    rules$cluster_id %in% c(
      "",
      "*",
      "all"
    ),
    ,
    drop = FALSE
  ]

  scoped <- rules[
    !(rules$cluster_id %in% c(
      "",
      "*",
      "all"
    )),
    ,
    drop = FALSE
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
# LOAD SUBMITTED DATA
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
    "Expected 231040 rows, found %d.",
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

missing_cols <- setdiff(
  required,
  names(d)
)

if (length(missing_cols) > 0) {
  stop(sprintf(
    "Missing columns: %s",
    paste(
      missing_cols,
      collapse = ", "
    )
  ))
}

lookup <- build_lookup(
  aliases_path
)

# ============================================================
# BUILD THEORETICAL CONFLICT CELLS
# ============================================================

conf <- d[
  d$experiment == "country_classification" &
    d$visual_cue %in% VISUAL_CUES &
    d$textual_cue_type %in% TEXTUAL_CUES,
  ,
  drop = FALSE
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

conf <- conf[
  conf$visual_target !=
    conf$textual_target,
  ,
  drop = FALSE
]

conf$cell_key <- make_cell_key(
  conf
)

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

cells <- unique(
  conf[
    ,
    cell_cols,
    drop = FALSE
  ]
)

rownames(cells) <- NULL

if (nrow(cells) != 4320) {
  stop(sprintf(
    "Expected 4320 conflict cells; found %d.",
    nrow(cells)
  ))
}

cell_sizes <- table(
  conf$cell_key
)

if (
  length(cell_sizes) != 4320 ||
  any(cell_sizes != 40)
) {
  stop(
    "Expected exactly 40 image trials per conflict cell."
  )
}

if (nrow(conf) != 172800) {
  stop(sprintf(
    "Expected 172800 conflict rows; found %d.",
    nrow(conf)
  ))
}

cell_image_key <- paste(
  conf$cell_key,
  conf$image,
  sep = "|||IMAGE|||"
)

if (anyDuplicated(cell_image_key) != 0) {
  stop(
    "Duplicate cell-image trial detected."
  )
}

# ============================================================
# VERIFY BASE-IMAGE UNITS
# ============================================================

base_images <- unique(
  conf[
    ,
    c(
      "cluster_id",
      "image"
    ),
    drop = FALSE
  ]
)

if (nrow(base_images) != 160) {
  stop(sprintf(
    "Expected 160 base images; found %d.",
    nrow(base_images)
  ))
}

for (cluster in CLUSTERS) {
  n <- length(unique(
    base_images$image[
      base_images$cluster_id == cluster
    ]
  ))

  if (n != 40) {
    stop(sprintf(
      "%s should contain 40 images; found %d.",
      cluster,
      n
    ))
  }
}

cat(
  "Validated: 4320 cells x 40 images; ",
  "160 base-image units.\n",
  sep = ""
)

# ============================================================
# NORMALIZE PREDICTIONS
# ============================================================

conf$success_bool <-
  tolower(conf$success) == "true"

conf$pred_norm <- NA_character_

successful <- which(
  conf$success_bool
)

conf$pred_norm[successful] <- vapply(
  conf$primary_country[successful],
  normalize_country,
  character(1),
  lookup = lookup
)

# ============================================================
# ELIGIBILITY POLICIES
# ============================================================

original_rules <- read_exclusions(
  original_exclusions_path
)

common_rules <- read_exclusions(
  common_exclusions_path
)

cells$eligible_original <-
  !excluded_cells(
    cells,
    original_rules
  )

cells$eligible_common <-
  !excluded_cells(
    cells,
    common_rules
  )

cells$eligible_all <- TRUE

observed_eligibility <- c(
  original = sum(
    cells$eligible_original
  ),
  common = sum(
    cells$eligible_common
  ),
  all = sum(
    cells$eligible_all
  )
)

expected_eligibility <- c(
  original = 3385L,
  common = 2920L,
  all = 4320L
)

if (!identical(
  as.integer(observed_eligibility),
  as.integer(expected_eligibility)
)) {
  stop(sprintf(
    paste0(
      "Eligibility mismatch. ",
      "Observed [%s], expected [%s]."
    ),
    paste(
      observed_eligibility,
      collapse = ", "
    ),
    paste(
      expected_eligibility,
      collapse = ", "
    )
  ))
}

cat(sprintf(
  paste0(
    "Eligibility validated: ",
    "original=%d common=%d all=%d\n"
  ),
  observed_eligibility["original"],
  observed_eligibility["common"],
  observed_eligibility["all"]
))

# ============================================================
# GLOBAL PREDICTION LABELS
# ============================================================

LABELS <- sort(unique(
  conf$pred_norm[
    conf$success_bool
  ]
))

label_lookup <- setNames(
  seq_along(LABELS),
  LABELS
)

conf$pred_code <- 0L

conf$pred_code[successful] <-
  unname(
    label_lookup[
      conf$pred_norm[successful]
    ]
  )

# ============================================================
# BOOTSTRAP WEIGHTS
#
# SAME scheme/seed as previous trial-level bootstrap.
# Each column is one bootstrap replicate.
# ============================================================

set.seed(SEED)

bootstrap_weights <- list()
image_lists <- list()

for (cluster in CLUSTERS) {
  images <- sort(unique(
    conf$image[
      conf$cluster_id == cluster
    ]
  ))

  if (length(images) != 40) {
    stop(sprintf(
      "%s does not have 40 images.",
      cluster
    ))
  }

  image_lists[[cluster]] <- images

  W <- matrix(
    0L,
    nrow = 40,
    ncol = B
  )

  for (b in seq_len(B)) {
    draws <- sample.int(
      40,
      size = 40,
      replace = TRUE
    )

    W[, b] <- tabulate(
      draws,
      nbins = 40
    )
  }

  bootstrap_weights[[cluster]] <- W
}

# ============================================================
# ACCUMULATORS
# ============================================================

POLICIES <- c(
  "original",
  "common",
  "all"
)

GROUPS <- c(
  "overall",
  MODELS
)

metrics <- c(
  "classified",
  "visual",
  "text",
  "tie",
  "other"
)

bootstrap_acc <- list()
point_acc <- list()

for (policy in POLICIES) {
  for (group in GROUPS) {
    key <- paste(
      policy,
      group,
      sep = "|||"
    )

    bootstrap_acc[[key]] <- list()

    for (metric in metrics) {
      bootstrap_acc[[key]][[metric]] <-
        numeric(B)
    }

    point_acc[[key]] <- setNames(
      rep(0L, length(metrics)),
      metrics
    )
  }
}

# ============================================================
# PROCESS ONE FOOD CLUSTER AT A TIME
#
# Important:
#   - mode is recomputed separately in every bootstrap replicate
#   - failed model calls contribute no prediction
#   - ties are preserved exactly
# ============================================================

for (cluster in CLUSTERS) {
  cat(sprintf(
    "Processing %s...\n",
    cluster
  ))

  cell_idx_global <- which(
    cells$cluster_id == cluster
  )

  cc <- cells[
    cell_idx_global,
    ,
    drop = FALSE
  ]

  rownames(cc) <- NULL

  C <- nrow(cc)

  if (C != 1080) {
    stop(sprintf(
      "Expected 1080 cells in %s; found %d.",
      cluster,
      C
    ))
  }

  images <- image_lists[[cluster]]
  W <- bootstrap_weights[[cluster]]

  cell_index <- setNames(
    seq_len(C),
    cc$cell_key
  )

  image_index <- setNames(
    seq_along(images),
    images
  )

  cx <- conf[
    conf$cluster_id == cluster,
    ,
    drop = FALSE
  ]

  local_cell <- unname(
    cell_index[
      cx$cell_key
    ]
  )

  local_image <- unname(
    image_index[
      cx$image
    ]
  )

  if (
    any(is.na(local_cell)) ||
    any(is.na(local_image))
  ) {
    stop(
      "Failed to map conflict rows to cell/image coordinates."
    )
  }

  # 1080 cells x 40 base images.
  pred_matrix <- matrix(
    0L,
    nrow = C,
    ncol = 40
  )

  success_matrix <- matrix(
    0L,
    nrow = C,
    ncol = 40
  )

  visual_matrix <- matrix(
    0L,
    nrow = C,
    ncol = 40
  )

  text_matrix <- matrix(
    0L,
    nrow = C,
    ncol = 40
  )

  ok <- which(
    cx$success_bool
  )

  coords <- cbind(
    local_cell[ok],
    local_image[ok]
  )

  pred_matrix[coords] <-
    cx$pred_code[ok]

  success_matrix[coords] <- 1L

  visual_matrix[coords] <-
    as.integer(
      cx$pred_norm[ok] ==
        cx$visual_target[ok]
    )

  text_matrix[coords] <-
    as.integer(
      cx$pred_norm[ok] ==
        cx$textual_target[ok]
    )

  # ----------------------------------------------------------
  # ORIGINAL, NON-BOOTSTRAPPED POINT CLASSIFICATION
  # ----------------------------------------------------------

  valid_point <- rowSums(
    success_matrix
  )

  visual_point <- rowSums(
    visual_matrix
  )

  text_point <- rowSums(
    text_matrix
  )

  max_point <- rep(
    0L,
    C
  )

  for (code in seq_along(LABELS)) {
    label_count <- rowSums(
      pred_matrix == code
    )

    max_point <- pmax(
      max_point,
      label_count
    )
  }

  has_visual_point <-
    valid_point > 0 &
    visual_point == max_point

  has_text_point <-
    valid_point > 0 &
    text_point == max_point

  point_visual <-
    has_visual_point &
    !has_text_point

  point_text <-
    has_text_point &
    !has_visual_point

  point_tie <-
    has_visual_point &
    has_text_point

  point_classified <-
    valid_point > 0

  point_other <-
    point_classified &
    !(
      point_visual |
      point_text |
      point_tie
    )

  # ----------------------------------------------------------
  # BOOTSTRAPPED CELL MODES
  # ----------------------------------------------------------

  valid_boot <-
    success_matrix %*% W

  visual_boot_count <-
    visual_matrix %*% W

  text_boot_count <-
    text_matrix %*% W

  max_boot <- matrix(
    0,
    nrow = C,
    ncol = B
  )

  # Find the highest weighted prediction count
  # independently for every cell and bootstrap replicate.
  for (code in seq_along(LABELS)) {
    label_indicator <-
      pred_matrix == code

    label_boot_count <-
      label_indicator %*% W

    max_boot <- pmax(
      max_boot,
      label_boot_count
    )

    rm(
      label_indicator,
      label_boot_count
    )
  }

  has_visual_boot <-
    valid_boot > 0 &
    visual_boot_count == max_boot

  has_text_boot <-
    valid_boot > 0 &
    text_boot_count == max_boot

  boot_visual <-
    has_visual_boot &
    !has_text_boot

  boot_text <-
    has_text_boot &
    !has_visual_boot

  boot_tie <-
    has_visual_boot &
    has_text_boot

  boot_classified <-
    valid_boot > 0

  boot_other <-
    boot_classified &
    !(
      boot_visual |
      boot_text |
      boot_tie
    )

  # ----------------------------------------------------------
  # ACCUMULATE BY POLICY AND MODEL
  # ----------------------------------------------------------

  for (policy in POLICIES) {
    eligibility_col <- paste0(
      "eligible_",
      policy
    )

    for (group in GROUPS) {
      keep <- cc[[eligibility_col]]

      if (group != "overall") {
        keep <- keep &
          cc$model == group
      }

      idx <- which(keep)

      key <- paste(
        policy,
        group,
        sep = "|||"
      )

      point_acc[[key]]["classified"] <-
        point_acc[[key]]["classified"] +
        sum(point_classified[idx])

      point_acc[[key]]["visual"] <-
        point_acc[[key]]["visual"] +
        sum(point_visual[idx])

      point_acc[[key]]["text"] <-
        point_acc[[key]]["text"] +
        sum(point_text[idx])

      point_acc[[key]]["tie"] <-
        point_acc[[key]]["tie"] +
        sum(point_tie[idx])

      point_acc[[key]]["other"] <-
        point_acc[[key]]["other"] +
        sum(point_other[idx])

      bootstrap_acc[[key]]$classified <-
        bootstrap_acc[[key]]$classified +
        colSums(
          boot_classified[
            idx,
            ,
            drop = FALSE
          ]
        )

      bootstrap_acc[[key]]$visual <-
        bootstrap_acc[[key]]$visual +
        colSums(
          boot_visual[
            idx,
            ,
            drop = FALSE
          ]
        )

      bootstrap_acc[[key]]$text <-
        bootstrap_acc[[key]]$text +
        colSums(
          boot_text[
            idx,
            ,
            drop = FALSE
          ]
        )

      bootstrap_acc[[key]]$tie <-
        bootstrap_acc[[key]]$tie +
        colSums(
          boot_tie[
            idx,
            ,
            drop = FALSE
          ]
        )

      bootstrap_acc[[key]]$other <-
        bootstrap_acc[[key]]$other +
        colSums(
          boot_other[
            idx,
            ,
            drop = FALSE
          ]
        )
    }
  }

  rm(
    pred_matrix,
    success_matrix,
    visual_matrix,
    text_matrix,
    valid_boot,
    visual_boot_count,
    text_boot_count,
    max_boot,
    boot_visual,
    boot_text,
    boot_tie,
    boot_other,
    boot_classified
  )

  gc(verbose = FALSE)
}

# ============================================================
# VERIFY POINT ESTIMATES AGAINST ALREADY VERIFIED REPORTS
# ============================================================

expected_points <- list(
  original = c(
    classified = 3385L,
    visual = 2215L,
    text = 628L,
    tie = 19L,
    other = 523L
  ),
  common = c(
    classified = 2920L,
    visual = 2040L,
    text = 504L,
    tie = 17L,
    other = 359L
  ),
  all = c(
    classified = 4320L,
    visual = 2474L,
    text = 873L,
    tie = 19L,
    other = 954L
  )
)

for (policy in POLICIES) {
  key <- paste(
    policy,
    "overall",
    sep = "|||"
  )

  observed <- point_acc[[key]]
  expected <- expected_points[[policy]]

  if (!identical(
    as.integer(observed),
    as.integer(expected)
  )) {
    cat(
      "\nExpected point counts:\n"
    )

    print(expected)

    cat(
      "\nObserved point counts:\n"
    )

    print(observed)

    stop(sprintf(
      "Point-estimate reproduction failed for %s.",
      policy
    ))
  }
}

cat(
  paste0(
    "Point estimates reproduced exactly for ",
    "original/common/all policies.\n"
  )
)

# ============================================================
# BUILD CI TABLES
# ============================================================

ci_rows <- list()
draw_rows <- list()

ci_i <- 1L
draw_i <- 1L

for (policy in POLICIES) {
  for (group in GROUPS) {
    key <- paste(
      policy,
      group,
      sep = "|||"
    )

    point <- point_acc[[key]]
    boot <- bootstrap_acc[[key]]

    denom <- boot$classified

    visual_rate <-
      boot$visual / denom

    text_rate <-
      boot$text / denom

    tie_rate <-
      boot$tie / denom

    other_rate <-
      boot$other / denom

    gap <-
      visual_rate - text_rate

    point_visual <-
      point["visual"] /
      point["classified"]

    point_text <-
      point["text"] /
      point["classified"]

    point_tie <-
      point["tie"] /
      point["classified"]

    point_other <-
      point["other"] /
      point["classified"]

    point_gap <-
      point_visual -
      point_text

    eligible_cell_count <-
      sum(
        cells[[
          paste0(
            "eligible_",
            policy
          )
        ]] &
          if (group == "overall") {
            TRUE
          } else {
            cells$model == group
          }
      )

    missing_boot <-
      eligible_cell_count -
      denom

    ci_rows[[ci_i]] <- data.frame(
      policy = policy,
      group = group,
      point_classified_cells =
        as.integer(
          point["classified"]
        ),

      visual_rate =
        as.numeric(point_visual),

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
        as.numeric(point_text),

      textual_ci_low =
        unname(quantile(
          text_rate,
          0.025,
          type = 7
        )),

      textual_ci_high =
        unname(quantile(
          text_rate,
          0.975,
          type = 7
        )),

      tie_rate =
        as.numeric(point_tie),

      tie_ci_low =
        unname(quantile(
          tie_rate,
          0.025,
          type = 7
        )),

      tie_ci_high =
        unname(quantile(
          tie_rate,
          0.975,
          type = 7
        )),

      other_rate =
        as.numeric(point_other),

      visual_minus_text =
        as.numeric(point_gap),

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

      max_missing_cells_any_bootstrap =
        max(missing_boot),

      bootstrap_replicates = B,

      stringsAsFactors = FALSE
    )

    ci_i <- ci_i + 1L

    draw_rows[[draw_i]] <- data.frame(
      policy = policy,
      group = group,
      replicate = seq_len(B),
      classified_cells = denom,
      visual_rate = visual_rate,
      textual_rate = text_rate,
      tie_rate = tie_rate,
      other_rate = other_rate,
      visual_minus_text = gap,
      stringsAsFactors = FALSE
    )

    draw_i <- draw_i + 1L
  }
}

ci_table <- do.call(
  rbind,
  ci_rows
)

draw_table <- do.call(
  rbind,
  draw_rows
)

write.csv(
  ci_table,
  file.path(
    output_dir,
    "modal_bootstrap_ci.csv"
  ),
  row.names = FALSE
)

write.csv(
  draw_table,
  file.path(
    output_dir,
    "modal_bootstrap_draws.csv"
  ),
  row.names = FALSE
)

# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_path <- file.path(
  output_dir,
  "modal_bootstrap_report.txt"
)

sink(report_path)

cat(
  "============================================================\n"
)

cat(
  "MODAL-CELL BASE-IMAGE BOOTSTRAP\n"
)

cat(
  "============================================================\n\n"
)

cat(sprintf(
  "Submitted rows:       %d\n",
  nrow(d)
))

cat(sprintf(
  "Conflict cells:       %d\n",
  nrow(cells)
))

cat(sprintf(
  "Base images:          %d\n",
  nrow(base_images)
))

cat(sprintf(
  "Bootstrap replicates: %d\n",
  B
))

cat(sprintf(
  "Bootstrap seed:       %d\n\n",
  SEED
))

cat(
  paste0(
    "Resampling unit: base image. ",
    "Within each food cluster, the 40 base images are ",
    "sampled with replacement. All repeated conditions ",
    "for a sampled image move together.\n\n"
  )
)

cat(
  paste0(
    "For every bootstrap replicate, each conflict cell's ",
    "modal prediction is recomputed from the resampled images. ",
    "Visual/textual ties are retained as ties. All remaining ",
    "modal outcomes are grouped as OTHER because the present ",
    "analysis targets uncertainty in the visual/textual headline.\n\n"
  )
)

for (policy in POLICIES) {
  cat(
    "============================================================\n"
  )

  cat(sprintf(
    "POLICY: %s\n",
    toupper(policy)
  ))

  cat(
    "============================================================\n\n"
  )

  key <- paste(
    policy,
    "overall",
    sep = "|||"
  )

  cat(
    "Point cell counts:\n"
  )

  print(
    point_acc[[key]]
  )

  cat(
    "\nBootstrap 95% intervals:\n"
  )

  print(
    ci_table[
      ci_table$policy == policy,
      ,
      drop = FALSE
    ],
    row.names = FALSE
  )

  cat("\n")
}

sink()

cat("\n[DONE]\n")

cat(sprintf(
  "Report: %s\n",
  report_path
))

cat(sprintf(
  "CI table: %s\n",
  file.path(
    output_dir,
    "modal_bootstrap_ci.csv"
  )
))

cat(sprintf(
  "Raw draws: %s\n",
  file.path(
    output_dir,
    "modal_bootstrap_draws.csv"
  )
))
