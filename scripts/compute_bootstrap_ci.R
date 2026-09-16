#!/usr/bin/env Rscript

if (!requireNamespace("boot", quietly = TRUE)) stop("Missing R package: boot")
if (!requireNamespace("dplyr", quietly = TRUE)) stop("Missing R package: dplyr")
library(boot)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
input_file <- if (length(args) >= 1) args[[1]] else "outputs/verified/experiment_runs.csv"
output_file <- if (length(args) >= 2) args[[2]] else "artifacts/tables/shapley_bootstrap_ci.csv"
if (!file.exists(input_file)) stop("Verified experiment CSV not found: ", input_file)

data <- read.csv(input_file, check.names = FALSE)
required <- c("scale_n", "phi_scope", "phi_facts", "phi_aggregation", "experiment_config_hash")
missing <- setdiff(required, names(data))
if (length(missing) > 0) stop("Missing measured/provenance columns: ", paste(missing, collapse = ", "))

bootstrap_mean <- function(values, samples = 2000) {
  values <- values[!is.na(values)]
  if (length(values) < 2) return(c(mean = ifelse(length(values), mean(values), NA), low = NA, high = NA, n = length(values)))
  statistic <- function(x, indices) mean(x[indices])
  fit <- boot(values, statistic, R = samples)
  interval <- boot.ci(fit, type = "perc")$percent[4:5]
  c(mean = mean(values), low = interval[[1]], high = interval[[2]], n = length(values))
}

rows <- list()
for (scale in sort(unique(data$scale_n))) {
  subset <- data[data$scale_n == scale, ]
  for (component in c("phi_scope", "phi_facts", "phi_aggregation")) {
    stats <- bootstrap_mean(subset[[component]])
    rows[[length(rows) + 1]] <- data.frame(scale_n = scale, component = component,
      mean = stats[["mean"]], ci_low = stats[["low"]], ci_high = stats[["high"]], n = stats[["n"]])
  }
}

result <- bind_rows(rows)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(result, output_file, row.names = FALSE)
message("Measured bootstrap intervals written to ", output_file)
