#!/usr/bin/env Rscript

# Generate figures exclusively from a verified experiment CSV. This script
# deliberately fails when measured or provenance columns are absent.

required_packages <- c("ggplot2", "dplyr", "tidyr", "svglite")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) stop("Missing R packages: ", paste(missing_packages, collapse = ", "))

library(ggplot2)
library(dplyr)
library(tidyr)
library(svglite)

args <- commandArgs(trailingOnly = TRUE)
input_file <- if (length(args) >= 1) args[[1]] else "outputs/verified/experiment_runs.csv"
out_dir <- if (length(args) >= 2) args[[2]] else "artifacts/figures"
if (!file.exists(input_file)) stop("Verified experiment CSV not found: ", input_file)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

runs <- read.csv(input_file, check.names = FALSE)
provenance_columns <- c("experiment_config_hash", "dataset_id", "query_spec_hash", "pipeline_id")
missing_provenance <- setdiff(provenance_columns, names(runs))
if (length(missing_provenance) > 0) stop("Input is not provenance-complete; missing: ", paste(missing_provenance, collapse = ", "))

save_plot <- function(name, plot, width = 8, height = 6) {
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot, width = width, height = height)
  ggsave(file.path(out_dir, paste0(name, ".svg")), plot, width = width, height = height)
}

if (all(c("scale_n", "is_correct") %in% names(runs))) {
  accuracy <- runs %>% filter(!is.na(is_correct)) %>% group_by(scale_n, pipeline_id) %>%
    summarise(accuracy = mean(as.logical(is_correct)), n = n(), .groups = "drop")
  write.csv(accuracy, file.path(out_dir, "accuracy_scale_degradation.csv"), row.names = FALSE)
  save_plot("accuracy_scale_degradation", ggplot(accuracy, aes(scale_n, accuracy, color = pipeline_id)) +
    geom_line() + geom_point() + scale_x_log10() +
    labs(title = "Measured Accuracy by Corpus Scale", x = "Corpus size", y = "Accuracy") + theme_minimal())
}

phi_columns <- c("phi_scope", "phi_facts", "phi_aggregation")
if (all(c("scale_n", phi_columns) %in% names(runs))) {
  shapley <- runs %>% select(scale_n, all_of(phi_columns)) %>%
    pivot_longer(all_of(phi_columns), names_to = "component", values_to = "shapley_value") %>%
    filter(!is.na(shapley_value)) %>% group_by(scale_n, component) %>%
    summarise(mean_shapley = mean(shapley_value), n = n(), .groups = "drop")
  write.csv(shapley, file.path(out_dir, "shapley_by_scale.csv"), row.names = FALSE)
  save_plot("shapley_stacked_bar", ggplot(shapley, aes(factor(scale_n), mean_shapley, fill = component)) +
    geom_col() + labs(title = "Measured Shapley Attribution", x = "Corpus size", y = "Mean Shapley value") + theme_minimal())
}

if (all(c("policy_decision", "loss") %in% names(runs))) {
  evaluable <- runs %>% filter(!is.na(loss))
  certified <- evaluable %>% filter(policy_decision == "certified")
  risk <- data.frame(
    operating_point = c("raw", "applied_policy"),
    coverage = c(1, if (nrow(evaluable) > 0) nrow(certified) / nrow(evaluable) else 0),
    risk = c(mean(evaluable$loss), if (nrow(certified) > 0) mean(certified$loss) else NA)
  )
  write.csv(risk, file.path(out_dir, "risk_coverage.csv"), row.names = FALSE)
  save_plot("abstention_risk_coverage", ggplot(risk, aes(coverage, risk, label = operating_point)) +
    geom_line() + geom_point() + geom_text(vjust = -0.5) +
    labs(title = "Measured Policy Operating Points", x = "Coverage", y = "Mean loss") + theme_minimal())
}

message("Generated only plots supported by measured columns in ", input_file)
