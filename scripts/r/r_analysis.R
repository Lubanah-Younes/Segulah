# Install required packages
install.packages(c("ggplot2", "dplyr", "tidyr", "readr"), repos = "https://cloud.r-project.org")

# Load libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Read the cleaned data
df <- read_csv("ic50_cleaned.csv")

# Summary
cat("\n========== SUMMARY STATISTICS ==========\n")
print(summary(df$ic50_nM))

# Potency categories
df <- df %>%
  mutate(potency_category = case_when(
    ic50_nM < 10 ~ "High (<10 nM)",
    ic50_nM < 100 ~ "Medium-High (10-100 nM)",
    ic50_nM < 1000 ~ "Medium (100-1000 nM)",
    ic50_nM < 10000 ~ "Low-Medium (1-10 uM)",
    TRUE ~ "Low (>10 uM)"
  ))

cat("\n========== POTENCY DISTRIBUTION ==========\n")
print(table(df$potency_category))

# Statistical test: High vs Low potency
high_potency <- df %>% filter(ic50_nM < 10) %>% pull(pchembl_value) %>% na.omit()
low_potency <- df %>% filter(ic50_nM > 1000) %>% pull(pchembl_value) %>% na.omit()

cat("\n========== T-TEST RESULTS ==========\n")
t_test_result <- t.test(high_potency, low_potency)
print(t_test_result)

# Save results to file
sink("r_analysis_results.txt")
cat("R Analysis Results - Drug Discovery Platform\n")
cat("============================================\n\n")
cat("Summary Statistics:\n")
print(summary(df$ic50_nM))
cat("\nPotency Distribution:\n")
print(table(df$potency_category))
cat("\nT-Test Results:\n")
print(t_test_result)
sink()

cat("\n\nResults saved to r_analysis_results.txt")