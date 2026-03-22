# Proteomics Analysis using Base R Functions
# Simulated LC-MS/MS data for drug discovery platform

# Set seed
set.seed(123)

# Create simulated proteomics data
# 100 proteins, 6 samples (3 control, 3 treated)
proteins <- paste0("Protein_", 1:100)
samples <- paste0("Sample_", 1:6)
conditions <- c(rep("Control", 3), rep("Treated", 3))

# Simulate intensity data (log2 scale)
intensity_data <- matrix(rnorm(600, mean = 25, sd = 3), nrow = 100, ncol = 6)
rownames(intensity_data) <- proteins
colnames(intensity_data) <- samples

# Add treatment effect to 20 proteins
set.seed(456)
treatment_effect <- sample(1:100, 20)
for(i in treatment_effect) {
    intensity_data[i, 4:6] <- intensity_data[i, 4:6] + rnorm(3, mean = 2, sd = 0.5)
}

# Perform t-test for each protein
p_values <- numeric(100)
log2FC <- numeric(100)

for(i in 1:100) {
    control_vals <- intensity_data[i, conditions == "Control"]
    treated_vals <- intensity_data[i, conditions == "Treated"]
    t_result <- t.test(control_vals, treated_vals)
    p_values[i] <- t_result$p.value
    log2FC[i] <- mean(treated_vals) - mean(control_vals)
}

# Adjust p-values
adj_p_values <- p.adjust(p_values, method = "BH")

# Create results table
results <- data.frame(
    Protein = proteins,
    log2FoldChange = log2FC,
    pvalue = p_values,
    adj_pvalue = adj_p_values
)

# Sort by p-value
results <- results[order(results$pvalue), ]

# Summary
cat("\n========== PROTEOMICS RESULTS ==========\n")
cat("Total proteins:", nrow(results), "\n")
cat("Significant proteins (adj.p < 0.05):", sum(results$adj_pvalue < 0.05, na.rm = TRUE), "\n")
cat("Up-regulated (log2FC > 1):", sum(results$log2FoldChange > 1 & results$adj_pvalue < 0.05, na.rm = TRUE), "\n")
cat("Down-regulated (log2FC < -1):", sum(results$log2FoldChange < -1 & results$adj_pvalue < 0.05, na.rm = TRUE), "\n")

# Top 10 proteins
cat("\n========== TOP 10 PROTEINS ==========\n")
print(head(results, 10))

# Save results
write.csv(results, "L:/drug_discovery/proteomics_results.csv", row.names = FALSE)
cat("\nResults saved to: L:/drug_discovery/proteomics_results.csv")

# Create volcano plot
png("L:/drug_discovery/proteomics_volcano.png", width = 800, height = 600)
with(results, plot(log2FoldChange, -log10(pvalue),
     pch = 20,
     col = ifelse(adj_pvalue < 0.05, "red", "gray"),
     xlab = "log2 Fold Change",
     ylab = "-log10(p-value)",
     main = "Proteomics - Volcano Plot"))
abline(h = -log10(0.05), col = "blue", lty = 2)
abline(v = c(-1, 1), col = "blue", lty = 2)
legend("topright", legend = c("Significant (adj.p < 0.05)", "Not significant"),
       col = c("red", "gray"), pch = 20)
dev.off()

cat("\nPlot saved to: L:/drug_discovery/proteomics_volcano.png\n")

# Histogram of p-values
png("L:/drug_discovery/proteomics_pvalues.png", width = 800, height = 600)
hist(results$pvalue, breaks = 20, col = "skyblue",
     main = "Distribution of p-values",
     xlab = "p-value", ylab = "Frequency")
abline(v = 0.05, col = "red", lty = 2)
legend("topright", legend = "p = 0.05", col = "red", lty = 2)
dev.off()

cat("P-value histogram saved to: L:/drug_discovery/proteomics_pvalues.png\n")

cat("\n========== ANALYSIS COMPLETE ==========\n")