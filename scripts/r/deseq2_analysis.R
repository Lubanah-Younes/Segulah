# DESeq2 Analysis - Simplified version
library(DESeq2)

# Create simulated count data
set.seed(123)
counts <- matrix(rnbinom(n = 6000, size = 10, mu = 100), nrow = 1000, ncol = 6)
rownames(counts) <- paste0("GENE_", 1:1000)
colnames(counts) <- paste0("sample_", 1:6)

condition <- factor(c("control", "control", "control", "treated", "treated", "treated"))

colData <- data.frame(row.names = colnames(counts), condition = condition)

# Run DESeq2
dds <- DESeqDataSetFromMatrix(countData = counts, colData = colData, design = ~ condition)
dds <- DESeq(dds)
res <- results(dds)

# Results
cat("\n========== DESeq2 RESULTS ==========\n")
summary(res)

# Top genes
res_sorted <- res[order(res$pvalue), ]
cat("\n========== TOP 10 GENES ==========\n")
print(head(res_sorted, 10))

# Save
write.csv(as.data.frame(res), "L:/drug_discovery/deseq2_results.csv")
cat("\nResults saved to deseq2_results.csv\n")

# Plot
png("L:/drug_discovery/deseq2_plot.png", width = 800, height = 600)
plotMA(res, main = "MA Plot - DESeq2")
dev.off()
cat("Plot saved to deseq2_plot.png\n")