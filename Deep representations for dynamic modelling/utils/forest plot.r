library(forestplot)
library(grid)

generate_forest_plot <- function(final_model) {
  coef_summary <- summary(final_model)
  coef_data <- data.frame(
    Variable = rownames(coef_summary$coefficients),
    OR = exp(coef_summary$coefficients[, "Estimate"]),
    LL = exp(coef_summary$coefficients[, "Estimate"] - 1.96 * coef_summary$coefficients[, "Std. Error"]),
    UL = exp(coef_summary$coefficients[, "Estimate"] + 1.96 * coef_summary$coefficients[, "Std. Error"]),
    p_val = coef_summary$coefficients[, "Pr(>|z|)"]
  )
  
  colnames(coef_data) <- c('Variable', 'Beta','LL', 'UL', 'p_val')
  coef_data <- coef_data[coef_data$Variable != "(Intercept)", ]
  coef_data <- round_df(coef_data, 5)
  coef_data$`               Unadjusted model` <- paste(rep(" ", 40), collapse = " ")
  coef_data$`OR (95% CI)` <- ifelse(is.na(coef_data$Beta), "",
                                    sprintf("%.3f (%.2f to %.2f)",
                                            coef_data$Beta, coef_data$LL, coef_data$UL))
  
  coef_data$p_val[coef_data$p_val == 0.000] <- " **"
  coef_data$p_val[coef_data$p_val < 0.05 & coef_data$p_val != " **"] <- " *"
  coef_data$p_val[coef_data$p_val >= 0.05] <- ""
  coef_data$` ` <- coef_data$p_val
  coef_data$se <- 0.75
  
  coef_data$Variable <- gsub("means", "(means)", coef_data$Variable)
  coef_data$Variable <- gsub("variance", "(variance)", coef_data$Variable)
  coef_data$Variable <- gsub("slope", "(slope)", coef_data$Variable)
  coef_data$Variable <- gsub("_", " ", coef_data$Variable)
  coef_data$Variable <- gsub("`", "", coef_data$Variable)
  
  
  
  #########  #########  #########  #########  #########  #########  #########  #########  #########
  empty_row <- data.frame(matrix(ncol = ncol(coef_data), nrow = 1))
  names(empty_row) <- names(coef_data)
  

  coef_data$`OR (95% CI)` <- ifelse(is.na(coef_data$`OR (95% CI)`), "", coef_data$`OR (95% CI)`)
  coef_data$`               Unadjusted model` <- ifelse(is.na(coef_data$`               Unadjusted model`), "", coef_data$`               Unadjusted model`)
  
  
  coef_data$" " <- ifelse(is.na(coef_data$" "), "", coef_data$" ")
  

  tm <- forest_theme(
    base_size = 14,
    xaxis_cex = 1,
    ci_pch = 16,
    ci_col = "black",
    ci_fill = "black",
    ci_alpha = 1,
    ci_lty = 1,
    ci_lwd = 1.5,
    ci_Theight = 0.4,
    refline_lwd = 1,
    refline_lty = "dashed",
    refline_col = "grey50",
    vertline_lwd = 0.2,
    vertline_lty = "dashed",
    vertline_col = "grey50",
    arrow_type = "open",
    arrow_label_just = "start",
    arrow_cex = 1,
    footnote_cex = 0.9,
    footnote_fontface = "italic"
  )
  
  p <- forest(
    coef_data[,c(1, 6, 7, 8)],
    est = coef_data$Beta,
    lower = coef_data$LL, 
    upper = coef_data$UL,
    sizes = coef_data$se,
    xlim = c(-1, 7),
    ci_column = 2,
    ref_line = 1,
    ticks_at = c(0, 1, 3, 5),
    vert_line = c(1),
    arrow_lab = c("Lower risk", "Higher risk"),
    theme= tm
  )
  
  p$heights <- rep(unit(10, "mm"), nrow(p)) 
  p <- edit_plot(p, which = "background",
                 gp = gpar(fill = "white"))
  
  return(p)
}

# Example usage:
# final_model <- lm(y ~ x1 + x2, data = your_data)
# forest_plot <- generate_forest_plot(final_model)
# print(forest_plot)
