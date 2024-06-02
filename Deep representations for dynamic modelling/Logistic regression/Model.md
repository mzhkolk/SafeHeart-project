```{r}
library(readr)
library(dplyr)
library(tibble)
library(pillar)
```

# Univariate
```{r}
library(tibble)
library(car)
df <- read_csv(path, show_col_types = FALSE)

df_withoutID <- df[, c(1:ncol(df) >= 1 & 1:ncol(df) <= 16) | names(df) == "y"]# | names(df) == "HF_HFrEF"]
df_withoutID <- df_withoutID[, !names(df_withoutID) %in% "ID"]
df_withoutID[, 1:16] <- df_withoutID[, 1:16] /10


log_model <- glm(formula = as.formula(paste('y', "~ .", collapse = " + ")), data = df_withoutID, family = binomial)

vif_values <- car::vif(log_model)
high_vif <- vif_values[vif_values > 5]
high_vif_names <- names(high_vif)

df_withoutID <- df_withoutID[, !(names(df_withoutID) %in% high_vif_names) | names(df_withoutID) == "y"]

log_model <- glm(formula = as.formula(paste('y', "~ .", collapse = " + ")), data = df_withoutID, family = binomial)

stepwise_model <- step(log_model, 
                       direction = "forward",
                       trace = FALSE,
                       #steps = 10,  
                       k =  2,
                       data = df_withoutID)
final_model <- glm(formula = formula(stepwise_model),family = binomial,data = df_withoutID)
summary(final_model)
p <- generate_forest_plot(final_model)

```


# Multivariate
```{r}
results_df <- data.frame(variable = character(),OR = numeric(),CI_lower = numeric(),CI_upper = numeric(),p_value = numeric(),stringsAsFactors = FALSE)
predictor_vars <- names(df_withoutID)[names(df_withoutID) != "y"]
df_withoutID[, 1:16] <- df_withoutID[, 1:16] /10

predictor_vars <- predictor_vars[1:16]

for(var in predictor_vars) {
    formula_str <- paste("y ~ Age+Sex+secondary_prevention_yn+DM+ AF+ HF_HFrEF+type_of_device_1+ICM+BMI+", var)
    log_model <- glm(formula = formula_str, data = df_withoutID, family = binomial)
    summary_obj <- summary(log_model)
    or <- exp(coef(summary_obj)[11])
    ci <- exp(confint(log_model)[11, ])
    p_val <- summary_obj$coefficients[11, 4]
    results_df <- rbind(results_df, data.frame(Variable = var,Beta = or,LL = ci[1],UL = ci[2], p_val = p_val))
}
significant_vars <- results_df[results_df$p_val < 0.05, ]
print(significant_vars)

significant_vars_str <- paste(significant_vars$Variable, " (OR=", round(significant_vars$Beta, 2),", 95% CI (", round(significant_vars$LL, 2),"-",round(significant_vars$UL, 2), "), p=", round(significant_vars$p_val, 4), ")", sep = "", collapse = ", ")
```
