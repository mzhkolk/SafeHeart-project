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


#Prediction k-fold cross-validation

```{r}
library(pROC)
path <- "//NN_16latents.csv"

df <- read_csv(path, show_col_types = FALSE)
df <- subset(df, HF_HFrEF == 1)
df <- df[, !names(df) %in% "HF_HFrEF"]

df_withoutID <- df[, !names(df) %in% c("ID ", "SubjectCode")]

set.seed(124) 
num_folds <- 5
unique_ids <- unique(df$ID)
num_ids <- length(unique_ids)
shuffled_ids <- sample(unique_ids)
ids_per_fold <- ceiling(num_ids / num_folds)

auroc_values <- numeric(num_folds)
roc_curves <- list()
brier_values <- numeric(num_folds)
results_df <- data.frame(actual = numeric(0), prediction = numeric(0))

suppressWarnings({
  for (i in 1:num_folds) {
    test_ids <- shuffled_ids[((i - 1) * ids_per_fold + 1):(min(i * ids_per_fold, num_ids))]
    train_ids <- setdiff(unique_ids, test_ids)
    train_data <- df[df$ID %in% train_ids, ]
    test_data <- df[df$ID %in% test_ids, ]
    test_data <- test_data[, !names(test_data) %in% "ID"]
    train_data <- train_data[, !names(train_data) %in% "ID"]

    log_model <- glm(formula = as.formula(paste('y', "~ .", collapse = " + ")), 
                 data = df_withoutID, 
                 family = binomial)

    stepwise_model <- step(log_model, direction = "backward",trace = FALSE,k =  2, data = train_data)
    
    final_model <- glm(formula = formula(stepwise_model),family = binomial,data = train_data)
    test_data$pred <- predict.glm(final_model, newdata = test_data, type = "response")
    
    roc_curve <- roc(test_data$y, test_data$pred)
    auroc_values[i] <- auc(roc_curve)
    print(auc(roc_curve))
    roc_curves[[i]] <- roc_curve
    pred.prob  <- predict.glm(final_model, newdata = test_data, type = "response")
    results_df <- rbind(results_df, data.frame(actual = test_data$y, prediction = test_data$pred))
    brier_values[i] <- mean((pred.prob - test_data$y)^2)}
})

mean_auroc <- mean(auroc_values)
print( mean_auroc)
std_auroc <- sd(auroc_values)
print( std_auroc)

mean_brier <- mean(brier_values)
print(mean_brier)
std_brier <- sd(brier_values)
print( std_brier)

write.csv(results_df, "HC_deLong.csv", row.names = FALSE)
saveRDS(roc_curves, "HC_ROC.rds")

dfs <- list()
for (i in 1:5) {
  # Extract sensitivity and specificity for the ith fold
  d.roc <- data.frame(
    sensitivity = loaded_roc_curves[[i]]$sensitivities,
    specificity = loaded_roc_curves[[i]]$specificities
  )
  
  # Add a column for fold number
  d.roc$fold <- i
  
  # Append the dataframe to the list
  dfs[[i]] <- d.roc
}
df_combined <- bind_rows(dfs)
write.csv(df_combined, "HC_ROC.csv", row.names = FALSE)
cat("CSV file saved:", getwd(), "/HC_ROC.csv")
```
