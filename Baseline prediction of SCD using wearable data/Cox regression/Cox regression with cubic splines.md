
#Read data

```{r}
path <- '//summary_stats_and_clin.csv'

df <- read_csv(path)
df <- data.frame(df)
#df <- na.omit(df)

df$median_SleepIntervalDuration <- df$median_SleepIntervalDuration / 3600
df$mean_ExerciseDurationDay <- df$mean_ExerciseDurationDay / 3600
df$median_TotalSleepDuration <- df$median_TotalSleepDuration / 3600
df$median_InactiveDurationDay <- df$median_InactiveDurationDay / 3600
df$median_ActiveEventCountDay <- df$median_ActiveEventCountDay / 100
df$median_M6Intensity <- df$median_M6Intensity * 10
df$median_ActiveIntensityDay <- df$median_ActiveIntensityDay * 1000
df$median_ExerciseDurationDay <- df$median_ExerciseDurationDay / 3600

results_df <- data.frame(variable = character(),
                         OR = numeric(),
                         CI_lower = numeric(),
                         CI_upper = numeric(),
                         p_value = numeric(),
                         stringsAsFactors = FALSE)

predictor_vars <-  c("median_InactiveDurationDay", 
                   "median_ActiveEventCountDay",  "median_M6Intensity", "median_MeanCadenceDay",
                    "median_ActivityVolumeDay", "median_Cadance95Day",
                   "median_TotalSleepDuration", "median_SleepEfficiency", 
                "median_WASOCount", "median_SleepEventsNumber" , "median_ExerciseDurationDay")

#"Age", "Sex", "DM", "AF",  "HF",  "Secprev", "BMI", "IHD", "ACEi", "Bblock"
for(var in predictor_vars) {
  #formula_str <- as.formula(paste("Surv(time_first_at, first_at_yn) ~ ", var, "+ Age +Sex + DM + HF +IHD + Secprev + CRT.D"))
  formula_str <- as.formula(paste("Surv(time_first_at, first_at_yn) ~ ", var))#, "+ Age +Sex + DM + HF +IHD + Secprev + CRT.D"))

  model <- coxph(formula = formula_str, data = df)
  test <- cox.zph(model)
  
  # Display results
  print(test)
  summary_obj <- summary(model)
  #print(summary_obj)
  hr <- exp((coef(summary_obj)[1]))
  ll <- signif(summary_obj$conf.int[,"lower .95"][1], 3)
  ul <- signif(summary_obj$conf.int[,"upper .95"][1], 3)
  p_val <- signif(summary_obj$coefficients[,"Pr(>|z|)"][1], digits=3)

  results_df <- rbind(results_df, data.frame(variable = var, HR = hr, CI_lower = ll, CI_upper = ul, p_value = p_val))
}

significant_vars <- results_df[results_df$p_val < 0.10, ]
print(significant_vars)
#print(results_df)
```

# Cubic spline
```{r}
df <- read_csv(path)
df <- data.frame(df)
#df <- na.omit(df)

df$median_SleepIntervalDuration <- df$median_SleepIntervalDuration / 3600
df$mean_ExerciseDurationDay <- df$mean_ExerciseDurationDay / 3600
df$median_TotalSleepDuration <- df$median_TotalSleepDuration / 3600
df$median_InactiveDurationDay <- df$median_InactiveDurationDay / 3600
df$median_ActiveEventCountDay <- df$median_ActiveEventCountDay / 100
df$median_M6Intensity <- df$median_M6Intensity * 10
df$median_ActiveIntensityDay <- df$median_ActiveIntensityDay * 1000
df$median_ExerciseDurationDay <- df$median_ExerciseDurationDay / 3600
#df$median_WASOCount <- df$median_WASOCount / 10
#df$median_SleepEventsNumber <- df$median_SleepEventsNumber / 10
#df$median_ActiveEventCountDay <- df$median_ActiveEventCountDay / 10


source("//plotHR_0.11.R")

selected_cols <- c("time_first_at", "first_at_yn", "median_InactiveDurationDay", 
                   "median_ActiveEventCountDay",  "median_M6Intensity", 
                    "median_ActivityVolumeDay", "median_Cadance95Day",
                   "median_TotalSleepDuration", "median_SleepEfficiency", 
                   "median_MidSleepTime", "median_WASOCount",
                   "median_SleepEventsNumber", 
                   "median_ExerciseDurationDay", "median_ActiveIntensityDay")



model<-coxph(Surv(time_first_at, first_at_yn) ~ pspline(median_InactiveDurationDay , df = 4) , data = df)

termplot(model, term = 1, se = T,plot = T)
plotHR(model,  ylim=c(-3,3), rug='density', col.term='black', col.se='lightblue', col.ref='darkgray', main="Dose-response curve for inactive duration", xlab="Hours per day")

#options(width=1000) 
summary(model,maxlabel =1000)
```
