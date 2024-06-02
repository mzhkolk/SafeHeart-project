library(ggplot2)
library(dplyr)

calibration_plot <- function(data,
                             obs,
                             follow_up = NULL,
                             pred,
                             group = NULL,
                             nTiles = 10,
                             legendPosition = "right",
                             title = NULL,
                             x_lim = NULL,
                             y_lim = NULL,
                             xlab = "Prediction",
                             ylab = "Observation",
                             points_col_list = NULL,
                             colour_to_match = 'black',
                             data_summary = FALSE) {
  
  if (!exists("obs") | !exists("pred")) 
    stop("obs and pred cannot be null.")
  
  n_groups <- length(unique(data[, group]))
  
  if (is.null(follow_up)) 
    data$follow_up <- 1
  
  if (!is.null(group)) {
    data %>%
      group_by(!!sym(group)) %>%
      mutate(decile = ntile(!!sym(pred), nTiles)) %>%
      group_by(.data$decile, !!sym(group)) %>%
      summarise(obsRate = mean(!!sym(obs) / follow_up, na.rm = TRUE),
                obsRate_SE = sd(!!sym(obs) / follow_up, na.rm = TRUE) / sqrt(n()),
                obsNo = n(),
                predRate = mean(!!sym(pred), na.rm = TRUE)) -> dataDec_mods
    colnames(dataDec_mods)[colnames(dataDec_mods) == "group"] <- group
  } else {
    data %>%
      mutate(decile = ntile(!!sym(pred), nTiles)) %>%
      group_by(.data$decile) %>%
      summarise(obsRate = mean(!!sym(obs) / follow_up, na.rm = TRUE),
                obsRate_SE = sd(!!sym(obs) / follow_up, na.rm = TRUE) / sqrt(n()),
                obsNo = n(),
                predRate = mean(!!sym(pred), na.rm = TRUE)) -> dataDec_mods
  }
  
  dataDec_mods$obsRate_UCL <- dataDec_mods$obsRate + 1.96 * dataDec_mods$obsRate_SE
  dataDec_mods$obsRate_LCL <- dataDec_mods$obsRate - 1.96 * dataDec_mods$obsRate_SE
  
  dataDec_mods <- as.data.frame(dataDec_mods)

  
  multiply_and_add_percent <- function(x) {
    paste0(format(x * 100, digits = 4), "%")
  }
  
  calibPlot_obj <- ggplot(data = dataDec_mods, aes(y = .data$obsRate, x = .data$predRate)) +
    geom_point(color = colour_to_match) +
    lims(x = x_lim,
         y = y_lim) +
    geom_errorbar(aes(ymax = .data$obsRate_UCL, ymin = .data$obsRate_LCL),linewidth = 0.3,
                  col = colour_to_match) +
    geom_smooth(method = "lm", se = FALSE, color = colour_to_match, linewidth = 0.8, fullrange = TRUE) + # Add regression line
    geom_ribbon(aes(ymin = .data$obsRate_LCL, ymax = .data$obsRate_UCL), fill = colour_to_match, alpha = 0.3) + # Add shaded area for confidence interval
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = 'black', size = 0.3) +
    scale_color_manual(values = ifelse(is.null(points_col_list),
                                       ggplot2::scale_colour_brewer(palette = "Set3")$palette(8)[5],
                                       points_col_list)) +
    scale_x_continuous(labels = function(x) multiply_and_add_percent(x), limits = x_lim) + # Update x-axis scale to multiply values by 100 and add percentage sign
    scale_y_continuous(labels = function(y) multiply_and_add_percent(y), limits = y_lim) + # Update y-axis scale to multiply values by 100 and add percentage sign
    labs(x = ifelse(is.null(xlab), "pred", xlab),
         y = ifelse(is.null(ylab), "obs", ylab),
         title = title) +
    theme(panel.grid.major = element_line(color = "grey80", size = 0.1),
          #panel.grid.minor = element_line(color = "grey90", size = 0.05),
          panel.background = element_rect(fill = "white"),
          axis.line = element_line(colour = "black"),
          text = element_text(size = 10),
          
          axis.text.x = element_text(size = 9, color = "black"),
          axis.text.y = element_text(size = 9, color = "black"),
          
          
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
          legend.position = legendPosition,
          axis.ticks = element_line(color = "grey60"))
  res_list <- list(calibration_plot = calibPlot_obj)
  if (data_summary) res_list$data_summary <- dataDec_mods
  
  return(res_list)
}
