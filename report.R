# Load required packages
packages <- c("caret", "randomForest", "glmnet", "xgboost", "gbm", "nnet", "ggplot2", "reshape2", "pROC")

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

set.seed(114514)

# Load Data

data_url <- "https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv"
heart_failure <- read.csv(data_url, header = TRUE)

print(summary(heart_failure))

## Show continuous variables and categorical variables

continuous_vars <- c("age", "creatinine_phosphokinase", "ejection_fraction",
                     "platelets", "serum_creatinine", "serum_sodium", "time")

dev.new(width = 8, height = 4)
par(mfrow=c(2,4))
for (var in continuous_vars) {
  hist(heart_failure[[var]], main = paste("Histogram of", var), 
       xlab = var, col="skyblue")
}

categorical_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "fatal_mi")

dev.new(width = 8, height = 4)
par(mfrow = c(2, 3))
for (var in categorical_vars) {
  barplot(table(heart_failure[[var]]),
          main = paste("Distribution of", var),
          col = c("skyblue", "salmon"),
          names.arg = c("No", "Yes"))
}

# Preprocess Data

heart_failure_preprocessed <- heart_failure

heart_failure_preprocessed$log_creatinine_phosphokinase <- log(heart_failure_preprocessed$creatinine_phosphokinase + 1)
heart_failure_preprocessed$log_serum_creatinine         <- log(heart_failure_preprocessed$serum_creatinine + 1)

fatal_mi_table <- table(heart_failure_preprocessed$fatal_mi)

continuous_vars_preprocessed <- c("age", "log_creatinine_phosphokinase", "ejection_fraction",
                                  "platelets", "log_serum_creatinine", "serum_sodium", "time")

heart_failure_preprocessed[continuous_vars_preprocessed] <- scale(heart_failure_preprocessed[continuous_vars_preprocessed])

class_counts <- as.numeric(fatal_mi_table)
weights <- ifelse(heart_failure_preprocessed$fatal_mi == 1, 
                  1 / class_counts[2], 
                  1 / class_counts[1])
heart_failure_preprocessed$weight <- weights

print(summary(heart_failure_preprocessed[continuous_vars_preprocessed]))

## Show Preprocess Data

dev.new(width = 8, height = 4)
par(mfrow = c(2, 4))
for (var in continuous_vars_preprocessed) {
  hist(heart_failure_preprocessed[[var]],
       main = paste("Histogram of", var),
       xlab = var, col = "skyblue")
}

# Test set and Valid set
set.seed(114514)
n             <- nrow(heart_failure_preprocessed)
train_indices <- sample(1:n, size = round(0.7 * n))
train_data    <- heart_failure_preprocessed[train_indices, ]
test_data     <- heart_failure_preprocessed[-train_indices, ]

cat("Number for test set：",  nrow(train_data), "\n")
cat("Number for valid set：", nrow(test_data), "\n")

train_data$fatal_mi <- factor(train_data$fatal_mi, levels = c(0, 1), labels = c("No", "Yes"))
test_data$fatal_mi  <- factor(test_data$fatal_mi,  levels = c(0, 1), labels = c("No", "Yes"))

ctrl <- trainControl(method = "cv",
                     number = 10,
                     savePredictions = "final",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Tuning

## Best combo for mtry, ntree and nodesize In Random Forest

mtry_values     <- c(2, 3, 4, 5, 6)
ntree_values    <- c(500, 750, 1000, 1250, 1500)
nodesize_values <- c(5, 10, 15, 20, 25)

tune_grid <- expand.grid(mtry = mtry_values, ntree = ntree_values, nodesize = nodesize_values)
tune_grid$OOB_Error <- NA

set.seed(114514)
for(i in 1:nrow(tune_grid)){
  current_mtry     <- tune_grid$mtry[i]
  current_ntree    <- tune_grid$ntree[i]
  current_nodesize <- tune_grid$nodesize[i]
  
  rf_fit <- randomForest(as.factor(fatal_mi) ~ . - weight, 
                         data     = train_data, 
                         mtry     = current_mtry, 
                         ntree    = current_ntree, 
                         nodesize = current_nodesize)
  
  tune_grid$OOB_Error[i] <- rf_fit$err.rate[current_ntree, "OOB"]
}

dev.new(width = 8, height = 4)
p <- ggplot(tune_grid, aes(x = factor(mtry), y = OOB_Error, fill = factor(nodesize))) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ ntree, labeller = label_both) +
  labs(title = "OOB Error Rate by mtry, ntree and nodesize",
       x = "mtry (Number of Variables Tried at Each Split)",
       y = "OOB Error Rate",
       fill = "nodesize") +
  theme_minimal()
print(p)

print(tune_grid[which.min(tune_grid$OOB_Error), ])

## Best combo for alpha and lambda in Logistic Regression

lr_grid <- expand.grid(alpha  = c(0, 0.5, 1),   # 0: Ridge, 1: Lasso, 0.5: Elastic Net
                       lambda = seq(0.0001, 0.1, length.out = 10))

set.seed(114514)
lr_tuned <- train(fatal_mi ~ . - weight,
                  data = train_data,
                  method = "glmnet",
                  trControl = ctrl,
                  tuneGrid = lr_grid,
                  metric = "ROC")

dev.new(width = 8, height = 4)
print(plot(lr_tuned))

print(lr_tuned$bestTune)

# Building the model
set.seed(114514)
#### 1. Random Forest (RF) - using optimal parameters: mtry = 2, ntree = 1500, nodesize = 20
rf_model <- train(fatal_mi ~ . - weight,
                  data = train_data,
                  method = "rf",
                  trControl = ctrl,
                  metric = "ROC",
                  tuneGrid = data.frame(mtry = 2),
                  ntree = 1500,
                  nodesize = 20,
                  importance = TRUE)

set.seed(114514)
##### 2. Logistic Regression (LR) - using glmnet with optimal parameters: alpha = 0.5, lambda = 0.1
lr_model <- train(fatal_mi ~ . - weight,
                  data = train_data,
                  method = "glmnet",
                  trControl = ctrl,
                  metric = "ROC",
                  tuneGrid = expand.grid(alpha = 0.5, lambda = 0.1),
                  weights = train_data$weight)

rf_preds <- rf_model$pred[, c("rowIndex", "Yes")]
lr_preds <- lr_model$pred[, c("rowIndex", "Yes")]

stack_data <- merge(rf_preds, lr_preds, by = "rowIndex", suffixes = c("_rf", "_lr"))
stack_data <- merge(stack_data,
                    data.frame(rowIndex = 1:nrow(train_data), fatal_mi = train_data$fatal_mi),
                    by = "rowIndex")

meta_model <- glm(fatal_mi ~ Yes_rf + Yes_lr, data = stack_data, family = binomial)

rf_test_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]
lr_test_prob <- predict(lr_model, newdata = test_data, type = "prob")[, "Yes"]

stack_test <- data.frame(Yes_rf = rf_test_prob, Yes_lr = lr_test_prob)
meta_pred_prob <- predict(meta_model, newdata = stack_test, type = "response")

## Best combo for threshold in Stacking Ensemble

threshold_values <- seq(0.3, 0.7, by = 0.05)
metrics_list     <- list()

for (th in threshold_values) {
  preds_th <- ifelse(meta_pred_prob > th, "Yes", "No")
  preds_th <- factor(preds_th, levels = c("No", "Yes"))
  
  conf_mat_th <- table(True = test_data$fatal_mi, Predicted = preds_th)
  
  TN <- conf_mat_th["No",   "No"]
  FP <- conf_mat_th["No",  "Yes"]
  FN <- conf_mat_th["Yes",  "No"]
  TP <- conf_mat_th["Yes", "Yes"]
  
  acc       <- (TP + TN) / sum(conf_mat_th)
  sens      <- if ((TP + FN) == 0) NA else TP / (TP + FN)
  spec      <- if ((TN + FP) == 0) NA else TN / (TN + FP)
  precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
  F1        <- if (is.na(precision) || is.na(sens) || (precision + sens) == 0) NA else 2 * precision * sens / (precision + sens)
  
  metrics_list[[as.character(th)]] <- data.frame(Threshold   = th,
                                                 Accuracy    = acc,
                                                 Sensitivity = sens,
                                                 Specificity = spec,
                                                 Precision   = precision,
                                                 F1          = F1)
}

metrics_df <- do.call(rbind, metrics_list)
rownames(metrics_df) <- NULL

metrics_melt <- melt(metrics_df, id.vars = "Threshold")

dev.new(width = 8, height = 4)
p <- ggplot(metrics_melt, aes(x = Threshold, y = value, color = variable)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  labs(title = "Performance Metrics vs. Custom Threshold",
       x = "Threshold",
       y = "Metric Value",
       color = "Metric") +
  theme_minimal()
print(p)

# Comparison

### Train Base Models
set.seed(114514)
##### 3. LDA Method
lda_model <- train(fatal_mi ~ . - weight,
                   data = train_data,
                   method = "lda",
                   trControl = ctrl,
                   metric = "ROC")

set.seed(114514)
##### 4. Gradient Boosting Machine (GBM)
gbm_model <- train(fatal_mi ~ . - weight,
                   data = train_data,
                   method = "gbm",
                   trControl = ctrl,
                   metric = "ROC",
                   verbose = FALSE,
                   tuneLength = 3)

set.seed(114514)
##### 5. Neural Network (NN) - using method "nnet"
nnet_model <- train(fatal_mi ~ . - weight,
                    data = train_data,
                    method = "nnet",
                    trControl = ctrl,
                    metric = "ROC",
                    tuneLength = 3,
                    trace = FALSE,
                    maxit = 200)

##### Comparison function

evaluate_on_test <- function(model, test_data, positive_class = "Yes", threshold = 0.5) {
  pred_prob <- predict(model, newdata = test_data, type = "prob")[, positive_class]
  pred_label <- factor(ifelse(pred_prob > threshold, "Yes", "No"), levels = c("No", "Yes"))
  cm <- table(True = test_data$fatal_mi, Predicted = pred_label)
  accuracy <- sum(diag(cm)) / sum(cm)
  roc_obj <- roc(response = test_data$fatal_mi, predictor = pred_prob, levels = c("No","Yes"))
  auc_val <- as.numeric(auc(roc_obj))
  TN <- cm["No", "No"]
  FP <- cm["No", "Yes"]
  FN <- cm["Yes", "No"]
  TP <- cm["Yes", "Yes"]
  sens <- TP / (TP + FN)
  spec <- TN / (TN + FP)
  prec <- TP / (TP + FP)
  F1 <- 2 * prec * sens / (prec + sens)
  return(c(Accuracy = accuracy, AUC = auc_val, Sensitivity = sens, Specificity = spec, F1 = F1))
}

base_models <- list(
  RandomForest       = rf_model,
  LogisticRegression = lr_model,
  LDA                = lda_model,
  GBM                = gbm_model,
  NeuralNetwork      = nnet_model
)

base_results <- lapply(base_models, evaluate_on_test, test_data = test_data)
base_df      <- do.call(rbind, base_results)
base_df      <- data.frame(Model = rownames(base_df), base_df, row.names = NULL)

##### 6. Stacking Ensemble

custom_threshold <- 0.6
stack_pred       <- factor(ifelse(meta_pred_prob > custom_threshold, "Yes", "No"), levels = c("No", "Yes"))
cm_stack         <- table(True = test_data$fatal_mi, Predicted = stack_pred)
stack_acc        <- sum(diag(cm_stack)) / sum(cm_stack)
roc_obj_stack    <- roc(response = test_data$fatal_mi, predictor = meta_pred_prob, levels = c("No","Yes"))
stack_auc        <- as.numeric(auc(roc_obj_stack))

TN <- cm_stack["No", "No"]
FP <- cm_stack["No", "Yes"]
FN <- cm_stack["Yes", "No"]
TP <- cm_stack["Yes", "Yes"]

stack_sens <- TP / (TP + FN)
stack_spec <- TN / (TN + FP)
stack_prec <- TP / (TP + FP)
stack_F1   <- 2 * stack_prec * stack_sens / (stack_prec + stack_sens)

stack_metrics <- c(Accuracy = stack_acc, AUC = stack_auc,
                   Sensitivity = stack_sens, Specificity = stack_spec, F1 = stack_F1)

##### Combine the results

compare_df <- rbind(base_df, data.frame(Model = "StackingEnsemble", t(stack_metrics)))
print(compare_df)

dev.new(width = 8, height = 4)
compare_melt <- melt(compare_df, id.vars = "Model")
p <- ggplot(compare_melt, aes(x = Model, y = as.numeric(value), fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Performance Comparison on Test Data",
       x = "Model",
       y = "Metric Value",
       fill = "Metric") +
  theme_minimal() +
  coord_flip()
print(p)