##### INFO: This code is designed for machine learning model training and evaluation with a focus 
# on hyperparameter tuning and resampling techniques, particularly for classification tasks. 
# It uses mostly mlr3* package for these tasks.

library("RPostgreSQL")
library("progressr")
library("mlr3")
library("mlr3verse")
library("mlr3learners")
#library("mlr3viz")
library("mlr3tuning")
library("plyr")
library("dplyr")
library("dbplyr")
library("DBI")
library("glue")
library("tidyverse")
library("janitor")
library("data.table")
library("ParallelLogger")
library("doParallel")
library("foreach")
library(PRROC)
library(boot)
library("mlr3db")
library(pROC)

#-----------------------------------------------------------------Connection
# Load configuration from config.R
source("R/config.R")
db_info <- paste("user=", db_user, " password=", db_password,
                           " dbname=", db, " host=", host_db, " port=", db_port, sep = ",")

con <- dbConnect(RPostgres::Postgres(), dbname = db, host = host_db,
                 port = db_port, user = db_user, password = db_password)
#-----------------------------------------------------------------Connection END

#-----------------------------------------------------------------Preprocess 
ParallelLogger::logInfo("Creating Cohorts")

cohort <- tbl(con, in_schema("synpuf_cdm", "target_cohort" )) %>% 
  rename("person_id" = "subject_id") %>% 
  filter(cohort_definition_id == 3) %>% select("person_id", "cohort_start_date")
condition_era <- tbl(con, in_schema("synpuf_cdm", "condition_era"))
death <- tbl(con, in_schema("synpuf_cdm", "death"))
observation <- tbl(con, in_schema("synpuf_cdm", "observation"))
person <- tbl(con, in_schema("synpuf_cdm", "person"))
drug_era <- tbl(con, in_schema("synpuf_cdm", "drug_era"))


ParallelLogger::logInfo("Creating Outcome Cohort")
final_cohort <- left_join(cohort, death, by='person_id') %>%
  select("death_type_concept_id", "person_id", "cohort_start_date") %>% collect()

ParallelLogger::logInfo("-----Creating Features-----")
ParallelLogger::logInfo("Joining Demographic Data")
final_cohort <- cohort %>% left_join(person, by='person_id') %>%
  
  select("gender_concept_id", "person_id") %>% collect() %>% 
  gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id") 

ParallelLogger::logInfo("Joining Condition Era Data")
final_cohort <- cohort %>% left_join(condition_era, by='person_id') %>%
  filter(condition_era_start_date <= cohort_start_date) %>%
  select("condition_concept_id", "person_id") %>% collect() %>% 
  gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")

ParallelLogger::logInfo("Joining Observation Data")
final_cohort <- cohort %>% left_join(observation, by='person_id') %>% 
  filter(observation_date <= cohort_start_date) %>%
  select("observation_concept_id", "person_id") %>% collect() %>% 
  gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")

ParallelLogger::logInfo("Joining Drug Era Data")
final_cohort <- cohort %>% left_join(drug_era, by='person_id') %>%
  filter(drug_era_start_date <= cohort_start_date) %>%
  select("drug_concept_id", "person_id") %>% collect() %>% 
  gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")  
gc()

ParallelLogger::logInfo("-----Preprocessing Data-----")
ParallelLogger::logInfo("Removing column \'person_ID\'")

final_cohort <- subset( final_cohort, select = -person_id)
final_cohort <- subset( final_cohort, select = -cohort_start_date)
ParallelLogger::logInfo("Standardizing column types to Integer")
final_cohort[] <- lapply(final_cohort, as.integer)
ParallelLogger::logInfo("Replacing NAs with 0")

final_cohort <- final_cohort %>% replace(is.na(.), 0)
ParallelLogger::logInfo("Standardizing non-zero values to 1")

final_cohort[final_cohort != 0] <- 1
logInfo("Converting target column type to factor")
final_cohort$death_type_concept_id = as.factor(final_cohort$death_type_concept_id)
gc()

#final_cohort_test = final_cohort%>% head(1000) #%>% select(01, 02,03, 04,05,death_type_concept_id)
#-----------------------------------------------------------------PreProcess END

#-----------------------------------------------------------------MLR3 TASK
logInfo("Creating mlr3-task")

task_cadaf = mlr3::as_task_classif(final_cohort, target="death_type_concept_id", "cadaf")

task_cadaf$positive = "1"
split = partition(task_cadaf, ratio = 0.75, stratify = TRUE)
task_cadaf$set_row_roles(split$test, "test")

lasso = lrn("classif.glmnet", predict_type = "prob")
gradient = lrn("classif.xgboost", predict_type = "prob", nrounds = 500, 
               early_stopping_rounds = 25, early_stopping_set = "test", eval_metric = "auc")
forest = lrn("classif.ranger", predict_type = "prob", max.depth = 17, seed = 12345, 
             mtry = as.integer(sqrt(length(task_cadaf$feature_names))))

terminator = trm("evals", n_evals = 5)
fselector = fs("random_search")

#-----------------------------------------------------------------MLR3 TASK END

#-----------------------------------------------------------------Hyperspaces
# Set a random seed for reproducibility
set.seed(7832)

#Set the logging threshold for mlr3 to "warn"
lgr::get_logger("mlr3")$set_threshold("warn")


### Hyper-parameters for tuning ###
# Define hyperparameter search spaces for each learner

# Hyperparameters for 'lrGradient'
spGradient = ps(
  scale_pos_weight = p_dbl(40,40), 
  eta = p_dbl(-4, 0), 
  .extra_trafo = function(x, param_set) {
    x$eta = round(10^(x$eta)) 
    x
  }
)

# Hyperparameters for 'lrForest'
spForest = ps(
  num.trees = p_int(500,2000) 
)

# Hyperparameters for 'lrLasso'
spLasso = ps(
  s = p_dbl(-12, 12),
  alpha = p_dbl(1,1), 
  .extra_trafo = function(x, param_set) {
    x$s = round(2^(x$s))
    x
  }
)

# Hyperparameters for 'lrElastic'
spElastic = ps(
  s = p_dbl(-12, 12),
  alpha = p_dbl(0,1), 
  .extra_trafo = function(x, param_set) { 
    x$s = round(2^(x$s))
    x
  }
)

#-----------------------------------------------------------------Hyperspaces END

#----------------------------------------------------------------Learners and tuning

# Define resampling strategy for inner cross-validation with 3 folds
inner_cv3 = rsmp("cv", folds = 3)
# Define the performance measure to optimize (Area under the Precision-Recall Curve)
measure = msr("classif.prauc")
# Define a termination criterion for early stopping after 5 evaluations
terminator2 = trm("evals", n_evals = 5)
terminator3 =trm("evals", n_evals = 1)

# Define an auto-tuner for gradient model
gradientAuto = auto_tuner(
  tuner = tnr("random_search"),
  learner = gradient,
  resampling = inner_cv3,
  measure = measure,
  search_space = spGradient,
  terminator = terminator2
)

# Define an auto-tuner for forest model
forestAuto = auto_tuner(
  tuner = tnr("random_search"),
  learner = forest,
  resampling = inner_cv3,
  measure = measure,
  search_space = spForest,
  terminator = terminator2
)

# Create an AutoFSelector for gradient model
lrGradient = AutoFSelector$new(
  learner = gradientAuto,
  resampling = rsmp("holdout"),
  measure = msr("classif.prauc"),
  terminator = terminator3,
  fselector = fselector
)

# Create an AutoFSelector for forest model
lrForest = AutoFSelector$new(
  learner = forestAuto,
  resampling = rsmp("holdout"),
  measure = msr("classif.prauc"),
  terminator = terminator3,
  fselector = fselector
)
# Define an auto-tuner for lasso model
lrLasso = auto_tuner(
  tuner = tnr("random_search"),
  learner = lasso,
  resampling = inner_cv3,
  measure = measure,
  search_space = spLasso,
  terminator = terminator2
)
# Define an auto-tuner for elastic net model
lrElastic = auto_tuner(
  tuner = tnr("random_search"),
  learner = lasso,
  resampling = inner_cv3,
  measure = measure,
  search_space = spLasso,
  terminator = terminator2
)

#----------------------------------------------------------------Learners and tuning END
#p = as.numeric(as.character(predictions$prob[,1]))
#t = as.numeric(as.character(predictions$truth))
#mean((p - t)^2)

#calculate_brier_score <- function(predictions) {
#  actual_outcomes <- as.numeric(predictions$truth)
#  brier_score <- mean((predictions$prob - actual_outcomes)^2)
#  return(brier_score)
#}


# Function to calculate Brier Score for binary classification with two probabilities
calculate_brier_score <- function(predictions) {
  # Extract the true outcomes (0 or 1)
  actual_outcomes <- as.numeric(as.character(predictions$truth))
  
  # Extract the predicted probabilities for class 1
  predicted_probabilities_class_1 <- as.numeric(as.character(predictions$prob[,1]))
  
  # Calculate the Brier Score as the mean squared difference between predicted probabilities and actual outcomes
  brier_score <- mean((predicted_probabilities_class_1 - actual_outcomes)^2)
  
  return(brier_score)
}

# Function to calculate ROC AUC
calculate_roc_auc <- function(predictions) {
  actual_outcomes_truth <- as.numeric(predictions$truth)
  actual_outcomes_response <- as.numeric(predictions$response)
  roc <- roc(actual_outcomes_truth, actual_outcomes_response)
  roc_auc <- auc(roc)
  return(roc_auc)
}

# Function to calculate PRC AUC
calculate_prc_auc <- function(predictions) {
  actual_outcomes_truth <- as.numeric(predictions$truth)
  actual_outcomes_response <- as.numeric(predictions$response)
  prc <- pr.curve(actual_outcomes_truth, actual_outcomes_response)
  prc_auc <- prc$auc.integral
  return(prc_auc)
}

# Create a list of models
models <- list(lrGradient, lrForest, lrLasso, lrElastic)
model_names <- c("lrGradient", "lrForest", "lrLasso", "lrElastic")

# Create a list to store results
results <- list()

# Define a random feature selection method
#fselector = fs("random_search")

file = system.file(file.path("extdata", "spam.parquet"), package = "mlr3db")

# Create a backend on the file
backend = as_duckdb_backend(file)

# Initialize an empty vector to store running times
running_times <- numeric(length(models))

# Iterate through models 
for (i in seq_along(models)) {
  model <- models[[i]]
  model_name <- model_names[i]
  
  # Record the start time
  start_time <- Sys.time()
  
  # Train the model
  trained_model <- model$train(task_cadaf, split$train)
  
  # Make predictions
  predictions <- model$predict_newdata(final_cohort[split$test, ])
  
  # Calculate Brier Score
  brier_score <- calculate_brier_score(predictions)
  
  # Calculate ROC AUC
  roc_auc <- calculate_roc_auc(predictions)
  
  # Calculate PRC AUC
  prc_auc <- calculate_prc_auc(predictions)
  
  # Record the end time
  end_time <- Sys.time()
  
  # Calculate the running time
  running_time <- end_time - start_time
  running_times[i] <- running_time
  
  cat(paste("Model:", model_name, "\n"))
  cat("Running Time:", running_time, "\n")
  
  results[[model_name]] <- list(
    brier_score = brier_score, 
    roc_auc = roc_auc,
    prc_auc = prc_auc
  
  )
}

# ----------------------------- Calculate 95% confidence intervals

# ----------------------- Function to calculate 95% CI for ROC AUC
calculate_roc_auc_ci <- function(model, task, n_bootstrap = 1000) {
  y_true <- as.numeric(task$truth())
  
  # Initialize an empty vector to store ROC AUC values
  roc_auc_values <- numeric(n_bootstrap)
  
  # Perform bootstrapping
  for (j in 1:n_bootstrap) {
    # Generate a random sample with replacement
    indices <- sample(seq_along(y_true), replace = TRUE)
    
    # Make predictions
    predictions <- model$predict_newdata(final_cohort[indices, ])
    
    # Calculate ROC AUC
    roc_auc_values[j] <- calculate_roc_auc(predictions)
  }
  
  # Sort AUC values
  sorted_auc_values <- sort(roc_auc_values)
  
  # Identify confidence interval bounds
  lower_bound <- sorted_auc_values[round(0.025 * n_bootstrap)]
  upper_bound <- sorted_auc_values[round(0.975 * n_bootstrap)]
  
  # Return results
  result <- list(
    lower_bound = lower_bound,
    upper_bound = upper_bound
  )
  
  cat("95% CI for ROC AUC:", "(", lower_bound, ",", upper_bound, ")\n")
  
  return(result)
}

# Example usage for (e.g.,lrGradient) models
lrGradient_ci <- calculate_roc_auc_ci(lrGradient, task_cadaf)
lrForest_roc_ci <- calculate_roc_auc_ci(lrForest, task_cadaf)
lrLasso_roc_ci <- calculate_roc_auc_ci(lrLasso, task_cadaf)
lrElastic_roc_ci <- calculate_roc_auc_ci(lrElastic, task_cadaf)
#-------------------------------- Function to calculate 95% CI for PRC AUC

# Function to calculate 95% CI for PRC AUC
calculate_prc_auc_ci <- function(model, task, n_bootstrap = 1000) {
  y_true <- as.numeric(task$truth())
  
  # Initialize an empty vector to store PRC AUC values
  prc_auc_values <- numeric(n_bootstrap)
  
  # Perform bootstrapping
  for (j in 1:n_bootstrap) {
    # Generate a random sample with replacement
    indices <- sample(seq_along(y_true), replace = TRUE)
    
    # Make predictions
    predictions <- model$predict_newdata(final_cohort[indices, ])
    
    # Calculate PRC AUC
    prc_auc_values[j] <- calculate_prc_auc(predictions)
  }
  
  # Sort AUC values
  sorted_auc_values <- sort(prc_auc_values)
  
  # Identify confidence interval bounds
  lower_bound <- sorted_auc_values[round(0.025 * n_bootstrap)]
  upper_bound <- sorted_auc_values[round(0.975 * n_bootstrap)]
  
  # Return results
  result <- list(
    lower_bound = lower_bound,
    upper_bound = upper_bound
  )
  
  cat("95% CI for PRC AUC:", "(", lower_bound, ",", upper_bound, ")\n")
  
  return(result)
}

# Example usage for lrGradient model
lrGradient_prc_ci <- calculate_prc_auc_ci(lrGradient, task_cadaf)
lrForest_prc_ci <- calculate_prc_auc_ci(lrForest, task_cadaf)
lrLasso_prc_ci <- calculate_prc_auc_ci(lrLasso, task_cadaf)
lrElastic_prc_ci <- calculate_roc_auc_ci(lrElastic, task_cadaf)

#-------------------------------- Function to calculate 95% CI for Brier score







#conf_intervals <- lapply(results, function(model_result) {
#  list(
#    brier_score = quantile(model_result$brier_score, c(0.025, 0.975)),
#    roc_auc = quantile(model_result$roc_auc, c(0.025, 0.975)),
#    prc_auc = quantile(model_result$prc_auc, c(0.025, 0.975))
#  )
#})




# -------------------- Print the results with confidence intervals and running times
for (i in seq_along(model_names)) {
  model_name <- model_names[i]
  cat(paste("Model:", model_name, "\n"))
  cat("Brier Score: ", results[[i]]$brier_score, "\n")
  #cat("95% Confidence Interval for Brier Score: ", conf_intervals[[i]]$brier_score, "\n")
  cat("ROC AUC: ", results[[i]]$roc_auc, "\n")
  #cat("95% Confidence Interval for ROC AUC: ", conf_intervals[[i]]$roc_auc, "\n")
  cat("PRC AUC: ", results[[i]]$prc_auc, "\n")
  #cat("95% Confidence Interval for PRC AUC: ", conf_intervals[[i]]$prc_auc, "\n")
  cat("Running Time:", running_times[i], "\n")
}

# -----------------------------Confidence interval part

  


