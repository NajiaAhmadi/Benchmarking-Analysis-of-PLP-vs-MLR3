##### INFO: This code is designed for machine learning model training and evaluation with a focus 
# on hyperparameter tuning and resampling techniques, particularly for classification tasks. 
# It uses mostly mlr3* package for these tasks.

library("RPostgreSQL")
library("progressr")
library("mlr3")
library("mlr3verse")
library("mlr3learners")
library("mlr3tuning")
library("dplyr")
library("dbplyr")
library("DBI")
library("glue")
library("tidyverse")
library("data.table")
library("ParallelLogger")
library("doParallel")
library("foreach")
library("PRROC")
library("boot")
library("pROC")
library(furrr)
library(future)

#library("mlr3db")
#library("janitor")
#library("plyr")
#library("mlr3viz")
#-----------------------------------------------------------------Connection START
# Load configuration from config.R

# Check and set the working directory
if (!file.exists("config.R")) {
  stop("Error: 'config.R' not found in the current directory.")
}

# Set the working directory to the location of the script
# This assumes that both the script and 'config.R' are in the same directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Now source the 'config.R' file
source("config.R")

# Database Connection
tryCatch({
  ParallelLogger::logInfo("Connecting to the database...")
  
  db_info <- paste("user=", db_user, " password=", db_password,
                   " dbname=", db, " host=", host_db, " port=", db_port, sep = ",")
  
  con <- dbConnect(RPostgres::Postgres(), dbname = db, host = host_db,
                   port = db_port, user = db_user, password = db_password)
  
  ParallelLogger::logInfo("Connected to the database successfully.")
}, error = function(e) {
  ParallelLogger::logError("Error connecting to the database:", conditionMessage(e))
  ParallelLogger::logError("Connection details:", db_info)
})

#-----------------------------------------------------------------Connection END

#-----------------------------------------------------------------Preprocess START 
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

#-----------------------------------------------------------------Preprocess END

#-----------------------------------------------------------------mlr3 TASK START
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

#-----------------------------------------------------------------mlr3 TASK END

#-----------------------------------------------------------------Hyperspaces START
# Set a random seed for reproducibility
set.seed(123)

#Set the logging threshold for mlr3 to "warn"
lgr::get_logger("mlr3")$set_threshold("warn")

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

#----------------------------------------------------------------Learners and tuning START

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
  terminator = terminator2,
  fselector = fselector
)

# Create an AutoFSelector for forest model
lrForest = AutoFSelector$new(
  learner = forestAuto,
  resampling = rsmp("holdout"),
  measure = msr("classif.prauc"),
  terminator = terminator2,
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

#--------------------------------------------------------------- Task and metrics START 

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
models <- list(lrGradient, 
               lrForest, lrLasso, lrElastic)
model_names <- c("lrGradient", 
                 "lrForest", "lrLasso", "lrElastic")

# Function to train and evaluate a model
train_and_evaluate <- function(model, model_name, task, split, final_cohort) {
  start_time <- Sys.time()
  
  # Train the model
  trained_model <- model$train(task, split$train)
  
  # Make predictions
  predictions <- model$predict_newdata(final_cohort[split$test, ])
  
  # Calculate evaluation metrics
  brier_score <- calculate_brier_score(predictions)
  roc_auc <- calculate_roc_auc(predictions)
  prc_auc <- calculate_prc_auc(predictions)
  
  end_time <- Sys.time()
  running_time <- end_time - start_time
  
  cat(paste("Model:", model_name, "\n"))
  cat("Running Time:", running_time, "\n")
  
  return(list(
    brier_score = brier_score, 
    roc_auc = roc_auc,
    prc_auc = prc_auc,
    running_time = running_time
  ))
}

# Train and evaluate models
results <- list()

for (i in seq_along(models)) {
  model <- models[[i]]
  model_name <- model_names[i]
  
  results[[model_name]] <- train_and_evaluate(model, model_name, task_cadaf, split, final_cohort)
} 

results_model_train_test <- results
#results_model_train_test_lrGradient <-results_model_train_test
#results_model_train_test_otherModels <- results

#results_all_models <- c(results_model_train_test_lrGradient,results_model_train_test_otherModels)

# Print
for (i in seq_along(model_names)) {
  model_name <- model_names[i]
  cat(paste("Model:", model_name, "\n"))
  cat("Brier Score: ", results_model_train_test[[model_name]]$brier_score, "\n")
  cat("ROC AUC: ", results_model_train_test[[model_name]]$roc_auc, "\n")
  cat("PRC AUC: ", results_model_train_test[[model_name]]$prc_auc, "\n")
  cat("Running Time:", results_model_train_test[[model_name]]$running_time, "\n")
}

# Set the maximum size for globals
options(future.globals.maxSize = Inf)

# Set the number of parallel workers
#plan(future::multisession, workers = 4)  # Adjust the number of workers based on your system

plan(list(
  tweak(future::multisession, workers = availableCores() %/% 4),
  tweak(future::multisession, workers = 4)
))

### future_map_dbl is a function for parallel computation. It applies a function (~ { ... }) 
### to each element in the sequence 1:n_bootstrap and returns a double vector.
#options(future.rng.onMisuse = "ignore")

calculate_metric_dist <- function(model, task, final_cohort, metric_function, n_bootstrap = 10, seed = NULL) {
  set.seed(seed)  # This seed is local to the function
  metric_values <- future_map_dbl(1:n_bootstrap, ~ {
    indices <- sample(seq_along(as.numeric(task$truth())), replace = TRUE)
    predictions <- model$predict_newdata(final_cohort[indices, ])
    metric_function(predictions)
  })
  return(metric_values)
}


#brier_score_result <- calculate_metric_dist(lrElastic, task_cadaf, final_cohort, calculate_brier_score, seed = 123)
#numeric_values <- unlist(brier_score_result)
#standard_deviation = sd(numeric_values)
#xbar = mean(as.numeric(numeric_values)) # mean of the score ditribution (mean is also the AUC value itself)

#margin = qt(0.975, df = n_bootstrap-1)*standard_deviation/sqrt(n_bootstrap)

#upperinterval= xbar + margin
#lowerinterval= xbar - margin

calculate_summary_statistics <- function(metric_result) {
  numeric_values <- unlist(metric_result)
  n_bootstrap <- length(numeric_values)
  
  xbar <- mean(as.numeric(numeric_values))
  standard_deviation <- sd(numeric_values)
  
  margin <- qt(0.975, df = n_bootstrap - 1) * standard_deviation / sqrt(n_bootstrap)
  
  upper_interval <- xbar + margin
  lower_interval <- xbar - margin
  
  return(list(
    xbar = xbar,
    standard_deviation = standard_deviation,
    lower_interval = lower_interval,
    upper_interval = upper_interval
  ))
}

calculate_metrics_summary <- function(model, task, final_cohort, seed = NULL) {
  brier_score_result <- calculate_metric_dist(model, task, final_cohort, calculate_brier_score, n_bootstrap = 10, seed = seed)
  roc_auc_result <- calculate_metric_dist(model, task, final_cohort, calculate_roc_auc, n_bootstrap = 10, seed = seed)
  prc_auc_result <- calculate_metric_dist(model, task, final_cohort, calculate_prc_auc, n_bootstrap = 10, seed = seed)
  
  brier_info <- calculate_summary_statistics(brier_score_result)
  roc_info <- calculate_summary_statistics(roc_auc_result)
  prc_info <- calculate_summary_statistics(prc_auc_result)
  
  metric_values <- list(
    brier = brier_score_result,
    roc = roc_auc_result,
    prc = prc_auc_result
  )
  
  return(list(
    brier_summary = brier_info,
    roc_summary = roc_info,
    prc_summary = prc_info,
    metric_values = metric_values,
    metric_info = list(
      brier = brier_info,
      roc = roc_info,
      prc = prc_info
    )
  ))
}


# call the function for each model
lrGradient_metrics_summary <- calculate_metrics_summary(lrGradient, task_cadaf, final_cohort, seed = 123)
lrForest_metrics_summary <- calculate_metrics_summary(lrForest, task_cadaf, final_cohort, seed = 123)
lrLasso_metrics_summary <- calculate_metrics_summary(lrLasso, task_cadaf, final_cohort, seed = 123)
lrElastic_metrics_summary <- calculate_metrics_summary(lrElastic, task_cadaf, final_cohort, seed = 123)


# save them in my computer 
write.csv(lrGradient_metrics_summary, "lrGradient_metrics_summary.csv", row.names = FALSE)
write.csv(lrForest_metrics_summary, "lrForest_metrics_summary.csv", row.names = FALSE)
write.csv(lrLasso_metrics_summary, "lrLasso_metrics_summary.csv", row.names = FALSE)
write.csv(lrElastic_metrics_summary, "lrElastic_metrics_summary.csv", row.names = FALSE)

# Accessing metrics:
print_metric_summary <- function(metric_summary) {
  metric_names <- c("brier", "roc", "prc")
  
  for (metric_name in metric_names) {
    cat("Metric:", tolower(metric_name), "\n")
    cat("xbar:", metric_summary[[paste0(metric_name, "_summary")]]$xbar, "\n")
    cat("Lower Interval:", metric_summary[[paste0(metric_name, "_summary")]]$lower_interval, "\n")
    cat("Upper Interval:", metric_summary[[paste0(metric_name, "_summary")]]$upper_interval, "\n\n")
  }
}

# Example usage:
cat("Model: ","Gradient Boosting Machines")
print_metric_summary(lrGradient_metrics_summary)
cat("Model: ","Random Forest")
print_metric_summary(lrForest_metrics_summary)
cat("Model: ","Lasso")
print_metric_summary(lrLasso_metrics_summary)
cat("Model: ","Elastic")
print_metric_summary(lrElastic_metrics_summary)

# Stop parallel processing
plan(sequential)

#--------------------------------------------------------------- Task and metrics END 



