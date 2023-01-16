library("RPostgreSQL")
library("mlr3verse")
library("dplyr")
library("dbplyr")
library("DBI")
library("glue")
library("mlr3viz")
library("tidyverse")
library("janitor")
library("data.table")
library("mlr3verse")
library(ParallelLogger)
library("progressr")

#-----------------------------------------------------------------Connection

ParallelLogger::logInfo("Connecting to DB")
drv <- dbDriver("PostgreSQL")
db <- 'omop'  
host_db <- "localhost"  
db_port <- '5432'  
db_user <- "postgres"  
db_password <- "1234"

con <- dbConnect(RPostgres::Postgres(), dbname = db, host=host_db, port=db_port, 
                 user=db_user, password=db_password)

#-----------------------------------------------------------------Connection END


#-----------------------------------------------------------------Preprocess 
ParallelLogger::logInfo("Creating Cohorts")
cohort <- tbl(con, in_schema("results", "cohort" )) %>% rename("person_id" = "subject_id") %>% filter(cohort_definition_id == 2) %>% select("person_id")
condition_era <- tbl(con, in_schema("cmd", "condition_era"))
death <- tbl(con, in_schema("cmd", "death"))
observation <- tbl(con, in_schema("cmd", "observation"))
person <- tbl(con, in_schema("cmd", "person"))
drug_era <- tbl(con, in_schema("cmd", "drug_era"))


tables <- c(condition_occurrence, death, observation, person, drug_exposure)
features <- c("observation_concept_id", "death_type_concept_id", "person_id", 
              "drug_concept_id") 

ParallelLogger::logInfo("Creating Outcome Cohort")
final_cohort <- left_join(cohort, death, by='person_id') %>%
  select("death_type_concept_id", "person_id") %>% collect()

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
  select("condition_concept_id", "person_id") %>% collect() %>% 
  gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")

ParallelLogger::logInfo("Joining Observation Data")
final_cohort <- cohort %>% left_join(observation, by='person_id') %>% 
  select("observation_concept_id", "person_id") %>% collect() %>% gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")

ParallelLogger::logInfo("Joining Drug Era Data")
final_cohort <- cohort %>% left_join(drug_era, by='person_id') %>%
  select("drug_concept_id", "person_id") %>% collect() %>% gather(variable, value, -(c(person_id))) %>% 
  mutate(value2 = value)  %>% unite(temp, variable, value2) %>% 
  distinct(.keep_all = TRUE) %>% 
  spread(temp, value) %>% left_join(final_cohort, by="person_id")

gc()

ParallelLogger::logInfo("-----Preprocessing Data-----")
ParallelLogger::logInfo("Removing column \'person_ID\'")

final_cohort <- subset( final_cohort, select = -person_id)
ParallelLogger::logInfo("Standardizing column types to Integer")

final_cohort[] <- lapply(final_cohort, as.integer)
ParallelLogger::logInfo("Replacing NAs with 0")

final_cohort <- final_cohort %>% replace(is.na(.), 0)
ParallelLogger::logInfo("Standardizing non-zero values to 1")

final_cohort[final_cohort != 0] <- 1
logInfo("Converting target column type to factor")
final_cohort$death_type_concept_id = as.factor(final_cohort$death_type_concept_id)
gc()


#-----------------------------------------------------------------PreProcess END

#-----------------------------------------------------------------MLR3 TASK
logInfo("Creating mlr3-task")
task_cadaf = mlr3::as_task_classif(final_cohort, target="death_type_concept_id", "cadaf")
print(task_cadaf)
task_cadaf$positive = "1"
split = partition(task_cadaf, ratio = 0.75, stratify = TRUE)
task_cadaf$set_row_roles(split$test, "test")

lasso = lrn("classif.glmnet", predict_type = "prob")
gradient = lrn("classif.xgboost", predict_type = "prob", nrounds = 500, early_stopping_rounds = 25, early_stopping_set = "test",
               eval_metric = "auc", ntreelimit = 1000)
forest = lrn("classif.ranger", predict_type = "prob", max.depth = 17, seed = 12345, mtry = as.integer(sqrt(length(task_cadaf$feature_names))))

terminator = trm("evals", n_evals = 5)
fselector = fs("random_search")

#-----------------------------------------------------------------MLR3 TASK END

#-----------------------------------------------------------------Hyperspaces---

set.seed(7832)
lgr::get_logger("mlr3")$set_threshold("warn")

po_under = po("classbalancing",
              id = "undersample", adjust = "major",
              reference = "major", shuffle = FALSE, ratio = 1 / 6)

gr_smote =
  po("colapply", id = "int_to_num",
     applicator = as.numeric, affect_columns = selector_type("integer")) %>>%
  po("smote", dup_size = 6) %>>%
  po("colapply", id = "num_to_int",
     applicator = function(x) as.integer(round(x, 0L)), affect_columns = selector_type("numeric"))




###No sampling techniques
spGradient = ps(
  scale_pos_weight = p_dbl(40,40),
  eta = p_dbl(-4, 0),
  .extra_trafo = function(x, param_set) {
    x$eta = round(10^(x$eta))
    x
  }
)

spForest = ps(
  num.trees = p_int(500,2000)
)

spLasso = ps(
  s = p_dbl(-12, 12),
  alpha = p_dbl(1,1),
  .extra_trafo = function(x, param_set) {
    x$s = round(2^(x$s))
    x
  }
)

spElastic = ps(
  s = p_dbl(-12, 12),
  alpha = p_dbl(0,1),
  .extra_trafo = function(x, param_set) {
    x$s = round(2^(x$s))
    x
  }
)

###Smote BEGIN

lrLassoSmote = as_learner(gr_smote %>>% lasso)

spLassoSmote = ps(
  classif.glmnet.s = p_dbl(-9.21034, 9.21034),
  classif.glmnet.alpha = p_dbl(1,1),
  smote.dup_size = p_int(1, 6),
  smote.K = p_int(1, 6),
  .extra_trafo = function(x, param_set) {
    x$classif.glmnet.s = round(2^(x$classif.glmnet.s))
    x
  }
)


###SMOTE END

###UNDER BEGIN

lrLassoUnder = as_learner(po_under %>>% lasso)

spLassoUnder = ps(
  classif.glmnet.s = p_dbl(-9.21034, 9.21034),
  classif.glmnet.alpha = p_dbl(1,1),
  undersample.ratio = p_dbl(1/6, 1),
  .extra_trafo = function(x, param_set) {
    x$classif.glmnet.s = round(2^(x$classif.glmnet.s))
    x
  }
)
###UNDER END

### DEFINE SAMPLING END

###Learner Creation
inner_cv3 = rsmp("cv", folds = 3)
measure = msr("classif.prauc")
terminator2 = trm("evals", n_evals = 5)
terminator3 =trm("evals", n_evals = 1)

gradientAuto = auto_tuner(
  method = "random_search",
  learner = gradient,
  resampling = inner_cv3,
  measure = measure,
  search_space = spGradient,
  terminator = terminator2
)

forestAuto = auto_tuner(
  method = "random_search",
  learner = forest,
  resampling = inner_cv3,
  measure = measure,
  search_space = spForest,
  terminator = terminator2
)

lrGradient = AutoFSelector$new(
  learner = gradientAuto,
  resampling = rsmp("holdout"),
  measure = msr("classif.prauc"),
  terminator = terminator3,
  fselector = fselector
)

lrForest = AutoFSelector$new(
  learner = forestAuto,
  resampling = rsmp("holdout"),
  measure = msr("classif.prauc"),
  terminator = terminator3,
  fselector = fselector
)

lrLasso = auto_tuner(
  method = "random_search",
  learner = lasso,
  resampling = inner_cv3,
  measure = measure,
  search_space = spLasso,
  terminator = terminator2
)

lrElastic = auto_tuner(
  method = "random_search",
  learner = lasso,
  resampling = inner_cv3,
  measure = measure,
  search_space = spLasso,
  terminator = terminator2
)


learns = list(
  lrForest,
  lrGradient,
  lrLasso,
  lrElastic
)



learnsSmote = list(
  auto_tuner(
    method = "random_search",
    learner = lrLassoSmote,
    resampling = inner_cv3,
    measure = measure,
    search_space = spLassoSmote,
    terminator = terminator2
  )
)

learnsUnder = list(
  auto_tuner(
    method = "random_search",
    learner = lrLassoUnder,
    resampling = inner_cv3,
    measure = measure,
    search_space = spLassoUnder,
    terminator = terminator2
  )
)



crossValidation = rsmp("cv", folds=3)

design = benchmark_grid(
  tasks = task_cadaf,
  learners = learns,
  resamplings = crossValidation)

designUnder = benchmark_grid(
  tasks = task_cadaf,
  learners = learnsUnder,
  resamplings = crossValidation)

designSmote = benchmark_grid(
  tasks = task_cadaf,
  learners = learnsSmote,
  resamplings = crossValidation)

bmr = benchmark(design, store_models = TRUE)
bmrSmote = benchmark(designSmote, store_models = TRUE)
bmrUnder = benchmark(designUnder, store_models = TRUE)
bmr$aggregate(measure)
autoplot(bmr, type = "roc")
autoplot(bmrSmote, type = "roc")
autoplot(bmrUnder, type = "roc")

#-----------------------------------------------------------------Pipeop END---

#LassoLogsitic without tuning
lasso_model = lasso$train(task_cadaf, split$train)
lasso$model
prediction_lasso = lasso_model$predict_newdata(final_cohort[split$test, ])
autoplot(prediction_lasso, type="roc")
autoplot(prediction_lasso, type="prc")

prediction_lasso$score(msr("classif.auc"))