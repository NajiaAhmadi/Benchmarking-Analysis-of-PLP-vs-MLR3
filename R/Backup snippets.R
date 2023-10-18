## tuned learners
learns = list(
  lrForest,
  lrGradient,
  lrLasso,
  lrElastic
)

# lrGradient model 
model_LR <- lrGradient$train(task_cadaf, split$train)
prediction_LR = lrGradient$predict_newdata(final_cohort_test[split$test, ])
autoplot(prediction_LR, type="roc")
autoplot(prediction_LR, type="prc")
prediction_LR$score(msr("classif.auc"))
actual_outcomes_LR <- as.numeric(prediction_LR$truth)
brier_score_LR <- mean((prediction_LR$prob - actual_outcomes_LR)^2)
brier_score_LR


# lrForest model 
model_RF <- lrForest$train(task_cadaf, split$train)
prediction_RF = lrForest$predict_newdata(final_cohort_test[split$test, ])
autoplot(prediction_RF, type="roc")
autoplot(prediction_RF, type="prc")
prediction_RF$score(msr("classif.auc"))
actual_outcomes_RF <- as.numeric(prediction_RF$truth)
brier_score_RF <- mean((prediction_RF$prob - actual_outcomes_RF)^2)
brier_score_RF


# lrLasso model 
model_lasso <- lrLasso$train(task_cadaf, split$train)
prediction_lasso = lrLasso$predict_newdata(final_cohort_test[split$test, ])
autoplot(prediction_lasso, type="roc")
autoplot(prediction_lasso, type="prc")
prediction_lasso$score(msr("classif.auc"))
actual_outcomes_lasso <- as.numeric(prediction_lasso$truth)
brier_score_lasso <- mean((prediction_lasso$prob - actual_outcomes_lasso)^2)
brier_score_lasso

# lrElastic model 
model_elastic <- lrElastic$train(task_cadaf, split$train)
prediction_elastic = lrElastic$predict_newdata(final_cohort_test[split$test, ])
autoplot(prediction_elastic, type="roc")
autoplot(prediction_elastic, type="prc")
prediction_elastic$score(msr("classif.auc"))
prediction_elastic$score(msr("classif.auc"))
actual_outcomes_elastic <- as.numeric(prediction_elastic$truth)
brier_score_elastic <- mean((prediction_elastic$prob - actual_outcomes_elastic)^2)
brier_score_elastic



# Create a reliability plot
library(ggplot2)

# Assuming 'prediction_LR' contains predicted probabilities and 'actual_outcomes' is the actual binary outcomes
reliability_data <- data.frame(prob = prediction_LR$prob, actual = prediction_LR$truth)
reliability_data
brierScorePlot <-
  ggplot(data = reliability_data, aes(x = prob.1, y = actual)) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  xlab("Mean Predicted Probability") +
  ylab("Proportion of Positives") +
  ggtitle("Reliability Plot") +
  theme_minimal()





#####--------------------------Parallelization
# Load the future package
library(future)
library(doParallel)

# Set up a parallel backend with doParallel
cl <- makeCluster(detectCores(6))
registerDoParallel(cl)

# Now, parallelize the benchmarking process
bmr <- future(benchmark(design, store_models = TRUE))

# Clean up parallel backend when done
stopCluster(cl)




## cross validation

#crossValidation = rsmp("cv", folds=3)

####### Benchmark task definition
#design = benchmark_grid(
#  tasks = task_cadaf,
#  learners = learns,
#  resamplings = crossValidation)

#bmr = benchmark(design, store_models = TRUE)
#bmr$aggregate(measure)
#autoplot(bmr, type = "roc")

#####--------------------------Parallelization END

#-----------------------------------------------------------------Pipeop END---
#LassoLogsitic without tuning
lasso_model = lasso$train(task_cadaf, split$train)
lasso$model
prediction_lasso = lasso_model$predict_newdata(final_cohort[split$test, ])
autoplot(prediction_lasso, type="roc")
autoplot(prediction_lasso, type="prc")
prediction_lasso$score(msr("classif.auc"))










#-----------------------------------------------------------------Cohort definition 
# Create a SQL query to find person_ids meeting the conditions
query <- "
  SELECT DISTINCT co.person_id AS person_id
  FROM synpuf_cdm.condition_occurrence co
  INNER JOIN synpuf_cdm.condition_occurrence co1 ON co.person_id = co1.person_id
  INNER JOIN synpuf_cdm.death d ON co.person_id = d.person_id
  WHERE co.condition_concept_id = '4185932' -- Condition 1
    AND co1.condition_concept_id = '313217' -- Condition 2
    AND d.death_type_concept_id IS NOT NULL -- Condition 3 (Death Type exists)
"

# Execute the query and store the result in a data frame
cohort <- dbGetQuery(omopDB, query) 
#print(cohort)
#typeof(cohort)

# Write the result into the cohort table in the database
#dbWriteTable(omopDB, schema = 'synpuf_cdm', name = 'cohort', value= df, append = TRUE)

#-----------------------------------------------------------------Cohort definition END



#sql_script <- readLines("cohort.sql", warn = FALSE)

# Execute the SQL script
#dbExecute(omopDB, paste(sql_script, collapse = " "))








#-----------------------------------------------------------------Cohort definition 

# Demegraphic data 
person_data <- dbGetQuery(omopDB, "
  SELECT person_id, gender_concept_id, year_of_birth
  FROM synpuf_cdm.person
  WHERE person_id IN (
    SELECT DISTINCT person_id
    FROM synpuf_cdm.condition_occurrence
    WHERE condition_concept_id IN ('4185932', '313217')
  )
")

# retrieve the medical conditions for the person_ids included in the cohort. 
condition_occurrence_data <- dbGetQuery(omopDB, "
  SELECT person_id, condition_concept_id
  FROM synpuf_cdm.condition_occurrence
  WHERE condition_concept_id IN ('4185932', '313217', '201820', '312327', '315286', '316139', '317898', 
  '319844', '321052', '433753', '435243', '437312', '440417', '4024552', '4027663', '4218106')
")

# retrieve the observation for the person_ids included in the cohort. 
observation_data <- dbGetQuery(omopDB, "
  SELECT o.person_id, o.observation_concept_id, o.observation_date, o.observation_type_concept_id
  FROM synpuf_cdm.observation o
  INNER JOIN (
    SELECT DISTINCT person_id
    FROM synpuf_cdm.condition_occurrence
    WHERE condition_concept_id IN ('4185932', '313217')
  ) co
  ON o.person_id = co.person_id
  WHERE o.observation_concept_id IN ('443372', '43530634')
")

# retrieve the drug exposure data for the person_ids included in the cohort. 
drug_exposure_data <- dbGetQuery(omopDB, "
  SELECT d.person_id, d.drug_concept_id, d.drug_exposure_start_date
  FROM synpuf_cdm.drug_exposure d
  INNER JOIN (
    SELECT DISTINCT person_id
    FROM synpuf_cdm.condition_occurrence
    WHERE condition_concept_id IN ('4185932', '313217')
  ) co
  ON d.person_id = co.person_id
  WHERE d.drug_concept_id IN (‘1112807’, ‘1309944’, ‘1310149’, ‘1315865’, ‘1322184’, ‘1326303’, 
  ‘1353256’, ‘1361711’, ‘1367571’, ‘1383815’, ‘1383925’, ‘19017067’)
")

procedure_occurrence_data <- dbGetQuery(omopDB, "
  SELECT p.person_id, p.procedure_concept_id
  FROM synpuf_cdm.procedure_occurrence p
  INNER JOIN (
    SELECT DISTINCT person_id
    FROM synpuf_cdm.condition_occurrence
    WHERE condition_concept_id IN ('4185932', '313217')
  ) co
  ON p.person_id = co.person_id
  WHERE p.procedure_concept_id IN ('4353741')
")

death_data <- dbGetQuery(omopDB, "
  SELECT death.person_id, death.death_type_concept_id, death.death_date
  FROM synpuf_cdm.death death
  INNER JOIN (
    SELECT DISTINCT person_id
    FROM synpuf_cdm.condition_occurrence
    WHERE condition_concept_id IN ('4185932', '313217')
  ) co
  ON death.person_id = co.person_id
")

# Perform joins between the tables
merged_data <- condition_occurrence_data %>%
  left_join(observation_data, by = "person_id") %>%
  left_join(drug_exposure_data, by = "person_id") %>%
  left_join(person_data, by = "person_id") %>%
  #left_join(procedure_occurrence_data) ="person_id"%>%
  left_join(death_data, by = "person_id")

# cohort defination

final_cohort <- dbGetQuery(con, "
  SELECT p.person_id, 
         p.gender_concept_id, 
         p.year_of_birth,
         p.month_of_birth,
         p.race_concept_id,
         p.ethnicity_concept_id,
         o.observation_concept_id,
         o.observation_date,
         o.observation_type_concept_id,
         d.drug_concept_id,
         d.drug_exposure_start_date,
         d.drug_exposure_end_date
         
  FROM synpuf_cdm.person p
  INNER JOIN synpuf_cdm.observation o
  ON p.person_id = o.person_id
  LEFT JOIN synpuf_cdm.drug_exposure d
  ON p.person_id = d.person_id
  WHERE p.person_id IN (
    SELECT DISTINCT p.person_id
    FROM synpuf_cdm.person p
    WHERE EXISTS (
      SELECT 1
      FROM synpuf_cdm.condition_occurrence co1
      JOIN synpuf_cdm.condition_occurrence co2
      ON p.person_id = co1.person_id AND p.person_id = co2.person_id
      WHERE co1.condition_concept_id = '4185932'  -- Ischemic Heart disease
        AND co2.condition_concept_id = '313217'  -- Ischemic Heart disease
    )
    AND EXISTS (
      SELECT 1
      FROM synpuf_cdm.death death
      WHERE death.person_id = p.person_id
      AND death.death_type_concept_id = '38003565'
      AND death.death_date >= '2008-01-01'  -- Start date
      AND death.death_date <= '2010-12-31'  -- End date
    )
  )
")




final_cohort <- dbGetQuery(omopDB, "
  SELECT p.person_id, 
         p.gender_concept_id, 
         p.year_of_birth,
         
         o.observation_concept_id,
         o.observation_type_concept_id,
         
         d.drug_concept_id,
         d.drug_exposure_start_date,
         
         death.death_type_concept_id,
         death.death_date,
         
         co.condition_concept_id AS condition_concept_id
  FROM synpuf_cdm.person p
  INNER JOIN synpuf_cdm.observation o
  ON p.person_id = o.person_id
  LEFT JOIN synpuf_cdm.drug_exposure d
  ON p.person_id = d.person_id
  LEFT JOIN synpuf_cdm.death death
  ON p.person_id = death.person_id
  LEFT JOIN synpuf_cdm.condition_occurrence co
  ON p.person_id = co.person_id
  WHERE p.person_id IN (
    SELECT DISTINCT p.person_id
    FROM synpuf_cdm.person p
    WHERE EXISTS (
      SELECT 1
      FROM synpuf_cdm.condition_occurrence co1
      JOIN synpuf_cdm.condition_occurrence co2
      ON p.person_id = co1.person_id AND p.person_id = co2.person_id
      WHERE co1.condition_concept_id = '4185932'  -- Ischemic heart disease
        AND co2.condition_concept_id = '313217'  -- Atrial fibrillation
    )
    AND (
      co.condition_concept_id IN ('201820', '312327', '315286')
      OR o.observation_concept_id IN ('443372', '43530634')
      OR d.drug_concept_id IN ('1112807', '1309944', '1310149')
    )
  )
")



columns <-c("gender_concept_id","year_of_birth", "observation_concept_id", "observation_type_concept_id", 
            "drug_concept_id", "death_type_concept_id")

final_cohort <- merged_data[, columns]