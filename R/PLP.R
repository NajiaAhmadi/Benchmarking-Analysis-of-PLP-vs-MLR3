library(devtools)
library("FeatureExtraction")
library("DatabaseConnector")
library("PatientLevelPrediction")


## download the .jar file for postgresql if necessary
#downloadJdbcDrivers(
#  dbms <- 'postgresql',
#  pathToDriver = Sys.getenv("DATABASECONNECTOR_JAR_FOLDER"),
#  method = "auto",
#)

# database setup
Sys.setenv("DATABASECONNECTOR_JAR_FOLDER" = "~/R/")

# Check and set the working directory
if (!file.exists("config.R")) {
  stop("Error: 'config.R' not found in the current directory.")
}
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("config.R")

db_info <- paste("user=", db_user, 
                 " password=", db_password,
                 " dbname=", db, 
                 " host=", host_db, 
                 " port=", db_port, 
                 "dbms =", dbms, 
                  sep = ",")

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                server = paste(host_db, "/", db, sep = ""),
                                                                user = db_user,
                                                                password = db_password,
                                                                port = db_port )
                                                             

cdmDatabaseName <- 'ohdsi'
cdmDatabaseSchema <- 'synpuf_cdm'
cohortDatabaseSchema <- 'synpuf_cdm'
tempEmulationSchema <- NULL
cohortTable <- 'target_cohort' 

# general patient feature setting: Gender, condition from condition era table, observation from observation and 
# drug exposure from drug era table prior to being diagnosed with ischemic heart disease.
covariateSettings <- FeatureExtraction::createCovariateSettings( useDemographicsGender = TRUE,
                                              useConditionEraAnyTimePrior = TRUE,
                                              useObservationAnyTimePrior = TRUE,
                                              useDrugEraAnyTimePrior = TRUE)

                                             
databaseDetails <- createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cdmDatabaseName = cdmDatabaseName,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = cohortTable,
  targetId = 3,
  outcomeDatabaseSchema = cohortDatabaseSchema,
  outcomeTable = cohortTable,
  outcomeIds = 4, # the id is coming from the outcome cohort patients that has a condition death. 
  cdmVersion = 5
)

# sample size restriction, if needed. not used in this analysis 
restrictPlpDataSettings <- createRestrictPlpDataSettings(sampleSize = 1000) #sampleSize = 1000


# retrieves the cohorts from the database
plpData <- getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings,
  restrictPlpDataSettings = restrictPlpDataSettings
)

savePlpData(plpData, "~/Benchmarking-Analysis-of-PLP-vs-MLR3/savePlpData")
plpData <- loadPlpData("~/Benchmarking-Analysis-of-PLP-vs-MLR3/savePlpData")

# time at risk and observation window definition
populationSettings <- createStudyPopulationSettings(
  washoutPeriod = 0, 
  firstExposureOnly = TRUE,
  removeSubjectsWithPriorOutcome = FALSE,
  priorOutcomeLookback = 999999,
  riskWindowStart = 1,
  riskWindowEnd = 1825,
  minTimeAtRisk = 365,
  requireTimeAtRisk = FALSE,
  includeAllOutcomes = TRUE
)

# training and test set splitting for model training. 
splitSettings <- createDefaultSplitSetting(
  trainFraction = 0.75,
  testFraction = 0.25,
  type = 'stratified',
  nfold = 3,
  splitSeed = 1234
)

# feature normalization 
preprocessSettings<- createPreprocessSettings(
  minFraction = 0, # sort out the features if the features has a lower incidence as the value that is provided (0) that
  normalize = T,
  removeRedundancy = T
)

# sampling techniques application
sampleSettings <- createSampleSettings(type = "none") #(type = "underSample", numberOutcomestoNonOutcomes = 1/20, sampleSeed=1234)


featureEngineeringSettings <- createFeatureEngineeringSettings()

path = file.path("~/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults", "plpMlResults")
# running configuration

runPlpModel <- function(model, analysisId, logFilePath) {
  # running configuration with logSettings
  logSettings <- createLogSettings()
  
  # Call runPlp for the given model and analysisId
  lrResults <- runPlp(
    plpData = plpData,
    outcomeId = 4,
    analysisId = analysisId,
    analysisName = paste("Run PLP for", analysisId, "with", model),
    populationSettings = populationSettings,
    splitSettings = splitSettings,
    sampleSettings = sampleSettings,
    featureEngineeringSettings = featureEngineeringSettings,
    preprocessSettings = preprocessSettings,
    modelSettings = model,
    logSettings = logSettings,
    executeSettings = createExecuteSettings(
      runSplitData = T,
      runSampleData = T,
      runfeatureEngineering = T,
      runPreprocessData = T,
      runModelDevelopment = T,
      runCovariateSummary = T
    ),
    saveDirectory = logFilePath
  )
  
  # Return the results if needed
  return(lrResults)
}

# model definition (random forest, gradient boosting machines and logistic regression) and the used hyperparameters
lrForest <- setRandomForest(ntrees = list(500, 750, 1000, 1250, 1500, 2000), 
                            maxDepth = list(17), 
                            minSamplesSplit = list(5), 
                            minSamplesLeaf = list(10), 
                            mtries = list("sqrt"), #maxSamples = list(NULL), 
                            classWeight = list("balanced_subsample"), 
                            seed = 132500)

lrGradient <- setGradientBoostingMachine(scalePosWeight=40, ntrees = 1000, 
                                         learnRate = c(0.005, 0.01, 0.1, 0.05, 0.001), 
                                         seed = 132500)

lrLogiReg <- setLassoLogisticRegression(seed=132500)

modellist <- list(lrForest, lrLogiReg, lrGradient)

lrResults_lrLogiReg <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg", logFilePath = path)
#call the outcomes
logi <- lrResults_lrLogiReg$performanceEvaluation$evaluationStatistics
logi_roc <- subset(logi, metric == "AUROC" & evaluation == "CV")
logi_roc_lower <- subset(logi, metric == "95% lower AUROC" & evaluation == "CV")
logi_roc_upper <- subset (logi, metric == "95% upper AUROC" & evaluation == "CV")

lrResults_lrForest <- runPlpModel(model = lrForest, analysisId = "lrForest", logFilePath = path)
#call the outcomes
rf <- lrResults_lrForest$performanceEvaluation$evaluationStatistics
rf_roc <- subset(rf, metric == "AUROC" & evaluation == "CV")
rf_roc_lower <- subset(rf, metric == "95% lower AUROC" & evaluation == "CV")
rf_roc_upper <- subset (rf, metric == "95% upper AUROC" & evaluation == "CV")

lrResults_lrGradient <- runPlpModel(model = lrGradient, analysisId = "lrGradient", logFilePath = path)
#call the outcomes
GBM <- lrResults_lrGradient$performanceEvaluation$evaluationStatistics
GBM_roc <- subset(GBM, metric == "AUROC" & evaluation == "CV")
GBM_roc_lower <- subset(GBM, metric == "95% lower AUROC" & evaluation == "CV")
GBM_roc_upper <- subset (GBM, metric == "95% upper AUROC" & evaluation == "CV")



# model outcome 
plpResult_lrLogiReg <- loadPlpResult(file.path("~/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults/plpMlResults/lrLogiReg/plpResult/"))
viewPlp(plpResult_lrLogiReg)

plpResult_lrForest <- loadPlpResult(file.path("~/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults/plpMlResults/lrForest/plpResult/"))
viewPlp(plpResult_lrForest)

plpResult_lrGradient <- loadPlpResult(file.path("~/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults/plpMlResults/lrGradient/plpResult/"))
viewPlp(plpResult_lrGradient)

# ----------------- oversample

sampleSettings_oversample <- createSampleSettings(type = "overSample")

lrResults_oversampling <- runPlp(
  plpData = plpData,
  outcomeId = 4,
  analysisId = 'lrForest_oversample',
  analysisName = 'Demonstration of runPlp for training single PLP models',
  populationSettings = populationSettings,
  splitSettings = splitSettings,
  sampleSettings = sampleSettings_oversample,
  featureEngineeringSettings = featureEngineeringSettings,
  preprocessSettings = preprocessSettings,
  modelSettings = lrForest,
  logSettings = createLogSettings(),
  executeSettings = createExecuteSettings(
    runSplitData = T,
    runSampleData = T,
    runfeatureEngineering = T,
    runPreprocessData = T,
    runModelDevelopment = T,
    runCovariateSummary = T
  ),
  saveDirectory = path
)

plpResult_lrForest <- loadPlpResult(file.path("~/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults/plpMlResults/lrForest_oversample/plpResult/"))
viewPlp(plpResult_lrForest)


# ---------------------------------------------- CI 

# RUN 1 of the functions with seed 123
lrResults_lrLogiReg_seed123 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg123", logFilePath = path)
lrResults_lrForest_seed123 <- runPlpModel(model = lrForest, analysisId = "lrForest123", logFilePath = path)
lrResults_lrGradient_seed123 <- runPlpModel(model = lrGradient, analysisId = "lrGradient123", logFilePath = path)

# RUN 2 of the functions with seed 124
lrResults_lrLogiReg_seed124 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg124", logFilePath = path)
lrResults_lrForest_seed124 <- runPlpModel(model = lrForest, analysisId = "lrForest124", logFilePath = path)
lrResults_lrGradient_seed124 <- runPlpModel(model = lrGradient, analysisId = "lrGradient124", logFilePath = path)

# RUN 3 of the functions with seed 125
lrResults_lrLogiReg_seed125 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg125", logFilePath = path)
lrResults_lrForest_seed125 <- runPlpModel(model = lrForest, analysisId = "lrForest125", logFilePath = path)
lrResults_lrGradient_seed125 <- runPlpModel(model = lrGradient, analysisId = "lrGradient125", logFilePath = path)

# RUN 4 of the functions with seed 126
lrResults_lrLogiReg_seed126 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg126", logFilePath = path)
lrResults_lrForest_seed126 <- runPlpModel(model = lrForest, analysisId = "lrForest126", logFilePath = path)
lrResults_lrGradient_seed126 <- runPlpModel(model = lrGradient, analysisId = "lrGradient126", logFilePath = path)

# RUN 5 of the functions with seed 127
lrResults_lrLogiReg_seed127 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg127", logFilePath = path)
lrResults_lrForest_seed127 <- runPlpModel(model = lrForest, analysisId = "lrForest127", logFilePath = path)
lrResults_lrGradient_seed127 <- runPlpModel(model = lrGradient, analysisId = "lrGradient127", logFilePath = path)

# RUN 6 of the functions with seed 128
lrResults_lrLogiReg_seed128 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg128", logFilePath = path)
lrResults_lrForest_seed128 <- runPlpModel(model = lrForest, analysisId = "lrForest128", logFilePath = path)
lrResults_lrGradient_seed128 <- runPlpModel(model = lrGradient, analysisId = "lrGradient128", logFilePath = path)

# RUN 7 of the functions with seed 129
lrResults_lrLogiReg_seed129 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg129", logFilePath = path)
lrResults_lrForest_seed129 <- runPlpModel(model = lrForest, analysisId = "lrForest129", logFilePath = path)
lrResults_lrGradient_seed129 <- runPlpModel(model = lrGradient, analysisId = "lrGradient129", logFilePath = path)

# RUN 8 of the functions with seed 130
lrResults_lrLogiReg_seed130 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg130", logFilePath = path)
lrResults_lrForest_seed130 <- runPlpModel(model = lrForest, analysisId = "lrForest130", logFilePath = path)
lrResults_lrGradient_seed130 <- runPlpModel(model = lrGradient, analysisId = "lrGradient130", logFilePath = path)

# RUN 9 of the functions with seed 131
lrResults_lrLogiReg_seed131 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg131", logFilePath = path)
lrResults_lrForest_seed131 <- runPlpModel(model = lrForest, analysisId = "lrForest131", logFilePath = path)
lrResults_lrGradient_seed131 <- runPlpModel(model = lrGradient, analysisId = "lrGradient131", logFilePath = path)

# RUN 10 of the functions with seed 132
lrResults_lrLogiReg_seed132 <- runPlpModel(model = lrLogiReg, analysisId = "lrLogiReg132", logFilePath = path)
lrResults_lrForest_seed132 <- runPlpModel(model = lrForest, analysisId = "lrForest132", logFilePath = path)
lrResults_lrGradient_seed132 <- runPlpModel(model = lrGradient, analysisId = "lrGradient132", logFilePath = path)


lrResults_lrLogiReg_list = list (lrResults_lrLogiReg_seed123$performanceEvaluation$evaluationStatistics, 
                                 lrResults_lrLogiReg_seed124$performanceEvaluation$evaluationStatistics, 
                                 lrResults_lrLogiReg_seed125$performanceEvaluation$evaluationStatistics, 
                                 lrResults_lrLogiReg_seed126$performanceEvaluation$evaluationStatistics, 
                                 lrResults_lrLogiReg_seed127$performanceEvaluation$evaluationStatistics,
                                 lrResults_lrLogiReg_seed128$performanceEvaluation$evaluationStatistics,
                                 lrResults_lrLogiReg_seed129$performanceEvaluation$evaluationStatistics,
                                 lrResults_lrLogiReg_seed130$performanceEvaluation$evaluationStatistics,
                                 lrResults_lrLogiReg_seed131$performanceEvaluation$evaluationStatistics,
                                 lrResults_lrLogiReg_seed132$performanceEvaluation$evaluationStatistics)

lrResults_lrForest_list = list (lrResults_lrForest_seed123$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrForest_seed124$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrForest_seed125$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrForest_seed126$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrForest_seed127$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrForest_seed128$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrForest_seed129$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrForest_seed130$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrForest_seed131$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrForest_seed132$performanceEvaluation$evaluationStatistics)


lrResults_lrGradient_list = list (lrResults_lrGradient_seed123$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrGradient_seed124$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrGradient_seed125$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrGradient_seed126$performanceEvaluation$evaluationStatistics, 
                                  lrResults_lrGradient_seed127$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrGradient_seed128$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrGradient_seed129$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrGradient_seed130$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrGradient_seed131$performanceEvaluation$evaluationStatistics,
                                  lrResults_lrGradient_seed132$performanceEvaluation$evaluationStatistics)

# get the AUPRC and brier score from the list
combineEvaluationData <- function(evaluation_list) {
  result_list <- lapply(evaluation_list, function(evaluation_df) {
    cv_auprc <- subset(evaluation_df, metric == "AUPRC" & evaluation == "CV")
    cv_brier <- subset(evaluation_df, metric == "brier score" & evaluation == "CV")
    
    # Combine the information into a data frame
    result_df <- data.frame(
      AUPRC = as.numeric(cv_auprc$value),
      Brier = as.numeric(cv_brier$value)
      # Add more columns if needed
    )
    
    return(result_df)
  })
  
  # Combine the individual data frames into one list
  combined_results <- do.call(rbind, result_list)
  
  return(combined_results)
}

lrLogiReg_combined_results <- combineEvaluationData(lrResults_lrLogiReg_list)
lrForest_combined_results <- combineEvaluationData(lrResults_lrForest_list)
lrGradient_combined_results <- combineEvaluationData(lrResults_lrGradient_list)


# Confidence interval for AUPRC and brier score

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

cat("Model: Gradient Boosting Machine") 
lrLogiReg_prc_CI = calculate_summary_statistics(lrLogiReg_combined_results$AUPRC)
lrLogiReg_brier_CI = calculate_summary_statistics(lrLogiReg_combined_results$Brier)

cat("Model: Gradient Boosting Machine") 
lrForest_prc_CI = calculate_summary_statistics(lrForest_combined_results$AUPRC)
lrForest_brier_CI = calculate_summary_statistics(lrForest_combined_results$Brier)

cat("Model: Gradient Boosting Machine") 
lrGradient_prc_CI = calculate_summary_statistics(lrGradient_combined_results$AUPRC)
lrGradient_brier_CI = calculate_summary_statistics(lrGradient_combined_results$Brier)


# Function to print the list with definitions and name
printResults <- function(results_list, list_name) {
  cat("List Name:", list_name, "\n")
  cat("Mean (xbar):\n[1] ", results_list$xbar, "\n\n")
  cat("Standard Deviation:\n[1] ", results_list$standard_deviation, "\n\n")
  cat("Lower Interval:\n[1] ", results_list$lower_interval, "\n\n")
  cat("Upper Interval:\n[1] ", results_list$upper_interval, "\n")
}

# Print the given list with definitions and name
printResults(lrLogiReg_prc_CI, "lrLogiReg_prc_CI")
printResults(lrLogiReg_brier_CI, "lrLogiReg_brier_CI")
printResults(lrForest_prc_CI, "lrForest_prc_CI")
printResults(lrForest_brier_CI, "lrForest_brier_CI")
printResults(lrGradient_prc_CI, "lrGradient_prc_CI")
printResults(lrGradient_brier_CI, "lrGradient_brier_CI")















