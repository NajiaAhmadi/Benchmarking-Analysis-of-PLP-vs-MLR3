library("FeatureExtraction")
library("DatabaseConnector")
library("PatientLevelPrediction")

reticulate::install_miniconda()
configurePython(envname = 'r-reticulate', envtype = 'conda')
#reticulate::conda_list()
usethis::edit_r_profile()
#/Users/ahmadinai/Library/r-miniconda/envs/r-reticulate/bin/python

#usethis::edit_r_profile()
#Sys.setenv(PATH = paste("/Users/ahmadinai/opt/anaconda3/bin/python", Sys.getenv("PATH"), sep = ";"))

# database setup
Sys.setenv("DATABASECONNECTOR_JAR_FOLDER" = "R/")

source("R/config.R")
db_info <- paste("user=", db_user, " password=", db_password,
                 " dbname=", db, " host=", host_db, " port=", db_port, sep = ",")

dbms <- 'postgresql'
user <- "ohdsi_admin_user"
password <- "iOF10AcQC5W+ga+kgMC0oFEOScBTvw"
server <- "sv-diz-omop-demo.med.tu-dresden.de/ohdsi"  
port <- '5432'

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                server = server,
                                                                user = user,
                                                                password = password,
                                                                port = port)


## download the .jar file for postgresql 
#downloadJdbcDrivers(
#  dbms <- 'postgresql',
#  pathToDriver = Sys.getenv("DATABASECONNECTOR_JAR_FOLDER"),
#  method = "auto",
#)


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
restrictPlpDataSettings <- createRestrictPlpDataSettings() #sampleSize = 1000


# retrieves the cohorts from the database
plpData <- getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings,
  restrictPlpDataSettings = restrictPlpDataSettings
)


savePlpData(plpData, "/Users/ahmadinai/Documents/GitHub/Benchmarking-Analysis-of-PLP-vs-MLR3/savePlpData")
plpData <- loadPlpData("/Users/ahmadinai/Documents/GitHub/Benchmarking-Analysis-of-PLP-vs-MLR3/savePlpData")

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

# model definition (random forest, gradient boosting machines and logistic regression) and the used hyperparameters
lrForest <- setRandomForest(ntrees = list(500, 750, 1000, 1250, 1500, 2000), 
                            maxDepth = list(17), minSamplesSplit = list(5), minSamplesLeaf = list(10), 
                            mtries = list("sqrt"), #maxSamples = list(NULL), 
                            classWeight = list("balanced_subsample"), 
                            seed = sample(12345))
lrGradient <- setGradientBoostingMachine(scalePosWeight=40, ntrees = 1000, learnRate = c(0.005, 0.01, 0.1, 0.05, 0.001))
lrLogiReg <- setLassoLogisticRegression(seed=1234)

modellist <- list(lrForest,lrLogiReg, lrGradient)

# sampling techniques application
sampleSettings <- createSampleSettings() #(type = "underSample", numberOutcomestoNonOutcomes = 1/20, sampleSeed=1234)

featureEngineeringSettings <- createFeatureEngineeringSettings()

path = file.path("/Users/ahmadinai/Documents/GitHub/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults", "plpMlResults")
# running configuration

lrResults <- runPlp(
  plpData = plpData,
  outcomeId = 4,
  analysisId = 'LassoUnderSampling',
  analysisName = 'Demonstration of runPlp for training single PLP models',
  populationSettings = populationSettings,
  splitSettings = splitSettings,
  sampleSettings = sampleSettings,
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

# model outcome 
plpResult <- loadPlpResult(file.path("/Users/ahmadinai/Documents/GitHub/Benchmarking-Analysis-of-PLP-vs-MLR3/plpResults/plpMlResults/LassoUnderSampling/plpResult/"))
# visualization of the outcome
viewPlp(plpResult)

