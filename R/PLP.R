library("FeatureExtraction")
library("DatabaseConnector")
library("PatientLevelPrediction")

Sys.setenv("DATABASECONNECTOR_JAR_FOLDER" = "C:/postgresdriver")
dbms <- "postgresql"
user <- 'postgres'
pw <- '1234'
server <- 'localhost/omop'
port <- '5432'

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                server = server,
                                                                user = user,
                                                                password = pw,
                                                                port = port)

cdmDatabaseSchema <- 'cmd'
cdmDatabaseName <- 'Ischemic Heartdisease with A Fib'
cohortDatabaseSchema <- 'results'
oracleTempSchema <- NULL
cohortTable <- 'cohort'


selectedCovs = c(201820,312327,315286,316139,317898,319844,321052,433753,435243,
                 437312,440417,443372,1112807,1309944,1310149,1315865,1322184,
                 1326303,1353256,1361711,1367571,1383815,1383925,4024552,4027663,
                 4185932,4218106,4353741,19017067,43530634,313217)

covariateSettings <- createCovariateSettings( useDemographicsGender = TRUE,
                                              useConditionEraAnyTimePrior = TRUE,
                                              useObservationAnyTimePrior = TRUE,
                                              useDrugEraAnyTimePrior = TRUE)
                                             
databaseDetails <- createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cdmDatabaseName = cdmDatabaseName,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = cohortTable,
  targetId = 2,
  outcomeDatabaseSchema = cohortDatabaseSchema,
  outcomeTable = cohortTable,
  outcomeIds = 3,
  cdmVersion = 5
)

restrictPlpDataSettings <- createRestrictPlpDataSettings()#sampleSize = 1000

plpData <- getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings,
  restrictPlpDataSettings = restrictPlpDataSettings
)

savePlpData(plpData, "C:/Users/win10/Documents/ResultsPLP/ischemic_in_af_data")
plpData <- loadPlpData("C:/Users/win10/Documents/ResultsPLP/ischemic_in_af_data")

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

splitSettings <- createDefaultSplitSetting(
  trainFraction = 0.75,
  testFraction = 0.25,
  type = 'stratified',
  nfold = 3,
  splitSeed = 1234
)


featureEngineeringSettings <- createFeatureEngineeringSettings()

preprocessSettings<- createPreprocessSettings(
  minFraction = 0,
  normalize = T,
  removeRedundancy = T
)
setGradient
lrForest <- setRandomForest(ntrees = list(500, 750, 1000, 1250, 1500, 2000), 
                            maxDepth = list(17), minSamplesSplit = list(5), minSamplesLeaf = list(10), 
                            mtries = list("sqrt"), maxSamples = list(NULL), classWeight = list("balanced_subsample"), seed = sample(12345))
lrGradient <- setGradientBoostingMachine(scalePosWeight=40, ntrees = 1000, learnRate = c(0.005, 0.01, 0.1, 0.05, 0.001))
lrLogiReg <- setLassoLogisticRegression(seed=1234)

sampleSettings <- createSampleSettings(type = "underSample", numberOutcomestoNonOutcomes = 1/20, sampleSeed=1234)
#sampleSettings <- createSampleSettings()

lrResults <- runPlp(
  plpData = plpData,
  outcomeId = 3,
  analysisId = 'LassoUnderSampling',
  analysisName = 'Demonstration of runPlp for training single PLP models',
  populationSettings = populationSettings,
  splitSettings = splitSettings,
  sampleSettings = sampleSettings,
  featureEngineeringSettings = featureEngineeringSettings,
  preprocessSettings = preprocessSettings,
  modelSettings = lrLogiReg,
  logSettings = createLogSettings(),
  executeSettings = createExecuteSettings(
    runSplitData = T,
    runSampleData = T,
    runfeatureEngineering = T,
    runPreprocessData = T,
    runModelDevelopment = T,
    runCovariateSummary = T
  ),
  saveDirectory = file.path("C:/Users/win10/Documents/ResultsPLP", "plpResults")
)

plpResult <- loadPlpResult(file.path("C:/Users/win10/Documents/ResultsPLP","plpResults", "LassoUnderSampling", "plpResult"))
viewPlp(plpResult)

