# Benchmarking-Analysis-of-PLP-vs-MLR3 
This repo contains the model script and the cohort definitions used for a comparison of PLP and mlr3 R-based package using the SynPUF 5% dataset.

# Cohort definition
Cohort definition was performed using the [ATLAS](https://atlas-demo.ohdsi.org/#/home) platform. The data was imported into a locally running ATLAS and the cohort definition option was used. The "cohort_target.sql" file include the target population definition used in the study. In addition "cohort_outcome_death.sql" file describes our outcome cohort within the target population.

# prerequisites
We loaded the target and outcome cohorts data in a postgresql database. But any database format that OMOP CDM support can be used to store the SynPUF 5% dataset.

# Configuration

"PLPvsMLR3.Rproj" (and its dependent files, NAMESPACE, and DESCRIPTION) include the required R/R-studio settings for the project and can be directly imported into the R-studio. 

## R and RStudio
RStudio version 2022.07.2 Build 576 with R version 4.3.1 was used. 

## ml3 package
MLR3 version 0.16.1

# PatientLevelPrediction package 
PatientLevelPrediction version 6.3.5

