# Benchmarking-Analysis-of-PLP-vs-MLR3 
This repo contains the model scripts and the cohort definitions used for a comparison of PLP and mlr3 R-based package using the SynPUF 5% dataset.

# Configuration

"PLPvsMLR3.Rproj" (and its dependent files, NAMESPACE, and DESCRIPTION) include the required R/R-studio settings for the project and can be directly imported into the R-studio. 

## R and RStudio
RStudio version 2022.07.2 Build 576 with R version 4.2.2 was used. 

## ml3 package
MLR3 version 0.14.1

# PLP package 
PLP version 6.0.8

# Package dependencies
"renv.lock" include the package dependencies that can be imported. 

# Cohort definition
Cohort definition was performed using the [ATLAS](https://atlas-demo.ohdsi.org/#/home) platform. The data was imported into a locally running ATLAS and the cohort definition option was used. The "Target Population (Ischemic heart disease and atrial fibrilition.json" file include the target population definition used in the study. In addition "Death.json" file describes our outcome cohort within the target population.