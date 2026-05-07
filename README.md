# **Predicting Cognitive Performance and Environmental Influences**
CS 210: Data Management for Data Science\
Students: Hemadharshinii Sendhilvel (hs1244)

## **Project Overview**
This project investigates how cognitive performance is influenced by demographic, behavioral, health, and environmental factors.

It combines multiple large-scale public datasets to:

- Predict cognitive performance (DSST scores) using machine learning
- Analyze relationships between cognition and health behaviors
- Study environmental impacts (air quality) at population level
- Validate findings using an external dataset (BRFSS)

The project follows a multi-level analysis approach:

- Individual-level prediction (NHANES)
- Population-level trends (NHANES + EPA AQI)
- External validation (BRFSS + EPA AQI)

### **Datasets Used**

#### **NHANES 2013–2014**
- Cognitive performance (DSST score)
- Demographics: age, gender, education
- Health: BMI, smoking status, physical activity
#### **EPA Air Quality Data (2011–2014 + 2022)**
- Daily AQI measurements
- Aggregated to yearly and state-level averages
#### **BRFSS 2022**
- Self-reported cognitive impairment
- Health and demographic indicators
- State-level aggregation

## **How to Run the Project**
This project was developed and tested in Codebench using Jupyter notebooks.

Run the notebooks sequentially in this order: 

1. **NHANES_Individual_Analysis.ipynb**
- Data cleaning (NHANES datasets)
- Feature engineering
- SQL-based merging
- EDA + visualization
- Machine learning models (regression)
2. **Population_Air_Quality_Analysis.ipynb**
- NHANES multi-cycle aggregation
- EPA AQI processing (2011–2014)
- Population-level comparison plots
3. **BRFSS_External_Validation.ipynb**
- BRFSS preprocessing
- AQI merging at state level
- Classification models
- Feature importance analysis

**Note:** Each notebook is independent but builds on the same thematic pipeline
