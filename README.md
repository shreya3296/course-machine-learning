# CS 109A: Introduction to Data Science - Fall 2023 - Harvard University

This repository contains course materials, problem sets, exams, and solutions for the **Machine Learning** course taught by Prof. Pavlos Protopapas at Harvard Universityy. This course focuses on machine learning methods using Python.

## Course Overview

**Course Title**: Introduction to Data Science / Machine Learning
**Instructor**: Prof. Pavlos Protopapas (SEAS) & Kevin Rader (Statistics)
**Institution**: Harvard University

CS 109A introduces key concepts and techniques used in data science, such as regression, classification, regularization, decision trees, random forests, and boosting. The course emphasizes hands-on learning, with Python being the primary programming language and tools like pandas, scikit-learn, and matplotlib being used extensively.

## Assignments

### Assignment 1: Web Scraping, Data Parsing, and Exploratory Data Analysis

- **Problem Statement:**
The goal of this assignment is to practice web scraping and parsing HTML data pulled from the IMDb Starmeter page. Students are tasked with extracting information about each person listed on a historical snapshot of IMDb's Starmeter, performing Exploratory Data Analysis (EDA), and answering various questions about the dataset, such as trends in acting credits, career start times, and actor demographics.

- **Dataset:**
The dataset is scraped directly from the Internet Archive's Wayback Machine, which provides a snapshot of the IMDb Starmeter page from September 13, 2023. The data contains information about:

- Name: The actor's name.
- Roles: Roles the actor is most known for (e.g., actor, producer, director).
- Star URL: A link to the actor’s IMDb page.
- Credits URL: A link to the actor’s IMDb credits page.
- Gender, Year of Birth, First Credit, Year of First Credit, and Number of Credits.

-**Approach:**
The assignment is divided into several steps:

- Web Scraping: Using the requests library to scrape data from IMDb's Starmeter snapshot, and BeautifulSoup for HTML parsing.
- Data Parsing: Extracting relevant information, including actor names, roles, and URLs, from the scraped HTML.
- Data Cleaning: Processing the extracted data to handle missing values and parse fields correctly.
- EDA (Exploratory Data Analysis): Analyzing trends in the dataset, such as relationships between career start times, the number of acting credits, and gender-based trends in acting careers.

-**Techniques Used:**

- Web Scraping (using requests and BeautifulSoup)
- Data Cleaning and Parsing
- Exploratory Data Analysis (EDA) using Python’s Pandas, Matplotlib, and Seaborn libraries
- Data Visualization (scatter plots, histograms, box plots)


### Assignment 2: k-Nearest Neighbors (kNN) and Linear Regression (cs109a_hw2.ipynb)

- **Problem Statement:**  
  This assignment is divided into two parts. The first part involves predicting the selling price of used cars on CarDekho.com using k-Nearest Neighbors (kNN) and Linear Regression. The second part focuses on analyzing simulated data from the Annual Social and Economic (ASEC) Supplement, with a focus on investigating income trends and calculating the Gini coefficient.

- **Dataset:**  
  The assignment uses two datasets:
  1. **CarDekho Dataset:** Contains 601 used cars with the following columns:
     - `Year`: The year the car was manufactured.
     - `Current_Selling_Price`: The current selling price of the car (in lakhs of Indian Rupees).
     - `MRP`: Maximum retail price of the car when it was new (in lakhs of Indian Rupees).
     - `Kms_Driven`: The number of kilometers the car has been driven.

  2. **Simulated ASEC Dataset:** Contains information from the 2021 US Annual Social and Economic (ASEC) Supplement. Key features include:
     - `age`, `hourly_pay`, `hours_per_week`, `weeks_worked`, `sex`, `education`, `marital_status`, `military_service`, `race`, `income`: Various socioeconomic and demographic factors.
     - The dataset simulates continuous income values based on income brackets originally found in the ASEC survey.

- **Approach:**  
  The assignment is divided into several steps:

  **Part 1: CarDekho Data**
  - **Exploratory Data Analysis (EDA):** Initial analysis was conducted to identify relationships between features and the target variable (`Current_Selling_Price`). Scatter plots and correlation matrices indicated that `MRP` was the strongest predictor.
  - **k-Nearest Neighbors (kNN):** kNN models were trained with different values of k (1, 2, 3, 5, 7, 10, 50, 100). Each model was evaluated using Mean Squared Error (MSE), and the model with k=7 had the best performance with the lowest test MSE and highest R-squared score.
  - **Linear Regression:** A linear regression model was fitted using `MRP` as the predictor. The slope and intercept were calculated, and the model was evaluated on both training and testing sets. The kNN model with k=7 outperformed the linear regression model.

  **Part 2: ASEC Data**
  - **Investigating Trends:** The dataset was used to investigate trends in income across various demographic factors such as gender, education, and age. For example, a strong correlation between income and education level was observed, with income increasing as education level increased.
  - **Gini Coefficient Calculation:** The Gini coefficient was calculated to quantify income inequality in the simulated dataset. The resulting coefficient was 0.473, indicating moderate income inequality, similar to the actual Gini coefficient of the United States.
  - **Critique of Simulated Data:** The assignment also required critiquing the simulated data, focusing on the implications of the simplifications and preprocessing steps used.
![download](https://github.com/user-attachments/assets/54819a8f-61e1-4043-ad20-0ee72f6f2e85)

- **Techniques Used:**
  - Exploratory Data Analysis (EDA) using `Pandas`, `Seaborn`, and `Matplotlib`
  - k-Nearest Neighbors (kNN) regression using `Scikit-learn`
  - Linear Regression using `Scikit-learn`
  - Model evaluation using Mean Squared Error (MSE) and R-squared
  - Gini Coefficient Calculation for income inequality
  - Data visualization to analyze income trends based on gender, education, and other demographic factors


### Assignment 3: Polynomial and LASSO Regression (cs109a_hw3.ipynb)

- **Problem Statement:**  
  The goal of this assignment is to model the percentage change in a bacteria population after 4 hours as a function of the bacteria's spreading factor using polynomial regression. The task includes fitting polynomial models, using cross-validation to find the optimal degree, and applying LASSO regularization to improve model consistency.

- **Dataset:**  
  The dataset has already been split into training and test sets and contains the following columns:
  - `Spreading_factor`: The predictor variable that quantifies how bacteria spread.
  - `Perc_population`: The response variable representing the percentage change in the bacteria population after 4 hours.

- **Approach:**  
  The assignment is divided into several steps:
  - **Exploratory Data Analysis (EDA):** Initial visualizations were generated to assess the relationship between the spreading factor and the percentage change in bacteria population. The scatter plot indicated a non-linear relationship.
  - **Polynomial Regression:** Polynomial regression models of varying degrees were fitted to the data. Models were evaluated using Mean Squared Error (MSE), and cross-validation was employed to find the best-fitting degree. Both a single validation set and k-fold cross-validation were used to evaluate model performance.
  - **LASSO Regularization:** LASSO (L1 regularization) was applied to a polynomial model of degree 30 to shrink coefficients of insignificant features to zero. This helped identify the most important polynomial terms while avoiding overfitting. The best regularization parameter (alpha) was found using LassoCV, and bootstrapping was used to evaluate the consistency of significant coefficients.
  - **Model Comparison:** Various polynomial regression models were compared based on their training and test MSEs. The best model was selected based on its performance on the test set, balancing complexity and generalization.

![hw3](https://github.com/user-attachments/assets/5fa258c0-2fde-4646-b6db-198ec5afb85a)

- **Techniques Used:**
  - Polynomial Regression using `Scikit-learn`
  - Exploratory Data Analysis (EDA) using `Pandas`, `Seaborn`, and `Matplotlib`
  - Cross-Validation (single validation set and k-fold)
  - LASSO Regularization (LassoCV) for feature selection and regularization
  - Bootstrapping for model consistency analysis

### Assignment 4: Missing Data & Principal Component Analysis (cs109a_hw4.ipynb)

- **Problem Statement:**  
  This assignment is divided into two parts. The first part focuses on using Principal Component Analysis (PCA) to reduce the dimensionality of a dataset and analyze patterns. The second part involves predicting the selling price of used cars by handling missing data using different imputation methods.

- **Dataset:**  
  1. **Communities and Crime Dataset (Part 1):**  
     This dataset contains 122 predictor variables and a response variable, `ViolentCrimesPerPop`, representing the number of violent crimes per 100K population. Predictors include features like household size, median income, police officer availability, and various demographic factors.  
     - **Predictors:** Variables related to population, income, housing, and crime rates.
     - **Response:** `ViolentCrimesPerPop` — total violent crimes per 100K population.
  
  2. **Vehicle Dataset (Part 2):**  
     This dataset contains information about used cars and includes missing data. Features include:
     - `year`: Year the car was bought.
     - `mileage`: Car mileage.
     - `max_power`: Maximum engine power (in bhps).
     - `selling_price`: The car's selling price (in lakh rupees).

- **Approach:**  
  **Part 1: Principal Component Analysis (PCA)**
  - **Exploratory Data Analysis (EDA):** The correlation matrix was computed to identify highly correlated variables.
  - **Dimensionality Reduction with PCA:** PCA was applied to reduce the dimensionality of the dataset. Several models were trained using different numbers of principal components, and cross-validation was performed to find the best model.
  - **Model Evaluation:** Linear regression models were fitted using both the original predictors and the PCA-transformed data. The R-squared values were compared to assess model performance and understand the impact of dimensionality reduction.

  **Part 2: Handling Missing Data and Prediction**
  - **Exploratory Data Analysis:** Missing data patterns were explored using visualizations.
  - **Imputation Methods:** Three imputation methods were applied:
    1. **Mean Imputation**: Replaced missing values with the mean of observed values.
    2. **k-NN Imputation**: Imputed missing values using k-nearest neighbors.
    3. **Indicator Method**: Added a binary indicator variable to flag rows with missing data before performing mean imputation.
  - **Modeling:** Linear regression and k-nearest neighbors regression models were trained for each imputation method. The models were evaluated using R-squared values on the test set to assess the impact of different imputation strategies on prediction performance.

- **Techniques Used:**
  - Principal Component Analysis (PCA) using `Scikit-learn`
  - Exploratory Data Analysis (EDA) using `Pandas`, `Seaborn`, and `Matplotlib`
  - Cross-Validation using `Scikit-learn`
  - Missing Data Handling using Mean Imputation, k-NN Imputation, and Indicator Method
  - Regression Modeling using Linear Regression and k-Nearest Neighbors (kNN)
  - Model evaluation using R-squared scores


### Assignment 5: Predicting College Admissions (cs109a_hw5.ipynb)

- **Problem Statement:**  
  The goal of this assignment is to model the chances of high school students being admitted into elite undergraduate colleges based on various predictors. We explore logistic regression models, k-Nearest Neighbors (kNN), and LASSO regression to predict college admissions.

- **Dataset:**  
  The dataset contains information on applicants and their admissions status, with features such as:
  - `test_score`: Standardized test scores (ACT/SAT).
  - `ap`: Number of Advanced Placement (AP) tests taken.
  - `avg_ap`: Average score on AP tests taken.
  - `sat_subjects`: Number of SAT subject tests taken.
  - `gpa`: Grade Point Average.
  - `female`: Binary variable for gender (1 = female, 0 = male).
  - `minority`: Binary variable indicating under-represented minority status.
  - `international`: Binary variable indicating international status.
  - `sports`: Binary indicator for high school all-American athlete.
  - `school`: Binary variable for school applied to (Harvard = 1, Yale = 0).
  - `early_app`: Binary indicator for early action applications.
  - `alumni`: Binary indicator for parents' alumni status.

- **Approach:**  
  The assignment is divided into several steps:
  - **Exploratory Data Analysis (EDA):** Visualizations were created to explore the relationship between the predictors and the admissions outcome.
![hw5](https://github.com/user-attachments/assets/e425fbd4-05f5-48d5-8324-2d582fd4fa6c)

  - **Logistic Regression Models:** Several logistic regression models were built to predict the chances of admission based on different combinations of predictors. The models included interaction terms to account for relationships between the variables, such as the interaction between applying to Harvard and test scores.
  - **k-Nearest Neighbors (kNN):** A kNN classifier was trained to predict admission, and cross-validation was used to identify the best number of neighbors (k) for the model.
  - **LASSO Regression:** LASSO (L1 regularization) was applied to reduce the complexity of the logistic regression model by shrinking unimportant coefficients to zero, thus selecting only the most relevant features.

- **Techniques Used:**
  - Logistic Regression using `Scikit-learn`
  - Exploratory Data Analysis (EDA) using `Pandas`, `Seaborn`, and `Matplotlib`
  - Cross-Validation for kNN and logistic regression models
  - LASSO Regularization using `LogisticRegressionCV`
  - Model evaluation using accuracy, ROC curves, and AUC scores

### Assignment 6: Trees, Bagging, Random Forests, and Boosting (cs109a_hw6.ipynb)

- **Problem Statement:**  
  The goal of this assignment is to explore various ensemble techniques such as bagging, random forests, and boosting, and understand their impact on model performance and generalization. The task involves using decision trees as the base learners and comparing how bagging and boosting differ in terms of bias-variance tradeoff.

- **Dataset:**  
  The dataset contains particle collision data from Monte-Carlo simulations used to discover the Higgs boson. It consists of 28 features representing the kinematic properties of particles in a collider, with a class label indicating whether the collision produced a Higgs boson (1) or background noise (0).

  - **Features:** 28 kinematic and derived properties from particle collisions.
  - **Target:** Binary label indicating whether the collision produced a Higgs boson (1) or not (0).

- **Approach:**  
  The assignment is divided into several steps:
  - **Decision Trees:** A range of tree depths (from 1 to 20) was explored, using cross-validation to find the optimal depth and avoid overfitting. A single decision tree was fitted using the optimal depth, and its performance was compared to deeper trees that tend to overfit.
  - **Bagging:** Bagging was applied by generating multiple bootstrapped samples and training decision trees on each sample. The results were aggregated to make predictions, and the impact of the number of trees on performance was analyzed.
  - **Random Forests:** A random forest model was trained, which added randomness by selecting random subsets of features for splitting at each node. The performance of the random forest was compared to the bagging model.
  - **Boosting:** A simplified boosting algorithm was implemented using two decision trees, followed by using AdaBoost to fit a more sophisticated ensemble model. The performance of AdaBoost was analyzed by tuning the number of iterations and the base learner depth.

- **Techniques Used:**
  - Decision Trees using `Scikit-learn`
  - Bagging using bootstrapped trees
  - Random Forests using `RandomForestClassifier`
  - Boosting using both a simplified custom implementation and `AdaBoostClassifier`
  - Model evaluation using cross-validation, accuracy scores, and bias-variance tradeoff analysis
