# Predicting Points Scored in NBA Games
This was a collaborative final project done for the Intro to Machine Learning class at UT Austin.

## Project Overview

This project focuses on developing a machine learning model to predict the points scored by NBA players based on their past game statistics. Utilizing datasets from the 2022-23 and 2023-24 NBA seasons, the goal is to assist coaches and managers in optimizing team rosters by providing data-driven insights into player performance.

## Contributions

- Built k-Nearest Neighbors and Linear Regression models using R packages such as `caret` and `glm` to predict points scored based on past game statistics.

## Important Findings

1. **Key Performance Indicators Identified**: Features such as minutes played, field goals per game, and field goal attempts were highly correlated with points scored, making them essential predictors in our models.

2. **Model Performance Comparison**: Among the models tested, Random Forest performed best in terms of accuracy and reliability, while k-Nearest Neighbors (k=5) showed moderate predictive capability but higher RMSE, indicating limited effectiveness in capturing complex relationships.

3. **Overfitting in Linear Models**: The high adjusted R-squared values in the Linear Regression model suggested potential overfitting, as the model’s performance decreased with cross-validation, indicating it might not generalize well to new data.

4. **Boosting for Enhanced Predictive Power**: Boosting methods demonstrated improved accuracy, identifying field goals per game and attempts as the most impactful features. However, the model required careful tuning to avoid overfitting due to the wide range of input variables.

5. **Significance of Player Position**: Positions like Center and Power Forward emerged as influential in scoring predictions, suggesting that physical and strategic roles impact scoring performance, especially in certain models like Random Forest.

6. **Practical Implications for NBA Management**: This model could support NBA trading, sports betting, and coaching decisions by providing data-backed performance forecasts, aiding managers in strategic team planning and resource allocation.

7. **Future Model Improvement Potential**: Further improvements could involve training on a more extensive dataset (e.g., a decade’s worth of stats) and integrating team-based statistics to capture inter-player dynamics, potentially enhancing model robustness and predictive power.

## Technologies Used

- **Programming Languages**: R
- **Packages**: `caret`, `glm`, `randomForest`, `gbm`
- **Data Sources**: NBA player statistics from the 2022-23 and 2023-24 seasons

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/nba-player-points-prediction.git
   ```
2. **Install Dependencies**:
```r
install.packages(c("caret", "glm", "randomForest", "gbm"))
```
3. **Load the Data**:
Ensure the NBA player statistics datasets for the 2022-23 and 2023-24 seasons are available in the working directory.

4. **Run the Models**:
Execute the R scripts to train and evaluate the k-Nearest Neighbors, Linear Regression, Random Forest, and Boosting models.

5. **Analyze Results**:
Review the output metrics to assess model performance and accuracy.

6. **Future Applications**:
* Extended Data Integration: Incorporate additional seasons and player metrics to improve model accuracy and generalizability.
* Real-Time Predictions: Develop a real-time prediction tool for live game analysis and decision-making.
* Advanced Feature Engineering: Explore advanced statistical techniques to identify new predictive features and enhance model performance.
  
## Contact
For questions or further information, please contact:
* Email: gayathreegopi@utexas.edu
* LinkedIn: linkedin.com/in/gayathreegopi
