# Stock Price Prediction Pipeline: Leveraging Multiple Machine Learning Models for Accurate and Efficient Forecasting 📈📉

# Overview
The project involved building a machine learning pipeline to predict stock prices, utilizing a range of models, including Linear Regression, Random Forest, Decision Trees, SVR, XGBoost, KNN, Artificial Neural Network (ANN), Recurrent Neural Networks (RNN), and Temporal Fusion Transformers (TFT). Given the time constraints, especially when dealing with large datasets and the complexity of model selection, the approach was adjusted to balance accuracy and efficiency.

## Table of Contents
- [How to Run the Project](#How-to-Run)
- [Challenges and Approach](#Challenges-and-Approach)
- [Ethics and AI Use](#Ethics-and-AI-Use)
- [Current Progress](#Current-Progress)
- [Contributions](#Contributions)

## How to Run
- Install Required Libraries: Ensure that the necessary libraries are installed. The code will automatically install scikit-learn if missing.
- Prepare the Model Files: Place the following files in the same directory as the code:
- best_model.pkl (pre-trained model)
- label_encoder.pkl (label encoder for decoding predictions)
- Run the Code: Execute the script to launch the app.

GUI: https://huggingface.co/spaces/vjl004/StockMarketPrediction

## Challenges and Approach
- Time Constraints in Model Development: Developing and tuning machine learning models within limited time led to a focus on exploring a variety of algorithms while still ensuring proper preprocessing and validation. The initial struggle was managing time effectively for feature engineering, such as extracting date features and creating lag features to capture temporal dependencies. Lag features were essential for understanding how past stock prices and macroeconomic indicators influence future movements. However, managing these within a tight timeline required prioritizing models that could handle the data efficiently without excessive computation time.
- Preprocessing and Data Normalization: Given the nature of stock price data and macroeconomic indicators, a significant portion of the time was spent on cleaning and normalizing the data. Ensuring that features were correctly scaled to prevent issues like overfitting or improper learning in the models was crucial. The focus was on normalizing the data in a way that would allow all models, especially more complex ones like the ANN and XGBoost, to function optimally.
- Cross-validation and Hyperparameter Tuning: To prevent overfitting and ensure the models would generalize well to unseen data, the pipeline implemented cross-validation and hyperparameter tuning. The challenge here was determining the best parameters for each model in a time-efficient manner. For models like Random Forests and XGBoost, the complexity of tuning hyperparameters (e.g., the number of trees or learning rate) added to the time constraints.
- Model Selection and Testing: Despite the time pressures, it was crucial to evaluate different models for their predictive capabilities. The ANN provided a deep learning approach to stock prediction, but traditional models like Linear Regression, Random Forest, and SVR also offered valuable insights into how simpler models could perform. Comparing the performance of these models involved balancing complexity and accuracy, while also taking into account how long it would take to test and tune each model.

## Ethics and AI Use
- The project involved a consideration of ethics, particularly in terms of how the machine learning models were developed and deployed. A checklist was included in the Hugging Face app, which helped reflect on the ethical implications of using AI in financial predictions. This checklist emphasized transparency in the model’s decision-making, the potential biases in training data, and the importance of ensuring that the model’s predictions were not misused in financial decisions that could affect individuals or markets in harmful ways.

### The ethics checklist prompted the evaluation of:
- Bias and Fairness: Ensuring that the model did not inadvertently prioritize certain factors, such as specific industries or economic events, that could skew predictions in an unfair way.
- Transparency: Making sure that users could understand how the model’s predictions were derived, especially for stakeholders who may rely on these predictions for investment decisions.
- Accountability: Considering how the predictions would be used in real-world scenarios and ensuring that the AI model’s output was not solely relied upon for making financial decisions without human oversight.
- This combination of model experimentation, ethics consideration, and practical constraints shaped the direction of the project and its outcomes, providing valuable insights into the use of machine learning in financial predictions.

## Current Progress
- Data Collection: You have gathered historical stock data, which likely includes features such as stock prices (open, close, high, low), volume, and potentially other technical indicators like moving averages or RSI (Relative Strength Index). You've also collected macroeconomic data, such as interest rates, GDP, inflation, and possibly other relevant economic factors.

### 1. Data Preprocessing
- Cleaning:  removed missing values, outliers, and irrelevant data from your dataset.
Normalization/Standardization: Stock data and macroeconomic indicators typically require scaling to bring all features to a comparable range. This ensures the RNN can learn efficiently without being biased by the scale of certain variables.
- Feature Engineering: Have created additional features such as moving averages or lag features to capture trends and patterns over time.
- Data Splitting: The dataset has been divided into training, validation, and test sets to ensure proper evaluation of the model's performance.

### 2. Model Design
- Selected an Recurrent Neural Networks (RNN) architecture for the project. RNN is an advanced machine learning architectures primarily used for time-series data.
- Model Training: The model has been trained using historical data, employing mean squared error (MSE) or a similar loss function for training the regression model, which minimizes the difference between predicted and actual stock prices.

### 3. Model Evaluation
- After training, the model's performance has been evaluated using the test dataset. Metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or R-squared may have been used to assess the accuracy and reliability of the model’s predictions.

### 4. Model Optimization
- In the process of fine-tuning the model by adjusting the architecture (e.g., adding more layers or neurons) or optimizing hyperparameters to improve prediction accuracy.Techniques like dropout or early stopping could be used to prevent overfitting and ensure the model generalizes well to new data.

### 5. Future Steps
- This project is a step toward leveraging machine learning techniques but needs refinning to analyze stock market data and make informed predictions. The combination of stock price and macroeconomic data will help build a more comprehensive model for predicting future movements in the stock market.

## Contributions
Viola Lin
 - Created the machine learning pipeline for stock price prediction, including model selection, data preprocessing, cross-validation, and hyperparameter tuning.
 - Developed the code to handle stock price data and macroeconomic indicators.
 - Implemented models such as Linear Regression, Random Forest, Decision Trees, SVR, XGBoost, KNN, ANN, RNN, and TFT
 - Set up a preprocessing pipeline, including the extraction of date features, creation of lag features, and normalization of data.
 - Evaluated and compared models using various metrics and conducted model optimization for better performance.
 - Ethics and AI Considerations: Addressed the ethical considerations in the project, integrating a checklist in the Hugging Face app. Focused on transparency, fairness, and accountability in using machine learning for financial predictions. Developed a framework for assessing potential biases and risks associated with AI-driven stock price predictions.

AI Model and Technical Guidance
- ChatGPT (OpenAI): Provided guidance on model selection, feature engineering, and ethical considerations related to the use of machine learning in financial predictions.

Project Oversight and Academic Support
- Professor Okolie: Provided academic guidance and feedback throughout the project.
