import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib 

def predict_and_return_close(files, year, month, day, model, scaler):
    try:
        if not files:
            return {"message": "Please upload at least one CSV file."}

        # Initialize an empty DataFrame to store all combined data
        all_data = pd.DataFrame()

        # Required columns for prediction
        required_columns = ['Adjusted Close', 'EBITDA', 'EV/EBITDA', 'Enterprise Value', 
                            'Market Cap (USD)', 'Net Income TTM', 'Debt/Equity Ratio', 
                            'PE Ratio', 'Revenue TTM', 'Shareholder Equity', 'Total Debt', 
                            'Volume', 'High', 'Low']

        for file in files:
            # Read each file
            data = pd.read_csv(file.name)

            if 'Date' not in data.columns:
                return {"message": f"File {file.name} must contain a 'Date' column."}

            # Convert Date column to datetime
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

            # Filter data for the specified date
            data_filtered = data[(data['Date'].dt.year == year) & 
                                  (data['Date'].dt.month == month) & 
                                  (data['Date'].dt.day == day)]

            # Keep only relevant columns (if they exist in the file)
            columns_to_keep = ['Date'] + [col for col in required_columns if col in data.columns]
            data_filtered = data_filtered[columns_to_keep]

            # Append filtered data to all_data
            all_data = pd.concat([all_data, data_filtered], ignore_index=True)

        if all_data.empty:
            return {"message": "No data found for the specified year, month, and day."}

        # Deduplicate rows based on the 'Date' column (if there are overlaps between files)
        all_data = all_data.drop_duplicates(subset=['Date'])

        # Check if 'Adjusted Close' already exists and is valid
        if not all_data['Adjusted Close'].isnull().all():
            # Return only the Adjusted Close values for the app output
            adjusted_close_data = all_data[['Adjusted Close']]
            return {
                "Adjusted Close": adjusted_close_data['Adjusted Close'].values.tolist()
            }

        # Ensure that there is enough data to make predictions
        if all_data[required_columns].isnull().any().any():
            return {"message": "Insufficient data to make predictions."}

        # Scale the data and predict
        scaled_features = scaler.transform(all_data[required_columns])
        predictions = model.predict(scaled_features)

        # Add predictions to the original DataFrame
        all_data['Predicted Close'] = predictions

        # Return the Date, Adjusted Close, and Predicted Close
        return {
            "message": "Prediction successful",
            "Data": all_data[['Date', 'Adjusted Close', 'Predicted Close']].to_dict(orient="records")
        }

    except Exception as e:
        return {"message": str(e)}
        
datacard_content = """
# Ethics DataCard for Stock Prediction Model
## Dataset Overview
Input Variables: Date, Open, High, Low, Volume, Net Income TTM, Market Cap, PE Ratio, Enterprise Value, EBITDA, EV/EBITDA, Total Debt, Shareholder Equity,	Debt/Equity Ratio, Revenue TTM
Output Variables: Adjusted Close
## Data Collection Process
Data sources include:
- Financial Market Data Providers: Platforms such as Bloomberg, Yahoo Finance, Alpha Vantage, or Quandl, which provide historical stock price and financial performance data.
- Corporate Financial Reports: Publicly available data from company 10-K and 10-Q filings with the SEC for metrics such as Net Income (TTM), Total Debt, and Shareholder Equity.
- Stock Exchanges: Direct feeds from stock exchanges like NYSE or NASDAQ for trading-related data (Open, High, Low, Volume).
- Collection Methods: Data was updated periodically to ensure relevance to market conditions.
## Bias Considerations
Potential Bias:
- Historical Data Bias: The dataset may reflect historical market conditions that do not account for future changes in market structure, regulations, or external shocks.
- Survivorship Bias: Only companies with available and consistent historical data are included, potentially excluding delisted or bankrupt companies.
- Sector-Specific Overrepresentation: Certain sectors (e.g., technology or healthcare) may dominate the dataset, skewing model predictions.
- Economic Cycles Bias: The dataset may be skewed by specific economic cycles (e.g., bull or bear markets).
- Data Source Bias: Reliance on specific data providers may introduce biases due to errors, incomplete data, or specific reporting standards.
Mitigation: The model is designed to minimize biases by :
Data Preprocessing: 
- Normalizing variables to reduce scale-related biases.
- Handling missing or incomplete data using techniques such as imputation or exclusion with justification.
- Ensuring Representation:
- Balancing the dataset to include diverse sectors, company sizes, and economic periods.
- Incorporating data from a wide range of sources to reduce dependence on a single provider.
## Fairness & Justice
Equity in Prediction:
- The model aims to ensure equitable predictions across companies and sectors by avoiding favoritism toward specific industries, market caps, or economic conditions.
- Equity is promoted through dataset balancing, ensuring equal representation of small-cap, mid-cap, and large-cap stocks, as well as diverse industries.
- The model evaluates prediction performance across different segments to identify and mitigate potential disparities.
Balancing Risks:
- The model acknowledges the risks of overfitting to historical data, which may reinforce past patterns of inequity in market evaluations. To counteract this, techniques like regularization and diverse data augmentation are used.
- Predictions are stress-tested under different economic scenarios (e.g., recession, bull markets) to ensure robustness and reduce the risk of misleading stakeholders.
Community Engagement:
- The project seeks feedback from diverse stakeholders, including financial analysts, data scientists, and ethical AI experts, to refine the model.
- Ongoing discussions with regulators or academic researchers help align the model's goals with broader societal objectives, such as fairness in financial predictions.
## Privacy and Security
- Data Anonymization: Individual financial records or personal data are not included in the dataset to preserve privacy. Only aggregated and publicly available data (e.g., market prices, financial metrics) is used.
- Consent Protocols: Publicly available data is obtained from reliable sources that have explicit permissions for use in research and modeling.
## Sustainability and Environmental Impact
Land Management Support: 
- Long-Term Sustainability: The model design supports efficient computation, minimizing energy consumption to align with principles of sustainable AI. Regular updates to the model ensure its longevity and adaptability to evolving market conditions, reducing the need for complete redevelopment.
- The model also considers periods of market recovery following significant economic downturns or external shocks. By analyzing historical trends and financial metrics, it provides insights into how companies and sectors rebound over time. These insights can help investors and policymakers identify opportunities for strategic investment and resource allocation, promoting long-term market stability and resilience.
## Model Limitations
- Regional Variability: The model may exhibit reduced accuracy when applied to stocks from regions not well-represented in the dataset. For instance, predictions for emerging markets may not reflect unique economic and regulatory environments. To address this, regional data augmentation and fine-tuning are planned as future improvements.
- Adaptive Updates: The model relies on periodic updates to incorporate new financial metrics, market trends, and regulatory changes. During the data preprocessing phase, gaps in the dataset were identified where certain dates lacked complete information due to the absence of specific financial metrics across multiple combined CSV files. To address this, forward and backward filling techniques were applied to interpolate missing values, ensuring the dataset remained comprehensive without introducing significant bias. Removing all rows with missing data would have resulted in an unrepresentative dataset by excluding a large portion of the target values. This approach preserves data integrity while maintaining sufficient representation for accurate predictions. Additionally, adaptive mechanisms like re-training on the latest data ensure that the model continues to provide relevant and accurate predictions over time, even as market conditions evolve.
## Accountability and Transparency
- Ongoing Monitoring: The model’s performance is regularly evaluated using out-of-sample testing and performance metrics such as RMSE, MAE, and bias indicators.
- Stakeholder Communication: Clear and accessible reports are provided to stakeholders (e.g., investors, analysts) detailing the model’s methodology, limitations, and results. Any significant changes to the model are communicated promptly to ensure trust and usability.
- Feedback Mechanisms: Regular reviews with domain experts ensure that the model aligns with industry needs and ethical standards.
## Societal Impact
- Informing Policy: Insights generated by the model can aid policymakers in understanding market trends, corporate sustainability efforts, and the financial implications of regulatory changes. The model can also support taxation policies or investment incentives targeting high-impact sectors.
- Regional Variability: The overarching goal of the model is to contribute to economic stability and sustainable investment practices. By identifying market trends and providing accurate predictions, it enables better financial decision-making, supporting the growth of companies with strong ESG (Environmental, Social, and Governance) practices. This indirectly promotes environmental conservation, ethical labor practices, and long-term societal well-being.
- Informing Policy: The insights provided by the model have the potential to inform policy changes in areas such as market regulations, corporate transparency, and sustainable investing. By highlighting key financial metrics and trends, the model aids policymakers and regulators in creating a more equitable and resilient financial system, ensuring long-term economic sustainability.
- Educational Outreach: The model’s framework and outputs can serve as a learning tool for students and professionals to understand financial modeling, ethical AI practices, and the impact of market predictions. Open-sourcing portions of the methodology or providing tutorials could further contribute to public knowledge.
"""

# Function to display the DataCard
def display_datacard():
    return datacard_content

md_content = """
# Ethics Checklist for Stock Prediction Model
## 1. Data Collection
Input Variables: Date, Open, High, Low, Volume, Net Income (TTM), Market Cap, PE Ratio, Enterprise Value, EBITDA, EV/EBITDA, Total Debt, Shareholder Equity, Debt/Equity Ratio, Revenue (TTM)
Output Variables: Adjusted Close
- Are the data sources on stock prices and market trends properly licensed and legally available?
    - Yes, all data comes from reputable and publicly available sources, including financial market data providers such as Bloomberg, Yahoo Finance, Alpha Vantage, and Quandl, as well as corporate financial reports (10-K and 10-Q filings with the SEC) and stock exchanges (e.g., NYSE, NASDAQ). These sources provide legally accessible data for research and modeling.
- Has any sensitive information, such as private property or personal location data, been anonymized?
    - Yes, the dataset does not contain any sensitive personal information. Only aggregated and publicly available financial data is used, ensuring privacy and anonymity for individuals or companies involved.
- Have you obtained consent for data collected from private or proprietary sources?
    - Yes, data from proprietary sources is obtained through explicit permissions for use in research and modeling, ensuring that the data collection process adheres to consent protocols.
## 2. Fairness & Justice
- How will you ensure that the model’s predictions are fair and do not disproportionately affect specific sectors or regions?
    - We will balance the dataset to include diverse sectors, company sizes, and economic conditions. The model will be tested for performance consistency across different sectors, ensuring no favoritism toward specific industries, such as technology or healthcare.
- What biases might exist in the historical data (e.g., market cycles, sector representation)? How will you address these to ensure the model does not unfairly target or neglect specific areas?
    - Biases in the data may stem from sector-specific overrepresentation, survivorship bias, or skewed economic cycles. To mitigate these, we will ensure balanced representation across various sectors and include data from a wide range of sources. Techniques such as data augmentation and regularization will be employed to address potential overfitting.
- How will you balance fairness in handling both false positives (predicting higher stock prices when there is none) and false negatives (failing to predict a price rise)?
    - We will evaluate and adjust the model’s performance using appropriate metrics such as RMSE, MAE, and bias indicators. Regular performance audits and testing across different economic conditions will help to minimize the risks of both false positives and false negatives.
- Have you tested the model across different sectors and company sizes to ensure consistent performance?
    - Yes, we have tested the model on stocks from different sectors (e.g., technology, healthcare, financials) and across small-cap, mid-cap, and large-cap companies. Adjustments are made to ensure balanced predictions across these categories.
## 3. Transparency
- How will you ensure transparency about the data sources, algorithms, and decision-making process of the model?
    - We will maintain transparency by documenting data sources, modeling techniques, and the rationale behind model design choices. All information will be available to stakeholders and the public via regular reports and open-access documentation.
- What information will you make available to investors, analysts, and the public?
    - We will provide detailed reports that include the model’s predictions, methodologies, assumptions, and limitations. These reports will be accessible to stakeholders and will be updated regularly to reflect changes in market conditions.
- How will you communicate the model’s predictions and limitations to decision-makers so that they understand the risks involved?
    - Clear, concise summaries will be provided, highlighting the model’s predictions along with an explanation of its limitations. Visual aids and risk assessments will accompany these summaries to help decision-makers understand potential risks and uncertainties.
- How will you explain false positives and false negatives to affected stakeholders?
    - We will create clear communication strategies to explain the impact of false positives and false negatives, particularly during critical market events. This will include informational materials and briefings aimed at helping stakeholders understand model outcomes and how to respond.
## 4. Privacy and Security
- How will you ensure the privacy of individuals whose data might be inadvertently captured?
    - The model uses aggregated, anonymized data sourced from public financial reports and stock exchanges. No personal information or sensitive data about individuals is used in the dataset, ensuring privacy protection.
- What steps will you take to prevent the misuse of financial data, especially regarding personal or company-specific financial information?
    - Strict data access controls will be enforced, and guidelines will be established to limit data usage to its intended purpose—predicting stock price movements. Any misuse will be met with clear penalties as per established data use agreements.
- How will you balance the need for accurate predictions with protecting individual privacy?
    - By using only publicly available, aggregated data and anonymizing any sensitive data, we ensure that privacy is upheld without compromising the model’s predictive power.
## 5. Accountability
- Who will be held accountable if the model incorrectly predicts stock movements, resulting in significant financial loss?
    - Accountability will rest with the model development team, which will include financial analysts, data scientists, and project managers. An oversight committee will be responsible for ensuring the model's integrity and addressing any issues that arise from predictions.
- What system will you establish to monitor and adjust the model over time, ensuring it adapts to changing market conditions?
    - We will implement continuous monitoring through performance metrics, regular audits, and periodic retraining using up-to-date market data. Feedback from stakeholders will help guide model adjustments as necessary.
- How will you communicate accountability measures to the public?
    - We will proactively communicate our accountability measures through detailed reports and stakeholder briefings. Any changes to the model or its performance will be clearly explained to maintain trust.
## 6. Inclusivity
- How will you ensure the model includes diverse data from different sectors and regions?
    - We will seek to diversify the dataset by including stocks from a broad range of sectors (e.g., technology, energy, industrials) and company sizes. We will also incorporate data from emerging markets to ensure the model is inclusive of various global economic conditions.
- How will you ensure the model accounts for the needs of different communities or stakeholders?
    - We will engage with stakeholders from different regions and sectors to understand their unique needs and concerns. This will include feedback from financial analysts, investors, and regulatory bodies to ensure the model’s predictions are relevant and fair.
- If certain sectors or regions lack sufficient data, how will you address this to avoid biased predictions?
    - We will use data augmentation techniques, seek collaborations with local researchers, and explore partnerships with organizations to gather missing data from underrepresented regions or sectors. Synthetic data generation may also be used to fill gaps in the dataset.
## 7. Sustainability
- How will the model’s predictions affect long-term market trends and sustainability efforts?
    - The model will assist investors in making informed decisions based on long-term market trends and corporate sustainability efforts. By providing accurate financial predictions, the model can help identify opportunities in sectors with strong environmental, social, and governance (ESG) practices.
- How will you ensure the model remains sustainable, considering the evolving nature of global financial markets and economic conditions?
    - Regular updates will be incorporated to ensure the model stays relevant. This includes adjusting the model’s inputs and algorithms to account for new market conditions, regulations, and emerging trends such as ESG investing and sustainable finance.
- What are the broader social and environmental implications if this model becomes widely adopted?
    - The broader social impact of the model will be positive, promoting transparency in financial markets and encouraging investments in sustainable and responsible companies. It may also contribute to global economic stability and sustainable market growth by aiding informed decision-making.
"""
import gradio as gr

# Gradio Interface
with gr.Blocks() as interface:
    with gr.Tabs():
        with gr.Tab("Stock Prediction"):
            file_input = gr.File(label="Upload CSV Files", file_types=[".csv"], file_count="multiple")
            year_input = gr.Number(label="Year", value=2023)
            month_input = gr.Number(label="Month", value=1)
            day_input = gr.Number(label="Day", value=1)
            output = gr.Textbox(label="Results")
            submit_button = gr.Button("Submit")

            # Ensure correct function reference
            submit_button.click(predict_and_return_close, [file_input, year_input, month_input, day_input], output)

        with gr.Tab("Ethics Data Card"):
            gr.Markdown(datacard_content)

        with gr.Tab("Ethics Checklist"):
            gr.Markdown(md_content)

# Launch the app
interface.launch()
