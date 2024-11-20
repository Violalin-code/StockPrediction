import gradio as gr
import pandas as pd
import joblib

# Load your pre-trained model and scalar
model = joblib.load("best_model.pkl")
scalar = joblib.load("scaler.pkl")  # Load the scalar for preprocessing

# Function to process and predict based on uploaded CSVs
def predict_stock_price(files, year, month):
    try:
        # Ensure at least one file is uploaded
        if len(files) == 0:
            return "Please upload at least one CSV file."
        
        # Initialize an empty dataframe for merging
        merged_data = None
        
        # Process each CSV file
        for file in files:
            data = pd.read_csv(file.name)  # Read the current CSV file
            
            # Debugging: print columns of the current file
            print(f"Columns in {file.name}: {data.columns}")
            
            # Check if 'Date' column exists in the current file
            if 'Date' not in data.columns:
                return f"The file {file.name} must contain a 'Date' column."
            
            # If merged_data is None, set it as the first file's data
            if merged_data is None:
                merged_data = data
            else:
                # Merge with the already loaded data on 'Date'
                merged_data = pd.merge(merged_data, data, on='Date', how='inner')

        # Ensure the required columns like 'Adjusted Close', 'EBITDA', 'EV/EBITDA', etc. are present
        required_columns = ['Date', 'Adjusted Close', 'EBITDA', 'EV/EBITDA', 'Enterprise Value', 'Market Cap (USD)', 
                            'Net Income TTM', 'Debt/Equity Ratio', 'PE Ratio', 'Revenue TTM', 'Shareholder Equity', 
                            'Total Debt', 'Volume', 'Year', 'High', 'Low']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        
        if missing_columns:
            return f"The following required columns are missing: {', '.join(missing_columns)}"

        # Convert 'Date' column to datetime, catching errors
        merged_data['Date'] = pd.to_datetime(merged_data['Date'], errors='coerce')
        
        # Check for invalid dates (NaT values)
        invalid_dates = merged_data[merged_data['Date'].isnull()]
        if not invalid_dates.empty:
            return f"Some of the dates could not be parsed. The following rows have invalid dates:\n{invalid_dates[['Date']].head(10)}"

        # Filter by the specified year and month
        merged_data = merged_data[
            (merged_data['Date'].dt.year == year) & (merged_data['Date'].dt.month == month)
        ]

        # Check if any data remains after filtering
        if merged_data.empty:
            return f"No data available for the specified year ({year}) and month ({month})."

        # Extract 'Year' from the 'Date' column
        merged_data['Year'] = merged_data['Date'].dt.year

        # Ensure all required columns are present
        for column in required_columns:
            if column not in merged_data.columns:
                merged_data[column] = 0  # Or you can use np.nan, mean, etc.

        # Reorder columns to match the expected feature order
        merged_data = merged_data[required_columns]

        # Scale the features using the scalar
        scaled_features = scalar.transform(merged_data)

        # Predict using the model
        predictions = model.predict(scaled_features)

        return f"Predictions for {year}-{month}: {predictions}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_stock_price,
    inputs=[
        gr.File(label="Upload CSV Files", file_count="multiple", type="filepath"),  # Multiple file input
        gr.Number(label="Year"),
        gr.Number(label="Month")
    ],
    outputs="text",
    description="Upload multiple CSV files containing stock price data and financial ratios to get stock price predictions."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
