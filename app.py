from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
np.float_ = np.float64
import matplotlib.pyplot as plt
import io
import os
import base64
from prophet import Prophet
import joblib

# Load the dataset to get unique values for validation
dataset = pd.read_csv('future_predictions_2023_2029.csv')
unique_guri_nums = dataset['guri_num'].unique()

# Initialize Flask application
app = Flask(__name__)

# Function to load the models dynamically
def load_models():
    models = {}
    models_folder = 'models'
    
    for filename in os.listdir(models_folder):
        if filename.endswith('.pkl'):
            guri = filename.split('_')[2].split('.')[0]  # Extract guri_num from filename
            with open(os.path.join(models_folder, filename), 'rb') as f:
                models[guri] = pd.read_pickle(f)
                
    return models

# Load the Prophet models into memory
models = load_models()

# Route to render the welcome page
@app.route('/')
def home():
    return render_template('index.html')

# Function to create future dataframe for predictions
def create_future_dataframe(year, month=None):
    if month:
        future = pd.DataFrame({
            'ds': [pd.Timestamp(f'{year}-{month}-01')],
            'month': [month],
            'year': [year]
        })
    else:
        future_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')
        future = pd.DataFrame({
            'ds': future_dates,
            'month': future_dates.month,
            'year': future_dates.year
        })
    return future

# Common prediction function
def predict_consumption(guri_num, year, month=None):
    future = create_future_dataframe(year, month)
    model = models[guri_num]
    forecast = model.predict(future)
    if month:
        return forecast.loc[forecast['ds'].dt.month == month, 'yhat'].values[0]
    else:
        return forecast[['ds', 'yhat']].set_index('ds')['yhat']

# Route to render the prediction form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            guri_num = request.form['guri_num']
            month = int(request.form['month'])
            year = int(request.form['year'])
            provider = request.form['provider']
        except ValueError:
            return render_template('predict.html', error="Please enter valid numbers for all fields.")

        # Validate the inputs
        if guri_num not in unique_guri_nums:
            return render_template('predict.html', error="Invalid guri number")
        if month not in range(1, 13):
            return render_template('predict.html', error="Invalid month")
        if year not in range(2024, 2031):
            return render_template('predict.html', error="Invalid year")

        # Predict consumption
        predicted_total_KW = predict_consumption(guri_num, year, month)

        # Determine the multiplier based on the provider
        if provider == 'beco':
            multiplier = 0.41
        elif provider == 'mogadishu':
            multiplier = 0.45
        elif provider == 'bluesky':
            multiplier = 0.35
        else:
            return render_template('predict.html', error="Invalid provider selected")

        # Calculate the scaled prediction
        predicted_total_price = predicted_total_KW * multiplier

        # Prepare prediction result to display
        kw = f'for House number {guri_num} in {month}-{year}'
        kw_prediction = f'Predicted Total KW: {predicted_total_KW:.2f}'
        price_prediction = f'Predicted price: {predicted_total_price:.2f}'

        return render_template('result.html', kw=kw, kw_prediction=kw_prediction, price_prediction=price_prediction, guri_num=guri_num, month=month, year=year)
    return render_template('predict.html')

@app.route('/predict_year', methods=['GET', 'POST'])
def predict_year():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            provider = request.form['provider']
        except ValueError:
            return render_template('form.html', error="Please enter a valid year and provider.")

        # Validate the inputs
        if year not in range(2024, 2031):
            return render_template('form.html', error="Invalid year")

        # Load the Prophet model
        with open('prophet_model1.pkl', 'rb') as file:
            model = joblib.load(file)

        # Generate a DataFrame for each month of the requested year
        future_df = pd.DataFrame({'ds': pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')})

        # Make prediction using the loaded model
        forecast = model.predict(future_df)

        # Extract the predictions for each month
        forecast['month'] = forecast['ds'].dt.strftime('%B')
        monthly_predictions = forecast[['month', 'yhat']]

        # Convert KW to MW for each month
        monthly_predictions['yhat_mw'] = monthly_predictions['yhat'] / 1000

        # Plotting the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_predictions['month'], monthly_predictions['yhat'], marker='o', linestyle='-', color='b')
        plt.title(f'Predicted Monthly Electricity Consumption for in {year}')
        plt.xlabel('Month')
        plt.ylabel('Electricity Consumption (KW)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Render the result template with the monthly predictions and plot
        return render_template('prediction_result.html', year=year, provider=provider, monthly_predictions=monthly_predictions.to_dict(orient='records'), plot_url=plot_url)

    return render_template('form.html')
# Route to render the visualization input form
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    if request.method == 'POST':
        try:
            guri_num = request.form['guri_num']
            year = int(request.form['year'])
            provider = request.form['provider']
        except ValueError:
            return render_template('visualization.html', error="Please enter valid numbers for all fields.")

        # Validate the inputs
        if guri_num not in unique_guri_nums:
            return render_template('visualization.html', error="Invalid guri number")
        if year not in range(2024, 2031):
            return render_template('visualization.html', error="Invalid year")

        # Predict monthly consumption for the entire year
        monthly_consumption = []
        monthly_prices = []
        for month in range(1, 13):
            kw = predict_consumption(guri_num, year, month)
            if provider == 'beco':
                multiplier = 0.41
            elif provider == 'mogadishu':
                multiplier = 0.45
            elif provider == 'bluesky':
                multiplier = 0.35
            else:
                return render_template('visualization.html', error="Invalid provider selected")
            price = kw * multiplier
            monthly_consumption.append(kw)
            monthly_prices.append(price)

        # Calculate total yearly consumption and price
        total_yearly_kw = sum(monthly_consumption)
        total_yearly_price = sum(monthly_prices)

        # Plot the results
        months = list(range(1, 13))
        plt.figure(figsize=(10, 5))
        plt.plot(months, monthly_consumption, marker='o', label='Total KW')
        plt.plot(months, monthly_prices, marker='o', label='Price ($)')
        plt.title(f'Monthly Electricity Consumption and Price for House {guri_num} for {year}')
        plt.xlabel('Month')
        plt.ylabel('Value')
        plt.xticks(months)
        plt.grid(True)
        plt.ylim(0, max(max(monthly_consumption), max(monthly_prices)) * 1.1)
        plt.legend()

        # Annotate the plot with predicted values
        for i, (kw, price) in enumerate(zip(monthly_consumption, monthly_prices)):
            plt.text(months[i], kw, f'{kw:.2f}KW', ha='center', va='bottom')
            plt.text(months[i], price, f'{price:.2f}$', ha='center', va='top')

        # Save plot to a string in base64 format
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('visualize_result.html', plot_url=plot_url, total_yearly_kw=total_yearly_kw, total_yearly_price=total_yearly_price)
    return render_template('visualization.html')

# Route to render the About Us page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
