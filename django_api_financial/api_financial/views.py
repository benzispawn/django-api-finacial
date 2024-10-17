import io

import matplotlib
from fontTools.subset import subset

matplotlib.use('Agg')
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.conf import settings
from .models import StockData, StockTimeline
from django.db import transaction, IntegrityError
import pandas as pd
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import pickle
import sklearn
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from django.http import FileResponse

def load_model():
    model_path = settings.PKL_MODEL_PATH  # Add this to your settings file, path to your .pkl file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_future_stock_prices(model, historical_data):
    # You will need to preprocess the historical_data to match the model's expected input format
    X = historical_data[['close_price']]  # Example feature used for prediction
    future_predictions = model.predict(X)  # Predict future stock prices
    return future_predictions

# Create your views here.
@api_view(['GET'])
def api_overview(request):
    return Response({ "message": "Welcome to the API" })

def custom_page_not_found_view(request, exception):
    return JsonResponse({'error': 'The requested resource was not found'}, status=404)

def safe_decimal_conversion(value):
    """
    Safely convert a value to Decimal.
    If conversion fails, log an error and return None.
    """
    try:
        if value is None or value == '':
            print(f"Skipping conversion: value is {value}")
            return None
        return Decimal(str(value).replace(',', '').strip())
    except (InvalidOperation, ValueError) as e:
        print(f"Error converting {value} to Decimal: {e}")
        return None

def calculate_moving_average(stock_data, window_size):
    """
    Calculates the moving average over a specified window size.
    :param stock_data: List of stock closing prices
    :param window_size: Number of days for moving average (e.g., 50 or 200)
    :return: Moving average as a pandas series
    """
    return stock_data['close_price'].rolling(window=window_size).mean()

def calculate_max_drawdown(stock_data):
    """
    Calculate the maximum drawdown of the stock.
    :param stock_data: List of stock closing prices
    :return: Max drawdown value
    """
    rolling_max = stock_data['close_price'].cummax()
    drawdown = (stock_data['close_price'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # The maximum drawdown
    return max_drawdown

def execute_trades(stock_data, investment):
    cash = investment  # The initial amount of cash
    holdings = 0  # Number of stocks owned
    total_trades = 0
    trades = []

    stock_data['50_day_sma'] = calculate_moving_average(stock_data, 50).apply(safe_decimal_conversion)
    stock_data['200_day_sma'] = calculate_moving_average(stock_data, 200).apply(safe_decimal_conversion)
    stock_data = stock_data.dropna(subset=['close_price', '50_day_sma', '200_day_sma'])
    stock_data['close_price'].ffill()
    stock_data['50_day_sma'].ffill()
    stock_data['200_day_sma'].ffill()

    for i, row in stock_data.iterrows():
        # Buy condition: stock price dips below the 50-day SMA
        if row['close_price'] < row['50_day_sma'] and cash > 0:
            holdings = cash / row['close_price']  # Buy as many shares as possible
            cash = 0
            trades.append(('buy', row.name, row['close_price'], holdings))
            total_trades += 1

        # Sell condition: stock price goes above the 200-day SMA
        elif row['close_price'] > row['200_day_sma'] and holdings > 0:
            cash = holdings * row['close_price']  # Sell all holdings
            holdings = 0
            trades.append(('sell', row.name, row['close_price'], cash))
            total_trades += 1

    final_value = cash if cash > 0 else holdings * stock_data.iloc[-1]['close_price']
    total_return = (final_value - investment) / investment * 100
    return {
        'total_return': total_return,
        'trades': trades,
        'total_trades': total_trades,
        'final_value': final_value
    }

def prepare_dataframe_for_trades(stock_data_df):
    """
       Convert all relevant columns in the DataFrame to Decimal for precise calculations.
       """
    stock_data_df['close_price'] = stock_data_df['close_price'].apply(safe_decimal_conversion)
    stock_data_df['50_day_sma'] = stock_data_df['50_day_sma'].apply(safe_decimal_conversion)
    stock_data_df['200_day_sma'] = stock_data_df['200_day_sma'].apply(safe_decimal_conversion)

    # Drop any rows where conversions failed
    stock_data_df = stock_data_df.dropna(subset=['close_price', '50_day_sma', '200_day_sma'])
    return stock_data_df


@api_view(['GET'])
def fetch_stock_data(request):
    try:
        symbol = request.GET.get('symbol', None)
        investment = safe_decimal_conversion(request.GET.get('investment', 10000))
        isPDF = request.GET.get('isPDF', 'false').lower() == 'true'
        if symbol is None:
            # Return a JSON error if the symbol is not provided
            return Response({'error': 'paramError','message': 'Stock symbol is required'}, status=400)
        # Define date range: today to 2 years ago
        today = datetime.now()
        two_years_ago = today - relativedelta(years=2)

        # Make the API request to Alpha Vantage
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': settings.ALPHAVANTAGE_API_KEY  # Ensure this is in your settings
        }
        response = requests.get(settings.ALPHAVANTAGE_API_URL, params=params)
        # Check for rate limiting or errors in response
        if response.status_code != 200:
            return JsonResponse({
                'error': 'apiError',
                'message': 'API request failed'
            }, status=500)

        data = response.json()
        # print(data)
        if 'Time Series (Daily)' not in data:
            return JsonResponse({
                'error': 'invalidResponse',
                'message': 'Invalid API response or data unavailable'
            }, status=500)
        # Insert the data into the database, limiting to the last two years
        time_series = data['Time Series (Daily)']
        # Lists
        stock_timelines_to_insert = []
        stock_data_to_insert = []
        stock_data_to_update = []
        # print(time_series.items())
        existing_timelines = StockTimeline.objects.filter(st_day__gte=two_years_ago).values_list('st_day', flat=True)
        existing_timelines_set = set(existing_timelines)

        existing_stock_data = StockData.objects.filter(sd_symbol=symbol).select_related('st_id')#StockData.objects.filter(sd_symbol=symbol).select_related('st_id')
        existing_stock_data_dict = {sd.st_id.st_day: sd for sd in existing_stock_data}

        for date_str, daily_data in time_series.items():
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            # Skip dates older than 2 years
            if date_obj < two_years_ago.date():
                break

            if date_obj not in existing_timelines_set:
                stock_timelines_to_insert.append(StockTimeline(st_day=date_obj))
            else:
                print(f"StockTimeline for {date_obj} already exists. Skipping insertion.")

        # print(stock_timelines_to_insert)
        try:
            with transaction.atomic():
                if len(stock_timelines_to_insert) > 0:
                    print(f"Inserting {len(stock_timelines_to_insert)} stock timeline records.")
                    StockTimeline.objects.bulk_create(stock_timelines_to_insert)

                    print("Stock timeline records inserted successfully.")

        except IntegrityError as e:
            print(f"IntegrityError during bulk creation of stock timelines: {e}")
            return JsonResponse(
                {'error': 'dbError', 'message': 'Failed to insert stock timeline data due to integrity error.'},
                status=500)

        all_timelines = StockTimeline.objects.filter(st_day__gte=two_years_ago)
        # print(all_timelines)
        timelines_dict = {timeline.st_day: timeline for timeline in all_timelines}
        # print(timelines_dict)
        two_years_ago_date = two_years_ago.date()
        for date_str, daily_data in time_series.items():
            # print("Inside the loop")
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()

            if date_obj < two_years_ago_date:
                break
            # print("antes de alocar timeline")
            timeline = timelines_dict.get(date_obj)
            # print(timeline)
            if not timeline:
                print(f"Warning: No timeline found for {date_obj}")
                continue
            # print("depois do continue")
            open_price = safe_decimal_conversion(daily_data['1. open'])
            close_price = safe_decimal_conversion(daily_data['4. close'])
            high_price = safe_decimal_conversion(daily_data['2. high'])
            low_price = safe_decimal_conversion(daily_data['3. low'])
            volume = safe_decimal_conversion(daily_data['5. volume'])
            if None in (open_price, close_price, high_price, low_price, volume):
                print(f"Skipping {date_obj} due to conversion errors")
                continue
            # Check if stock data for this date exists
            if date_obj in existing_stock_data_dict:
                # If it exists, update the data
                stock = existing_stock_data_dict[date_obj]
                stock.sd_open_price = open_price
                stock.sd_close_price = close_price
                stock.sd_high_price = high_price
                stock.sd_low_price = low_price
                stock.sd_volume = volume
                stock_data_to_update.append(stock)
            else:
                # If it doesn't exist, prepare a new StockData object for insertion
                print("Stock data to insert")
                stock_data_to_insert.append(StockData(
                    sd_symbol=symbol,
                    st_id=timeline,  # Link to the saved timeline
                    sd_open_price=open_price,
                    sd_close_price=close_price,
                    sd_high_price=high_price,
                    sd_low_price=low_price,
                    sd_volume=volume
                ))
        try:
            with transaction.atomic():
                if stock_data_to_insert:
                    print(f"Inserting {len(stock_data_to_insert)} stock data records.")
                    StockData.objects.bulk_create(stock_data_to_insert)
                    print("Stock data records inserted successfully.")

                    # Bulk update existing StockData records
                if stock_data_to_update:
                    StockData.objects.bulk_update(stock_data_to_update, [
                        'sd_open_price', 'sd_close_price', 'sd_high_price', 'sd_low_price', 'sd_volume'
                    ])
                    print(f"Updated {len(stock_data_to_update)} stock data records.")
        except IntegrityError as e:
            print(f"IntegrityError during bulk creation of stock data: {e}")
            return JsonResponse({'error': 'dbError', 'message': 'Failed to insert stock data due to integrity error.'},
                                status=500)
        transaction.commit() # commit

        # Start backtest
        # stock_data_df = pd.DataFrame.from_records(
        #     list(StockData.objects.filter(sd_symbol=symbol).values('sd_close_price', 'st_id__st_day'))
        # )
        stock_data_df = pd.DataFrame.from_records(
            list(
                StockData.objects.filter(sd_symbol=symbol)
                .select_related('st_id')  # This performs the inner join with StockTimeline
                .values('sd_close_price', 'st_id__st_day')  # Select the needed fields
            )
        )
        # stock_data_df = stock_data_df.rename(columns={'st_id__st_day': 'date', 'sd_close_price': 'close_price'})
        stock_data_df = stock_data_df.rename(columns={'st_id__st_day': 'date', 'sd_close_price': 'close_price'})
        stock_data_df.set_index('date', inplace=True)

        stock_data_df['close_price'] = stock_data_df['close_price'].apply(safe_decimal_conversion)
        stock_data_df = stock_data_df.sort_index()

        # Calculate moving averages
        # stock_data_df['50_day_sma'] = calculate_moving_average(stock_data_df, 50)
        # stock_data_df['200_day_sma'] = calculate_moving_average(stock_data_df, 200)

        # Backtesting: Execute trades based on moving averages
        performance = execute_trades(stock_data_df, investment)
        #
        max_drawdown = calculate_max_drawdown(stock_data_df)
        #
        if stock_data_df.empty:
            print("No numeric data to plot")
            return JsonResponse({'error': 'No valid stock data to plot'}, status=400)  #

        if isPDF:
            try:
                # Setting up the report
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.setTitle(f"Stock Report for {symbol}")

                # Add a Title
                p.setFont("Helvetica-Bold", 16)
                p.drawString(100, 750, f"Stock Report for {symbol}")
                p.setFont("Helvetica", 12)
                p.drawString(100, 730, f"Generated on {today.strftime('%Y-%m-%d')}")

                # Add Key Metrics
                p.drawString(100, 710, f"Total Return: {round(performance['total_return'], 2)}")
                p.drawString(100, 690, f"Total Trades: {performance['total_trades']}")
                p.drawString(100, 670, f"Max Drawdown: {round(max_drawdown, 2)}")
                p.drawString(100, 650, f"Final Portfolio Value: {round(performance['final_value'], 2)}")

                # List of Trades
                p.setFont("Helvetica-Bold", 12)
                p.drawString(100, 620, "List of Trades:")
                p.setFont("Helvetica", 10)

                y_position = 600  # Starting Y position for the trade list

                trade_days = performance['trades']
                for i, trade in enumerate(trade_days):
                    trade_type, date, price, portfolio_value = trade
                    trade_text = f"{i + 1}. {trade_type.capitalize()} on {date.strftime('%Y-%m-%d')} at ${round(price, 2)} - Portfolio: ${round(portfolio_value, 2)}"
                    p.drawString(100, y_position, trade_text)
                    y_position -= 15  # Move the text up for the next line

                    # Check if the y_position is too low and start a new page if necessary
                    if y_position < 50:
                        p.showPage()
                        y_position = 750

                if y_position < 200:  # Check if there’s enough space for the graph on the same page
                    p.showPage()

                stock_data_df.index = pd.to_datetime(stock_data_df.index)

                stock_data_df['50_day_sma'] = calculate_moving_average(stock_data_df, 50).apply(safe_decimal_conversion)
                stock_data_df['200_day_sma'] = calculate_moving_average(stock_data_df, 200).apply(safe_decimal_conversion)
                # stock_data_df = stock_data_df.dropna(subset=['close_price', '50_day_sma', '200_day_sma'])
                stock_data_df['close_price'].ffill()
                stock_data_df['50_day_sma'].ffill()
                stock_data_df['200_day_sma'].ffill()
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
                stock_data_df['scaled_close_price'] = scaler.fit_transform(stock_data_df[['close_price']])
                # print(stock_data_df)
                stock_data_df = stock_data_df.astype(float)
                # Plot and add graph to PDF
                plt.figure(figsize=(10, 6))
                plt.clf()
                # stock_data_df['close_price'].plot()
                # date_df = pd.to_datetime((list(stock_data_df["Date"])))
                plt.plot(stock_data_df.index, stock_data_df['scaled_close_price'], label='Close Price', color='blue', linewidth=2)
                plt.title(f"Stock Price for {symbol}")
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Close Price (USD)", fontsize=12)
                # Trades

                # print(trade_days)
                buy_days = [trade[1] for trade in trade_days if trade[0] == 'buy']
                # print(buy_days)
                sell_days = [trade[1] for trade in trade_days if trade[0] == 'sell']
                plt.scatter(stock_data_df.loc[buy_days].index, stock_data_df.loc[buy_days]['scaled_close_price'],
                            color='green', label='Buy', marker='^', s=45)

                plt.scatter(stock_data_df.loc[sell_days].index, stock_data_df.loc[sell_days]['scaled_close_price'],
                            color='red', label='Sell', marker='v', s=45)

                plt.grid(True)
                plt.legend()

                # Save the plot to the PDF
                imgdata = io.BytesIO()#BytesIO()
                plt.savefig(imgdata, format='png')
                imgdata.seek(0)
                img = ImageReader(imgdata)
                p.drawImage(img, 100, 400, width=400, height=200)

                # Finish PDF
                p.showPage()
                p.save()

                buffer.seek(0)
                pdf_size = len(buffer.getvalue())
                print(f"PDF size: {pdf_size} bytes")  # Debug statement to check the size of the PDF
                if pdf_size == 0:
                    print("PDF generation failed, buffer is empty!")

                return FileResponse(buffer, as_attachment=True, content_type='application/pdf', filename=f'{symbol}_stock_report.pdf')
            except Exception as e:
                print(f"Request Error: {e}")
                return JsonResponse({
                    'error': 'apiError',
                    'message': 'API request failed'
                }, status=500)

        return JsonResponse({
            'success': f"Stock data for {symbol} has been successfully updated",
            'total_return': performance['total_return'],
            'total_trades': performance['total_trades'],
            'final_portfolio_value': performance['final_value'],
            'trades': performance['trades'],
            'max_drawdown': max_drawdown,
        }, status=200)
        # return JsonResponse({'success': f"Stock data for {symbol} has been successfully updated"}, status=200)
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'error': str(e),
            'message': 'API request failed'
        }, status=500)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'An error occurred while processing the data'
        }, status=500)

@api_view(['GET'])
def fetch_stock_data_with_predictions(request):
    try:
        symbol = request.GET.get('symbol', None)
        investment = safe_decimal_conversion(request.GET.get('investment', 10000))
        if symbol is None:
            # Return a JSON error if the symbol is not provided
            return Response({'error': 'paramError','message': 'Stock symbol is required'}, status=400)
        # Define date range: today to 2 years ago
        today = datetime.now()
        two_years_ago = today - relativedelta(years=2)

        # Make the API request to Alpha Vantage
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': settings.ALPHAVANTAGE_API_KEY  # Ensure this is in your settings
        }
        response = requests.get(settings.ALPHAVANTAGE_API_URL, params=params)
        # Check for rate limiting or errors in response
        if response.status_code != 200:
            return JsonResponse({
                'error': 'apiError',
                'message': 'API request failed'
            }, status=500)

        data = response.json()
        # print(data)
        if 'Time Series (Daily)' not in data:
            return JsonResponse({
                'error': 'invalidResponse',
                'message': 'Invalid API response or data unavailable'
            }, status=500)
        # Insert the data into the database, limiting to the last two years
        time_series = data['Time Series (Daily)']
        # Lists
        stock_timelines_to_insert = []
        stock_data_to_insert = []
        stock_data_to_update = []
        # print(time_series.items())
        existing_timelines = StockTimeline.objects.filter(st_day__gte=two_years_ago).values_list('st_day', flat=True)
        existing_timelines_set = set(existing_timelines)

        existing_stock_data = StockData.objects.filter(sd_symbol=symbol).select_related('st_id')
        existing_stock_data_dict = {sd.st_id.st_day: sd for sd in existing_stock_data}

        for date_str, daily_data in time_series.items():
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            # Skip dates older than 2 years
            if date_obj < two_years_ago.date():
                break

            if date_obj not in existing_timelines_set:
                stock_timelines_to_insert.append(StockTimeline(st_day=date_obj))

        # print(stock_timelines_to_insert)
        try:
            with transaction.atomic():
                if len(stock_timelines_to_insert) > 0:
                    print(f"Inserting {len(stock_timelines_to_insert)} stock timeline records.")
                    StockTimeline.objects.bulk_create(stock_timelines_to_insert)

                    print("Stock timeline records inserted successfully.")

        except IntegrityError as e:
            print(f"IntegrityError during bulk creation of stock timelines: {e}")
            return JsonResponse(
                {'error': 'dbError', 'message': 'Failed to insert stock timeline data due to integrity error.'},
                status=500)

        all_timelines = StockTimeline.objects.filter(st_day__gte=two_years_ago)
        # print(all_timelines)
        timelines_dict = {timeline.st_day: timeline for timeline in all_timelines}
        # print(timelines_dict)
        two_years_ago_date = two_years_ago.date()
        for date_str, daily_data in time_series.items():
            # print("Inside the loop")
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()

            if date_obj < two_years_ago_date:
                break
            # print("antes de alocar timeline")
            timeline = timelines_dict.get(date_obj)
            # print(timeline)
            if not timeline:
                print(f"Warning: No timeline found for {date_obj}")
                continue
            # print("depois do continue")
            open_price = safe_decimal_conversion(daily_data['1. open'])
            close_price = safe_decimal_conversion(daily_data['4. close'])
            high_price = safe_decimal_conversion(daily_data['2. high'])
            low_price = safe_decimal_conversion(daily_data['3. low'])
            volume = safe_decimal_conversion(daily_data['5. volume'])
            if None in (open_price, close_price, high_price, low_price, volume):
                print(f"Skipping {date_obj} due to conversion errors")
                continue
            # Check if stock data for this date exists
            if date_obj in existing_stock_data_dict:
                # If it exists, update the data
                stock = existing_stock_data_dict[date_obj]
                stock.sd_open_price = open_price
                stock.sd_close_price = close_price
                stock.sd_high_price = high_price
                stock.sd_low_price = low_price
                stock.sd_volume = volume
                stock_data_to_update.append(stock)
            else:
                # If it doesn't exist, prepare a new StockData object for insertion
                print("Stock data to insert")
                stock_data_to_insert.append(StockData(
                    sd_symbol=symbol,
                    st_id=timeline,  # Link to the saved timeline
                    sd_open_price=open_price,
                    sd_close_price=close_price,
                    sd_high_price=high_price,
                    sd_low_price=low_price,
                    sd_volume=volume,
                ))
        try:
            with transaction.atomic():
                if stock_data_to_insert:
                    print(f"Inserting {len(stock_data_to_insert)} stock data records.")
                    StockData.objects.bulk_create(stock_data_to_insert)
                    print("Stock data records inserted successfully.")

                    # Bulk update existing StockData records
                if stock_data_to_update:
                    StockData.objects.bulk_update(stock_data_to_update, [
                        'sd_open_price', 'sd_close_price', 'sd_high_price', 'sd_low_price', 'sd_volume'
                    ])
                    print(f"Updated {len(stock_data_to_update)} stock data records.")
        except IntegrityError as e:
            print(f"IntegrityError during bulk creation of stock data: {e}")
            return JsonResponse({'error': 'dbError', 'message': 'Failed to insert stock data due to integrity error.'},
                                status=500)
        # transaction.commit()
        # commit

        stock_data_df = pd.DataFrame.from_records(
            list(StockData.objects.filter(sd_symbol=symbol).values('sd_close_price', 'st_id__st_day')))
        stock_data_df = stock_data_df.rename(columns={'st_id__st_day': 'date', 'sd_close_price': 'close_price'})
        stock_data_df.set_index('date', inplace=True)
        stock_data_df['close_price'] = stock_data_df['close_price'].apply(safe_decimal_conversion)
        stock_data_df = stock_data_df.dropna(subset=['close_price'])

        model = load_model()
        # stock_data_df = stock_data_df.rename(columns={'close_price': 'close'})
        future_predictions = predict_future_stock_prices(model, stock_data_df)
        #
        future_dates = pd.date_range(start=today, periods=30)
        prediction_data_to_insert = []
        for i, future_date in enumerate(future_dates):
            prediction_price = future_predictions[i]

            # Insert or update records with predicted prices
            stock_timeline, created = StockTimeline.objects.get_or_create(st_day=future_date)
            prediction_price = Decimal(prediction_price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            # print(prediction_price)
            # Create new StockData entry with predicted price, leaving other fields as null
            stock_data_entry = StockData(
                sd_symbol=symbol,
                st_id=stock_timeline,
                sd_price_prediction=prediction_price,
                sd_open_price=None,  # Set to null
                sd_close_price=None,  # Set to null
                sd_high_price=None,  # Set to null
                sd_low_price=None,  # Set to null
                sd_volume=None  # Set to null
            )
            prediction_data_to_insert.append(stock_data_entry)

        # Insert the predicted stock data in bulk
        try:
            with transaction.atomic():
                if prediction_data_to_insert:
                    StockData.objects.bulk_create(prediction_data_to_insert)
        except IntegrityError as e:
            print(f"IntegrityError during bulk creation of stock data: {e}")
            return JsonResponse({'error': 'dbError', 'message': 'Failed to insert stock data due to integrity error.'},
                                status=500)

        transaction.commit()
        return JsonResponse({
            'success': f"Stock data for {symbol} has been successfully updated",
            'future_predictions': future_predictions.tolist()
        }, status=200)
        # return JsonResponse({'success': f"Stock data for {symbol} has been successfully updated"}, status=200)
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'error': str(e),
            'message': 'API request failed'
        }, status=500)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'message': 'An error occurred while processing the data'
        }, status=500)