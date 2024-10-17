# Django Financial API Project

This project is a Django-based financial API that allows users to fetch stock data from Alpha Vantage, calculate financial metrics, and generate stock performance reports, including PDF outputs.

## Features

- Fetch stock data from Alpha Vantage.
- Perform financial backtesting and generate reports.
- Provide stock data visualization.
- Support JSON and PDF output formats for stock data.
- Dockerized for easy deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [API Routes and Usage](#api-routes-and-usage)
4. [Environment Variables](#environment-variables)
5. [Running with Docker](#running-with-docker)
6. [Running Migrations](#running-migrations)
7. [Error Handling](#error-handling)
8. [Deploying to DigitalOcean](#deploying-to-digitalocean)
9. [License](#license)

---

## Prerequisites

Before running this project, ensure that you have the following installed:

- **Python 3.10** or higher
- **Docker** and **Docker Compose**
- **Alpha Vantage API Key** for fetching stock data

---

## Installation

### Local Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/django-api-financial.git
   cd django-api-financial

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Set up environment variables**:\
   Create a .env file in the django-api-financial directory and add your API keys:
   ```bash
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
   DJANGO_SECRET_KEY=your_django_secret_key
   DJANGO_DEBUG=False
   DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,coral-app-pcdhu.ondigitalocean.app
   PORT=8000

## API Routes and Usage

### Fetch Stock Data
   Example: Fetch Stock Data in JSON
   ```bash
   curl "http://localhost:8000/fetch-stock-data/q?symbol=IBM&investment=100000"
   ```
   Example: Fetch Stock Data Prediction(30 days) in JSON
   ```bash
   curl "http://localhost:8000/fetch-stock-data-prediciton/q?symbol=IBM&investment=100000"
   ```
   Example: Fetch Stock Data in JSON
   ```bash
   curl -o stock_report.pdf "http://localhost:8000/fetch-stock-data/q?symbol=IBM&investment=100000&isPDF=true"
   ```
   Example JSON Response
   ```bash
   {
      "success": "Stock data for IBM has been successfully updated",
      "total_return": 12.34,
      "total_trades": 15,
      "final_portfolio_value": 11345.00,
      "trades": [
        {
          "action": "buy",
          "date": "2024-05-08",
          "price": 169.90,
          "value": 10000
        },
        {
          "action": "sell",
          "date": "2024-05-09",
          "price": 166.27,
          "value": 10123
        }
      ],
      "max_drawdown": 5.67
   }
   ```

## Environment Variables

- ALPHAVANTAGE_API_KEY:  Your Alpha Vantage API key for fetching stock data.
- DJANGO_SECRET_KEY:  Secret key for Django.
- DJANGO_PORT/PORT:  The port on which the app runs (default: 8000).
- DB_NAME: database name
- DB_PASSWORD: password database
- DB_HOST: host of the database
- DB_PORT: port of the database
- DB_USER: user of the database

## Running with Docker
The project is Dockerized for easier deployment.
1. **Build Docker Image**:
   ```bash
   docker compose build
2. **Start the Application**:
   ```bash
   docker compose up
3. **Start the Application**:
   ```bash
   docker compose down

The application will run at http://localhost:8000.

## Running Migrations
Ensure the database is set up by running the following command after Docker setup:
   ```bash
   docker-compose run django python django_api_financial/manage.py migrate
   ```

## Error Handling
When a requested route does not exist, the API returns a 404 JSON response:
   ```bash
    {
      "error": "not_found",
      "message": "The requested resource was not found"
    }
   ```

## Deploying to DigitalOcean
### DigitalOcean App Platform Setup:
1. Ensure the .env file contains valid API keys and allowed hosts for the DigitalOcean domain.
2. Set up your app on the DigitalOcean App Platform.
3. Ensure that the PORT environment variable is properly set in the DigitalOcean settings.

### Configuration for Production
For production, you'll run the application using Gunicorn, a production-grade WSGI HTTP server.

## License
This project is licensed under the MIT License.

Feel free to adjust any part of the file to suit your specific needs or setup. Let me know if you need further modifications.
