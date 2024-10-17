from django.urls import path
from . import views

app_name = 'api_financial'
urlpatterns = [
    # path('overview/', views.api_overview, name='api-overview'),
    # path('fetch-stock-data/<str:symbol>', views.fetch_stock_data, name='fetch-stock-data'),
    path('fetch-stock-data/q', views.fetch_stock_data, name='fetch-stock-data'),
    path('fetch-stock-data-prediciton/q', views.fetch_stock_data_with_predictions, name='fetch-stock-data-predict'),
    # path('fetch-stock-data?symbol=<str:symbol>/', views.fetch_stock_data, name='fetch-stock-data'),
]

handler404 = 'api_financial.views.custom_page_not_found_view'