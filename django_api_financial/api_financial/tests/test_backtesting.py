import unittest
from decimal import Decimal
import pandas as pd
from datetime import datetime

from api_financial.views import prepare_dataframe_for_trades, execute_trades


# from  import execute_trades, prepare_dataframe_for_trades


class TestBacktestingLogic(unittest.TestCase):

    def setUp(self):
        # Set up mock stock data in a pandas DataFrame
        data = {
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
                datetime(2024, 1, 4), datetime(2024, 1, 5)],
            'close_price': [100, 90, 80, 110, 130],
            '50_day_sma': [105, 95, 85, 90, 100],
            '200_day_sma': [110, 108, 107, 106, 105]
        }
        self.stock_data_df = pd.DataFrame(data)
        self.stock_data_df.set_index('date', inplace=True)

        # Prepare data for testing (converts values to Decimal)
        self.stock_data_df = prepare_dataframe_for_trades(self.stock_data_df)

    def test_basic_buy_sell(self):
        """
        Test a simple case where a Buy is triggered when the price dips below
        the 50-day SMA and a Sell is triggered when it rises above the 200-day SMA.
        """
        investment = 10000  # Initial investment of $10,000

        performance = execute_trades(self.stock_data_df, investment)

        # Check if a buy and a sell were triggered
        self.assertEqual(len(performance['trades']), 2)
        self.assertEqual(performance['trades'][0][0], 'Buy')  # First trade is a buy
        self.assertEqual(performance['trades'][1][0], 'Sell')  # Second trade is a sell

        # Validate the total return
        self.assertGreater(performance['total_return'], 0)

    def test_no_trades(self):
        """
        Test a scenario where no trades occur (prices never cross the SMAs).
        """
        # Modify stock data so no trades should happen
        self.stock_data_df['close_price'] = [120, 115, 112, 109, 108]  # Always above SMA

        investment = 10000

        performance = execute_trades(self.stock_data_df, investment)

        # Expect no trades
        self.assertEqual(performance['total_trades'], 0)
        self.assertEqual(performance['total_return'], 0)

    def test_multiple_buy_sell_cycles(self):
        """
        Test multiple buy/sell cycles as the stock price fluctuates across moving averages.
        """
        # Add more data for multiple cycles
        data = {
            'date': [datetime(2024, 1, 6), datetime(2024, 1, 7), datetime(2024, 1, 8),
                     datetime(2024, 1, 9), datetime(2024, 1, 10)],
            'close_price': [70, 60, 80, 140, 150],
            '50_day_sma': [80, 70, 75, 130, 140],
            '200_day_sma': [110, 105, 102, 108, 112]
        }
        new_data_df = pd.DataFrame(data)
        new_data_df.set_index('date', inplace=True)
        self.stock_data_df = pd.concat([self.stock_data_df, new_data_df])
        self.stock_data_df = prepare_dataframe_for_trades(self.stock_data_df)

        investment = 10000

        performance = execute_trades(self.stock_data_df, investment)

        # Multiple buy/sell trades should be triggered
        self.assertGreater(performance['total_trades'], 2)  # More than one trade
        self.assertTrue(performance['total_return'] != 0)  # Validate return is calculated

    def test_single_trade(self):
        """
        Test edge case where only one Buy happens and no Sell opportunity arises.
        """
        # Modify stock data so only one Buy happens
        self.stock_data_df['close_price'] = [90, 85, 80, 75, 70]  # Always below SMA

        investment = 10000

        performance = execute_trades(self.stock_data_df, investment)

        # Expect only one trade (Buy)
        self.assertEqual(performance['total_trades'], 1)
        self.assertEqual(performance['trades'][0][0], 'Buy')  # Ensure first trade is a buy

        # Since no Sell happened, cash should still be 0
        self.assertEqual(performance['final_value'], 0)
        self.assertLess(performance['total_return'], 0)  # Loss since no sell happened


if __name__ == '__main__':
    unittest.main()
