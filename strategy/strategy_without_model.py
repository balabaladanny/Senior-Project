# ========== Import Modules ========== #
import datetime
import logging

import backtrader as bt
import numpy as np 
import pandas as pd
import talib as ta

from enum import Enum
from tqdm import tqdm
# ================================================== #

# ========== Hyper parameters and parameters ========== #
BEGIN_TIME = datetime.time(9, 15)
END_TIME = datetime.time(13, 15)

LOG_FILE_PATH = './strategy.log'
# ================================================== #

# ========== Classes ========== #
class signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class strategy(bt.Strategy):
    def __init__(self):        
        self.open_price = self.datas[0].open
        self.high_price = self.datas[0].high
        self.low_price = self.datas[0].low
        self.close_price = self.datas[0].close
        self.volume = self.datas[0].volume
        self.datetime = self.datas[0].datetime

        self.enter_squeeze = False
        self.pioneer_signal = signal.HOLD
        self.cover_pioneer_signal = signal.HOLD
        self.cur_KDJ = None
        self.J_hold_threshold = None
        self.J_counter = 0
        self.sl_price = 0

    def log(self, txt, log_into_file = True):
        print(txt)
        if log_into_file:
            logging.info(txt)

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.log(f'Current time: {self.datetime.datetime().time()}')
            if trade.pnlcomm > 0:
                self.log(f'[Profit]\t Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')
                self.log(f'Current cash: {self.broker.get_cash()}\n')
            else:
                self.log(f'[Loss]\t Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')
                self.log(f'Current cash: {self.broker.get_cash()}\n')

    def is_squeeze(self, indicators):
        return indicators['BBAND_UPPER'] <= indicators['KELTNER_UPPER'] and indicators['BBAND_LOWER'] >= indicators['KELTNER_LOWER']

    def do_long(self):
        self.buy(size = 200)
        self.log(f'[Long]\t Open at {self.datetime.datetime() + datetime.timedelta(minutes = 1)} with price {self.open_price[1]}')
        self.J_hold_threshold = 80
        self.J_counter = 0
        self.cover_pioneer_signal = signal.HOLD
        self.sl_price = min(self.low_price[i] for i in range(-1, -6, -1))

    def do_short(self):
        self.sell(size = 200)
        self.log(f'[Short]\t Open at {self.datetime.datetime() + datetime.timedelta(minutes = 1)} with price {self.open_price[1]}')
        self.J_hold_threshold = 20
        self.J_counter = 0
        self.cover_pioneer_signal = signal.HOLD
        self.sl_price = max(self.high_price[i] for i in range(-1, -6, -1))
        
    def cover_long(self):
        self.close()
        self.log(f'[Long]\t Close at {self.datetime.datetime() + datetime.timedelta(minutes = 1)} with price {self.open_price[1]}')
        self.J_hold_threshold = None
        self.sl_price = 0
        self.pioneer_signal = signal.HOLD
        self.cover_pioneer_signal = signal.HOLD
        self.enter_squeeze = False

    def cover_short(self):
        self.close()
        self.log(f'[Short]\t Close at {self.datetime.datetime() + datetime.timedelta(minutes = 1)} with price {self.open_price[1]}')
        self.J_hold_threshold = None
        self.sl_price = 0
        self.pioneer_signal = signal.HOLD
        self.cover_pioneer_signal = signal.HOLD
        self.enter_squeeze = False

    def next(self):
        # ===== calculate indicators ===== #
        prev_close = np.array([self.close_price[i+1] for i in range(-30, 0)])
        prev_high = np.array([self.high_price[i+1] for i in range(-30, 0)])
        prev_low = np.array([self.low_price[i+1] for i in range(-30, 0)])

        bband_upper, _, bband_lower = ta.BBANDS(prev_close, timeperiod = 15)
        keltner_upper, _, keltner_lower = keltner_bands(
            prev_close,
            prev_high,
            prev_low,
            period = 15, multiplier = 1.5
        )
        self.cur_KDJ = KDJ(prev_high, prev_low, prev_close, 30, 3, 3, self.cur_KDJ)
        indicators = {
            'BBAND_UPPER': bband_upper[-1],
            'BBAND_LOWER': bband_lower[-1],
            "KELTNER_UPPER": keltner_upper[-1],
            "KELTNER_LOWER": keltner_lower[-1],
            "J": self.cur_KDJ['J'],
        }
        # print(indicators)
        # ======================================== #

        # ===== check time and volume ===== #
        cur_time = self.datetime.datetime().time()
        if cur_time < BEGIN_TIME or cur_time > END_TIME or self.volume[0] == 0:
            return

        if cur_time == END_TIME:
            self.enter_squeeze = False
            self.pioneer_signal = signal.HOLD
            self.cover_pioneer_signal = signal.HOLD
            self.J_hold_threshold = None
            self.J_counter = 0
            self.sl_price = 0

            if self.position.size != 0:
                self.close(size = self.position.size)
                self.log(f'Close position at {cur_time}')
                self.log(f'Since the market is about to close')
            return
        # ======================================== #

        # ===== trading logic ===== #
        if self.position.size == 0:
            if self.is_squeeze(indicators):
                if self.enter_squeeze == False:     # First time enter squeeze
                    self.pioneer_signal = signal.HOLD
                self.enter_squeeze = True

                do_long_precondition_1 = self.close_price[0] < indicators['BBAND_LOWER']
                do_long_precondition_2 = indicators['J'] < 20
                if do_long_precondition_1 and do_long_precondition_2:
                    self.pioneer_signal = signal.BUY
                    return

                do_short_precondition_1 = self.close_price[0] > indicators['BBAND_UPPER']
                do_short_precondition_2 = indicators['J'] > 80
                if do_short_precondition_1 and do_short_precondition_2:
                    self.pioneer_signal = signal.SELL
                    return

                do_long_condition_1 = self.pioneer_signal == signal.BUY
                do_long_condition_2 = self.open_price[0] < self.close_price[0] 
                do_long_condition_3 = indicators['J'] > 20
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3:
                    self.do_long()
                    return
                
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3:
                    self.do_short()
                    return

            else:
                self.enter_squeeze = False

                do_long_condition_1 = self.pioneer_signal == signal.BUY
                do_long_condition_2 = self.open_price[0] < self.close_price[0] 
                do_long_condition_3 = indicators['J'] > 20
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3:
                    self.do_long()
                    return
                
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3:
                    self.do_short()
                    return

        elif self.position.size > 0:
            if indicators['J'] > self.J_hold_threshold + 5:
                self.J_hold_threshold = (indicators['J'] // 5) * 5 # Update the threshold

            cover_precondition_1 = indicators['J'] > 80
            cover_precondition_2 = self.cover_pioneer_signal == signal.HOLD
            if cover_precondition_1 and cover_precondition_2:
                self.cover_pioneer_signal = signal.SELL
                return
            
            cover_condition_1 = indicators['J'] <= self.J_hold_threshold
            cover_condition_2 = self.cover_pioneer_signal == signal.SELL
            # cover_condition_3 = any(self.model_result) if self.model is not None else True
            cover_condition_3 = True
            if cover_condition_1 and cover_condition_2 and cover_condition_3:
                self.cover_long()
                return
            
            sl_condition = self.close_price[0] < self.sl_price
            if sl_condition:
                self.cover_long()
                self.log(f'Stop loss at {self.datetime.datetime()}')
                return

        elif self.position.size < 0:
            if indicators['J'] < self.J_hold_threshold - 5:
                self.J_hold_threshold = (indicators['J'] // 5) * 5 # Update the threshold

            cover_precondition_1 = indicators['J'] < 20
            cover_precondition_2 = self.cover_pioneer_signal == signal.HOLD
            if cover_precondition_1 and cover_precondition_2:
                self.cover_pioneer_signal = signal.BUY
                return
            
            cover_condition_1 = indicators['J'] >= self.J_hold_threshold
            cover_condition_2 = self.cover_pioneer_signal == signal.BUY
            # cover_condition_3 = not any(self.model_result) if self.model is not None else True
            cover_condition_3 = True
            if cover_condition_1 and cover_condition_2 and cover_condition_3:
                self.cover_short()
                return
            
            sl_condition = self.close_price[0] > self.sl_price
            if sl_condition:
                self.cover_short()
                self.log(f'Stop loss at {self.datetime.datetime()}')
                return
        # ======================================== #
        
        return
# ================================================== #


# ========== Functions ========== #
def keltner_bands(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int, multiplier: int) -> tuple:
    mid = ta.EMA(close, timeperiod = period)
    # mid = np.nan_to_num(mid, nan = mid.iloc[period - 1])
    kelt_trange = np.array([])

    for i in range(len(close)):
        tem_trange = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i - 1]),
            abs(low[-i] - close[-i - 1])
        )
        kelt_trange = np.append(tem_trange, kelt_trange)
    # kelt_trange = np.append(high[0] - low[0], kelt_trange)
    atr = ta.EMA(kelt_trange, timeperiod = period)
    # atr = np.nan_to_num(atr, nan = atr[period - 1])
    upper = mid + atr * multiplier
    lower = mid - atr * multiplier

    return upper, mid, lower

def KDJ(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, signal_k: int, signal_d: int, prev_data: dict) -> tuple:
    # high = high[-period + 1 : ]
    # low = low[-period + 1 : ]
    # close = close[-period + 1 : ]
    RSV = int(((close[-1] - np.min(low[-period:])) / (np.max(high[-period:]) - np.min(low[-period:])) * 100) + 0.5)

    _alpha_k = 2 / (signal_k + 1)
    _alpha_d = 2 / (signal_d + 1)
    prev_k = prev_data['K'] if prev_data is not None else 50
    prev_d = prev_data['D'] if prev_data is not None else 50

    K = int((_alpha_k * ((prev_k + 2 * RSV) / 3) + (1 - _alpha_k) * prev_k) + 0.5)
    D = int((_alpha_d * ((prev_d + 2 * K) / 3) + (1 - _alpha_d) * prev_d) + 0.5)
    J = 3 * K - 2 * D

    return {'K': K, 'D': D, 'J': J}

def load_and_check(data_path: str, start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    begin_time = datetime.time(8, 46)
    end_time = datetime.time(13, 45)

    data = pd.read_csv(data_path, dtype = {
        'datetime': str,
        'open': np.int16,
        'high': np.int16,
        'low': np.int16,
        'close': np.int16,
        'volume': np.int16
    }, index_col = 0)
    data.index = pd.to_datetime(data.datetime)
    data = data.between_time(begin_time, end_time)
    # data = data.between_time(BEGIN_TIME, END_TIME)

    if start_date is not None:
        data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
 
    IS_MISSING_DATA = False
    for i in tqdm(range(1, len(data.index)), desc = 'Check missing data'):
        if data.index[i].date() - data.index[i - 1].date() == datetime.timedelta(days = 1) and \
           data.index[i] - data.index[i - 1] != datetime.timedelta(hours = 19, minutes = 1, seconds = 0):
            if IS_MISSING_DATA == False:
                IS_MISSING_DATA = True
                logging.info('Not continuous time: ')
                
            logging.info(f'Missing data between {data.index[i - 1]} and {data.index[i]}')
            logging.info(f'{data.index[i].date() - data.index[i - 1].date()}, {data.index[i] - data.index[i - 1]}')
    if IS_MISSING_DATA:
        raise ValueError(f'Missing data in the data at {data_path}')

    for i in tqdm(range(1, len(data.index)), desc = 'Check missing data in day'):
        if (data.index[i] - data.index[i - 1] != datetime.timedelta(minutes = 1) and data.index[i].time() != begin_time and data.index[i - 1].time() != end_time):
                if IS_MISSING_DATA == False:
                    IS_MISSING_DATA = True
                    logging.info('Not continuous time: ')
                    
                logging.info(f'Missing data between {data.index[i - 1]} and {data.index[i]}')
    if IS_MISSING_DATA:
        raise ValueError(f'Missing data in the data at {data_path}')

    logging.info(f"Succeed to load and check the data at '{data_path}'")
    return data

def log_and_print(msg = ''):
    logging.info(msg)
    print(msg)
# ================================================== #


if __name__ == '__main__':
    # ===== Initialize the log file ===== #
    with open(LOG_FILE_PATH, 'w') as f:
        f.write('')
    logging.basicConfig(filename = LOG_FILE_PATH, filemode = 'a', format = '%(asctime)s [%(levelname)s] %(message)s', level = logging.INFO)
    # ======================================== #

    # ===== Load with backtesting data ===== #
    log_and_print('Start to load test data')
    TXF_test = load_and_check('../TXFR1_1min_backtesting.csv', start_date = datetime.date(2022, 1, 1), end_date = datetime.date(2024, 8, 16))
    log_and_print(f'Succeed to load test data, the length of test data is {len(TXF_test)}')
    log_and_print('=' * 50)
    # ======================================== #


    # ===== start backtesting ===== #
    cerebro = bt.Cerebro()

    log_and_print('Add data to cerebro')
    cerebro.adddata(bt.feeds.PandasData(dataname = TXF_test))
    cerebro.addstrategy(strategy)
    
    analyzers = cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name = 'drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name = 'returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name = 'sqn')

    log_and_print('Set cash and commission')
    cerebro.broker.setcash(500_000)
    cerebro.broker.setcommission(commission = 0.00025, margin = 0.05, mult = 1)

    log_and_print('Start to backtesting')
    log_and_print('=' * 50)
    result = cerebro.run()

    log_and_print('Trade Analyzer: ')
    for key, value in result[0].analyzers.trade_analyzer.get_analysis().items():
        log_and_print(f'{key}: {value}')
    log_and_print()

    log_and_print('Sharpe Ratio Analyzer: ')
    for key, value in result[0].analyzers.sharpe_ratio.get_analysis().items():
        log_and_print(f'{key}: {value}')
    log_and_print()


    log_and_print('Drawdown Analyzer: ')
    for key, value in result[0].analyzers.drawdown.get_analysis().items():
        log_and_print(f'{key}: {value}')
    log_and_print()

    log_and_print('Returns Analyzer: ')
    for key, value in result[0].analyzers.returns.get_analysis().items():
        log_and_print(f'{key}: {value}')
    log_and_print()

    log_and_print('SQN Analyzer: ')
    for key, value in result[0].analyzers.sqn.get_analysis().items():
        log_and_print(f'{key}: {value}')
    log_and_print()

    cerebro.plot()
    log_and_print('Succeed to backtesting')
    # ======================================== #
    
