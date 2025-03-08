import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*convert_dtype parameter.*')

import os
import sys
import torch
import backtrader as bt
import numpy as np
import pandas as pd
import talib as ta
import datetime
from enum import Enum

sys.path.append(os.path.dirname(__file__))

from model import Informer 
from utils.timefeatures import time_features  

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#設定交易時間
BEGIN_TIME = datetime.time(9, 15)
END_TIME = datetime.time(13, 15)

class signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class strategy(bt.Strategy):
    def __init__(self, model_path=None):
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

        #加載模型
        if model_path:
            self.model = Informer(
                enc_in=5, dec_in=5, c_out=1, seq_len=120, label_len=60, out_len=60,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                d_ff=2048, dropout=0.05, attn='prob', embed='timeF', freq='t',
                activation='gelu', output_attention=False, distil=True, mix=True,
                device=torch.device('cpu')
            )
            self.label_len = 60
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only = True))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            self.model = None

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.log(f'Current time: {self.datetime.datetime().time()}')
            if trade.pnlcomm > 0:
                self.log(f'[Profit]\t Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')
                self.log(f'Current cash: {self.broker.get_cash()}\n')
            else:
                self.log(f'[Loss]\t Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')
                self.log(f'Current cash: {self.broker.get_cash()}\n')

    def get_model_prediction(self, input_data, time_enc, time_dec):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        time_enc_tensor = torch.tensor(time_enc, dtype=torch.float32).unsqueeze(0)
        time_dec_tensor = torch.tensor(time_dec, dtype=torch.float32).unsqueeze(0)
        dec_input = input_tensor[:, -self.label_len:, :]
        with torch.no_grad():
            predictions = self.model(input_tensor, time_enc_tensor, dec_input, time_dec_tensor)
        predicted_1min = predictions[0, 0].item()
        predicted_30min = predictions[0, 29].item()
        return predicted_1min, predicted_30min

    def next(self):
        # ===== 計算指標 ===== #
        prev_close = np.array([self.close_price[i + 1] for i in range(-30, 0)])
        prev_high = np.array([self.high_price[i + 1] for i in range(-30, 0)])
        prev_low = np.array([self.low_price[i + 1] for i in range(-30, 0)])

        bband_upper, _, bband_lower = ta.BBANDS(prev_close, timeperiod=15)
        keltner_upper, _, keltner_lower = self.keltner_bands(prev_close, prev_high, prev_low, period=15, multiplier=1.5)
        self.cur_KDJ = self.KDJ(prev_high, prev_low, prev_close, 30, 3, 3, self.cur_KDJ)
        indicators = {
            'BBAND_UPPER': bband_upper[-1],
            'BBAND_LOWER': bband_lower[-1],
            "KELTNER_UPPER": keltner_upper[-1],
            "KELTNER_LOWER": keltner_lower[-1],
            "J": self.cur_KDJ['J'],
        }
        # ======================================== #

        # ===== 檢查時間和成交量 ===== #
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
                self.close(size=self.position.size)
                self.log(f'Close position at {cur_time}')
                self.log(f'Since the market is about to close')
            return
        # ======================================== #

        #使用模型生成預測
        if self.model:
            input_data = [[self.open_price[0], self.high_price[0], self.low_price[0], self.close_price[0], self.volume[0]]]
            time_enc = self.generate_time_features(seq_len=120)
            time_dec = self.generate_time_features(seq_len=60)
            predicted_1min, predicted_30min = self.get_model_prediction(input_data, time_enc, time_dec)
        else:
            predicted_1min, predicted_30min = None, None
        predicted_1min, predicted_30min
        # ===== 交易邏輯 ===== #
        if self.position.size == 0:
            if self.is_squeeze(indicators):
                if not self.enter_squeeze:
                    self.pioneer_signal = signal.HOLD
                self.enter_squeeze = True

                do_long_precondition_1 = self.close_price[0] < indicators['BBAND_LOWER']
                do_long_precondition_2 = indicators['J'] < 20
                do_long_precondition_3 = predicted_30min - predicted_1min
                if do_long_precondition_1 and do_long_precondition_2 and do_long_precondition_3:
                    self.pioneer_signal = signal.BUY
                    return

                do_short_precondition_1 = self.close_price[0] > indicators['BBAND_UPPER']
                do_short_precondition_2 = indicators['J'] > 80
                do_short_precondition_3 = -predicted_1min + predicted_30min
                if do_short_precondition_1 and do_short_precondition_2 and do_short_precondition_3:
                    self.pioneer_signal = signal.SELL
                    return

                #long
                do_long_condition_1 = self.pioneer_signal == signal.BUY
                do_long_condition_2 = self.open_price[0] < self.close_price[0]
                do_long_condition_3 = indicators['J'] > 20
                do_long_condition_4 = predicted_30min - predicted_1min
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3 and do_long_condition_4:
                    self.do_long()
                    return

                #short
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                do_short_condition_4 = -predicted_1min + predicted_30min

                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3 and do_short_condition_4:
                    self.do_short()
                    return

            else:
                self.enter_squeeze = False

                #long
                do_long_condition_1 = self.pioneer_signal == signal.BUY
                do_long_condition_2 = self.open_price[0] < self.close_price[0]
                do_long_condition_3 = indicators['J'] > 20
                do_long_condition_4 = predicted_30min - predicted_1min
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3 and do_long_condition_4:
                    self.do_long()
                    return

                #short
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                do_short_condition_4 = -predicted_1min + predicted_30min
                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3 and do_short_condition_4:
                    self.do_short()
                    return

        elif self.position.size > 0:

            entry_price = self.position.price
            current_price = self.close_price[0]
            loss_percentage = (entry_price - current_price) / entry_price * 100
            if loss_percentage >= 4:
                self.close()
                return
            
            if self.J_hold_threshold is None:
                self.J_hold_threshold = indicators['J']

            if indicators['J'] > self.J_hold_threshold + 5:
                self.J_hold_threshold = (indicators['J'] // 5) * 5

            cover_precondition_1 = indicators['J'] > 80
            cover_precondition_2 = self.cover_pioneer_signal == signal.HOLD
            if cover_precondition_1 and cover_precondition_2:
                self.cover_pioneer_signal = signal.SELL
                return

            cover_condition_1 = indicators['J'] <= self.J_hold_threshold
            cover_condition_2 = self.cover_pioneer_signal == signal.SELL
            if cover_condition_1 and cover_condition_2:
                self.cover_long()
                return

            sl_condition = self.close_price[0] < self.sl_price
            if sl_condition:
                self.cover_long()
                self.log(f'Stop loss at {self.datetime.datetime()}')
                return

        elif self.position.size < 0:
            entry_price = self.position.price
            current_price = self.close_price[0]
            loss_percentage = (current_price - entry_price) / entry_price * 100


            if loss_percentage >= 4:
                self.close()
                return
            
            if self.J_hold_threshold is None:
                self.J_hold_threshold = indicators['J']

            if indicators['J'] < self.J_hold_threshold - 5:
                self.J_hold_threshold = (indicators['J'] // 5) * 5

            cover_precondition_1 = indicators['J'] < 20
            cover_precondition_2 = self.cover_pioneer_signal == signal.HOLD
            if cover_precondition_1 and cover_precondition_2:
                self.cover_pioneer_signal = signal.BUY
                return

            cover_condition_1 = indicators['J'] >= self.J_hold_threshold
            cover_condition_2 = self.cover_pioneer_signal == signal.BUY
            if cover_condition_1 and cover_condition_2:
                self.cover_short()
                return

            sl_condition = self.close_price[0] > self.sl_price
            if sl_condition:
                self.cover_short()
                self.log(f'Stop loss at {self.datetime.datetime()}')
                return




    def log(self, txt):
        print(txt)

    def do_long(self):
        self.buy(size=200)
        self.log(f"[Long] Open at {self.datetime.datetime()} with price {self.open_price[0]}")

    def do_short(self):
        self.sell(size=200)
        self.log(f"[Short] Open at {self.datetime.datetime()} with price {self.open_price[0]}")

    def cover_long(self):
        self.close()
        self.log(f"[Long] Close at {self.datetime.datetime()}")

    def cover_short(self):
        self.close()
        self.log(f"[Short] Close at {self.datetime.datetime()}")

    def is_squeeze(self, indicators):
        return indicators['BBAND_UPPER'] <= indicators['KELTNER_UPPER'] and indicators['BBAND_LOWER'] >= indicators['KELTNER_LOWER']

    def generate_time_features(self, seq_len):
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp['date'] = [bt.num2date(self.datetime[-i]) for i in range(seq_len)]
        return time_features(df_stamp, timeenc=0, freq='t')

    def keltner_bands(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int, multiplier: int) -> tuple:
        mid = ta.EMA(close, timeperiod=period)
        kelt_trange = np.array([])

        for i in range(len(close)):
            tem_trange = max(
                high[-i] - low[-i],
                abs(high[-i] - close[-i - 1]),
                abs(low[-i] - close[-i - 1])
            )
            kelt_trange = np.append(tem_trange, kelt_trange)
        atr = ta.EMA(kelt_trange, timeperiod=period)
        upper = mid + atr * multiplier
        lower = mid - atr * multiplier

        return upper, mid, lower


    def KDJ(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, signal_k: int, signal_d: int, prev_data: dict) -> dict:
        RSV = int(((close[-1] - np.min(low[-period:])) / (np.max(high[-period:]) - np.min(low[-period:])) * 100) + 0.5) if np.max(high[-period:]) != np.min(low[-period:]) else 0

        _alpha_k = 2 / (signal_k + 1)
        _alpha_d = 2 / (signal_d + 1)
        prev_k = prev_data['K'] if prev_data is not None else 50
        prev_d = prev_data['D'] if prev_data is not None else 50

        K = int((_alpha_k * ((prev_k + 2 * RSV) / 3) + (1 - _alpha_k) * prev_k) + 0.5)
        D = int((_alpha_d * ((prev_d + 2 * K) / 3) + (1 - _alpha_d) * prev_d) + 0.5)
        J = 3 * K - 2 * D

        return {'K': K, 'D': D, 'J': J}


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    data_path = '../TXFR1_1min_backtesting.csv'

    TXF_test = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    TXF_test = TXF_test[TXF_test.index >= '2022-01-01']
    data_feed = bt.feeds.PandasData(dataname=TXF_test)
    cerebro.adddata(data_feed)

    model_path = 'informer_custom_ftMS_sl120_ll60_pl60_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_mxTrue_TaiwanFuturesPrediction_final_model.pth'

    
    cerebro.addstrategy(strategy, model_path=model_path)
    
    cerebro.broker.setcash(500_000)
    cerebro.broker.setcommission(commission=0.00025, margin=0.05, mult=1)
    
    analyzers = cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name = 'drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name = 'returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name = 'sqn')
    
    print("Start backtesting")
    result = cerebro.run()

    print("Backtest completed. Trade analysis results:")

    print('Trade Analyzer: ')
    for key, value in result[0].analyzers.trade_analyzer.get_analysis().items():
        print(f'{key}: {value}')
    print()

    print('Sharpe Ratio Analyzer: ')
    for key, value in result[0].analyzers.sharpe_ratio.get_analysis().items():
        print(f'{key}: {value}')
    print()

    print('Drawdown Analyzer: ')
    for key, value in result[0].analyzers.drawdown.get_analysis().items():
        print(f'{key}: {value}')
    print()

    print('Returns Analyzer: ')
    for key, value in result[0].analyzers.returns.get_analysis().items():
        print(f'{key}: {value}')
    print()

    print('SQN Analyzer: ')
    for key, value in result[0].analyzers.sqn.get_analysis().items():
        print(f'{key}: {value}')
    print()

    cerebro.plot()
    print('Succeed to backtesting')
    # fig = cerebro.plot()[0][0]
    # fig.savefig('backtest_result.png')
    # print("回測結果已保存為 'backtest_result.png'")
