import datetime
import math

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import talib as ta
import torch

from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_num_threads(8)

## Hyperparameters
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 25

D_MODEL = 24
NUM_HEADS = 3
NUM_ENCODER_LAYERS = 6
DROPOUT_RATE = 0.1
LEARNING_RATE = 2e-4

## Parameters
MIN_LATER = 10  # The minute we want to predict in the future
DEVICE = (torch.cuda.is_available() and 'cuda:0') or 'cpu'

FEATURE_NUM=5

class PositionEmbedding(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.linear = torch.nn.Linear(FEATURE_NUM, d_model, dtype=torch.double)

        # Create a matrix of shape (max_len, feature_num)
        pe = torch.zeros(SEQ_LEN, d_model)
        position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.linear(x)
        x = x + self.pe[:, :x.size(1)]
        return x
    
class EnhancedMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int) -> None:
        super(EnhancedMultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len

        self.V_linear = torch.nn.Linear(d_model, d_model, dtype=torch.double)
        self.K_linear = torch.nn.Linear(d_model, d_model, dtype=torch.double)
        self.Q_linear = torch.nn.Linear(d_model, d_model, dtype=torch.double)
        self.output_linear = torch.nn.Linear(d_model, d_model, dtype=torch.double)

        # Multi-Scale Gaussian Prior
        self.gaussian_masks = self.create_gaussian_masks()

    def create_gaussian_masks(self):
        masks = []
        window_sizes = [5, 10, 20, 40]  # Corresponds to different Dh (σ_h)
        for window_size in window_sizes:
            mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.double)
            for i in range(self.seq_len):
                for j in range(self.seq_len):
                    if j <= i:  # Apply Gaussian only if j <= i
                        mask[i, j] = math.exp(-((i - j) ** 2) / (2 * (window_size ** 2)))
                    else:
                        mask[i, j] = 0  # Set to 0 for j > i
            masks.append(mask)
        return torch.stack(masks)


    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply Multi-Scale Gaussian Prior
        batch_size = attention.size(0)
        gaussian_masks = self.gaussian_masks.unsqueeze(0).repeat(batch_size, self.num_heads, 1, 1, 1).to(attention.device)
        attention = attention.unsqueeze(2) + gaussian_masks
        attention = attention.mean(dim=2)  # Average over the different scales

        attention_prob = torch.nn.functional.softmax(attention, dim=-1)
        output = torch.matmul(attention_prob, V)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q = self.split_heads(self.Q_linear(Q))
        K = self.split_heads(self.K_linear(K))
        V = self.split_heads(self.V_linear(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.output_linear(self.concat_heads(attention_output))

        return output

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForward, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=torch.double)
        self.conv1d_2 = torch.nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=torch.double)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        nonlinear_output_1 = self.relu(self.conv1d_1(x))
        output = self.conv1d_2(nonlinear_output_1)
        return output.transpose(1, 2)

class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, seq_len: int) -> None:
        super(Encoder, self).__init__()
        self.multi_head_attention = EnhancedMultiHeadAttention(d_model, num_heads, seq_len)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.norm_1 = torch.nn.LayerNorm(d_model, dtype=torch.double)
        self.norm_2 = torch.nn.LayerNorm(d_model, dtype=torch.double)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_output = self.multi_head_attention(x, x, x)
        feed_forward_input = self.norm_1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(feed_forward_input)
        output = self.norm_2(feed_forward_input + self.dropout(feed_forward_output))
        return output
    
class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout_rate: float, seq_len: int) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model 
        self.position_embedding = PositionEmbedding(d_model)
        self.encoder = torch.nn.ModuleList([Encoder(d_model, num_heads, dropout_rate, seq_len) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.prepare_output = torch.nn.Linear(seq_len * d_model, 64, dtype=torch.double)
        self.relu = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(64, 1, dtype=torch.double)

    def forward(self, x):
        x = self.position_embedding(x)
        encoder_output = x
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output)
        encoder_output = encoder_output.view(encoder_output.shape[0], -1)
        flatten_encoder_output = self.dropout(self.prepare_output(encoder_output))
        output = self.final_layer(flatten_encoder_output)
        return output

#define model
transformer = Transformer(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_ENCODER_LAYERS,
    dropout_rate=DROPOUT_RATE,
    seq_len=SEQ_LEN
).to(device=DEVICE)

criterion = torch.nn.MSELoss().to(device=DEVICE)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4)

#backtest

#%matplotlib inline

# ========== Import Modules ========== #
from sklearn.preprocessing import StandardScaler
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
        # Initialize transformer model
        self.model = Transformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_ENCODER_LAYERS,
            dropout_rate=DROPOUT_RATE,
            seq_len=SEQ_LEN
        )
        self.model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
        self.model.eval()
        self.scaler = StandardScaler()
        
    def prepare_model_input(self):
        # Prepare last SEQ_LEN data points
        data = {
            'open': np.array([self.open_price[i] for i in range(-SEQ_LEN, 0)]),
            'high': np.array([self.high_price[i] for i in range(-SEQ_LEN, 0)]),
            'low': np.array([self.low_price[i] for i in range(-SEQ_LEN, 0)]),
            'close': np.array([self.close_price[i] for i in range(-SEQ_LEN, 0)]),
            'volume': np.array([self.volume[i] for i in range(-SEQ_LEN, 0)])
        }
        
        features = np.stack([data['open'], data['high'], data['low'], data['close'], data['volume']], axis=1)
        scaled_features = self.scaler.fit_transform(features)
        model_input = torch.tensor(scaled_features, dtype=torch.double).unsqueeze(0)
        return model_input

    def get_model_prediction(self):
        with torch.no_grad():
            model_input = self.prepare_model_input()
            prediction = self.model(model_input)
            current_price = self.close_price[0]
            predicted_price = prediction.item()
            return predicted_price > current_price
        
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
        model_prediction = self.get_model_prediction()
        if model_prediction:  # Only go long if model predicts price increase
            self.buy(size=200)
            self.log(f'[Long]\t Open at {self.datetime.datetime() + datetime.timedelta(minutes=1)} with price {self.open_price[1]}')
            self.J_hold_threshold = 80
            self.J_counter = 0
            self.cover_pioneer_signal = signal.HOLD
            self.sl_price = min(self.low_price[i] for i in range(-1, -6, -1))
            
    def do_short(self):
        model_prediction = self.get_model_prediction()
        if not model_prediction:  # Only go short if model predicts price decrease
            self.sell(size=200)
            self.log(f'[Short]\t Open at {self.datetime.datetime() + datetime.timedelta(minutes=1)} with price {self.open_price[1]}')
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
            cover_condition_3 =  not self.get_model_prediction()
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
            cover_condition_3 = self.get_model_prediction()
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
