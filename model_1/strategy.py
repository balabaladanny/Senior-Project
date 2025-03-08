# ========== Import Modules ========== #
import datetime
import logging
import math
import os

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import talib as ta
import torch

from enum import Enum
from sklearn.model_selection import KFold
import torch.utils
from tqdm import tqdm
# ================================================== #

# ========== Hyper parameters and parameters ========== #
## Hyperparameters
SEQ_LEN = int(120)
BATCH_SIZE = int(64)
EPOCHS = 20
D_MODEL = 21
NUM_HEADS = 3
NUM_ENCODER_LAYERS = 3
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
K = 5
PATIENCE = 2
MIN_DELTA_PERCENTAGE = 0.075

## Parameters
MIN_LATER = 15  # The minute we want to predict in the future
DEVICE = (torch.cuda.is_available() and 'cuda:0') or 'cpu'

BEGIN_TIME = datetime.time(9, 15)
END_TIME = datetime.time(13, 15)

TQDM_LEAVE = True
WANDB_RELATED = True

LOG_FILE_PATH = './strategy.log'
MODEL_PATH = './model.pth'

# logging flag
first_batch = True
# ================================================== #

# ========== Classes ========== #
class PositionEmbedding(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.linear = torch.nn.Linear(d_model, d_model, dtype = torch.double)

        # Create a matrix of shape (max_len, feature_num)
        pe = torch.zeros(SEQ_LEN, d_model)
        position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(1000.0) / d_model // 2))
        pe[:, 0:d_model // 2:2] = torch.sin(position * div_term)
        
        div_term = torch.exp(torch.arange(1, d_model // 2, 2).float() * (-math.log(1000.0) / d_model // 2))
        pe[:, 1:d_model // 2:2] = torch.cos(position * div_term[ : d_model // 2])
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x, validate):
        # if first_batch and not validate:
        #     logging.info(f'    PositionEmbedding output[0][0]: {[num for num in x[0][0].tolist()]}')

        x = self.linear(x)

        # if first_batch and not validate:
        #     logging.info(f'    PositionEmbedding output[0][0]: {[num for num in x[0][0].tolist()]}')

        x = x + self.pe[:, :x.size(1)]
        # if first_batch and not validate:
        #     logging.info(f'    PositionEmbedding output[0][0]: {[num for num in x[0][0].tolist()]}')
        return x

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, q, k, v):
        # q, k, v: [batch_size, num_head, seq_len, d_k]
        _, _, _, d_k = q.size()

        # k: [batch_size, num_head, seq_len, d_k] -> k: [batch_size, num_head, d_k, seq_len]
        k_t = k.transpose(-2, -1)
        attention = (q @ k_t) / math.sqrt(d_k)

        attention = self.softmax(attention)
        v = attention @ v

        return v

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        
        # d_model should be divisible by num_heads
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.V_linear = torch.nn.Linear(self.d_model, self.d_model, dtype = torch.double)
        self.K_linear = torch.nn.Linear(self.d_model, self.d_model, dtype = torch.double)
        self.Q_linear = torch.nn.Linear(self.d_model, self.d_model, dtype = torch.double)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_model)
        self.output_linear = torch.nn.Linear(self.d_model, self.d_model, dtype = torch.double)
    
    def split_heads(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, d_model = x.size()

        # x: [batch_size, seq_len, d_model] -> x: [batch_size, num_heads, seq_len, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        return x
    
    def concat_heads(self, x: torch.tensor) -> torch.tensor:
        batch_size, _, seq_len, d_k = x.size()

        # x: [batch_size, num_heads, seq_len, d_k] -> x: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return x
    
    def forward(self, Q: torch.tensor, K: torch.tensor, V: torch.tensor, validate: bool) -> torch.tensor:
        # if first_batch and not validate:
        #     logging.info(f'    linaer_weight: {self.Q_linear.weight.data}')
        #     logging.info(f'    weight: {self.Q_linear.weight.data == 0}')
        #     logging.info(f'{torch.all(self.Q_linear.weight.data == 0)}')
        #     logging.info(f'    Q[0][0]: {[num for num in Q[0][0].tolist()]}')
        #     logging.info(f'    K[0][0]: {[num for num in K[0][0].tolist()]}')
        #     logging.info(f'    V[0][0]: {[num for num in V[0][0].tolist()]}')
        #     logging.info(f'    Q.shape: {Q.shape}')
        #     logging.info(f'    d_model: {self.d_model}')

        assert not torch.all(self.Q_linear.weight.data == 0), "Q_linear weight is all zero"

        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        V = self.V_linear(V)
        # if first_batch and not validate:
        #     logging.info(f'    Q[0][0]: {[num for num in Q[0][0].tolist()]}')
        #     logging.info(f'    K[0][0]: {[num for num in K[0][0].tolist()]}')
        #     logging.info(f'    V[0][0]: {[num for num in V[0][0].tolist()]}')
        # print("-> Split head succeeded!")

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # if first_batch and not validate:
        #     logging.info(f'    Q\'[0][0][0]: {[num for num in Q[0][0][0].tolist()]}')
        #     logging.info(f'    K\'[0][0][0]: {[num for num in K[0][0][0].tolist()]}')
        #     logging.info(f'    V\'[0][0][0]: {[num for num in V[0][0][0].tolist()]}')

        V_out = self.scaled_dot_product_attention(Q, K, V)
        # if first_batch and not validate:
        #     logging.info(f'    Scaled dot product attention output[0][0]: {[num for num in V_out[0][0].tolist()]}')
        # print("-> Scaled dot product attention succeeded!")

        output = self.output_linear(self.concat_heads(V_out))
        # if first_batch and not validate:
        #     logging.info(f'    MultiHeadAttention output[0][0]: {[num for num in output[0][0].tolist()]}')
        # print("-> Concat head succeeded!")
        

        return output

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForward, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(d_model, d_ff, dtype = torch.double)
        self.fully_connected_2 = torch.nn.Linear(d_ff, d_model, dtype = torch.double)
        # self.conv1d_1 = torch.nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = 1, dtype = torch.double)
        # self.conv1d_2 = torch.nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = 1, dtype = torch.double)
        self.dropout = torch.nn.Dropout(DROPOUT_RATE)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.tensor, validate: bool) -> torch.tensor:
        # x: [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        # x = x.transpose(1, 2)
        # nonliear_output_1 = self.relu(self.conv1d_1(x))
        # output = self.conv1d_2(nonliear_output_1)
        output = self.fully_connected_2(self.dropout(self.gelu(self.fully_connected_1(x))))

        # if first_batch and not validate:
        #     logging.info(f'    FeedForward output[0][0]: {[num for num in output[0][0].tolist()]}')
        # return output.transpose(1, 2)
        return output

class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: int) -> None:
        super(Encoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.norm_1 = torch.nn.LayerNorm(d_model, dtype = torch.double)
        self.norm_2 = torch.nn.LayerNorm(d_model, dtype = torch.double)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x, validate):
        # print("Start to encode...")
        attention_output = self.multi_head_attention(x, x, x, validate)
        # print("Get the attention output!\n")

        # print("Start to feed forward...")
        feed_forward_input = self.norm_1(x + self.dropout1(attention_output))
        feed_forward_output = self.feed_forward(feed_forward_input, validate)
        output = self.norm_2(feed_forward_input + self.dropout2(feed_forward_output))
        # print("Get the feed forward output!\n")
        
        # if first_batch and not validate:
        #     logging.info(f'    Encoder output[0][0]: {[num for num in output[0][0].tolist()]}')
        return output

class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout_rate: int) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.position_embedding = PositionEmbedding(d_model)
        self.encoder = torch.nn.ModuleList([Encoder(d_model, num_heads, dropout_rate) for _ in range(num_layers)])
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.prepare_output = torch.nn.Linear(SEQ_LEN * D_MODEL, 1, dtype = torch.double)
        # self.prepare_output = torch.nn.Linear(SEQ_LEN * D_MODEL, math.floor(math.sqrt(SEQ_LEN * D_MODEL)), dtype = torch.double)
        # self.relu1 = torch.nn.GELU()
        # self.dropout2 = torch.nn.Dropout(dropout_rate)
        # self.flatten1 = torch.nn.Linear(math.floor(math.sqrt(SEQ_LEN * D_MODEL)), 1, dtype = torch.double)
        # self.gelu = torch.nn.GELU()
        # self.dropout3 = torch.nn.Dropout(dropout_rate)
        # self.relu2 = torch.nn.LeakyReLU()
        # self.flatten2 = torch.nn.Linear(64, 1, dtype = torch.double)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, validate):
        x = self.position_embedding(x, validate)
        # if first_batch and not validate:
        #     logging.info(f'    PositionEmbedding output[0][0]: {[num for num in x[0][0].tolist()]}')

        encoder_output = x
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, validate)

        encoder_output = encoder_output.view(encoder_output.shape[0], -1)

        # [batch_size, seq_len * d_model] -> [batch_size, 1]
        flatten_encoder_output = self.dropout1(self.prepare_output(encoder_output))
        flatten_encoder_output = self.sigmoid(flatten_encoder_output)
        # flatten_encoder_output = self.flatten1(self.dropout2(self.relu1(flatten_encoder_output)))
        # flatten_encoder_output = self.sigmoid(flatten_encoder_output)
        # if first_batch and not validate:
        #     logging.info(f'    encoder_output[0]: {[num for num in encoder_output[0].tolist()]}')
        #     logging.info(f'    flatten_encoder_output: {[num[0] for num in flatten_encoder_output.tolist()]}')
        return flatten_encoder_output

        # tem_output = self.relu1(self.flatten1(flatten_encoder_output))
        # # tem_output = self.relu2(self.flatten2(tem_output))
        # output = self.sigmoid(tem_output)
        # if first_batch and not validate:
        #     logging.info(f'    encoder_output[0]: {[num for num in encoder_output[0].tolist()]}')
        #     logging.info(f'    flatten_encoder_output[0]: {[num for num in flatten_encoder_output[0].tolist()]}')
        #     logging.info(f'    tem_output: {[num for num in tem_output.tolist()]}')
        #     logging.info(f'    output: {[num for num in output.tolist()]}')

        # return output

class signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class strategy(bt.Strategy):
    def __init__(self, model, threshold):        
        self.model = model
        self.threshold = threshold
        self.output_len = 15
        self.output_list = [None] * self.output_len

        self.KDJ_model_input = None
        self.J_model_input = [None] * SEQ_LEN

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

    def get_model_input(self):
        def minmax(data):
            return (data - data.min()) / (data.max() - data.min())

        Open = np.array([self.open_price[i + 1] for i in range(-SEQ_LEN - 30, 0)])
        High = np.array([self.high_price[i + 1] for i in range(-SEQ_LEN - 30, 0)])
        Low = np.array([self.low_price[i + 1] for i in range(-SEQ_LEN - 30, 0)])
        Close = np.array([self.close_price[i + 1] for i in range(-SEQ_LEN - 30, 0)])
        Volume = np.array([self.volume[i + 1] for i in range(-SEQ_LEN - 30, 0)])

        sma_short = ta.SMA(Close, 3); sma_long = ta.SMA(Close, 15)
        bias_short = (Close[-SEQ_LEN : ] - sma_short[-SEQ_LEN : ]) / sma_short[-SEQ_LEN : ]
        bias_long = (Close[-SEQ_LEN : ] - sma_long[-SEQ_LEN : ]) / sma_long[-SEQ_LEN : ]

        adosc = ta.ADOSC(High, Low, Close, Volume, fastperiod = 6, slowperiod = 15)
        bband_upper, bband_mid, bband_lower = ta.BBANDS(Close, timeperiod = 20)
        keltner_upper, keltner_mid, keltner_lower = keltner_bands(Close, High, Low, period = 20, multiplier = 1.5)
        J = np.array(self.J_model_input)
        solid_kbar_diff = (High - Low)
        upper_shadow = ((High - np.maximum(Open, Close)))
        lower_shadow = ((np.minimum(Open, Close) - Low))

        cci_short = ta.CCI(High, Low, Close, timeperiod = 14)
        cci_long = ta.CCI(High, Low, Close, timeperiod = 21)
                
        model_input = torch.cat((
            torch.from_numpy(minmax(Open[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(High[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(Low[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(Close[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(Volume[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(bias_short[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(bias_long[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(adosc[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(bband_upper[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(bband_mid[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(bband_lower[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(keltner_upper[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(keltner_mid[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(keltner_lower[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(J[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(solid_kbar_diff[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(upper_shadow[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(lower_shadow[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(solid_kbar_diff[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(cci_short[-SEQ_LEN : ])).double().view(-1, 1),
            torch.from_numpy(minmax(cci_long[-SEQ_LEN : ])).double().view(-1, 1),
        ), -1)
        # print(model_input.shape)
        model_input = model_input.view(1, SEQ_LEN, D_MODEL)
        return model_input.to(device = DEVICE)

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
        prev_close = np.array([self.close_price[i + 1] for i in range(-30, 0)])
        prev_high = np.array([self.high_price[i + 1] for i in range(-30, 0)])
        prev_low = np.array([self.low_price[i + 1] for i in range(-30, 0)])

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

        # ===== generate model input and output ===== #
        self.KDJ_model_input = KDJ(prev_high, prev_low, prev_close, 25, 3, 3, self.KDJ_model_input)
        self.J_model_input = self.J_model_input[1:] + [self.KDJ_model_input['J']]

        if cur_time < datetime.time(11, 15):
            self.output_list = [None] * self.output_len
        else:
            model_input = self.get_model_input()
            self.model.eval()
            with torch.no_grad():
                model_output = self.model(model_input, validate = True)
                rise = model_output > self.threshold
                self.output_list = self.output_list[1:] + [rise]
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
                # do_long_condition_4 = self.output_list.count(True) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else False
                do_long_condition_4 = False
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3 or do_long_condition_4:
                    self.do_long()
                    return
                
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                do_short_condition_4 = self.output_list.count(False) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else False
                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3 or do_short_condition_4:
                    self.do_short()
                    return

            else:
                self.enter_squeeze = False

                do_long_condition_1 = self.pioneer_signal == signal.BUY
                do_long_condition_2 = self.open_price[0] < self.close_price[0] 
                do_long_condition_3 = indicators['J'] > 20
                # do_long_condition_4 = self.output_list.count(True) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else False
                do_long_condition_4 = False
                if do_long_condition_1 and do_long_condition_2 and do_long_condition_3 or do_long_condition_4:
                    self.do_long()
                    return
                
                do_short_condition_1 = self.pioneer_signal == signal.SELL
                do_short_condition_2 = self.open_price[0] > self.close_price[0]
                do_short_condition_3 = indicators['J'] < 80
                do_short_condition_4 = self.output_list.count(False) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else False
                if do_short_condition_1 and do_short_condition_2 and do_short_condition_3 or do_short_condition_4:
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
            # cover_condition_3 = self.output_list.count(False) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else True
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
            # cover_condition_3 = self.output_list.count(True) >= len(self.output_list) * 2 / 3 if all(self.output_list) is not None else True
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
    high = high[-period - 1 : ]
    low = low[-period - 1 : ]
    close = close[-period - 1 : ]
    RSV = int((close[-1] - min(low)) / (max(high) - min(low)) * 100 + 0.5 if max(high) - min(low) != 0 else 0)

    _alpha_k = 2 / (signal_k + 1)
    _alpha_d = 2 / (signal_d + 1)
    prev_k = prev_data['K'] if prev_data is not None else 50
    prev_d = prev_data['D'] if prev_data is not None else 50

    K = int(_alpha_k * ((prev_k + 2 * RSV) / 3) + (1 - _alpha_k) * prev_k + 0.5)
    D = int(_alpha_d * ((prev_d + 2 * K) / 3) + (1 - _alpha_d) * prev_d + 0.5)
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

def init_model(loss_weight = None):
    transformer = Transformer(
        d_model = D_MODEL,
        num_heads = NUM_HEADS,
        num_layers = NUM_ENCODER_LAYERS,
        dropout_rate = DROPOUT_RATE
    ).to(device = DEVICE)

    criterion = torch.nn.BCELoss(weight = loss_weight)
    optimizer = torch.optim.SGD(transformer.parameters(), lr = LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 0)

    return transformer, criterion, optimizer, lr_scheduler

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

    # load model
    logging.info('Start to load model')
    transformer, _, _, _ = init_model()
    transformer.load_state_dict(torch.load(MODEL_PATH, weights_only = True))
    logging.info('Succeed to load model')
    logging.info('=' * 50)

    # start backtesting
    cerebro = bt.Cerebro()

    params = {
        'model': transformer,
        'threshold': 0.5
    }

    print()
    print('Add data and strategy')
    cerebro.adddata(bt.feeds.PandasData(dataname = TXF_test))
    cerebro.addstrategy(strategy, **params)
    analyzers = cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name = 'drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name = 'returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name = 'sqn')


    print('Set cash and commission')
    cerebro.broker.setcash(500_000)
    cerebro.broker.setcommission(commission = 0.00025, margin = 0.05, mult = 1)
    print()

    logging.info('Start to backtesting')
    logging.info('=' * 50)
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
