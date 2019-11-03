# -*- coding: utf-8 -*-

# Author: EunHwan Koh

import sqlite3
import pandas as pd
import numpy as np


#==============================================
# 데이터베이스에서 데이터 가져오기

def load_chart_data_fromDB(stock_code):

    # Connecting database 
    conn = sqlite3.connect("/Users/unanimous0/kospi.db")
    cur = conn.cursor()

    # Executing SQL Query
    stock_code_KS = stock_code + ".KS"
    cur.execute("SELECT * FROM '{}';".format(stock_code_KS))

    # Getting data
    rows = cur.fetchall()
    # Checking the data 
    # for row in rows:
    #     print(row)

    # Processing the data table
    header = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = pd.DataFrame(rows, columns=header)

    chart_data = chart_data[::-1]    # Sorting in reverse order by date

    strType_date = chart_data['date'].astype('str')         # 날짜 타입을 문자열로 변환
    dateType_date = strType_date.astype('datetime64[ns]')   # 날짜 포맷 변환
    strDateType_date = dateType_date.astype('str')          # 날짜 포맷 문자열로 재변환
    del chart_data['date']
    chart_data.insert(loc=0, column='date', value=strDateType_date)

    # 날짜로 인덱스 만들기    
    # date = chart_data['date']
    # date_list = list(date)
    # del chart_data['date']
    # chart_data.index = date_list


    conn.close()

    return chart_data
#==============================================


def load_chart_data_fromCSV(fpath="/Users/unanimous0/Desktop/000660_train.csv"):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # # <<보완>> 기관 순매수('inst')와 외국인 순매수('frgn') 데이터 추가
    # inst와 frgn 데이터는 2015년 9월 7일 날짜부터 존재하기 때문에 그 이전 날짜에서는 빈 문자열(empty string)이 채워져 있다. 따라서 빈 문자열을 nan값으로 바꿔준다.
    # chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'inst', 'frgn']
    # chart_data['inst'] = pd.to_numeric(chart_data['inst'].str.replace(',', ''), errors='coerce')      # 기관 순매수 데이터
    # chart_data['frgn'] = pd.to_numeric(chart_data['frgn'].str.replace(',', ''), errors='coerce')      # 외국인 순매수 데이터

    return chart_data
    # 과거의 주가와 현재의 주가가 크게 차이가 나기 때문에 이 값을 그대로 학습에 사용하기는 어렵다.
    # 따라서 현재 종가와 전일 종가의 비율, 이동평균 종가의 비율, 현재 거래량과 전일 거래량의 비율, 이동평균 거래량의 비율을 학습에서 사용한다.


# 종가와 거래량의 이동 평균 구하기
def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()      # 새로운 열(종가의 이동 평균) 추가
        prep_data['volume_ma{}'.format(window)] = prep_data['volume'].rolling(window).mean()    # 새로운 열(거래량의 이동 평균) 추가
        # TODO <<보완>> 추가된 기관 순매수('inst')와 외국인 순매수('frgn') 데이터 전처리 과정 추가
        #   prep_data['inst_ma{}'.format(window)] = prep_data['inst'].rolling(window).mean()
        #   prep_data['frgn_ma{}'.format(window)] = prep_data['frgn'].rolling(window).mean()
    return prep_data


# 주가와 거래량의 비율 구하기
def build_training_data(prep_data):
    training_data = prep_data

    # 시가/전일종가 비율
    # 시가와 전일 종가의 비율을 구하는 것이므로 open은 open[1:]이고, lastclose는 close[:-1]이 된다.
    # (open이 1부터 시작하는 이유는 첫 번째 행은 전일 값이 없거나 그 값이 있더라도 알 수 없기 때문에 전일 대비 종가 비율을 구하지 못한다.)
    # (lasoclose가 -1 전까지가 마지막인 이유는 위의 이유와 유사하다.)
    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['open_lastclose_ratio'].iloc[1:] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
        # ex) training_data['open'][1:].values의 결과로 ndarray 형식의 배열이 반환된다. --> 따라서 위의 식은 ndarray간에 element-wise로 계산이 되는 것이므로 문제 없다.

    # 고가/종가 비율
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / training_data['close'].values

    # 저가/종가 비율
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / training_data['close'].values

    # 종가/전일종가 비율
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['close_lastclose_ratio'].iloc[1:] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values

    # 거래량/전일거래량 비율
    # 단 이 비율을 구할 때는 거래량 값이 0이면 이전의 0이 아닌 값으로 바꾸어 준다.
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data['volume_lastvolume_ratio'].iloc[1:] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) \
            / training_data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values

    # # <<보완>> 기관 순매수 거래량의 전일 대비 비율 (기관 순매수 거래량/전일 거래량 비율)
    # training_data['inst_lastinst_ratio'] = np.zeros(len(training_data))
    # training_data.loc[1:, 'inst_lastinst_ratio'] = (training_data['inst'][1:].values - training_data['inst'][:-1].values) / \
    #     training_data['inst'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values

    # # <<보완>> 외국인 순매수 거래량의 전일 대비 비율 (외국인 순매수 거래량/전일 거래량 비율)
    # training_data['frgn_lastfrgn_ratio'] = np.zeros(len(training_data))
    # training_data.loc[1:, 'frgn_lastfrgn_ratio'] = (training_data['frgn'][1:].values - training_data['frgn'][:-1].values) / \
    #     training_data['frgn'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values

    windows = [5, 10, 20, 60, 120]
    for window in windows:
        # 이동평균 종가 비율
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / training_data['close_ma%d' % window]

        # 이동평균 거래량 비율
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / training_data['volume_ma%d' % window]

    return training_data
    # 이렇게 반환되는 trading_data는 {차트 데이터 열들(chart_data(DOHLCV)) + 전처리에서 추가된 열들(close_ma(5~120) & volume_ma(5~120)) + 학습 데이터의 열들(ratio data)} 로 이루어진 데이터이다.
    # 이처럼 여러 feature를 가진 training_data는 main 함수에서 필요한 부분들(DOHLCV의 차트 데이터와 15개의 feature를 가진 학습 데이터)만 떼어내서 강화학습에 사용한다.
