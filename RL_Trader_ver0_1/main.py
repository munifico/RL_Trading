# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 메인 모듈(main.py)은 주식투자 강화학습을 실행한다.

import os
import logging
import settings
import data_manager
from policy_learner import PolicyLearner

if __name__ == "__main__":
    stock_code = '000020'

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % stock_code):
        os.makedirs('logs/%s' % stock_code)
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, "%s_%s.log" %
                                                             (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 강화학습에 필요한 주식 데이터 준비
    # 1) csv 파일에서 데이터 불러오기
    # chart_data = data_manager.load_chart_data_fromCSV(os.path.join(settings.BASE_DIR, 'data/chart_data/{}.csv'.format(stock_code)))
    # 2) database에서 데이터 불러오기
    chart_data = data_manager.load_chart_data_fromDB(stock_code)
    prep_data = data_manager.preprocess(chart_data)     # 불러온 차트데이터 전처리하여 학습 데이터를 만들 준비
    training_data = data_manager.build_training_data(prep_data)     # 학습 데이터에 포함될 열들을 추가
                                                                    # 이 training_data는 차트 데이터의 열들, 전처리에서 추가된 열들, 학습 데이터의 열들이 모두 포함된 데이터이다.

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2017-01-01')
                                  & (training_data['date'] <= '2017-12-31')]
    training_data = training_data.dropna()

    # 데이터를 강화학습에 필요한 차트 데이터와 학습 데이터로 분리하기 --> 여러 feature를 가진 training_data는 필요한 부분들(DOHLCV의 차트 데이터와 15개의 feature를 가진 학습 데이터)로 떼어낸다.
    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]
    

    # # <<보완>> 학습 데이터 분리 + inst & frgn
    # # 위의 "학습 데이터 분리"에 기관 순매수 및 외국인 순매수 학습 데이터 추가
    # features_training_data = [
    #     'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    #     'close_lastclose_ratio', 'volume_lastvolume_ratio',
    #     'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    #     'close_ma5_ratio', 'volume_ma5_ratio',
    #     'inst_ma5_ratio', 'frgn_ma5_ratio',
    #     'close_ma10_ratio', 'volume_ma10_ratio',
    #     'inst_ma10_ratio', 'frgn_ma10_ratio',
    #     'close_ma20_ratio', 'volume_ma20_ratio',
    #     'inst_ma20_ratio', 'frgn_ma20_ratio',
    #     'close_ma60_ratio', 'volume_ma60_ratio',
    #     'inst_ma60_ratio', 'frgn_ma60_ratio',
    #     'close_ma120_ratio', 'volume_ma120_ratio',
    #     'inst_ma120_ratio', 'frgn_ma120_ratio',
    # ]
    # training_data = training_data[features_training_data]


    # 강화학습 시작
    # TODO max_trading_unit과 min_trading_unit의 차이가 2 이상이 되도록 시도해볼 것
    # TODO delayed_reward_threshold가 0.2면 너무 높은 것 같다. --> 0.05 등으로 변경 후 시도해볼 것
    # TODO discount_factor가 0 이므로, 제대로 할인이 될 수 있는 값을 지정 후 시도해볼 것
    policy_learner = PolicyLearner(stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                                   min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=0.2, l_rate=0.001)
    policy_learner.fit(balance=10000000, num_epoches=1000, discount_factor=0, start_epsilon=0.5)

    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)       # 저장할 폴더 경로 지정
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)               # 저장할 파일명 지정
    policy_learner.policy_network.save_model(model_path)                        # 정책 신경망 모델 저장