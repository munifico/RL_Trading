# Author: EunHwan Koh

import os
import logging
import settings
import data_manager
from policy_learner import PolicyLearner

if __name__ == "__main__":
    stock_code = '005930'       # 삼성전자
    model_ver = '20180202000545'

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

    # 데이터를 강화학습에 필요한 차트 데이터와 학습 데이터로 분리하기
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

    # 비학습(Without Training) 투자 시뮬레이션 시작
    # 종목 코드, 차트 데이터, 학습 데이터, 최소 및 최대 투자 단위를 지정해준다. 기존 main 모듈에서 지정해줬던 지연 보상 기준(delayed_reward_threshold)과
    # 학습 속도(l_rate)는 입력하지 않아도 된다. --> 비 학습 투자 시뮬레이션에서는 사용하지 않는 인자들이기 때문이다.
    policy_learner = PolicyLearner(stock_code=stock_code, chart_data=chart_data,
                                   training_data=training_data, min_trading_unit=1, max_trading_unit=3)

    # 정책 학습기 객체의 trade() 메서드를 호출한다. trade() 함수는 비 학습 투자 시뮬레이션을 수행하기 위해 인자들을 적절히 설정하여 fit() 메소드를 호출한다.
    policy_learner.trade(balance=10000000,
                         model_path=os.path.join(settings.BASE_DIR, 'models/{}/model_{}.h5'.format(stock_code, model_ver)))


    # 기존 main 모듈의 마지막 부분에 있는 "정책 신경망을 파일로 저장" 코드 블록은 제거해준다.
    # 이미 저장된 정책 신경망 모듈을 사용했고 추가적으로 학습을 하지 않았기 때문이다.
    # 만약 추가적인 학습을 수행하여 모델을 새로 저장하고 싶다면 코드 블록을 제거하지 않고 그대로 두면 된다.