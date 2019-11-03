# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 정책 학습기 모듈(policy_learner.py)은 정책 학습기 클래스(PolicyLearner)를 가지고 일련의 학습 데이터를 준비하고 정책 신경망을 학습한다.)


import os
import locale       # 통화(currency) 문자열 포맷을 위해 사용
import logging      # 학습 과정 중 정보를 기록하기 위해 사용
import settings     # 투자 설정, 로깅 설정 등을 위한 모듈로서 여러 상수 값들을 포함
import numpy as np
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)        # 특정 로거 설정
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=0.05, l_rate=0.01):
        self.stock_code = stock_code 
        self.chart_data = chart_data

        # 환경 객체
        self.environment = Environment(chart_data)
        
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        # 학습 데이터
        self.training_data = training_data 
        self.sample = None      # 여기서 sample도 training_data와 같이 17차원
        self.training_data_idx = -1

        # 정책 신경망의 Input layer에 들어오는 입력의 크기 또는 특징의 수(17) = 학습 데이터의 크기(15) + 에이전트 상태의 크기(2)
        # TODO self.training_data.shape의 [1]이 왜 학습 데이터의 크기인지 확인
        # TODO <해결> shape가 2차원이면 (n,15)일 것이고, 여기서 행은 전체 갯수이고, 열의 갯수가 학습 데이터의 크기로 들어갈 특징의 갯수일 것이다.
        #            shape가 2차원인 이유는 policy_network.py에서 설명했다.
        #              --> "Sequential 클래스의 predict() 함수는 여러 샘플을 한번에 받아서 신경망의 출력을 반환한다. 
        #                   하나의 샘플에 대한 결과만을 받고 싶어도 입력값을 샘플의 배열로 구성해야하기 때문에 2차원 배열로 재구성한다."
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        # 정책 신경망 객체
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, l_rate=l_rate)
        
        # 가시화기 객체 (에포크마다 투자 결과 가시화)
        self.visualizer = Visualizer()  


    # 에포크마다 호출하여 reset
    def reset(self):
        self.sample = None              # 읽어들인 데이터는 self.sample에 저장된다. 단, 이 초기화 단계에서는 읽어들인 데이터가 없으므로 None값을 갖는다.
        self.training_data_idx = -1     # 학습 데이터를 다시 처음부터 읽기 위해 -1로 재설정 (학습 데이터를 읽어가며 이 값은 1씩 증가하는데, 읽어 온 데이터는 self.sample에 저장된다.)


    # 학습 데이터 샘플 생성
    def _build_sample(self):
        # TODO self.environment.observe() 함수는 특정 행의 데이터를 전부(D,O,H,L,C,V) 반환한다. --> 그런데 반환을 받아주는 변수가 없다?
        #   --> 얘는 여기가 아니라 visualizer의 axes[0]을 그릴 때 필요한 것 같다. 그러나 심지어 거기서도 얘를 가공해서 쓰지않고 바로 chart_data에서 ohlcv를 가져와서 쓴다. --> 이 함수를 만든 이유를 파악할 수 없다.
        #       (envrironment.py의 get_price()가 사용하긴 한다. --> 근데 얘는 observe() 함수에서 갱신된 self.observation 값을 사용하려는 것이 목적이지 observe() 함수의 반환값은 필요없다.)
        #   --> 여기서 observe()를 사용하는 이유는 이 함수를 호출하여 차트 데이터의 현재 인덱스에서 다음 인덱스를 읽어들이기 위함이다. --> 이 목적은 알겠으나, self.obseravtion을 반환하는 이유를 알 수 없다.
        self.environment.observe()      # 환경 객체의 observe() 함수를 호출하여, 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽도록 한다.
        if len(self.training_data) > self.training_data_idx + 1:        # 학습 데이터의 다음 인덱스가 존재하는지 확인
            self.training_data_idx += 1
            # TODO sample은 policy_network.py에서 2차원으로 만들어주는 것이므로 여기서는 1차원 형태가 맞는지 확인 (※ 여기서 sample의 길이는 15이고, 아래에서 에이전트 상태가 추가되어야 feature 길이가 17이 되는 것이다. --> 이는 training_data의 feature 길이 역시 15[-1,15]라는 뜻이다.)
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())     # 현재까지 sample 데이터는 15개의 값으로 구성되어 있는데,extend()를 통해
            return self.sample                              # sample에 에이전트 상태(2개)를 추가하여 17개의 값으로 구성되도록 한다.
        return None

    
    # TODO memory, sample, 그리고  batch_size가 정확히 무엇인지 알아야한다.
    # TODO <해결> memory = [(memory_sample[i], memory_action[i], memory_reward[i]) for i in list(range(len(memory_action)))[-max_memory:]]
    #                   =  즉 (각 거래일 idx마다의 데이터가 모여 만들어진) 학습 데이터 sample 배열 / 에이전트의 action 배열 / 즉시 reward 배열을 모아서 만든 2차원 배열
    #                   ex) memory_sample = [1,2,3,4,5]
    #                       memory_action = [6,7,8,9,10]
    #                       memory_reward = [11,12,13,14,15]
    #                       memory = [(memory_sample[i], memory_action[i], memory_reward[i]) for i in list(range(len(memory_action)))]
    #                       memory = [(1, 6, 11), (2, 7, 12), (3, 8, 13), (4, 9, 14), (5, 10, 15)]
    #                       |
    #            sample: training_data가 저장되므로 15(학습데이터)+2(상태데이터) 총 17개의 데이터를 갖는 데이터)
    #            batch_size: 배치 데이터의 크기는 지연 보상이 발생할 때 결정되기 때문에 매번 다르다.
    # 미니 배치 데이터 생성
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        # TODO 아래 x의 np.zeros를 3차원이 아닌 2차원으로 만들면 안되나? --> np.zeros((batch_size, self.num_features))로.
        # TODO <해결??> 가운데 1을 없애고 2차원으로 만드는게 맞다. <-- 책을 보면 "x 배열의 형태는 배치 데이터 크기, 학습 데이터 특징 크기로 2차원으로 구성된다."가 맞다.
        # TODO <해결 아님> 가운데 1을 없애면, "expected lstm_1_input to have 3 dimensions, but got array with shape (60, 17)"와 같은 에러가 발생한다. 가운데 1이 있어야 한다. 
        # x = np.zeros((batch_size, self.num_features))
        x = np.zeros((batch_size, 1, self.num_features))            # 일련의 학습 데이터(15) 및 에이전트 상태(2) (np.zeros((1,2,3)) --> array([[[0., 0., 0.],[0., 0., 0.]]])이 된다. (인자는 튜플로 넘겨야한다. )
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)      # 일련의 지연 보상 (np.full() --> 첫 번째 인자는 shape이고(다차원일 경우 튜플로 넘겨야한다.), 두 번째 인자 값으로 배열을 채운다.

        # 배치 데이터의 크기는 지연 보상이 발생할 때 결정되기 때문에 매번 다른 반면, 학습 데이터 특징의 크기(17)와 에이전트 행동 수(2)는 고정되어있다.

        # TODO for문의 'reward'가 Unused variable이다. --> 확인 필수
        for i, (sample, action, reward) in enumerate(       # action은 0(매수)과 1(매도)로 이루어져있다.
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))     # 학습 데이터 특징 벡터를 지정하고,
            y[i, action] = (delayed_reward + 1) / 2                         # 지연 보상을 정답(=Label)으로 설정하여 학습/훈련 데이터 셋을 구성 --> 지연 보상이 1인 경우 1로 레이블을 지정하고, 지연 보상이 -1인 경우 0으로 레이블을 지정한다.
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i    # 더 과거에 더 큰 할인요인을 곱한다.
        return x, y


    # fit() 메서드: PolicyLearner 클래스의 핵심 함수
    """
    fit()의 Elements
        max_memory:      배치(batch) 학습 데이터를 만들기 위해 과거 데이터를 저장할 배열 (이 배열의 크기가 배치 학습 데이터의 크기와 같은지 확인)
        
        balance:         에이전트의 초기 투자 자본금을 정해주기 위한 인자
        
        discount_factor: !**지연 보상이 발생했을 때, 그 이전 지연 보상이 발생한 시점과 현재 지연 보상이 발생한 시점 사이에서 수행한 행동들 전체에 현재의! 지연 보상을 적용한다.
                         이때 과거로 갈수록 현재 지연 보상을 적용할 판단 근거가 흐려지기 때문에, 먼 과거의 행동일수록 할인 요인(discout factor)을 적용하여 지연 보상을 약하게 적용한다.**!
                         
        start_epsilon:   초기 탐험 비율 (학습이 되어 있지 않은 초기에 탐험 비율을 크게 해서 더 많은 탐험, 즉 무작위 투자를 수행하도록 해야한다. 이러한 탐험을 통해 특정 상황에서
                         좋은 행동과 그렇지 않은 행동을 결정하기 위한 경험을 쌓는다.) (탐험을 통한 학습이 지속적으로 쌓이게되면 탐험 비율을 줄여나간다.)
                         
        learning:        학습 유무를 정하는 boolean 값 (학습을 마치면 학습된 정책 신경망 모델이 만들어지는데, 이렇게 학습을 해서 정책 신경망 모델을 만들고자 한다면 learning을 True로,
                         이미 학습된 모델을 가지고 투자 시뮬레이션만 하려 한다면 False로 준다.)
    """
    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logger.info("Learning Rate: {l_rate}, Discount Factor: {discount_factor}, "
                    "Trading Unit[min,max]: [{min_trading_unit}, {max_trading_unit}], "
                    "Delayed Reward Threshold: {delayed_reward_threshold}".format(
            l_rate=self.policy_network.l_rate,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 (한 번만) 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과를 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (     # settings.BASE_DIR: 프로젝트 관련 파일들이 포함된 기본/루트 폴더 경로를 말한다.
                self.stock_code, settings.timestr))     # settings.timestr: 폴더 이름에 포함할 날짜와 시간 - 문자열 형식: %Y%m%d%H%M%S
        # epoch_summary_dir = os.path.join(         # --> 이렇게 코드를 작성하면 윈도우 계열, 리눅스 계열 OS에 상관없이 경로를 구성할 수 있다.
        #   settings.BASE_DIR, 'epoch_summary', '%s' % self.stock_code,
        #                      'epoch_summary_%s' % settings.timestr)   
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0     # max_portfolio_value: 학습 과정에서 달성한 최대 포트폴리오 가치를 저장하는 변수
        epoch_win_cnt = 0           # epoch_win_cnt: 수익이 발생한 에포크 수를 저장하는 변수

        ### 학습 반복 for문 시작 ###
        for epoch in range(num_epoches):    
            # 에포크 관련 정보 초기화
            loss = 0.               # 정책 신경망의 결과가 학습/실제 데이터와 얼만큼 차이가 있는지 저장
            # TODO itr_cnt와 win_cnt는 에포크 수를 (카운트해서) 저장하는게 아니라, 거래일 idx를 (카운트해서) 저장하는 변수가 맞는 듯 하다. (심지어 win_cnt는 이 주석 위에 epoch_win_cnt가 따로 있다.)
            itr_cnt = 0             # 수행한 에포크 수를 저장 (반복 카운팅 횟수)
            win_cnt = 0             # 수행한 에포크 중에서 수익이 발생한 에포크 수를 저장 - 즉 포트폴리오 가치가 초기 자본금보다 높아진 에포크 수를 저장
            exploration_cnt = 0     # 무작위 투자(탐험)를 수행한 횟수를 저장
            batch_size = 0          # batch의 크기
            pos_learning_cnt = 0    # 수익이 발생하여 긍정적 지연 보상을 준 횟수
            neg_learning_cnt = 0    # 손실이 발생하여 부정적 지연 보상을 준 수

            # 메모리 초기화
            # TODO Q_1.각 변수가 어떤 값과 모양을 갖는지 잘 모르겠음, Q_2.특히 탐험 위치 및 학습 위치, 정확히 어느 위치인지 잘 모르겠음 --> 체크 필수!
            # TODO <해결> A_1. 우선 memory가 붙은 변수들 중 위 6개의 범위는 Visualizer에서 만든 x축이 갖는 범위만큼이며,
            #                 x축이 갖는 각 idx에서의 샘플 데이터값, 행동값, 즉시보상값, 정책 신경망의 출력값, 포트폴리오의 가치값, 보유 주식 수 값을 각각 가진다.
            #                 즉 쉽게 말해 x축의 범위인 거래일자마다의 샘플 데이터값, 행동값, 즉시보상값, 정책 신경망의 출력값, 포트폴리오의 가치값, 보유 주식 수 값을 가지는 것이다.
            #            A_2. 탐험 위치와 학습 위치 역시 범위가 x축이 갖는 범위 만큼인 것은 맞지만, 이 둘은 x축이 갖는 범위만큼의 각 idx들 중에서 탐험한 idx와학습한 idx만을 선택해서 가지는 것이다.
            memory_sample = []          # 샘플
            memory_action = []          # 행동
            memory_reward = []          # 즉시 보상
            memory_prob = []            # 정책 신경망의 출력
            memory_pv = []              # 포트폴리오 가치
            memory_num_stocks = []      # 보유 주식 수

            memory_exp_idx = []         # 탐험 위치
            memory_learning_idx = []    # 학습 위치     # TODO memory_learning_idx가 무엇인지 정확히 파악 후 visualizer.py의 151번 줄 TODO 해결할 것 (151번줄에서 얘(memory_learning_idx == visualizer.py의 learning) 때문에 막힘)
                                                      # TODO <해결> 지연 보상(으로 학습한) 위치와 지연 보상 값으로 이루어진 배열 --> [[지연 보상 위치, 지연 보상 값], [지연 보상 위치, 지연 보상 값], [지연 보상 위치, 지연 보상 값], ...]
                                                      #            이 학습 위치 변수의 이름을 바꿔야 할 필요가 있는 것 같다. --> 아래에서 memory_learning_idx.append([itr_cnt, delayed_reward])가 실행되므로 memory_learning_idx_val로 바꾼다.
            
            # 아래 코드를 보면 알겠지만 첫 에포크를 포함하여 에포크가 시작할 때마다 우선 모두 초기화(reset & clear)부터 하고 진행한다.
            # 환경, 에이전트, 정책 신경망, 정책 학습기 초기화 (이 클래스들은 각자 reset 함수를 정의했었음 - 다른 클래스들은 따로 reset 메서드를 정의하지 않았음)
            self.environment.reset()                                                    # (cf. Visualizer는 clear 함수로 초기화 함)
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            self.visualizer.clear([0, len(self.chart_data)])

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))       # epoch는 학습을 반복하며 계속 1씩 증가할 것 --> epsilon은 계속 감소한다.
                # e = 1. / ((episode / 10) + 1)                                         # epoch / num_epoches는 학습 진행률을 의미하며,
            else:                                                                       # num_epoches에서 1을 빼는 이유는 0부터 시작하기 때문이다. (표기는 1부터 시작하지만 실제 학습시에는 0부터 시작)
                epsilon = 0

            # (현재 for문 속에서 선택된) 하나의 Epoch에 대해 돌아가는 while문
            while True:
                # 샘플 생성
                next_sample = self._build_sample()      # 행동을 결정하기 위한 데이터인 샘플 준비
                if next_sample is None:                 # next_sample이 None이라면 마지막까지 데이터를 다 읽은 것이므로 반복문을 종료한다.
                    break

                # 탐험 또는 정책 신경망에 의한 행동 결정 (E-Greedy: Exploration v.s Exploitation)
                # decide_action()의 반환값 3가지: --> action: 결정한 행동, confidence: 결정(된 행동)에 대한 확신도, exploration: 무작위 투자 유무 (반환 값 중 action은 매수 또는 매도 둘 중 하나가 된다. 관망 없음 - 이유는 5줄 아래 주석 참고)
                action, confidence, exploration = self.agent.decide_action(     # Exploitation, 즉 정책 신경망의 출력은 매수, 매도를 했을 때의 포트폴리오 가치를 높일 확률을 의미한다.
                    self.policy_network, self.sample, epsilon)                  # 즉 매수에 대한 정책 신경망의 출력이 매도에 대한 출력보다 높으면 매수를, 그 반대의 경우에는 매도를 선택한다.

                # 결정한 행동을 수행하고 "즉시 보상"과 "지연 보상" 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)       # agent 모듈에 가서 act 함수를 보면 알겠지만, (일단 이 프로젝트에서 관망은 매수 및 매도가 불가능한 경우에만
                                                                                            # action이 hold가 된다. 따라서) decide_action 함수에서 일단 action이 매수 또는 매도로 결정되고,
                                                                                            # act 함수 초반에 매수/매도 상황이 불가능한 경우인지 아닌지, 즉 관망인지 아닌지 아닌지를 판단하게 된다.
                                                                                            # 따라서 agent의 decide_action 함수가 반환하는 action은 우선 매수 또는 매도가 맞으며,
                                                                                            # 그 뒤에 act 함수가 호출 됐을 때, 이 함수의 초반에 validate 검사에 합격하여 실제로 행해지는 action이
                                                                                            # 그 매수 또는 매도가 될지, 혹은 검사에 실패하여 관망이 될지의 여부가 결정된다.

                # 행동 및 행동에 대한 결과를 기억
                # 아래의 메모리 변수들은 2가지 목적으로 사용된다. --> 1) 학습에서 배치 학습 데이터로 사용  2) Visualizer에서 차트를 그릴 때 사용
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [(
                    memory_sample[i],            # memory: 위의 항목들(학습 데이터 sample 배열 / 에이전트의 action 배열 / 즉시 reward 배열)을 모아서 하나의 2차원 배열로 만든다.
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]        # -max_memory의 수만큼을 반복시킨다. (max_memory가 60이므로 [-60:]은 60개이다.
                ]
                """
                위의 memory 부분 참고 (다시 볼 때 이해한 후 지울 것)

                *** [-max_memory:] 부분 --> 어렵게 생각하지 말 것
                a = np.array([1,2,3,4,5])
                a[-5:] --> 1,2,3,4,5
                a[-5:-1] --> 1,2,3,4
                a[-4:] --> 2,3,4,5

                *** memory가 3차원이 아닌 2차원인 이유 --> 제대로 볼 것
                배열만으로 이루어진 2차원 == [[], [], ...]
                배열과 튜플로 이루어진 2차원 == [(), (), ...]
                즉 memory는 [] 안에 여러 개의 ()이 반복문으로 들어간다.
                최종적으로 memory의 shape는 [None, 3] (== [i+1, 3]) 이다.
                """

                if exploration:
                    # TODO itr_cnt가 수행한 에포크 수(반복 카운팅 횟수)를 저장하는 변수인데, 현재의 인덱스라고? --> 체크 필수! (에포크 수(돌아가는 에포크의 넘버)를 인덱스로 가져가는 건가? --> 체크 필수!
                    #       --> 에포크 수를 세는 변수가 아니라 거래일 idx를 세는 변수가 맞는 것 같다.
                    memory_exp_idx.append(itr_cnt)      # 무작위 투자로 행동을 결정한 경우, memory_exp_idx에 현재의 인덱스를 저장한다.
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)        # memory_prob는 정책 신경망의 출력을 그대로 저장하는 배열인데, 무작위 투자에서는 정책 신경망의 출력이 없기 때문에 nan값을 준다.
                                                                            # [np.nan] * 2이므로 [nan, nan]이 된다. --> [매수확률이 nan, 매도확률이 nan]
                else:
                    memory_prob.append(self.policy_network.prob)            # 무작위 투자가 아닌 경우, 정책 신경망의 출력을 그대로 memory_prob에 저장한다. (정책 신경망의 출력: [매수확률, 매도확률])

                # 학습 함수의 반복에 대한 정보 갱신
                # TODO batch_size를 왜 1씩 증가시키지?? 내가 알고있는 batch의 개념이 아닌가? --> 체크 필수!
                batch_size += 1
                # TODO itr_cnt가 에포크 수를 세는 변수라고 해놓고 왜 한 에포크 내에서 반복되는 횟수(x축의 길이)를 카운트하고 있는가?
                #   --> 에포크 수를 세는 변수가 아니라 거래일 idx를 세는 변수가 맞는 것 같다.
                #       (exploration_cnt와 win_cnt도 마찬가지 - 다만 itr_cnt는 모든 횟수를 세기 때문에 index 역할도 되는 것이고, ex_cnt와 win_cnt는 특정 상황에만 카운트 되므로 횟수에 불과하다.)
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0      # (한 에포크 내에서 발생하는) 탐험 횟수 저장
                win_cnt += 1 if delayed_reward > 0 else 0       # (한 에포크 내에서 발생하는) 지연 보상 임계치를 초과하는 수익이 났으면 이전에 했던 행등들을 잘했다고 판단하여 긍정적으로(positive) 학습한다.

                # TODO 이 부분부터 마지막 끝까지 다시 이해하기 (헷갈림) --> 체크 필수!!!
                # 학습 함수의 정책 신경망 학습 부분: 지연 보상이 발생한 경우 학습을 수행하는 부분을 보여준다.
                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if delayed_reward == 0 and batch_size >= max_memory:    # 지연 보상이 있어야 그때까지의 데이터를 모아 한 번에 batch 사이즈로 학습을 할 수 있다. 그런데 지연 보상이 없다. 이러면 학습한 것도 없는 상황인데,
                    delayed_reward = immediate_reward                   # 메모리마저 최대 크기만큼 다 찼다면, 지연 보상을 즉시 보상으로 대체하여 학습을 진행한다.
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:        # 지연 보상이 발생한 경우
                    # 배치 학습 데이터 크기 (배치 학습에 사용할 배치 데이터 크기 결정)
                    batch_size = min(batch_size, max_memory)        # 배치 데이터 크기는 memory 변수의 크기인 max_memory 보다 작아야 한다.
                    # 배치 학습 데이터 생성
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)     # _get_batch() 함수를 통해 배치 데이터를 준비한다.
                    if len(x) > 0:
                        if delayed_reward > 0:      # 로그 기록을 위해 긍정 학습과 부정 학습 횟수를 카운트
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # 정책 신경망 갱신
                        loss += self.policy_network.train_on_batch(x, y)        # 준비한 배치 데이터로 학습 진행 --> 학습은 정책 신경망 객체의 train_on_batch() 함수로 수행한다.
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    batch_size = 0
                    # TODO 위의 batch 사이즈는 내가 아는 batch의 개념이 아닌 듯함 --> 왜 0으로 초기화를 하는지? 체크 필수!
                    # TODO <해결> 책 p.56 읽어볼 것 (내가 아는 batch의 개념이 맞기는 한데, 조건이 걸린 batch이다.)
                    #            ==> "주식투자를 하면서 지연 보상을 줄 수 있을지 판단한다. 즉 투자를 하다가 임계치를 넘는 이익이나 손실이 발생하면 이때까지의 상황과 행동들을 학습 데이터로 생성한다.
                    #                 그리고 이 학습 데이터들을 한 번에 적용하여 정책 신경망을 업데이트 하는데, 이를 배치(batch) 학습 방법이라 한다."
                    #                 ※ "배치 데이터의 크기는 지연 보상이 발생될 때 결정되기 때문에 매번 다르다."

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')    # ex. epoch가 50이고, num_epoches_digit이 4라면 --> 0050이 된다.

            # 에포크 수행 결과 가시화
            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            # 가시화한 에포크 수행 결과 저장
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"       # #Expl. 은 "무작위 투자를 수행한 횟수 / 수행한 에포크 수(반복 카운팅 횟수)" 이다.
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))

            # 학습 관련 (통계) 정보 갱신
            max_portfolio_value = max(      # max_portfolio_value의 초기값은 0이다. --> 에포크를 늘려가며 max포트폴리오 가치가 갱신되는 것.
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        ### 학습 반복 for문 종료 ###  

        # 최종 학습 (결과) 관련 정보 로그 기록
        logger.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

        ##### 여기까지가 fit() 함수의 영역이다. #####


    # (학습된 정책 신경망으로) 투자 시뮬레이션 진행
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)  # 학습된 정책 신경망 모델을 정책 신경망 객체의 load_model로 적용시킨다.
        self.fit(balance=balance, num_epoches=1, learning=False)
    # 이 trade() 함수는 학습된 정책 신경망으로 투자 시뮬레이션을 하는 것이므로 반복 투자를 할 필요가 없다.
    # 따라서 총 에포크 수 num_epoches를 1로 주고, learning 인자에 False를 넘겨준다.
    # 이렇게 하면 학습을 진행하지 않고 정책 신경망에만 의존하여 투자 시뮬레이션을 진행한다.
    # (물론 무작위 투자는 수행하지 않는다. fit() 함수 인자의 learning이 False가 되어 epsilon이 0이 되므로.)
