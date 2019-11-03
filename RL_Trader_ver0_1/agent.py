# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 에이전트 모듈(agent.py)은 투자 행동을 수행하고 투자금과 보유 주식을 관리하기 위한 에이전트 클래스를 가진다.
"""
<Attributes>
initial_balance: 초기 투자금
balance: 현금 잔고
num_stocks: 보유 주식 수
portfolio_value: 포트폴리오 가치(= 투자금 잔고 + 주식 현재가 * 보유 주식 수)

<Methods>
reset(): 에이전트의 상태를 초기화
set_balance(): 초기 자본금을 설정
get_states(): 에이전트 상태를 획득
decide_action(): 탐험 또는 정책 신경망에 의한 행동 결정
validate_action(): 행동의 유효성 판단
decide_trading_unit(): 매수 또는 매도할 주식 수 결정
act(): 행동 수행
"""

import numpy as np


class Agent:
    STATE_DIM = 2           # 에이전트 상태의 차원: 주식 보유 비율 & 포트폴리오 가치 비율
                            # 에이전트의 상태를 구성하는 값의 갯수
                            # RLTrader에서의 에이전트 상태는 2개의 값을 가지므로 2차원이다.

    # 매매 수수료 및 세금 미고려
    TRADING_CHARGE = 0      # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0         # 거래세 미고려 (실제 0.3%)

    # # <<보완>> 매매 수수료 및 세금 고려
    # TRADING_CHARGE = 0.00015        # 매수 또는 매도 수수료 0.015%
    # TRADING_TAX = 0.003             # 거래세 0.3%

    # 행동
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL]     # 인공 신경망에서 확률을 구할 행동들 (정책 신경망이 확률을 계산할 행동들 - 관망의 행동 확률은 구하지 않는다.)
                                            # 본 프로젝트에서는 매수와 매도에 대한 확률만 계산하고, 매수와 매도 중에서 결정한 행동을 할 수 없을 때만 관망을 한다.
                                            # 즉 우선 어느 상황이든 매수 및 매도에 대한 확률을 구하고, 그 뒤에 매수 혹은 매도가 불가능한 상황일 경우인지를 판단하여 불가능한 상황일 때만 관망을 하는 것이다.
    NUM_ACTIONS = len(ACTIONS)              # 인공 신경망에서 고려할 출력값의 갯수 (output의 갯수는 매수와 매도에 대한 확률 2가지이다. - 관망에 대한 확률을 구하지 않는다고 바로 위에서 밝혔다.)


    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=0.05):
        # Environment 객체
        self.environment = environment      # 현재 주식 가격을 가져오기 위해 환경을 참조

        # 최소 매매 단위, 최대 매매 단위, 지연 보상 임계치
        # max_trading_unit을 크게 잡으면 결정한 행동에 대한 확신이 높을 때 더 많이 매수 또는 매도할 수 있다.
        self.min_trading_unit = min_trading_unit       # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit       # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold    # 지연 보상 임계치 (손익률이 이 값을 넘으면 지연 보상이 발생한다.)

        # Agent 클래스의 속성
        self.initial_balance = 0        # 초기 자본금
        self.balance = 0                # 현재 현금 잔고
        self.num_stocks = 0             # 보유 주식 수
        self.portfolio_value = 0        # 포트폴리오 가치 (= balance + {num_stocks * 현재 주식의 가격})
        self.base_portfolio_value = 0   # 직전 학습 시점의 포트폴리오 가치 --> 이 기준 포트폴리오 가치는 목표 수익률 또는 기준 손실률을 달성하기 전의 과거 포트폴리오 가치로,
                                        # 현재 포트폴리오 가치가 증가했는지, 감소했는지를 비교할 기준이 된다. (현재 수익이 발생했는지 혹은 손실이 발생했는지를 판단할 수 있는 값이 된다.)
                                        # 이 변수는 "초기" 포트폴리오 가치가 아닌 "직전 학습 시점"의 포트폴리오 가치를 나타내므로 중간에 이 값을 계속 업데이트 시켜주는 과정이 있다. (초기 가치라면 처음 한 번 정의 후 업데이트 없을 것)

        self.num_buy = 0                # 매수 횟수
        self.num_sell = 0               # 매도 횟수
        self.num_hold = 0               # 관망 횟수
        self.immediate_reward = 0       # 즉시 보상 (에이전트가 가장 최근에 행한 행동에 대한 즉시 보상 값)
                                        # 이 즉시 보상은 행동을 수행한 시점에서 수익이 발생한 상태면 1을, 아니면 -1을 준다.

        # Agent의 상태
        self.ratio_hold = 0             # 주식 보유 비율 (최대로 보유할 수 있는 주식 수 대비 현재 보유하고 있는 주식 수의 비율)
        self.ratio_portfolio_value = 0  # "포트폴리오 가치 비율" (직전 지연 보상이 발생했을 때의 포트폴리오 가치 대비 현재의 포트폴리오 가치의 비율)


    # 에포크마다 당연히 에이전트의 상태는 초기화되어야 하므로(학습된 신경망이 초기화되는 것이 아님!) 학습 단계에서 한 에포크마다 에이전트의 상태를 초기화한다.
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0


    # 에이전트의 초기 자본금 설정
    def set_balance(self, balance):
        self.initial_balance = balance


    # 에이전트의 상태를 반환
    def get_states(self):
        # 주식 보유 비율은 현재 상태에서 가장 많이 가질 수 있는 주식 수(=포트폴리오가치/현재주가, =최대 보유 가능 주식 수) 대비 현재 보유한 주식의 비율이다.
        # 주식 보유 비율 = 보유 주식 수  / (포트폴리오 가치 / 현재 주가)
        # 주식 수가 너무 적으면 매수의 관점에서 투자에 임하고, 주식 수가 너무 많으면 매도의 관점에서 투자에 임하게 된다.
        # 즉 보유 주식 수 혹은 비율이 투자 행동 결정에 영향을 줄 수 있기 때문에 (같이 입력되는 다른 15개의 학습 데이터와 같이) 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력에 포함한다.
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())

        # "포트폴리오 가치 비율" = 포트폴리오 가치 / 기준 포트폴리오 가치
        # "포트폴리오 가치 비율" : 기준 포트폴리오 가치 대비 현재 포트폴리오 가치의 비율 (이때 기준 포트폴리오 가치는 직전에 목표 수익 또는 손익률을 달성했을 때의 포트폴리오 가치)
        # 이 포트폴리오 가치 비율이 0에 가까우면 손실이 큰 것이고, 1보다 크면 수익이 발생했다는 뜻이다.
        # 현재 수익률이 목표 수익률에 가까우면 매도 관점에서 투자할 수 있다. 즉 수익률이 투자 행동 결정에 영향을 줄 수 있기 때문에 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력값에 포함한다.
        # self.ratio_portfolio_value = self.portfolio_value / self.initial_balance
        #TODO 분모가 (self.initial_balance가 아닌) self.base_portfolio_value가 맞는 것 같으므로 확인해볼 것 (둘이 우선 0으로 같긴하다.) (책 설명으로는 아래 내가 쓴게 맞음)
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        return (self.ratio_hold, self.ratio_portfolio_value)


    # 행동 결정
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.0

        # 탐험 결정 by E-Greedy
        if np.random.rand() < epsilon:      # 엡실론의 확률로 무작위 행동을 한다.
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)    # 액션의 수는 2개이다(관망을 제외한 매수 및 매도) --> 그러나 randint(2)는 0~2 중 랜덤 값이므로 탐험 결과로 매수(0),매도(1),관망(2)s 중 하나를 선택한다.
        else:                               # (1-엡실론)의 확률로 정책 신경망을 통해 (최적화 된, argmax로) 행동을 결정한다.        (이 부분 관련 설명이 policy_network의 predict 메서드에 있다.)
            exploration = False
            probs = policy_network.predict(sample)      # 각 행동에 대한 확률 - 정책 신경망 클래스의 predict() 함수를 사용하여, 현재 상태에서의 매수와 매도의 확률을 받아온다. (해당 정책 신경망은 output으로 매수와 매도에대한 두 가지 확률 값을 반환한다.)
            action = np.argmax(probs)                   # 이렇게 받아온 확률 중에서 가장 큰 값을 선택하여 행동으로 결정한다. (action은 매수와 매도 중 가장 큰 확률 값을 갖는 행동의 위치를 나타낸다. (argmax() 함수를 사용했으므로.)
            confidence = probs[action]                  # probs가 매수 및 매도에 대한 확률을 의미하고, action은 가장 큰 확률 값을 가져 선택된 그 행동(매수 또는 매도 중 하나)의 위치 값을 의미하므로
                                                        # confidence는 그 선택된 행동의 확률 값을 갖는다. --> (단순히 바로 위의 action에 해당하는 (prob의) 확률 값)
        """
        # # <<보완>> 
        # # 위의 경우 정책 신경망 출력 값이, (즉 결과로 나온 확률 중 MAX값을 선택해도 그 확률 자체가 낮을 수도 있으므로, 예를 들어 0.05와 0.06의 싸움), 
        # # 매우 낮은 상황에서 (즉 선택한 행동이 최선인지 불확실한 경우에) 매수 또는 매도를 수행할 수 있다. --> 이런 경우 선택된 매수 또는 매도가 신뢰성있는 행동인지에 대한 의문이 남게된다.
        # # 따라서 정책 신경망의 출력 값이 어느 정도의 확률값(임계값, threshold) 이상일 경우에만 결정된 행동을 결정하도록 한다. 임계치를 넘치 못한다면 관망을 하도록 만든다.
        # else:
        #     exploration = False
        #     probs = policy_network.predict(sample)      # 각 행동에 대한 확률
        #     action = np.argmax(probs) if np.max(probs) >= 0.5 else Agent.ACTION_HOLD      # 이때의 threshold는 0.5이다.
        #     confidence = probs[action]
        
        # 유의사항:
        #    만일 이런식으로 threshold를 줘버리면, visualizer에서 구현한 plot 함수에서 가지는 값(매수 확률, 매도 확률) 중 하나를 argmax로 결정하고 있기 때문에,
        #    실제로는 관망을 선택했음에도 visulaizer에서는 매수 또는 매도 중 하나를 선택해버리는 상황이 생길 수 있다. 
        #    --> 따라서 plot 함수에도 위처럼 조건문을 주거나 agent.action을 사용하도록 만들어야 할 것이다. 
        """
        #TODO 확인해야할 사항 발견 --> 아래 TODO 확인

        return action, confidence, exploration      # action: 결정한 행동, confidence: 결정(된 행동)에 대한 확신도, exploration: 무작위 투자 유무


    # 결정한 행동이 특정 상황에서는 수행할 수 없을 수도 있다. 예를 들어 매수를 결정했는데 잔금이 1주를 매수하기에도 부족한 경우나,
    # 매도를 결정했는데 보유하고 있는 주식이 하나도 없는 경우에 결정한 행동을 수행할 수 없다.
    # 따라서 결정한 행동을 수행할 수 있는지를 validate_action()을 통해 확인한다.
    def validate_action(self, action):
        # 디폴트 값이 True이며 아래 조건에 걸리면 False가 된다.
        validity = True
        if action == Agent.ACTION_BUY:
            # (선택된 행동이 매수라면,) 적어도 최소 거래단위주를 살 수 있는지 확인
            if self.balance < (self.environment.get_price() * (1 + self.TRADING_CHARGE)) * self.min_trading_unit:     # 우리가 최소 거래 단위를 1로 해놨으니 self.min_trading_unit을
                validity = False                                                                                      # 안 곱해도 되지만, 최소 거래 단위가 1보다 크면 꼭 곱해줘야한다.
        elif action == Agent.ACTION_SELL:
            # (선택된 행동이 매도라면,) 매도할 주식 잔고가 있는지 확인
            # TODO 비교 기준이 0이 아니라, min_trading_unit가 되어야하지 않나?
            if self.num_stocks <= 0:
                validity = False
        return validity


    # 매수/매도 단위 결정
    # 정책 신경망이 결정한 행동의 확률(confidence)이 높을 수록 매수 또는 매도하는 단위를 크게 정해준다.
    # 높은 확률로 매수를 결정했으면 더 많은 주식을 매수하고, 높은 확률로 매도를 결정했으면 더 많은 보유 주식을 매도한다.
    # TODO 인공지능이 아닌 인간이 매수/매도 확률을 계산할 때 어느 하나가 높게 나오는 이유는 모멘텀 등이 있을텐데, 이러한 요소를 학습 파라미터로 설정하면 학습 시 반영 가능?
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading
    # TODO confidence가, 아니, confidence * (max-min)이 1을 넘지 못한다면 결국 max(0,0)이 되므로 추가되는 거래 단위는 없다.
    #      그러나 (max-min)이 1을 넘는다면(즉 2 이상이라면), confidence는 1을 넘게 된다. --> 그렇게 되면 max(1 또는 그 이상의 정수, 0) = 1 또는 그 이상의 정수가 되므로 추가되는 거래 단위는 1 이상이 된다.
    #      다시 말해, max = 3이고 min = 1로 고정이라면, max가 3인 시점부터 max-min의 값이 2 이상이 되어버린다. --> max값이 4,5,6... 증가할 수록 max-min의 값도 3,4,5...가 된다.
    # TODO 계속 아래 주석 read
    """
    정리해보면 일단 confidence는 두 확률 값 중 큰 값이며, 소프트맥스 확률 값이므로 선택된 행동의 확률 값 범위는 0.5 < x < 1이 된다. 
    따라서 confidence*(max-min)은 항상 (max-min)보다 작을 수 밖에 없다. (max-min이 2 이상이 되든 말든 상관없이)
    --> Q_1. 여기서 첫 번째 의문: min() 함수를 뭐하러 사용한 것인가? 그냥 int(confidence*(max-min))만 사용해도 됐을텐데.
    
    --> Q_1_2. 애초에 일단 max-min의 의미를 모르겠다.
    
    어찌됐든 그 다음 max() 함수를 살펴보면, min() 함수를 통해 선택된 int(confidence*(max-min))과 0을 비교하게 된다.
    이때, max-min이 1이라면 confidence*(max-min)은 항상 1보다 작고, 여기에 int()를 씌우면 항상 0이 되기 때문에 추가되는 거래단위는 없게 된다.  
    그러나 max-min이 2 이상이 되는 순간 부터는 confidence*(max-min)이 1보다 커지게 되고, int()를 씌우면 1이 나오기 때문에,
    이 순간 부터는 (max-min이 2 이상이 되는 순간 부터는) max() 함수의 결과가 항상 0을 선택하지 않게 되며, 이로 인해 거래 단위가 항상 증가할 수 밖에 (0보다 클 수 밖에) 없게 된다.
    --> Q_2. 여기서 두 번째 의문: 그렇다면 선택된 행동의 확률 값이 높을 수록 추가되는 거래 단위가 많아 지는게 아니라, max-min의 차가 클 수록 거래단위가 많아지는 것 아닌가?
    
    ==> A_2. 다시 생각해보니 max-min이 2 이상이 되는 순간 부터는 confidence가 0.51인 경우와 0.99인 경우 추가되는 거래 단위가 차이가 나게 된다. 
    그리고 max-min, 즉 이 둘의 차가 confidence에 미치는 영향의 정도는 confidence가 어떤 값을 갖든 일정할 것이므로, 
    max-min의 값이 클 수록 작은 경우보다 추가되는 거래가 많아지지만 이는 올바른 비교가 아니고, max-min의 영향이 같은 상황에서 
    confidence에 따라 추가되는 거래가 얼만큼 차이나는 지를 봐야하는 것이므로, confidence가 어느정도의 확률 값을 갖느냐에 따라 추가되는 거래의 양이 증가하거나 감소하는 것이 맞다. 
    즉 여기서 max-min은 그 차이가 1보다 커야만(2 이상이어야만) confidence에 따라 거래 단위에 유의미한 차이가 생기게 되는 것이다.
    """


    # 투자 행동 수행 함수 (행동 준비)
    # 인자로 들어오는 action은 탐험 또는 정책 신경망을 통해 결정한 행동으로 매수와 매도를 의미하는 0 또는 1의 값을 갖는다.
    # confidence는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률 값이다.
    # TODO confidence가 소프트맥스 확률 값이라면 두 행동에 대한 확률의 합은 1이 된다. 그렇다면 선택되는 행동의 값은 무조건 0.5 보다 높게 된다.
    #   --> 이럴 경우 위의 <<보완>>에서 작성한 선택된 확률 자체가 낮게 나오는 경우는 없게 될 것이다.
    #   --> Q. 그러나 이는 둘이 합쳐서 반드시 혹은 어쩔 수 없이 1이 나와야 하는 상황인데, 이 수치가 신뢰할 만한 수준인지 의문이 남는다. 확인 필수!
    """
        바로 위의 소프트맥스에 관한 이야기 [개발이야기 블로그] 글 참고
        - softmax
        : "여기서 풀려는 문제는 입력 이미지가 주어졌을 때 0~9 까지 각 숫자와 얼마나 비슷한지에 대한 확률을 얻으려는 것임을 기억해 두세요. 
        예를 들면 우리 모델이 어떤 이미지에 대해 “9” 일 확률이 80% 이고 “8” 일 확률이 5%(9의 애매한 꼬리 부분으로 인해) 이며 다른 숫자에 대해서는 낮은 확률을 예측할 수 있습니다. 
        손글씨 숫자를 인식하는 것엔 어느정도 불확실한 부분이 있고 100% 확신을 가지고 하긴 어렵습니다. 이런 경우에 확률 분포는 예측에 대해 얼마나 신뢰할 수 있는지에 대해 좋은 정보를 제공합니다.
        그래서 우리는 상호 배타적인 레이블에 대한 결과로 확률 분포를 담은 출력 벡터를 가집니다. 10개의 확률 값을 가진 이 벡터는 각각 0 에서 9 까지의 숫자에 대응되는 것이고 확률의 전체 합은 1입니다.
        앞서 언급한 것 처럼 이 벡터는 출력 레이어를 소프트맥스 활성화 함수로 구성하여 얻어집니다. 소프트맥스 함수를 사용한 뉴런의 출력 값은 그 레이어의 다른 뉴런의 출력 값에 영향을 받게 되고 그들의 출력 값 합은 1이 되어야 합니다."
        
        출처: [https://potensj.tistory.com/13]
    """
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻어오기 (매수 금액, 매도 금액, 포트폴리오 가치를 계산할 때 사용된다.)
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화 (에이전트가 행동할 때마다 (state가 바뀔 때 마다) 결정되기 때문에 초기화해야한다. (즉시 보상은 누적되는 값이 아니므로 행동마다 초기화하는 것이 맞다.)
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            # (decide_trading_unit 함수에 의해) confidence가 높다면 (그리고 max-min이 2 이상의 값을 갖는다면) 추가되는 거래 단위가 있을 수 있다.
            trading_unit = self.decide_trading_unit(confidence)

            # 매수 후의 잔금을 확인 (이는(바로 아래 식은) 실제 매수 전에 미리 가격을 계산해보고 매수 후 잔금이 어떻게 될지 확인하는 것이다.)
            balance = self.balance - (curr_price * (1 + self.TRADING_CHARGE) * trading_unit)

            # (위 식을 통해 확인한 결과) 보유 현금이 모자랄 경우 보유 현금으로 가능한만큼 최대한 매수 (매수 후에 잔금이 0보다 적으면 안되기 때문에 확인해준다.)
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                    self.max_trading_unit),
                    self.min_trading_unit)
                # (최소가 1, 최대가 2라면) 최소 1단위에서 최대 2단위까지 최대한 매수하며, 1.5단위의 경우에는 max보다 작고 min보다 크기 때문에
                # 그 단위만큼 사는 것이다. (이 1.5단위가 보유 현금으로 최대 구매 가능한 매수 단위인 것이다. 물론 int가 있어서 1.5는 1로 대체될 것이다.)
                # 즉 "self.balance / (curr_price * (1+self.TRADING_CHARGE))" 가 최대구매가능단위를 뜻하므로 1.5가 나오면
                # 앞의 min과 max에 따라 1.5만큼 최대 구매 하는 것 (다른 값이 나오면 최대 최소 비교해 본 상황에 따라 다를 것)
                # 또한, 적어도 1주를 살 수 있는지를 이 메서드 제일 첫 부분인 self.validate_action에서 확인하고 들어오므로 현 메서드 내에서의 잔고로 최소 1주는 살 수 있다.
                ### (max와 min을 통해, 결정한 매수 단위가 최대 단일 거래 단위를 넘어가면 최대 단일 거래 단위로 제한하고, 최소한 최소 거래 단위만큼을 살 수 있다.)

                # TODO 위의 주석은 옳으나, 한정적으로 옳다. 최대 살 수 있는 단위와 max와 min의 차이가 아래 주석과 같다면 문제가 생긴다. --> 확인 필수!
                """
                여기서 의문점이 있다. 
                confidece가 높기 때문에 거래 단위가 늘어나서 10개를 사도 된다는 decide_trading_unit의 결과가 나왔다고 하자. 
                그런데 현재 내 balance로 몇 개를 살 수 있는가 봤더니 (self.balance / (curr_price * (1+self.TRADING_CHARGE))) 7개만 살 수 있다고 한다.
                이때 만일 max_trading_unit이 5라면, "보유 현금이 모자랄 경우 보유 현금으로 가능한만큼 최대한 매수" 라고 해놓고 결국 5개 밖에 못사게 된다.
                (그 이유는 min(int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit) 이니까.)
                그럼 7개를 살 수 있는 balance를 가졌음에도, 즉 보유 현금이 모자라지 않은 경우(7개를 살 수 있는 상황)임에도 왜 보유 현금으로 가능한만큼 최대한 매수(5개 매수)라는 표현을 썼을까
                
                --> 이는 7개를 사면 좋다고 할지라도, max와 min의 범위 내에서만 거래가 이루어져야 (안전)하기 때문인 것인가? 
                
                --> 거래 단위를 max와 min으로 나누어 놓은 이유가 max는 상황이 좋을 경우 거래를 늘려 더 많은 이익을 보기 위해서이며, min은 상황이 좋지 않은 경우 손실을 피하기 위함이다.
                    그렇다면 지금 상황과 같은 경우(7개 살 수 있으며 이대로 사면 이득인데 5개 밖에 못사는 경우)는 문제가 있다고 생각.
                    
                --> 현재 trading_unit은 max(~, self.min_trading_unit)으로 되어있는데, 이를 max(~, self.max_trading_unit)으로 고치거나 (이렇게하면 max(7,5)니까 7만큼 매수 가능),
                    아예 max_trading_unit의 한도를 없애는 것은 어떠한가. max가 balance나 num_stock의 범위만 넘어가지 않도록 상황마다 조건을 걸어주면 되지 않나? (현재도 그렇게 하는 듯 하다.)
                    아래 코드 처럼.
                    
                    --> 
                    if balance < 0:     # --> (trading_unit만큼 구매하면) 보유 현금이 모자란 경우
                        if trading_unit > int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))):     # --> (당연히 그 trading_unit보다는 작게, 즉) 보유 현금으로 가능한만큼 최대한 매수 
                            trading_unit = int(self.balance / (curr_price * (1 + self.TRADING_CHARGE)))     # --> 이렇게해야 max_trading_unit보다 더 매수할 수 있음에도 매수를 못하는 경우를 방지할 수 있다. 
                            
                    --> 그리고 이렇게해야 맞다고 생각하는 또 다른 이유는 밑에서 매도의 trading_unit을 판단할 때는 max_trading_unit과 min_trading_unit으로 비교하거나 조건을 두고 따지지 않는다.
                """

            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount       # 보유 현금을 갱신
            self.num_stocks += trading_unit     # 보유 주식 수를 갱신
            self.num_buy += 1                   # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)

            # TODO 매도 후의 잔금 확인은 없는가? --> 잔금 확인이 아니라, 즉 balance<0이 아니라 self.num_stocks<0을 확인해야함
            #  --> A: 그래서 바로 아래 줄에서 min으로 최소 매도 수량을 정한다. 그렇다면 Q_1.최소 매도 수량은 1일텐데 아예 없으면 문제가 있지 않은가(, 즉 trading_unit이 0이 되면 어떻게하느냐)라는 질문에 대한 답은
            #  --> A_1: 이 메서드의 제일 첫 부분에 self.validate_action으로 매수/매도가 최소 단위로라도 가능한지 확인부터 하므로 문제없다.
            #  ==> 정리하자면 self.num_stocks<0 이어야 한다는 조건은 validate에서 했고, num_stocks>trading_unit을 따져야 하는데 이게 아래의 min(trading_unit, self.num_stocks)와 같다.

            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도 (결정한 매도 단위가 현재 보유한 주식 수보다 많으면 안되므로, 현재 보유 주식 수를 최대 매도 단위로 제한한다.)
            trading_unit = min(trading_unit, self.num_stocks)

            # 수수료를 적용하여 총 매도 금액 산정
            invest_amount = (curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE))) * trading_unit
            self.balance += invest_amount       # 보유 현금을 갱신
            self.num_stocks -= trading_unit     # 보유 주식 수를 갱신
            self.num_sell += 1                  # 매도 횟수 증가

        # 관망
        # 관망은 아무 것도 하지 않으므로 보유 주식 수나 잔고에 영향을 미치지 않는다. 대신 주가 변동은 있을 수 있으므로 포트폴리오 가치가 변경될 수 있다.
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1      # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        # profitloss: 기준 포트폴리오 가치에서 현재 포트폴리오 가치의 등락률을 계산한다.
        # base_portfolio_value: 기준 포트폴리오 가치는 직전 학습 시점의 포트폴리오 가치를 의미한다.
        self.portfolio_value = self.balance + (curr_price * self.num_stocks)      # 위의 과정을 통해 세 가지 변수 모두 값이 변경되었으므로 포트폴리오 가치를 갱신해준다.
        profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value     # 변화율

        # 즉시 보상 판단
        # TODO (profitloss가 0인 경우가 거의 없겠지만) 포트폴리오 가치의 등락률이 0인 경우에도 긍정 보상을 주는 경우는 재고의 여지가 있다고 판단된다.
        # 사람의 성향에 따라 다르다고 판단. 위험기피자의 경우 0을 긍정적으로 여길 수 있으며, 위험선호자의 경우 0을 부정적으로 여길 수 있다.
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        # delayed_reward_threshold는 지연 보상 임계치로, 손익률이 이 값을 넘으면 지연 보상이 발생한다.
        # 지연 보상 임계치를 넘어야 기준 포트폴리오 가치(base_portfolio_value)의 갱신이 이루어진다.
        if profitloss > self.delayed_reward_threshold:      # 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치를 수익으로 초과하는 경우
            delayed_reward = 1
            # 목표 수익률을 달성하였으므로(달성하고) 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value

        elif profitloss < -self.delayed_reward_threshold:   # 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치를 손실로 초과하는 경우
            delayed_reward = -1
            # 손실 기준치를 초과하였으므로(초과하고) 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value

        else:                                               # 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치와 같다면 긍정이든 부정이든 보상이 없다.
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
        """
        # RLTrader는 지연 보상(delayed_reward)이 0이 아닌! 경우 학습을 수행한다.
        # 즉 지연 보상 임계치를 초과하는 수익이 났으면 이전에 했던 행등들(긍정적 즉시 보상을 얻은 행동들)을 잘했다고 판단하여 긍정적으로(positive) 학습하고,
        # 지연 보상 임계치를 초과하는 손실이 났으면 이전 행동들(부정적 즉시 보상을 얻은 행동들)에 문제가 있다고 판단하여 부정적으로(negative) 학습한다.
        """

        """
        즉시 보상 판단을 하는 이유는 지연 보상 판단을 할 수 있게 될 때까지 시간이 걸리기 때문에 그 때까지의 학습에 대해 기준이 되는 방향을 정해주기 위해서이다.
        예를 들어 5일 거래를 한 뒤에 threshold를 넘었다고 할 경우, 1일차는 랜덤 결정이고 2,3,4,5일차의 행동 선택에 대한 기준/방향이 있어야 한다. 
        이때 매일 거래마다 즉시 보상을 해줌으로써 양의 수익을 얻으면 +1을, 음의 손해를 입으면 -1을 가감해줌으로써 지연 보상 시점 전까지의 학습 방향을 정해준다.
        
        --> Q. 이때 문제가 발행한다. 
            즉시 보상을 판단할 때도 지연 보상 판단과 마찬가지로 profitloss로 그 기준을 정하는데, profitloss는 현재 포트폴리오 가치와 기준 포트폴리오 가치의 등락률로 결정된다.
            이때 기준 포트폴리오 가치(base_portfolio_value)는 직전 학습 시점의 포트폴리오 가치이며, 이는 학습이 될 때마다 기준 포트폴리오 가치가 갱신된다는 뜻이다.
            그런데 즉시 보상을 판단할 때 profitloss를 사용한다면 이는 매 거래일 마다 profitloss가 갱신되어야한다는 것인데, 이것은 지연 보상이 발생하여 포트폴리오 가치를 갱신하는 것과 다른 것인가?
            즉시와 지연 보상을 판단할 때 profitloss는 같아야 하는가? 그렇다면 즉시 보상마다 계산을 해야하는데 이러면 리소스를 많이 잡아 먹지 않나?
            
        --> A. 지연 보상 임계치를 넘을 때, 그때까지의 행동을 모아서 이게 옳구나 옳지 않구나를 학습하는 것 
            그때까지의 행동들이 +3, -2면 임계치를 아슬아슬하게 넘고, +5면 확연히 넘을테니 다음에 행동할때는 최대한 +를 많이 받게 해야겠구나를 학습하는 것
            지연 보상을 받을 때까지 즉시 보상들이 +1이 많아야 좋구나 라고 학습할 수 있도록 하기 위해 즉시와 지연으로 보상을 나누는 것
            그러니 profitloss는 같이 쓰는 것 맞다.
            
        --> <정리> (표 참고)
            위에서 한 질문은 즉시 보상과 지연 보상과의 관계를 잘못 이해한 것이다. 
            우선 즉시 보상이나 지연 보상이나 둘 다 profitloss를 쓰는 것이 맞다. 즉시 보상도 이전 포트폴리오 가치로 부터 현재 포트폴리오 가치의 변화를 계산해서 보상을 내려야하므로
            profitloss를 사용하는 것이 맞다.
            그리고 즉시 보상할 때마다 에이전트가 학습을 하는 것이 아니라, 지연 보상이 발생할 때마다 학습하는 것이 맞다. 이때 이루어지는 학습은 지연 보상이 긍정적이라면,
            이전 학습 시점부터 현재 지연 보상이 이루어질 때까지 했던 행동(과 그 행동에 대한 즉시 보상을 그 행동들이)이 긍정적이 었구나라고 학습하는 것이다. 
            예를 들어 5일 동안 매매를 하고 5일 차에 이익률이 임계치를 넘었고 지연 보상 +1을 받는다면, 1일부터 5일차까지 진행했던 행동들과 즉시 보상들이 긍정적인 결과를 얻을 수 있었던
            행동들이었구나 하고 학습하는 것이다. 그리고 전체적으로는 계속 학습이 진행되면서 (에포크가 증가하면서) 그러한 행동들(과 즉시 보상들)의 순서/조합/+1과 -1의 수 등을 고려하여 점점 더 긍정적인
            지연 보상을 받는 방향으로 학습하게 되는 것이다.     
            이러한 과정을 거치면서 지연 보상이 발생할 때마다 기준 포트폴리오 가치는 갱신된다. (="지연 보상 임계치를 넘어야 기준 포트폴리오 가치(base_portfolio_value)의 갱신이 이루어진다.")       
        """
