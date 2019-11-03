# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 환경 모듈(environment.py)은 투자할 종목의 차트 데이터를 관리하는 모듈이다.
"""
<Attributes>
chart_data: 주식 종목의 차트 데이터
observation: 현재 관측치
idx: 차트 데이터에서의 현재 위치

<Methods>
reset(): idx와 observation을 초기화
observe(): idx를 다음 위치로 이동하고 obseravation을 업데이트
get_price(): 현재 observation에서 종가를 획득
"""

class Environment:
    PRICE_IDX = 4   # The place of End-Price (Date, Open, High, Low, Close, Volume)
    # PRICE_IDX = 5   # The place of End-Price (Date, Open, High, Low, Close, Adj Close, Volume)


    def __init__(self, chart_data=None):
        self.chart_data = chart_data    # 생성자에서 관리할 차트 데이터를 할당 - 차트 데이터인 chart_data를 입력으로 받아서 저장
        self.observation = None         # 차트 데이터에서 현재 위치의 관측 값이 self.observation에 저장된다.
        self.idx = -1                   # 현재 위치는 self.idx에 저장된다.


    # Initialize the state in the constructor above, and also make initializing method here.
    # (생성자에서 초기화를 하고, 여기서도 초기화 함수를 만든다.)
    # 다시 차트 데이터의 처음으로 돌아가게 한다. (self.idx가 -1이어서 처음이 아니지 않느냐고 할 수 있지만, 1을 더하면 0이 되므로 차트의 처음이 맞다.)
    def reset(self):
        self.observation = None
        self.idx = -1


    # 하루 앞으로 이동하며 차트 데이터에서 관측 데이터(observation)을 제공한다.
    # 더 이상 제공할 데이터가 없을 때는 None을 반환한다.
    """
    self.idx = -1인 이유와 여기에 1을 더해서 차트를 하나씩 훑는 이유는 초기 idx가 -1이지만, 
    바로 1을 더하고 state를 관찰하므로, idx=0부터 시작하는 것과 다름이 없다.
    self.idx: -1+1=0 --> 0+1=1 --> 1+1=2 ...
    """
    def observe(self):
        if len(self.chart_data) > self.idx + 1:     # 차트 데이터의 전체 길이보다 다음 위치가 작을 경우 가져올 데이터가 있다는 뜻이며,
            self.idx += 1                           # 이 경우 현재 위치 self.idx에 1을 더하고
            self.observation = self.chart_data.iloc[self.idx]   # 차트 데이터에서 이 위치의 요소를 가져와 self.observation에 저장한다..
            return self.observation                             # 이때 iloc()함수는 특정 행의 데이터를 전부(D,O,H,L,C,V) 가져오므로, self.observation에는 종가 뿐만 아니라 다른 데이터 역시 담겨있다.
        return None     # 더 이상 제공할 데이터가 없을 때는 None을 반환한다.

    def get_ohlcv(self):
        if self.observation is not None:
            return self.observation[1:]
        return None

    def get_ohlc(self):
        if self.observation is not None:
            return self.observation[1:-1]
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]     # iloc()로 가져온 self.observation에는 (D,) O,H,L,C,V가 다 들어있으므로 종가의 인덱스를 통해 종가를 가져온다.
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data


