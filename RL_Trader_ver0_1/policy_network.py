# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 정책 신경망 모듈(policy_network.py)은 투자 행동을 결정하기 위해 신경망을 관리하는 정책 신경망 클래스를 가진다.

    # 정책 경사(Policy Gradient)는 분류(Classfication) 문제를 푸는 강화학습으로 볼 수 있다. 즉 (Q-러닝 처럼 기대 손익(수익률)을 예측하는 것이 아니라)
    # 현재 상태에서 어떤 행동을 하는 것이 가장 좋을지를 확률적으로 판단하는 것이다. (어떠한 상태에서의 주어진 행동에 대한 확률 값들을 인공 신경망으로 모델링한다.)
    # 본 프로젝트에서 행동은 정책 신경망(Policy Network)에서 행동별 확률을 구하여 결정하고, 정책 경사는 결정된 행동을 바탕으로 투자를 진행하면서 발생하는 보상과 학습 데이터를 학습한다.
    # 인공신경망(LSTM)으로 각 행동에 대한 확률을 구해서 높은 확률을 갖는 행동을 선택하고, 그 선택에 따른 결과들을 즉시 보상과 지연 보상으로 긍적적인 학습과 부정적인 학습을 하는 것이다.
    # 그리고 에포크가 늘어날 수록, 즉 경험이 늘어날 수록 긍정적인 보상을 얻을 수 있는 방향으로 (=LSTM에서 얻는 확률값이 더 높게 나오는 방향으로) 행동을 하게 될 것이다.
    # 확률을 구하는 것 자체는 인공신경망(LSTM)의 역할이며, 여기서 선택된 행동을 바탕으로 즉시/지연 보상을 얻으며 긍정적 행동을 해나가는 방향으로 진행시키는 것이 정책 경사의 역할(정책 신경망의 역할이 아님)이다.
    # "정책 신경망은 강화학습에서 에이전트가 행동!을 결정하기 위한 두뇌 역할을 하는 인공 신경망이며, 강화학습은 그 기법으로 Q_러닝 혹은 정책 경사가 있다. Q_러닝은 특정 행동을 취했을 때 예측되는 포트폴리오 가치를
    #  회귀하고자 사용하며(기대 손익을 수치적으로 예측), 정책 경사는 어떤 행동이 현재 상태에서 가장 좋을지를 확률적으로 판단한다.(기대 손익 예측이 아닌 현 상황에서 어떤 행동이 더 좋을지를 판단하는 것)"
    # TODO <<근본적 중요성>> 바로 위에서 언급한 "긍정적 행동을 진행 방향으로 잡고 목표 함수(maximize 긍정 지연 보상)를 최대화 하는 정책 경사의 역할" 이 policy_learner.py의 코드 중 어느 부분에 해당되는가?
    #                     ==> (같은 질문) policy_learner.py 에서 epoch 마다 초기화가 진행되는데 학습한 내용은 초기화되지 않을 것이다. 그럼 이 학습한 내용을 담는 코드는 무엇인가?
    #                     <부족한 의견> policy_network.py에서 정책 신경망의 출력이 매수 및 매도를 했을 때 포트폴리오 가치를 높이는 확률을 의미하므로 [이 확률에 대한 신뢰도/가중치 or 이 확률의 수치 자체]를 높이는
    #                                방향으로 정책 신경망의 학습이 진행된다면, --> 자연스레 긍정 지연 보상도 높아지지 않을까? 이게 곧 정책 경사의 역할이고?
    #                                 ("주식 투자에는 정답이 없기 때문에 학습이 어렵지만, 강화학습으로 확률적으로 좋은! 결정을 내리게 함으로써 수익이 발생하도록 한다.")
    #                                --> 문제는 이 의견의 근거가 타당하지 않고, 설령 맞다면 강화 학습이라는 것이 지도 학습인 인공 신경망(LSTM)에 의해 좌우된다는 뜻인데, 그럼 강화학습은 스스로 학습한다는 정의로부터 벗어나게 된다.
    #                     !!!!! 그러니까 {지연 보상으로 학습을 진행하고 긍정 학습을 통해 (직전 포트폴리오 가치 보다) 현재 포트폴리오 가치를 높이고 --> 그럼 다시 긍정 지연 보상으로 긍정학습을 하고 ... --> 반복 학습}을 하는건데 이걸 누가 담당하냐고. !!!!!
    #                           (policy_learner.py의 trade() 함수를 보면 학습이 끝난 후 학습 모델을 바탕으로 거래를 진행하는데, 이때 활용하는 학습 모델을 "policy_network.load_model"을 통해 가져온다.
    #                            이러면 정책 신경망과 정책 경사 모두 policy_network.py에 있는 것인가? 아니면 정책 경사 자체가 아예 정책 신경망에 포함된 개념인 것인가? (후자면 강화학습이 아니라 지도 학습이 된다.)
    #                                                                                                                                       (포함된 개념이지만 지도 학습이 되는게 아니라 정책 경사를 바탕으로 정책 신경망이 학습 되는 것인가?)


    # (※ 참고) 지연 보상이 발생했을 때, 그 이전 지연 보상이 발생한 시점과 현재 지연 보상이 발생한 시점 사이에서 수행한 행동들 전체에 현재의! 지연 보상을 적용한다.
    #        이때 과거로 갈수록 현재 지연 보상을 적용할 판단 근거가 흐려지기 때문에, 먼 과거의 행동일수록 할인 요인(discout factor)을 적용하여 지연 보상을 약하게 적용한다.**!

    # TODO (강화학습 기법 중 하나인 정책 경사는 현 상태에서 어느 행동이 좋을지를 확률적으로 판단하는 것이니 인공 신경망과의 결합이 필수인 것인가?)
    #   "정책 (경사) 기반 강화학습(Policy(Gradient)-based Reinforcement Learning): 가치 함수를 토대로 행동을 선택하지 않고, 상태(State)에 따라 행동을 선택 <-- 인공신경망이 정책(Policy)를 근사(Gradient)한다.
    #    정책 기반 강화학습에서는 인공신경망이 정책을 근사한다." --> 인공신경망의 입력은 상태(의 특징 벡터)가 되고 출력은 각 행동에 대한 확률이 된다. 따라서, 정책신경망에서는 출력 층의 활성 함수로 Softmax 함수(각 확률의 합은 1)를 사용한다.
    #    (※ 정책 신경망(Policy Network): 정책 기반 강화학습에서 정책을 근사하는 인공신경망) & (※ 본 프로젝트에서 에이전트의 상태(State)는 '행동'과 '보유 주식 수'를 의미한다.)
    #    또한 이 정책 경사는 목표함수 J(𝛉)를 최적화하는 방법은 '경사 상승법(Gradient Ascent)'을 사용한다. 이는 '경사 하강법(Gradient Descent)'과 다르게 오류함수를 최소화하는 것이 아니라 목표함수를 최대화하는 것이다."
    #    출처: [파이썬과 케라스로 배우는 강화학습, 이웅원 외 4명 저, 위키북스]


"""
<Attributes>
model: LSTM 신경망 모델
prob: 가장 최근에 계산한 각 투자 행동별 확률    # Agent 모듈을 봐도 prob은 각 행동들에 대한 확률을 모아놓은 것이며, 이 확률은 predict 메서드를 통해 구핼 수 있다.

<Methods>
reset(): prob 변수를 초기화
predict(): 신경망을 통해 투자 행동별 확률을 계산
train_on_batch(): 입력으로 들어온 배치 데이터로 학습을 진행한다.
save_model(): 학습한 신경망을 파일로 저장
load_model(): 파일로 저장한 신경망을 로드

<LSTM 신경망의 구조>
Input Layer(17차원) - Hidden Layer1(256차원) - Hidden Layer2(256차원) - Hidden Layer3(256차원) - Output Layer(2차원)
입력층은 17차원: 학습 데이터 차원인 15차원과 에이전트 상태 차원인 2차원을 합함.
출력층은 2차원: 투자 행동인 매수와 매도에 대한 확률로 2차원을 가짐.
"""


import numpy as np
from keras.optimizers import sgd    # Learning Algorithm: Stochastic Gradient Descent (SGD)
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, l_rate=0.01):     # 여기서 input_dim은 15+2로 17차원이며, output_dim은 매수/매도에 대한 확률로 2차원이다.
        self.input_dim = input_dim
        self.l_rate = l_rate

        # LSTM 신경망
        self.model = Sequential()       # keras에서 Sequential 클래스: 전체 신경망을 구성하는 클래스

        self.model.add(LSTM(256, input_shape=(1, input_dim), return_sequences=True, stateful=False, dropout=0.5))    # Hidden Layer 1
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))    # Hidden Layer 2
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))   # Hidden Layer 3
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))       # Output Layer
        self.model.add(Activation('sigmoid'))   # Activation Function

        self.model.compile(optimizer=sgd(lr=l_rate), loss='mse')
        # TODO 책 [파이썬과 케라스로 배우는 강화학습]에 아래 주석과 같은 문장이 있으므로, self.model.compile() 함수를 다시 따져봐야 한다. (책 읽은 후 다시 확인 필수!)
        #   또한 마지막 activation 함수의 인자가 반드시 'softmax'일 필요는 없는가?
        """
          Softmax
            정책신경망에는 model.complie이 없습니다.
            딥살사에서는 인공신경망의 학습을 MSE 오차를 줄이는 방식으로 진행했습니다.
            정책신경망에서는 MSE 오차 함수를 사용하지 않기 때문에 따로 오류 함수를 정의해야 합니다.
            (※정책신경망 : 정책 기반 강화학습에서 정책을 근사하는 인공신경망)
        """

        self.prob = None


    def reset(self):
        self.prob = None


    # 신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17차원의 입력을 받아서(즉 state를 입력 받아서), 매수와 매도가 수익을 높일 것으로 판단되는 확률(즉 Q-Learning의 Qpred 같은)을 구한다.
    # sample을 2차원으로 reshape 하는 이유: Sequential 클래스의 predict() 함수는 여러 샘플을 한번에 받아서 신경망의 출력을 반환한다. 하나의 샘플에 대한 결과만을 받고 싶어도 입력값을 샘플의 배열로 구성해야하기 때문에 2차원 배열로 재구성한다.
    # TODO 아래 TODO에도 써놨지만 2차원으로 재구성하는 이유는 알겠지만, 왜 3차원으로 재구성했다가 2차원으로 돌아오는지를 모르겠다.
    #       --> 코드를 잘못 봤따. sample을 3차원으로 만들고 뒤의 [0]은 sample이 아닌 model.predict(sample)의 결과에 붙는다. 여전히 문제 있음. 아래 todo문 확인)

    # 이 predict 매서드는 Agent 클래스의 decide_action 매서드가 사용한다. decide_action 메서드는 탐험 시에는 매수,매도,관망 중 한 가지를 랜덤으로 선택하게 되며,
    # 탐험이 아닐 시에는 이 predict 함수가 17차원의 sample을 입력으로 받아 매수,매도 2가지를 output으로 내놓게 된다. 정책 신경망의 출력은 매수, 매도를 했을 때의 포트폴리오 가치를 높일 확률들(매수에 대한 확률 하나와 매도에 대한 확률 하나로 총 2차원)이다.
    # "관망은 어디 있느냐"라는 질문에 대한 답으로는, 탐험이 아니면 우선 실제 행동을 실행하는 Agent 클래스의 act 메서드가 시작 부분에 있는 validate_action 함수 호출을 통해 매수, 매도를 할 수 없는 상황인 경우 관망을 선택한다.
    # 매수, 매도를 할 수 있는 것이 validate되면(True값이 나오면) act 메서드는 decide_action 메서드에서 결정된 action (매수와 매도 중 확률값이 큰 행동)을 토대로 실제 행동을 실행하게 된다.
    def predict(self, sample):
        # TODO 만일 sample의 shape가 (1,17)이라면, "self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]"은
        #      self.prob = self.model.predict(np.array(sample)) 과 같을텐데 저런 식으로 굳이 reshape를 해준 이유가 무엇일까?
        #      (뭐하러 2차원을 3차원으로 만들었다가 뒤에 [0]을 붙여서 다시 2차원으로 빼는 것인가? 그리고 왜 괄호가 중복으로 쌓여있을까?
        # TODO <반정도 해결> 잘못 이해했다. 우선 sample을 3차원으로 만드는 것은 맞다. 그러나 [0]은 그 sample에 적용되는 것이 아니라, sample을 model.predict를 한 결과, 즉 정책 신경망의 output인 각 행동에 대한 확률에 적용되는 것이다.
        #                 그렇다면 아직 남아있는 의문으로 어찌됐든 sample을 3차원으로 만드는 이유와 output의 모양과 [0]을 해주는 이유는 무엇인지가 있다.
        #                 policy_learner.py의 _get_batch() 함수에서도 sample을 3차원으로 reshape 한다.
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob


    # 입력으로 들어온 학습 데이터 집합 x와 레이블 y로 정책 신경망을 학습시킨다.
    # keras의 train_on_batch()는 입력으로 들어온 학습 데이터 집합, 즉 "1개"의 배치로 신경망을 "한 번" 학습한다.
    # train_on_batch: "Runs a single gradient update on a single batch of data."
    #   Returns: "Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
    #   The attribute model.metrics_names will give you the display labels for the scalar outputs."
    #   출처: [https://keras.io/models/model/]
    
    # train_on_batch 또 다른 설명 (from stack overflow)
    # "train_on_batch allows you to expressly update weights based on a collection of samples you provide, without regard to any fixed batch size. 
    # You would use this in cases when that is what you want: to train on an explicit collection of samples. 
    # You could use that approach to maintain your own iteration over multiple batches of a traditional training set 
    # but allowing fit or fit_generator to iterate batches for you is likely simpler.
    
    # One case when it might be nice to use train_on_batch is for updating a pre-trained model on a single new batch of samples. 
    # Suppose you've already trained and deployed a model, and sometime later you've received a new set of training samples previously never used. 
    # You could use train_on_batch to directly update the existing model only on those samples. 
    # Other methods can do this too, but it is rather explicit to use train_on_batch for this case."

    # "train_on_batch() gives you greater control of the state of the LSTM, for example, 
    # when using a stateful LSTM and controlling calls to model.reset_states() is needed. 
    # You may have multi-series data and need to reset the state after each series, which you can do with train_on_batch(), 
    # but if you used .fit() then the network would be trained on all the series of data without resetting the state. 
    # There's no right or wrong, it depends on what data you're using, and how you want the network to behave."
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)


    # 정책 신경망을 학습하는데 많은 시간과 컴퓨팅 자원이 소모되므로, 한번 학습한 정책 신경망을 (HDF5 파일 형식으로) 저장해 놓고 필요할 때 불러와서 사용하도록 한다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)     # (weight를 network로 간주)


    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)