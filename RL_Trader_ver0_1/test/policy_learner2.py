import os
import locale
import logging
import time
import datetime
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

class PolicyLearner:
    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=0.05, l_rate=0.01):

        self.stock_code = stock_code
        self.chart_data = chart_data

        self.environment = Environment(chart_data)

        self.agent = Agent(self.environment, min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1

        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM

        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, l_rate=l_rate )

        self.visualizer = Visualizer()

    def reset(self):
        self.sample = None
        self.training_data_idx = -1


    def fit(self, num_epoches=1000, max_memory=60, balance=1000000, discount_factor=0,
            start_epsilon=0.5, learning=True):
        logger.info("LR: {l_rate}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            l_rate = self.policy_network.l_rate,
            discount_factor = discount_factor,
            min_trading_unit = self.agent.min_trading_unit,
            max_trading_unit = self.agent.max_trading_unit,
            delayed_reward_threshold = self.agent.delayed_reward_threshold
        ))

        self.visualizer.prepare(self.environment.chart_data)

        # epoch_summary_dir = os.path.join(settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
        #     self.stock_code, settings.timestr
        # ))
        epoch_summary_dir = os.path.join(settings.BASE_DIR, 'epoch_summary', '%s' % self.stock_code,
                                         'epoch_summary_%s' % settings.timestr)

        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        self.agent.set_balance(balance)

        max_portfolio_value = 0
        epoch_win_cnt = 0

        for epoch in range(num_epoches):
            loss = 0.0
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 0
            pos_learning_cnt = 0
            neg_learning_cnt = 0

            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []

            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            self.visualizer.clear([0, len(self.chart_data)])

            if learning:
                epsilon = start_epsilon * (1. - (float(epoch) / (num_epoches - 1)))
            else:
                epsilon = 0

            while True:
                next_sample = self._build_sample()
                if next_sample in None:
                    break

                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, epsilon
                )
                immediate_reward, delayed_reward = self.agent.act(action, confidence)