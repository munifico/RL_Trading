# Close가 아닌 Adj Close를 추가하여 IDX는 4가 아닌 5가 된다.

class Enviroment:
    ADJ_CLOSE_IDX = 5   # The place of Adj Close (Date, Open, High, Low, Close, Adj Close, Volume)

    def __init(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.ADJ_CLOSE_IDX]
        return None

