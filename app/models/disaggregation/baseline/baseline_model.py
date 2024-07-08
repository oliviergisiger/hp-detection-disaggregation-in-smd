import numpy as np
import matplotlib.pyplot as plt


class BaselineDisaggregatior:

    def __init__(self, M: np.ndarray, datetime: bool = False):
        if datetime:
            self.datetime = M[:, 0].astype('datetime64[s]')
        else:
            self.datetime = np.linspace(0, M.shape[0]-1, M.shape[0])
        self.aggregated_w = M[:, 0]
        self.temperature = M[:, 1]

    def predict(self, X):
        pass

    def decompose(self, p: int = 0.1, t: float = 12, plot: bool = False):
        hp_events = np.roll(self.get_hp_events(), -1)
        hp_state = np.clip(self.get_hp_state(p, t), 0, 1)
        dx = np.diff(self.aggregated_w, append=self.aggregated_w[-1])
        _dx = self._continuous(np.where(hp_events == 1, dx, 0))
        hh_load = np.clip(self.aggregated_w - np.roll(_dx, 1) * hp_state, 0, np.inf)
        if np.mean(self.temperature) > t:
            hh_load = self.aggregated_w
        hp_load = np.clip(self.aggregated_w - hh_load, 0, np.inf)
        y = np.column_stack([self.aggregated_w, hp_load, hh_load])
        if plot:
            self._plot(np.linspace(1, y.shape[0], y.shape[0]), y)
        return y

    def get_hp_state(self, p: int = 0.1, t: float = 16):
        return self._continuous(self.get_hp_events(p, t))

    def get_hp_events(self, p: int = 0.1, t=7):
        y = np.diff(self.aggregated_w, prepend=self.aggregated_w[0])
        on = (y > p).astype(int)
        off = (y < (-p)).astype(int)
        no = ((y > (-p)) & (y < p)).astype(int)
        _events = np.argmax(np.column_stack([off, no, on]), axis=1) - 1
        on_events = self._events(_events, 0, 1)
        off_events = self._events(_events, -1, 0)
        return np.where(off_events == -1, off_events, on_events)

    @staticmethod
    def _plot(x, y):
        plt.rcParams['font.size'] = 9
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 6))
        titles = ['input', 'heat pump', 'household']
        for i in range(3):
            ax[i].plot(x, y[:, i], linewidth=0.6)
            ax[i].set_title(titles[i], x=0.92, y=1.0, pad=-14)
            ax[i].set_xticks([0, 24, 48, 72, 96])
        plt.savefig('decomposition.png')

    @staticmethod
    def _events(x, l, u):
        return np.clip(np.diff(np.clip(x, l, u), prepend=x[0]), l, u)

    @staticmethod
    def _continuous(x):
        m = x == 0
        i = np.where(~m, np.arange(m.size), 0)
        np.maximum.accumulate(i, out=i)
        return x[i]

    @staticmethod
    def _interpolate(y, factor=2):
        x = np.arange(0, y.size)
        x_out = np.arange(0, y.size, (1 / factor))
        return np.interp(x_out, x, y)

    
if __name__ == '__main__':
    ...
