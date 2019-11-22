class Parameter:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys += ['s', 'a', 'r', 'm', 'v', 'q', 'pi', 'log_pi', 'ent', 'adv', 'ret', 'q_a', 'log_pi_a', 'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])