
class ModelCache:
    def __init__(self, max_size):
        self.cache_max_size = max_size or 3
        self.cache = {}  # {m_id:model}
        self.order = []  # [m_id] in self.order of calling/adding least called always at the bottom

    def clean(self):
        'uses LRU to clean up unused models from cache'
        lru = self.order.pop(-1)
        del self.cache[lru]

    def add(self, model):
        if len(self.order) + 1 > self.cache_max_size:
            self.clean()
        self._add(model)

    def _add(self, model):
        _id = model['id']
        self.cache[_id] = model
        self.order.insert(0, _id)

    def fetch(self, m_id, target_id, company):
        from ml import load_model
        model = self.cache.get(m_id)
        if model:
            self.order.insert(0, self.order.pop(m_id))
        else:
            if len(self.order) + 1 > self.cache_max_size:
                self.clean()
            model = load_model(company, m_id, target_id)
            self._add(model)
        return model