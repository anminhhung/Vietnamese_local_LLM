from queue import Empty, Queue


class BatchStreamer:
    def __init__(self, timeout: float = None):
        self.q = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, values):
        self.q.put(values)

    def end(self):
        self.q.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.q.get(timeout=self.timeout)
        if result == self.stop_signal:
            raise StopIteration()
        else:
            return result
