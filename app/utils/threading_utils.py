from concurrent.futures import ThreadPoolExecutor


class BackgroundRunner:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)

    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)
