class ProgressCallback:
    def __init__(self, total):
        self.total = total
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Progress: {self.count}/{self.total} iterations", end='\r')