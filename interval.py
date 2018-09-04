class interval():
    def __init__(self,interval):
        self.interval = interval
    def __lt__(self, other):
        return self.getDiam(self.interval) < self.getDiam(other.interval)

    def getDiam(self, interval):
        sum = 0
        for i in range(len(interval)):
            sum += interval[i].diam()
        return sum
