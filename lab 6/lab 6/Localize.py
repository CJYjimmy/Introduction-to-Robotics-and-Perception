
class Localize():
    def __init__(self):
        self.positions = np.array([])
    def push(self, position):
        if self.positions.size == 0:
            self.positions = np.array(position)
        else:
            self.positions = np.vstack((self.positions, position))
    def get(self):
        return np.average(self.positions, axis=0)
