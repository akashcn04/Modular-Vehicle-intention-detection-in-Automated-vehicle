class TrafficLightBuffer:
    def __init__(self, max_age=3):
        self.last_state = None
        self.age = 0
        self.max_age = max_age

    def update(self, current_state):
        """
        current_state: RED / YELLOW / GREEN / UNKNOWN
        """
        if current_state != "UNKNOWN":
            self.last_state = current_state
            self.age = 0
            return current_state

        if self.last_state is not None and self.age < self.max_age:
            self.age += 1
            return self.last_state

        return "UNKNOWN"
