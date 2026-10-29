from microban_model import JOINTS

I = {n: JOINTS.index(n) for n in ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow")}

# Ce script sert a bouger manuellement le bras gauche du robot avec des slider

class moveRightArmMove:
    description = "Test bras droit : un slider par articulation"
    variants = ("climber",)
    kp = 125
    options = {}
    option_docs = {}
    sliders = {
        "shoulder_pitch": (-2.0, 2.0, 0.0),
        "shoulder_roll":  (-2.0, 2.0, 0.0),
        "elbow":          (-2.0, 2.0, 0.0),
    }

    def start(self, obs):
        self.q0 = obs["q"].copy()

    def step(self, obs):
        t = self.q0.copy()
        t[I["right_shoulder_pitch"]] = self.q0[I["right_shoulder_pitch"]] + self.slider["shoulder_pitch"]
        t[I["right_shoulder_roll"]]  = self.q0[I["right_shoulder_roll"]]  + self.slider["shoulder_roll"]
        t[I["right_elbow"]]          = self.q0[I["right_elbow"]]          + self.slider["elbow"]
        return t