

class GraspPriorsList():

    def __init__(self):

        self.grasp_priors_list = []

    def add_grasp_prior(self, grasp_prior):
        self.grasp_priors_list.append(grasp_prior)

    def get_grasp_prior(self, index):
        return self.grasp_priors_list[index]


class GraspPrior():
    def __init__(self):

        self.joint_values = []
        self.dof_values = []
        self.wrist_roll = 0.0
        self.uvd = 0, 0, 0
