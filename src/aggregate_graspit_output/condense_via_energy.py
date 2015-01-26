import numpy as np


class CondenseGraspsViaEnergy():

    def __init__(self, threshold=None):
        self.threshold = threshold

    def getNumGrasps(self, grasps):
        return sum([len(grasps[model_name]) for model_name in grasps.keys()])

    def run(self, grasps):
        """
        Condense the grasps list by discarding low energy grasps. Grasps with an
        energy falling EITHER lower than a given threshold value OR Grasps with
        an energy falling below one standard deviations from the mean of grasps
        for that object type. This function returns the condensed grasp
        dictionary.

        :param grasps: A dictionary of grasps categorized by object
        :return: A condensed dictionary of grasps
        """

        energyByObj = dict([(model_name,
                             [grasp.energy
                              for grasp in grasps[model_name]])
                            for model_name in grasps.keys()])


        for object, energyList in energyByObj.items():
            mean = np.mean(energyList)
            std = np.std(energyList)

            for grasp in grasps[object]:
                if self.threshold is None:
                    if grasp.energy < (mean - std):
                        grasps[object].remove(grasp)
                else:
                    if grasp.energy < self.threshold:
                        grasps[object].remove(grasp)


        return grasps
