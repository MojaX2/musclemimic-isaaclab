import unittest
import torch
from g1_muscle_posture import torso_collapsed


class PostureTests(unittest.TestCase):
    def test_upright_and_moderate_lean_are_valid_but_folded_trunk_is_not(self):
        # A level, correctly elevated pelvis alone cannot detect these failures.
        initial_gap=.5985066
        gaps=torch.tensor([initial_gap,.50,.40,.20,0.,-.38])
        self.assertEqual(torso_collapsed(gaps,initial_gap).tolist(),
                         [False,False,False,True,True,True])


if __name__=='__main__':unittest.main()
