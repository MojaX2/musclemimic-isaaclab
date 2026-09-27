import math,unittest
import torch
from cartesian_reference_reward import CartesianReferenceReward,aligned_relative_sites


class CartesianReferenceTest(unittest.TestCase):
    def test_positive_score_bounds_and_phase_baseline_difference(self):
        standing=torch.zeros(2,3)
        coefficients=torch.zeros(3,2,3)
        coefficients[1,:,0]=torch.tensor([.2,-.2])
        coefficients[2,:,2]=torch.tensor([.04,-.04])
        legacy=CartesianReferenceReward(coefficients,standing)
        positive=CartesianReferenceReward(coefficients,standing,subtract_standing_baseline=False)
        phase=torch.tensor([0.,.7,1.6]);origin=torch.zeros(3,3);heading=torch.zeros(3);command=torch.ones(3)
        target=positive.target(phase)
        self.assertTrue(torch.equal(positive(target,origin,heading,phase,command),torch.ones(3)))
        standing_score=positive(standing.expand(3,-1,-1),origin,heading,phase,command)
        self.assertTrue(((standing_score>0)&(standing_score<1)).all())
        for offset in [0.,.03,10.]:
            sites=target+offset
            score=positive(sites,origin,heading,phase,command)
            self.assertTrue(((score>=0)&(score<=1)).all())
            self.assertTrue(torch.allclose(score-legacy(sites,origin,heading,phase,command),standing_score,atol=1e-7))
        self.assertTrue(torch.equal(positive(target,origin,heading,phase,torch.zeros(3)),torch.zeros(3)))

    def test_planar_rigid_transform_invariance_and_vertical_sensitivity(self):
        sites=torch.tensor([[[.1,.2,-.8],[-.3,.1,-.75]]]);pelvis=torch.zeros(1,3)
        angle=torch.tensor([.7]);c=angle.cos()[0];s=angle.sin()[0]
        rotation=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]])
        translation=torch.tensor([[4.,-2.,.9]])
        transformed=sites@rotation.T+translation[:,None]
        aligned=aligned_relative_sites(transformed,translation,angle)
        self.assertTrue(torch.allclose(aligned,sites,atol=1e-6))
        transformed[:,:,2]+=.1
        self.assertFalse(torch.allclose(aligned_relative_sites(transformed,translation,angle),sites))

    def test_target_beats_standing_and_phase_seam(self):
        standing=torch.tensor([[0.,0.,-.8],[0.,0.,-.8]])
        coefficients=torch.zeros(3,2,3);coefficients[0]=standing
        coefficients[1,:,0]=torch.tensor([.2,-.2]);coefficients[2,:,2]=torch.tensor([.04,-.04])
        reward=CartesianReferenceReward(coefficients,standing)
        phase=torch.tensor([0.,1.,2*math.pi]);command=torch.ones(3)
        target=reward.target(phase);origin=torch.zeros(3,3);heading=torch.zeros(3)
        self.assertTrue(torch.allclose(target[0],target[-1],atol=1e-6))
        score=reward(target,origin,heading,phase,command)
        baseline=reward(standing.expand(3,-1,-1),origin,heading,phase,command)
        self.assertTrue((score>baseline).all());self.assertTrue(torch.equal(baseline,torch.zeros(3)))
        self.assertTrue(torch.equal(reward(target,origin,heading,phase,torch.zeros(3)),torch.zeros(3)))
        perturbed=target.clone();perturbed[:,:,0]+=.05
        self.assertTrue((reward(perturbed,origin,heading,phase,command)<score).all())


if __name__=='__main__':unittest.main()
