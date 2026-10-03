"""Check tactile observation masking and finite-horizon PPO targets."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from infant_crawl_env import PALM_JOINTS
from infant_crawl_oscillator import oscillator_phase
from train_infant_crawl_ppo import (SupportPolicy, forward_crawl_reward, gae,
                                    hand_switch_reward, initialize_policy_from_checkpoint,
                                    observe, rollout, update)


class InfantCrawlPpoTests(unittest.TestCase):
    def test_masked_ppo_update_leaves_overridden_actor_dimension_unchanged(self):
        torch.manual_seed(89)
        policy = SupportPolicy(3, 2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        observations = torch.randn((2, 2, 3))
        actions = torch.randn((2, 2, 2))
        weights = torch.zeros_like(actions)
        weights[:, :, 0] = 1
        with torch.no_grad():
            old_log_probs = (policy.distribution(observations).log_prob(actions) *
                             weights).sum(-1)
            values = policy.critic(observations).squeeze(-1)
        rewards = torch.ones_like(values)
        before = policy.actor[-1].weight.detach().clone()
        batch = (observations, actions, old_log_probs, rewards, values,
                 torch.zeros(2), weights)
        update(policy, optimizer, batch, epochs=1, minibatch_size=4)
        after = policy.actor[-1].weight.detach()
        self.assertFalse(torch.equal(before[0], after[0]))
        torch.testing.assert_close(before[1], after[1])

    def test_reference_anchor_requires_pretrained_policy(self):
        policy = SupportPolicy(2, 1)
        optimizer = torch.optim.Adam(policy.parameters())
        observations = torch.zeros((1, 1, 2))
        actions = torch.zeros((1, 1, 1))
        scalars = torch.zeros((1, 1))
        batch = (observations, actions, scalars, scalars, scalars, scalars[0])
        with self.assertRaisesRegex(ValueError, "reference policy"):
            update(policy, optimizer, batch, reference_strength=1.0)

    def test_phase_observation_tracks_oscillator(self):
        parameters = torch.tensor([[1., 0., 0., 0., 0., 0., 0.]])
        torch.testing.assert_close(oscillator_phase(parameters, 0.25),
                                   torch.tensor([[1., 0.]]), atol=1e-6, rtol=0.)

    def test_phase_extension_preserves_pretrained_policy_outputs(self):
        original = SupportPolicy(5, 2)
        extended = SupportPolicy(7, 2)
        initialize_policy_from_checkpoint(extended, {"policy": original.state_dict()},
                                          extend_phase=True)
        observation = torch.randn((3, 5))
        phase = torch.randn((3, 2))
        torch.testing.assert_close(extended.actor(torch.cat((observation, phase), dim=1)),
                                   original.actor(observation))
        torch.testing.assert_close(extended.critic(torch.cat((observation, phase), dim=1)),
                                   original.critic(observation))

    def test_tactile_feature_extension_preserves_pretrained_policy_outputs(self):
        original = SupportPolicy(5, 2)
        extended = SupportPolicy(133, 2)
        initialize_policy_from_checkpoint(extended, {"policy": original.state_dict()},
                                          extra_observation_size=128)
        observation = torch.randn((3, 5))
        latent = torch.randn((3, 128))
        torch.testing.assert_close(extended.actor(torch.cat((observation, latent), dim=1)),
                                   original.actor(observation))

    def test_phase_and_tactile_extension_preserves_pretrained_policy_outputs(self):
        original = SupportPolicy(5, 2)
        extended = SupportPolicy(135, 2)
        initialize_policy_from_checkpoint(extended, {"policy": original.state_dict()},
                                          extend_phase=True, extra_observation_size=128)
        observation = torch.randn((3, 5))
        phase = torch.randn((3, 2))
        latent = torch.randn((3, 128))
        combined = torch.cat((observation, phase, latent), dim=1)
        torch.testing.assert_close(extended.actor(combined), original.actor(observation))
        torch.testing.assert_close(extended.critic(combined), original.critic(observation))

    def test_touch_ablation_only_masks_ground_touch(self):
        environment = SimpleNamespace(
            root_address=0,
            initial=torch.zeros((1, 8)),
            position=torch.zeros((1, 8)),
            velocity=torch.zeros((1, 8)),
            crawl_qpos_ids=torch.tensor([7]),
            crawl_qvel_ids=torch.tensor([7]),
            crawl_actuators=torch.tensor([0]),
            source=SimpleNamespace(nu=1),
            muscles=SimpleNamespace(activity=torch.tensor([[0.2, 0.4]])),
        )
        measure = {"ground_touch": torch.tensor([[0.0, 9.0]]),
                   "chest_height": torch.tensor([0.2]),
                   "head_height": torch.tensor([0.3])}
        with_touch = observe(environment, measure, True)
        without_touch = observe(environment, measure, False)
        self.assertEqual(with_touch.shape, without_touch.shape)
        self.assertEqual(int(torch.count_nonzero(with_touch - without_touch)), 1)
        self.assertGreater(float(with_touch[0, -3]), 0)
        environment.controlled_joint_names = PALM_JOINTS
        measure["palm_shin_force"] = torch.tensor([[8., 0., 4., 0.]])
        palm_touch = observe(environment, measure, True)
        palm_masked = observe(environment, measure, False)
        self.assertEqual(palm_touch.shape[-1], with_touch.shape[-1] + 4)
        self.assertEqual(int(torch.count_nonzero(palm_touch - palm_masked)), 3)

    def test_gae_terminal_bootstrap_is_zero(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros_like(rewards)
        advantage, target = gae(rewards, values, gamma=1.0, decay=1.0)
        torch.testing.assert_close(advantage[:, 0], torch.tensor([3.0, 2.0]))
        torch.testing.assert_close(target, advantage)

    def test_forward_reward_requires_elevated_contact_and_progress(self):
        previous = {"forward_displacement": torch.zeros(4)}
        current = {"forward_displacement": torch.tensor([0.01, 0.01, -0.01, -0.01]),
                   "palm_shin_force": torch.tensor([[5., 0., 0., 5.],
                                                     [0., 0., 5., 5.],
                                                     [5., 0., 5., 0.],
                                                     [0., 0., 0., 0.]]),
                   "chest_height": torch.tensor([0.2, 0.2, 0.2, 0.1]),
                   "head_height": torch.tensor([0.2, 0.2, 0.2, 0.1])}
        reward = forward_crawl_reward(previous, current, 0.05, 1.0)
        torch.testing.assert_close(reward, torch.tensor([0.2, 0.0, -0.2, -0.2]))

    def test_hand_switch_reward_requires_elevated_shin_support(self):
        previous = torch.tensor([1, 1, 1])
        forces = torch.tensor([[0., 5., 5., 0.], [0., 5., 0., 0.],
                               [0., 5., 5., 0.]])
        reward, remembered = hand_switch_reward(previous, forces,
                                                 torch.tensor([0.2, 0.2, 0.1]),
                                                 torch.full((3,), 0.2), 0.5)
        torch.testing.assert_close(reward, torch.tensor([0.5, 0., 0.]))
        torch.testing.assert_close(remembered, torch.tensor([-1, -1, -1]))

    def test_world_features_use_current_state_and_action_then_next_observation(self):
        class FakeEnvironment:
            worlds, device, root_address = 1, "cpu", 0
            source = SimpleNamespace(nu=1)
            crawl_qpos_ids = torch.tensor([7])
            crawl_qvel_ids = torch.tensor([7])
            crawl_actuators = torch.tensor([0])
            initial = torch.zeros((1, 8))
            velocity = torch.zeros((1, 8))
            muscles = SimpleNamespace(activity=torch.zeros((1, 2)))

            def reset(self, positions):
                self.position = positions.clone()

            def measure(self):
                return {"touch": torch.zeros((1, 15)),
                        "ground_touch": torch.zeros((1, 15)),
                        "support_force": torch.zeros((1, 4)),
                        "chest_height": torch.tensor([0.2]),
                        "head_height": torch.tensor([0.2])}

            def action_from_drives(self, drives):
                return torch.tensor([[0.2, 0.3]])

            def step(self, action, physics_steps):
                self.position[:, 7] += 1
                return self.measure()

        class FakePolicy:
            def sample(self, observation):
                captured.append(observation[0, -256:].clone())
                return torch.zeros((1, 22)), torch.zeros(1), torch.zeros(1)

        class FakeFeatures:
            def encode(self, environment, measure, action):
                calls.append((float(environment.position[0, 7]), action.clone()))
                return torch.full((1, 256), 17 + environment.position[0, 7].item())

        captured, calls = [], []
        environment = FakeEnvironment()
        with patch("train_infant_crawl_ppo.jittered_positions", return_value=environment.initial):
            rollout(environment, FakePolicy(), 2, 1, True, 0, FakeFeatures())
        self.assertEqual([state for state, _ in calls], [0.0, 1.0])
        torch.testing.assert_close(calls[0][1], torch.tensor([[0.2, 0.3]]))
        torch.testing.assert_close(captured[0], torch.zeros(256))
        torch.testing.assert_close(captured[1], torch.full((256,), 17.0))


if __name__ == "__main__":
    unittest.main()
