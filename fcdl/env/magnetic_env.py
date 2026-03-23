import copy
import logging

import gym
import numpy as np
from gym.utils import seeding

try:
    from robosuite.controllers import load_composite_controller_config
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BallObject, BoxObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
except ImportError as exc:
    _ROBOSUITE_IMPORT_ERROR = exc
else:
    _ROBOSUITE_IMPORT_ERROR = None

logging.getLogger("robosuite_logs").setLevel(logging.WARNING)


RED_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
BLACK_RGBA = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class RobosuiteMagneticTask(ManipulationEnv):
    """Robosuite task that matches the Magnetic environment description from Appendix C."""

    def __init__(
        self,
        robot,
        horizon,
        control_freq,
        table_full_size,
        table_friction,
        table_offset,
        action_low,
        action_high,
        ball_radius,
        box_size,
        ball_density,
        ball_friction,
        box_density,
        box_friction,
        magnetic_force,
        goal_height,
        reward_scale,
        reach_threshold,
        seed=None,
    ):
        self.robot = robot
        self.table_full_size = np.array(table_full_size, dtype=np.float32)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array(table_offset, dtype=np.float32)

        self.action_low = np.array(action_low, dtype=np.float32)
        self.action_high = np.array(action_high, dtype=np.float32)
        self.action_scale = np.maximum(np.abs(self.action_low), np.abs(self.action_high))

        self.ball_radius = float(ball_radius)
        self.box_size = np.array(box_size, dtype=np.float32)
        self.ball_density = float(ball_density)
        self.ball_friction = tuple(ball_friction)
        self.box_density = float(box_density)
        self.box_friction = tuple(box_friction)
        self.magnetic_force = float(magnetic_force)
        self.goal_height = float(goal_height)
        self.reward_scale = float(reward_scale)
        self.reach_threshold = float(reach_threshold)

        self.ball_color = 0
        self.box_color = 0
        self.override_box_xy = None
        self.fixed_box_xy = np.zeros(2, dtype=np.float64)
        self.fixed_box_quat = IDENTITY_QUAT_WXYZ.copy()
        self.arm_name = None

        controller_configs = self._build_controller_config()

        super().__init__(
            robots=robot,
            env_configuration="default",
            controller_configs=controller_configs,
            gripper_types=None,
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            hard_reset=False,
            renderer="mjviewer",
            seed=seed,
        )

    def _build_controller_config(self):
        controller_configs = copy.deepcopy(load_composite_controller_config(robot=self.robot))
        arm_name = next(iter(controller_configs["body_parts"]))
        arm_cfg = controller_configs["body_parts"][arm_name]
        arm_cfg["type"] = "OSC_POSITION"
        arm_cfg["input_max"] = 1
        arm_cfg["input_min"] = -1
        arm_cfg["output_max"] = self.action_high.tolist()
        arm_cfg["output_min"] = self.action_low.tolist()
        arm_cfg.pop("orientation_limits", None)
        arm_cfg.pop("uncouple_pos_ori", None)
        arm_cfg.pop("gripper", None)
        return controller_configs

    def set_episode_context(self, ball_color, box_color, override_box_xy=None):
        self.ball_color = int(ball_color)
        self.box_color = int(box_color)
        if override_box_xy is None:
            self.override_box_xy = None
        else:
            self.override_box_xy = np.asarray(override_box_xy, dtype=np.float64)

    def is_magnetic_context(self):
        return self.ball_color == 1 and self.box_color == 1

    def get_ball_xy(self):
        return np.array(self.sim.data.body_xpos[self.ball_body_id][:2], dtype=np.float32)

    def get_box_xy(self):
        return np.array(self.sim.data.body_xpos[self.box_body_id][:2], dtype=np.float32)

    def get_eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[self.arm_name]], dtype=np.float32)

    def reward(self, action=None):
        goal = np.array([*self.get_ball_xy(), self.goal_height], dtype=np.float32)
        distance = np.abs(self.get_eef_pos() - goal).sum()
        return float(1.0 - np.tanh(self.reward_scale * distance))

    def _check_success(self):
        goal = np.array([*self.get_ball_xy(), self.goal_height], dtype=np.float32)
        distance = np.abs(self.get_eef_pos() - goal).sum()
        return bool(distance < self.reach_threshold)

    def _sample_box_xy(self):
        if self.override_box_xy is not None:
            return self.override_box_xy

        placements = self.placement_initializer.sample()
        box_pos, _, _ = placements[self.box.name]
        return np.array(box_pos[:2], dtype=np.float64)

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.ball = BallObject(
            name="ball",
            size=[self.ball_radius],
            rgba=RED_RGBA.tolist(),
            density=self.ball_density,
            friction=self.ball_friction,
            joints="default",
        )
        self.box = BoxObject(
            name="box",
            size=self.box_size.tolist(),
            rgba=RED_RGBA.tolist(),
            density=self.box_density,
            friction=self.box_friction,
            joints="default",
        )

        x_range = [-self.table_full_size[0] / 2.0, self.table_full_size[0] / 2.0]
        y_range = [-self.table_full_size[1] / 2.0, self.table_full_size[1] / 2.0]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="BallSampler",
                mujoco_objects=self.ball,
                x_range=x_range,
                y_range=y_range,
                rotation=[0.0, 0.0],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )
        )
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="BoxSampler",
                mujoco_objects=self.box,
                x_range=x_range,
                y_range=y_range,
                rotation=[0.0, 0.0],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.ball, self.box],
        )

    def _setup_references(self):
        super()._setup_references()
        self.arm_name = self.robots[0].arms[0]
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)
        self.ball_geom_ids = [self.sim.model.geom_name2id(name) for name in self.ball.contact_geoms + self.ball.visual_geoms]
        self.box_geom_ids = [self.sim.model.geom_name2id(name) for name in self.box.contact_geoms + self.box.visual_geoms]

    def _set_object_rgba(self, geom_ids, rgba):
        for geom_id in geom_ids:
            self.sim.model.geom_rgba[geom_id] = rgba

    def _set_box_pose(self):
        box_z = float(self.table_offset[2] + self.box_size[2])
        qpos = np.concatenate([np.array([self.fixed_box_xy[0], self.fixed_box_xy[1], box_z]), self.fixed_box_quat])
        self.sim.data.set_joint_qpos(self.box.joints[0], qpos)
        self.sim.data.set_joint_qvel(self.box.joints[0], np.zeros(6))

    def _apply_magnetic_force(self):
        self.sim.data.xfrc_applied[self.ball_body_id, :] = 0.0
        if not self.is_magnetic_context():
            return

        ball_pos = np.array(self.sim.data.body_xpos[self.ball_body_id], dtype=np.float64)
        box_pos = np.array(self.sim.data.body_xpos[self.box_body_id], dtype=np.float64)
        direction = box_pos - ball_pos
        direction[2] = 0.0
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return

        force = self.magnetic_force * direction / norm
        self.sim.data.xfrc_applied[self.ball_body_id, :3] = force

    def _reset_internal(self):
        super()._reset_internal()

        placements = self.placement_initializer.sample()
        ball_pos, ball_quat, _ = placements[self.ball.name]
        self.sim.data.set_joint_qpos(self.ball.joints[0], np.concatenate([np.array(ball_pos), np.array(ball_quat)]))
        self.sim.data.set_joint_qvel(self.ball.joints[0], np.zeros(6))

        self.fixed_box_xy = self._sample_box_xy()
        self._set_box_pose()

        self._set_object_rgba(self.ball_geom_ids, RED_RGBA if self.ball_color else BLACK_RGBA)
        self._set_object_rgba(self.box_geom_ids, RED_RGBA if self.box_color else BLACK_RGBA)
        self.sim.data.xfrc_applied[self.ball_body_id, :] = 0.0
        self.sim.data.xfrc_applied[self.box_body_id, :] = 0.0
        self.sim.forward()

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self._set_box_pose()
        self._apply_magnetic_force()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        info["success"] = self._check_success()
        info["magnetic"] = self.is_magnetic_context()
        return reward, done, info

    def _check_robot_configuration(self, robots):
        robots = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        assert len(robots) == 1, "Magnetic only supports a single robot."


class Magnetic(gym.Env):
    """FCDL wrapper over a Robosuite magnetic manipulation task."""

    def __init__(self, params):
        if _ROBOSUITE_IMPORT_ERROR is not None:
            raise ImportError(
                "Robosuite and MuJoCo are required for the Magnetic environment."
            ) from _ROBOSUITE_IMPORT_ERROR

        self.params = params
        self.env_params = env_params = params.env_params
        self.magnetic_env_params = magnetic_env_params = env_params.magnetic_env_params

        self.name = magnetic_env_params.name
        self.max_steps = magnetic_env_params.max_steps
        self.reward_scale = magnetic_env_params.reward_scale
        self.goal_height = magnetic_env_params.goal_height
        self.reach_threshold = magnetic_env_params.reach_threshold

        self.action_low = np.array(magnetic_env_params.action_low, dtype=np.float32)
        self.action_high = np.array(magnetic_env_params.action_high, dtype=np.float32)
        self.action_scale = np.maximum(np.abs(self.action_low), np.abs(self.action_high))

        self.action_dim = 3
        self.action_spec = (self.action_low, self.action_high)
        self.num_action_variable = 3

        # ball color, ball x, ball y, box color, box x, box y, eef x, eef y, eef z
        self.feature_inner_dim = np.array([2, 1, 1, 2, 1, 1, 1, 1, 1], dtype=np.int64)
        self.continuous_state = False
        self.continuous_action = True
        self.continuous_factor = False

        self.np_random = None
        if params.seed != -1:
            self.seed(params.seed)
            robosuite_seed = params.seed
        else:
            self.seed()
            robosuite_seed = None

        self.robosuite_env = RobosuiteMagneticTask(
            robot=getattr(magnetic_env_params, "robot", "Panda"),
            horizon=self.max_steps,
            control_freq=getattr(magnetic_env_params, "control_freq", 20),
            table_full_size=getattr(magnetic_env_params, "table_full_size", [0.6, 0.9, 0.05]),
            table_friction=getattr(magnetic_env_params, "table_friction", [1.0, 5e-3, 1e-4]),
            table_offset=getattr(magnetic_env_params, "table_offset", [0.0, 0.0, 0.8]),
            action_low=self.action_low,
            action_high=self.action_high,
            ball_radius=getattr(magnetic_env_params, "ball_radius", 0.025),
            box_size=getattr(magnetic_env_params, "box_size", [0.03, 0.03, 0.03]),
            ball_density=getattr(magnetic_env_params, "ball_density", 1000.0),
            ball_friction=getattr(magnetic_env_params, "ball_friction", [1.0, 5e-3, 1e-4]),
            box_density=getattr(magnetic_env_params, "box_density", 5000.0),
            box_friction=getattr(magnetic_env_params, "box_friction", [1.0, 5e-3, 1e-4]),
            magnetic_force=getattr(magnetic_env_params, "magnetic_force", 0.5),
            goal_height=self.goal_height,
            reward_scale=self.reward_scale,
            reach_threshold=self.reach_threshold,
            seed=robosuite_seed,
        )

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_train_colors(self):
        return int(self.np_random.integers(0, 2)), int(self.np_random.integers(0, 2))

    def _sample_test_colors(self):
        if self.np_random.uniform() < 0.5:
            return 0, 1
        return 1, 0

    def _sample_colors(self):
        if self.name.startswith("test"):
            return self._sample_test_colors()
        return self._sample_train_colors()

    def _sample_box_override(self):
        if self.name.startswith("test") and getattr(self.magnetic_env_params, "ood_box", False):
            sigma = float(getattr(self.magnetic_env_params, "test_box_sigma", 100.0))
            return self.np_random.normal(0.0, sigma, size=2).astype(np.float64)
        return None

    def _normalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_low, self.action_high)
        return np.divide(action, self.action_scale, out=np.zeros_like(action), where=self.action_scale > 0)

    def get_state(self):
        return {
            "ball": np.array(
                [
                    self.robosuite_env.ball_color,
                    *self.robosuite_env.get_ball_xy(),
                ],
                dtype=np.float32,
            ),
            "box": np.array(
                [
                    self.robosuite_env.box_color,
                    *self.robosuite_env.get_box_xy(),
                ],
                dtype=np.float32,
            ),
            "eef": self.robosuite_env.get_eef_pos().astype(np.float32),
        }

    def get_gt_local_mask(self):
        lcm = np.zeros((len(self.feature_inner_dim), len(self.feature_inner_dim) + self.num_action_variable), dtype=np.float32)

        # colors are static
        lcm[0, 0] = 1.0
        lcm[3, 3] = 1.0

        # box is static
        lcm[4, 4] = 1.0
        lcm[5, 5] = 1.0

        # robot arm follows delta-position actions
        lcm[6, 6] = 1.0
        lcm[6, 9] = 1.0
        lcm[7, 7] = 1.0
        lcm[7, 10] = 1.0
        lcm[8, 8] = 1.0
        lcm[8, 11] = 1.0

        if self.robosuite_env.is_magnetic_context():
            lcm[1, [1, 2, 4, 5]] = 1.0
            lcm[2, [1, 2, 4, 5]] = 1.0
        else:
            lcm[1, 1] = 1.0
            lcm[2, 2] = 1.0

        return lcm

    def observation_spec(self):
        return self.get_state()

    def observation_dims(self):
        return {
            "ball": np.array([2, 1, 1], dtype=np.int64),
            "box": np.array([2, 1, 1], dtype=np.int64),
            "eef": np.array([1, 1, 1], dtype=np.int64),
        }

    def reset(self):
        ball_color, box_color = self._sample_colors()
        self.robosuite_env.set_episode_context(
            ball_color=ball_color,
            box_color=box_color,
            override_box_xy=self._sample_box_override(),
        )
        self.robosuite_env.reset()
        return self.get_state()

    def step(self, action):
        norm_action = self._normalize_action(action)
        _, reward, done, info = self.robosuite_env.step(norm_action)
        info["lcm"] = self.get_gt_local_mask()
        return self.get_state(), float(reward), bool(done), info

    def close(self):
        self.robosuite_env.close()
