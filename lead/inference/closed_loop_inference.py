from dataclasses import dataclass

import jaxtyping as jt
import numpy as np
import torch
from beartype import beartype

from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference, OpenLoopPrediction
from lead.tfv6.planning_decoder import BetaDistribution
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig

np.set_printoptions(suppress=True)


@dataclass
@beartype
class ClosedLoopPrediction(OpenLoopPrediction):
    steer: float
    throttle: float
    brake: float
    waypoints_steer: float
    waypoints_throttle: float
    waypoints_brake: float
    route_steer: float
    target_speed_throttle: float
    target_speed_brake: float
    raw_action_steer: float | None
    raw_action_throttle: float | None
    raw_action_brake: float | None


class ClosedLoopInference(OpenLoopInference):
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_closed_loop: ClosedLoopConfig,
        config_expert: ExpertConfig,
        model_path: str,
        device: torch.device,
        prefix: str,
    ):
        super().__init__(
            config_training=config_training,
            config_open_loop=config_closed_loop,
            model_path=model_path,
            device=device,
            prefix=prefix,
        )
        self.config_expert = config_expert
        self.config_training = config_training
        self.config_closed_loop = config_closed_loop

        # Auto-derive control modalities from training config if not explicitly set via
        # LEAD_CLOSED_LOOP_CONFIG. Explicit overrides in the env var always take precedence.
        _loaded = config_closed_loop._loaded_config
        if config_training.use_raw_action and not config_training.use_waypoints_action:
            if "steer_modality" not in _loaded:
                config_closed_loop.steer_modality = "raw_action"
            if "throttle_modality" not in _loaded:
                config_closed_loop.throttle_modality = "raw_action"
            if "brake_modality" not in _loaded:
                config_closed_loop.brake_modality = "raw_action"

        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)
        self.longitudinal_target_speed_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

        self.step = 4  # Constant so produced images start with 5, not really important

    @jt.jaxtyped(typechecker=beartype)
    def execute_raw_action(
        self,
        pred_alpha: jt.Float[torch.Tensor, "1 2"],
        pred_beta: jt.Float[torch.Tensor, "1 2"],
    ) -> tuple[float, float, float]:
        """Convert Beta distribution parameters to vehicle controls.

        Decodes E[Beta] = alpha/(alpha+beta), rescales from [0,1] to [-1,1],
        then maps to steer/throttle/brake following the reference convention:
        positive action[1] -> throttle, negative -> brake.

        Args:
            pred_alpha: Beta distribution alpha parameters, shape [1, 2].
            pred_beta: Beta distribution beta parameters, shape [1, 2].
        Returns:
            steer: Steering command in [-1, 1].
            throttle: Throttle command in [0, 1].
            brake: Brake command in [0, 1].
        """
        action_dist = BetaDistribution(action_dim=2)
        action_dist.proba_distribution(pred_alpha.float(), pred_beta.float())
        # E[Beta] in [0, 1]; rescale to [-1, 1]
        mean = action_dist.evaluate_mean()[0]  # shape [2]
        action = mean * 2.0 - 1.0
        steer = float(action[0].item())
        throttle_brake = float(action[1].item())
        if throttle_brake > 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        return steer, throttle, brake

    @jt.jaxtyped(typechecker=beartype)
    def execute_route_and_target_speed(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_target_speed: jt.Float[torch.Tensor, "1 1"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use route and target speed predictions.

        Args:
            pred_checkpoints: Predicted route checkpoints in ego-vehicle coordinates.
            pred_target_speed: Predicted target speed in m/s.
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        pred_checkpoints = pred_checkpoints[0].data.cpu().float().numpy()
        speed = float(speed[0].data.cpu().float().numpy())
        pred_target_speed = float(pred_target_speed[0].data.cpu().float().numpy())

        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio,
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            ego_vehicle_location,
            ego_vehicle_rotation,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake,
            pred_target_speed,
            speed,
            self.config_expert,
        )

        return steer, throttle, float(brake)

    @jt.jaxtyped(typechecker=beartype)
    def execute_waypoints(
        self,
        waypoints: jt.Float[torch.Tensor, "1 num_waypoints 2"],
        velocity: jt.Float[torch.Tensor, "1 1"],
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use waypoint predictions.

        Args:
            waypoints: Predicted future waypoints in ego-vehicle coordinates.
            velocity: Current speed of the vehicle in m/s.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        waypoints = waypoints[0].data.cpu().float().numpy()
        speed = velocity[0].data.cpu().float().numpy()

        one_second = int(
            self.config_training.carla_fps // self.config_training.data_save_freq,
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = np.clip(
            desired_speed - speed,
            0.0,
            self.config_closed_loop.wp_delta_clip,
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / desired_speed) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if (
            self.config_closed_loop.tuned_aim_distance
        ):  # In LB2, we go faster, so we need to choose waypoints farther ahead
            # range [2.4, 10.5] same as in the disentangled rep.
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            # To replicate the slow TransFuser behaviour we have a different distance
            # inside and outside of intersections (detected by desired_speed)
            if desired_speed < self.config_closed_loop.aim_distance_threshold:
                aim_distance = self.config_closed_loop.aim_distance_slow
            else:
                aim_distance = self.config_closed_loop.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]
        return float(steer), float(throttle), float(brake)

    @beartype
    def ensemble(
        self,
        data: dict[str, torch.Tensor],
        predictions: list[Prediction],
    ) -> ClosedLoopPrediction:
        """
        Args:
            data: Dictionary containing the input data
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            ClosedLoopPrediction: Dataclass containing all predictions
        """
        ego_speed = data["speed"].to(self.device, dtype=torch.float32).unsqueeze(1)
        open_loop_prediction: OpenLoopPrediction = super().ensemble(data, predictions)

        # Convert high-level commands to vehicle controls
        steer = throttle = brake = waypoint_steer = waypoint_throttle = (
            waypoint_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None
        raw_action_steer = raw_action_throttle = raw_action_brake = None

        # If any modality requests raw_action, compute it once and cache.
        _uses_raw_action = (
            self.config_closed_loop.steer_modality == "raw_action"
            or self.config_closed_loop.throttle_modality == "raw_action"
            or self.config_closed_loop.brake_modality == "raw_action"
        )
        if _uses_raw_action:
            if (
                open_loop_prediction.pred_action_beta_alpha is None
                or open_loop_prediction.pred_action_beta_beta is None
            ):
                raise ValueError(
                    "raw_action modality requires a model trained with use_raw_action=True "
                    "(predict_beta_action must be enabled)."
                )
            raw_action_steer, raw_action_throttle, raw_action_brake = (
                self.execute_raw_action(
                    open_loop_prediction.pred_action_beta_alpha,
                    open_loop_prediction.pred_action_beta_beta,
                )
            )

        if open_loop_prediction.pred_route is not None and open_loop_prediction.pred_target_speed_scalar is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.execute_route_and_target_speed(
                    open_loop_prediction.pred_route,
                    open_loop_prediction.pred_target_speed_scalar,
                    ego_speed,
                )
            )
        if open_loop_prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.execute_waypoints(
                    open_loop_prediction.pred_future_waypoints,
                    ego_speed,
                )
            )

        # Select which high-level modality we want to use for each control
        if self.config_closed_loop.steer_modality == "route":
            steer = route_steer
        elif self.config_closed_loop.steer_modality == "waypoint":
            steer = waypoints_steer
        elif self.config_closed_loop.steer_modality == "raw_action":
            steer = raw_action_steer
        else:
            raise ValueError(
                f"Invalid steer_modality: {self.config_closed_loop.steer_modality}",
            )

        if self.config_closed_loop.throttle_modality == "target_speed":
            throttle = target_speed_throttle
        elif self.config_closed_loop.throttle_modality == "waypoint":
            throttle = waypoints_throttle
        elif self.config_closed_loop.throttle_modality == "raw_action":
            throttle = raw_action_throttle
        else:
            raise ValueError(
                f"Invalid throttle_modality: {self.config_closed_loop.throttle_modality}",
            )

        if self.config_closed_loop.brake_modality == "target_speed":
            brake = target_speed_brake
        elif self.config_closed_loop.brake_modality == "waypoint":
            brake = waypoints_brake
        elif self.config_closed_loop.brake_modality == "raw_action":
            brake = raw_action_brake
        else:
            raise ValueError(
                f"Invalid brake_modality: {self.config_closed_loop.brake_modality}",
            )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        # Return all predictions
        return ClosedLoopPrediction(
            **vars(open_loop_prediction),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=waypoint_steer,
            waypoints_throttle=waypoint_throttle,
            waypoints_brake=waypoint_brake,
            route_steer=route_steer,
            target_speed_throttle=target_speed_throttle,
            target_speed_brake=target_speed_brake,
            raw_action_steer=raw_action_steer,
            raw_action_throttle=raw_action_throttle,
            raw_action_brake=raw_action_brake,
        )
