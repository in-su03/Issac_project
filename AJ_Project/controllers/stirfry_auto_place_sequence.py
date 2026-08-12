"""조리·준비 테이블 그릇 여러 곳 앞에 순서대로 자동 배치하는 구조 검증 시퀀스.

실제 파지는 하지 않는다. 각 target마다 StirfryAutoSequence의 stage 0에
해당하는 "그릇 림 접촉 기준 180 mm 위 elbow-up 안전 위치"만 다시 계산해
그 위치까지만 이동한다.

StirfryAutoSequence._build_stages()를 매 target마다 그대로 재사용하지 않는
이유: 그 메서드는 manual_handoff 여부와 무관하게, 원본 Ø250 mm 조리 그릇
전용으로 캘리브레이션된 하부 훅 최종 잠금 자세가 CAD 기준(BOWL_FROM_GRIPPER_LOCKED)과
0.2 mm 이내로 일치하는지부터 검증한다. 0.75배로 축소된 재료 그릇의
BOWL_NEAR_RIM_LOCAL을 넣으면 이 검증이 항상 실패하므로(그릇 크기와 무관하게
그리퍼 형상만 검증하는 하드코딩된 불변식), 여기서는 그 무관한 잠금 검증을
건너뛰고 안전 접근 위치 계산과 이동만 독립적으로 구현한다. 그릇을 순간이동
시키거나 gripper에 강제로 붙이지 않으며, 실제 도착은 link_6의 실측 자세로만
판정한다.

target 사이 이동은 항상 현재 관절각에서 StirfryAutoSequence.HOME_Q까지
부드럽게(smoothstep) 관절 보간한 뒤, 그 다음 target의 elbow-up 접근으로
이어간다. 첫 target도 같은 경로를 타므로(이미 0에 있어 사실상 대기만
발생) 별도 분기가 필요 없다. 어떤 target이 실패해도 나머지 target을 계속
시도하고, 끝에 성공/실패 요약을 출력한다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

from stirfry_auto_sequence import StirfryAutoSequence
from stirfry_auto_control import StirfryAutoJointControl


@dataclass(frozen=True)
class PlaceTarget:
    label: str
    bowl_actor: int
    slot_direction_xy: tuple
    bowl_near_rim_local: np.ndarray


class StirfryAutoPlaceSequence:
    """여러 그릇 앞 안전 위치에 순서대로 자동 접근한다(파지 없음)."""

    APPROACH_DURATION_S = 4.0
    RETURN_HOME_DURATION_S = 4.0
    SETTLE_S = 1.0
    ARRIVAL_HOLD_S = 1.0
    ARRIVAL_TIMEOUT_S = 10.0
    POSITION_TOLERANCE = 0.005
    ORIENTATION_TOLERANCE_DEG = 1.0

    def __init__(self, gym, sim, env, arm, targets, base_z, dt=1.0 / 60.0):
        if not targets:
            raise ValueError("targets는 최소 1개 이상이어야 합니다.")
        self.gym = gym
        self.sim = sim
        self.env = env
        self.arm = arm
        self.targets = list(targets)
        self.base_z = float(base_z)
        self.dt = float(dt)

        arm_body_names = self.gym.get_actor_rigid_body_names(self.env, self.arm.actor)
        self.link6_body_index = arm_body_names.index("link_6")

        self.auto_control = StirfryAutoJointControl(self.arm, dt=self.dt)
        self.auto_control.command(StirfryAutoSequence.HOME_Q)

        self.target_index = -1
        self.phase = None
        self.phase_elapsed = 0.0
        self.phase_start_joints = None
        self.phase_target_joints = None
        self.phase_target_link_pose = None
        self.arrival_wait = 0.0
        self.results = []
        self.finished = False
        self.failed = False
        self.handoff_ready = False

        print(
            f"[배치] 그릇 {len(self.targets)}곳에 순서대로 안전 위치까지만 "
            "자동 접근합니다(파지 없음)."
        )
        self._start_target(0)

    # ---------------------------------------------------------------- 공개
    def update(self):
        """시뮬레이션 루프에서 물리 스텝 전에 매 프레임 한 번 호출한다."""
        if self.finished:
            self.auto_control.update()
            return
        self._update_move(is_return_home=(self.phase == "return_home"))

    # ---------------------------------------------------------------- 내부
    def _start_target(self, index):
        self.target_index = index
        if self.target_index >= len(self.targets):
            self._finish_all()
            return
        self._begin_return_home()

    def _begin_return_home(self):
        self.phase = "return_home"
        self.phase_elapsed = 0.0
        self.arrival_wait = 0.0
        self.phase_start_joints = self.arm.current_joints().astype(np.float64)
        self.phase_target_joints = StirfryAutoSequence.HOME_Q.astype(np.float64)
        target = self.targets[self.target_index]
        print(f"\n[배치] '{target.label}' 목표로 이동 전 HOME 자세로 복귀합니다.")

    def _begin_approach(self):
        target = self.targets[self.target_index]
        print(
            f"\n[배치 {self.target_index + 1}/{len(self.targets)}] "
            f"{target.label} 접근을 계산합니다 "
            f"(slot_direction_xy={np.round(target.slot_direction_xy, 3)})."
        )
        try:
            staging_q, link_position, link_orientation, bowl_position = (
                self._solve_staging(target)
            )
        except RuntimeError as error:
            print(f"[배치][실패] {target.label}: {error}")
            self.results.append((target.label, "FAIL", str(error)))
            self._start_target(self.target_index + 1)
            return

        self.phase = "approach"
        self.phase_elapsed = 0.0
        self.arrival_wait = 0.0
        self.phase_start_joints = self.arm.current_joints().astype(np.float64)
        self.phase_target_joints = staging_q.astype(np.float64)
        self.phase_target_link_pose = (link_position, link_orientation)
        print(
            f"[배치] {target.label}: 그릇 world = {np.round(bowl_position, 4)} m, "
            f"목표 관절각(deg) = {np.round(np.rad2deg(staging_q), 1)}"
        )

    def _solve_staging(self, target):
        """StirfryAutoSequence의 stage 0(elbow-up 안전 접근)와 동일한 위치를
        하부 훅 잠금 검증 없이 계산한다."""
        bowl_position = self._bowl_position(target.bowl_actor)

        x_axis = np.array(
            [target.slot_direction_xy[0], target.slot_direction_xy[1], 0.0],
            dtype=np.float64,
        )
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        bowl_frame = np.column_stack((x_axis, y_axis, z_axis))

        entry_R = bowl_frame @ Rotation.from_euler(
            "y", StirfryAutoSequence.VERTICAL_ENTRY_DEG, degrees=True
        ).as_matrix()
        outer_rim_world = bowl_position + bowl_frame @ target.bowl_near_rim_local
        entry_origin = (
            outer_rim_world - entry_R @ StirfryAutoSequence.UPPER_ENTRY_REFERENCE_LOCAL
        )
        staging_origin = entry_origin + np.array(
            [0.0, 0.0, StirfryAutoSequence.VERTICAL_STAGING_CLEARANCE]
        )

        link_orientation = entry_R @ StirfryAutoSequence.LINK6_TO_GRIPPER_ROTATION.T
        link_position_world = (
            staging_origin
            - link_orientation @ StirfryAutoSequence.LINK6_TO_GRIPPER_POSITION
        )
        link_position = link_position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )

        staging_q, _, ik_error_mm = self.arm.solve_ik(
            link_position,
            target_R=link_orientation,
            seed_6=StirfryAutoSequence.ELBOW_UP_SEED,
        )
        if not np.all(np.isfinite(staging_q)) or ik_error_mm > 8.0:
            raise RuntimeError(
                f"elbow-up 안전 접근 IK 실패: 위치 오차 {ik_error_mm:.2f} mm"
            )
        return staging_q, link_position, link_orientation, bowl_position

    def _update_move(self, is_return_home):
        self.phase_elapsed += self.dt
        duration = (
            self.RETURN_HOME_DURATION_S if is_return_home else self.APPROACH_DURATION_S
        )
        alpha = min(1.0, self.phase_elapsed / duration)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        target_joints = (
            self.phase_start_joints
            + (self.phase_target_joints - self.phase_start_joints) * alpha
        )
        settling = self.phase_elapsed >= duration
        self.auto_control.command(target_joints.astype(np.float32), settling=settling)

        if self.phase_elapsed < duration:
            return

        if is_return_home:
            if self.phase_elapsed >= duration + self.SETTLE_S:
                self._begin_approach()
            return

        self._check_approach_arrival()

    def _check_approach_arrival(self):
        position_error, orientation_error = self._link_pose_error()
        if (
            position_error <= self.POSITION_TOLERANCE
            and orientation_error <= self.ORIENTATION_TOLERANCE_DEG
        ):
            self._on_arrival()
            return
        self.arrival_wait += self.dt
        if self.arrival_wait >= self.ARRIVAL_TIMEOUT_S:
            target = self.targets[self.target_index]
            print(
                f"[배치][실패] {target.label}: 안전 위치 미도착 "
                f"(위치 오차 {position_error * 1000:.1f} mm, "
                f"자세 오차 {orientation_error:.2f} deg)"
            )
            self.results.append((target.label, "FAIL", "arrival_timeout"))
            self._start_target(self.target_index + 1)

    def _on_arrival(self):
        target = self.targets[self.target_index]
        self.arrival_wait += self.dt
        if self.arrival_wait < self.ARRIVAL_HOLD_S:
            return
        joints_deg = np.round(np.rad2deg(self.arm.current_joints()), 1)
        print(f"[배치][도착] {target.label}: 관절각(deg) = {joints_deg}")
        self.results.append((target.label, "OK", None))
        self._start_target(self.target_index + 1)

    def _link_pose_error(self):
        position_world, orientation = self._rigid_body_pose(
            self.arm.actor, self.link6_body_index
        )
        position = position_world - np.array(
            [0.0, 0.0, self.base_z], dtype=np.float64
        )
        target_position, target_orientation = self.phase_target_link_pose
        position_error = float(np.linalg.norm(target_position - position))
        orientation_error = float(
            np.degrees(
                Rotation.from_matrix(target_orientation @ orientation.T).magnitude()
            )
        )
        return position_error, orientation_error

    def _bowl_position(self, bowl_actor):
        position, _ = self._rigid_body_pose(bowl_actor, 0)
        return position

    def _rigid_body_pose(self, actor, body_index):
        states = self.gym.get_actor_rigid_body_states(
            self.env, actor, gymapi.STATE_POS
        )
        position = states["pose"]["p"][body_index]
        orientation = states["pose"]["r"][body_index]
        position_array = np.array(
            [position["x"], position["y"], position["z"]], dtype=np.float64
        )
        orientation_matrix = Rotation.from_quat(
            [orientation["x"], orientation["y"], orientation["z"], orientation["w"]]
        ).as_matrix()
        return position_array, orientation_matrix

    def _finish_all(self):
        self.finished = True
        self.handoff_ready = True
        print("\n===== 배치 구조 검증 요약 =====")
        for label, status, detail in self.results:
            suffix = f" ({detail})" if detail else ""
            print(f"  [{status}] {label}{suffix}")
        ok_count = sum(1 for _, status, _ in self.results if status == "OK")
        print(f"{ok_count}/{len(self.results)} 위치 도착 성공")
        print("================================\n")
