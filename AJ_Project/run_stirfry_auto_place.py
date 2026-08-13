"""run_stirfry_auto_place.py — 조리·준비 테이블 그릇 다중 위치 자동 배치 구조 검증.

run_stirfry.py의 --auto는 조리 테이블 그릇 1개 앞 안전 위치까지만 자동
이동한다(그 뒤 TSC 키보드로 인계). 이 스크립트는 같은 안전 위치 개념을
조리 테이블 그릇 1개 + 준비 테이블 두 뱅크(ㄱ자 배치)의 대표 그릇 1개씩,
총 3곳에 순서대로 자동 접근시켜서 로봇팔이 구조적으로(관절 한계·IK 도달성)
세 위치 모두에 실제로 도착할 수 있는지 물리 시뮬레이션에서 검증한다.

실제 파지·상승·붓기는 하지 않는다. 그릇을 순간이동시키거나 gripper에
강제로 붙이지 않는다. 각 위치 사이 이동은 HOME 자세를 거쳐 스냅 없이
부드럽게 이어간다(controllers/stirfry_auto_place_sequence.py).

실행:  conda activate isaac_gym
       python run_stirfry_auto_place.py --auto-place
       python run_stirfry_auto_place.py    # 플래그 없으면 기존과 동일한 수동 키보드 조작
"""
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil   # torch보다 먼저

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers"))
from doosan_controller import DoosanController

from asset_config import get_asset_root
asset_root = get_asset_root()

CABINET_HEIGHT = 0.805
BASE_Z = CABINET_HEIGHT

BONITKIT_POS       = (0.0, 1.07, 0.0)
COMPLETE_TABLE_POS = (-0.625, 0.10, 0.0)
PREPARE_TABLE_POS  = (0.10, -0.25, 0.0)
TABLE_YAW_DEG      = 90.0
TABLE_TOP_Z        = 0.85
COOK_BOWL_Z        = 0.8225
INGREDIENT_BOWL_Z  = 0.8255
ROBOT_CABINET_URDF  = "urdf/robot_cabinetnplate/robot_cabinetnplate.urdf"
AIR_COMPRESSOR_URDF = "urdf/air_compressor/air_compressor.urdf"
DOOSAN_CONTROLLER_URDF = "urdf/doosan_controller/doosan_controller.urdf"
COMPLETE_TABLE_URDF = "urdf/complete_table/complete_table.urdf"
PREPARE_TABLE_URDF  = "urdf/prepare_table/prepare_table.urdf"
BOWL_URDF            = "urdf/stirfry_bowl/stirfry_bowl.urdf"
A0509_URDF           = "urdf/doosan_a0509/a0509.urdf"
A0509_GRIPPER_URDF   = "urdf/a0509_stirfry_gripper/a0509_stirfry_gripper.urdf"
GRIPPER_BODY_NAME    = "stirfry_gripper_link"
CABINET_INTERNAL_COLLISION_FILTER = 2
AIR_COMPRESSOR_POS = (-0.2293, -0.1591, 0.1960)
DOOSAN_CONTROLLER_POS = (-0.2017, 0.2472, 0.1090)

BOWL_FRICTION       = 0.50
GRIPPER_FRICTION    = 0.50
TABLE_FRICTION      = 0.50
CONTACT_RESTITUTION = 0.0
CONTACT_OFFSET      = 0.001
REST_OFFSET         = 0.0

COMPLETE_BOWL_LOCAL_XY = (0.25, 0.0)
PREPARE_BOWL_LOCAL_XY = (
    *((-0.475, y) for y in (-0.30, -0.10, 0.10, 0.30, 0.50)),
    *((x, -0.475) for x in (-0.30, -0.10, 0.10, 0.30, 0.50)),
)
INGREDIENT_BOWL_SCALE = 0.75

# 준비 테이블 두 뱅크(ㄱ자 배치)에서 검증용 대표 그릇 1개씩.
# CLAUDE.md 기구학 사전 검사에서 03번은 여유가 크고, 08번은 두 뱅크 중
# 로봇 베이스에서 비교적 가까운 쪽으로 먼저 검증하라고 권장된 그릇이다.
PREPARE_BANK1_INDEX = 3
PREPARE_BANK2_INDEX = 8


def pose(x, y, z=0.0, yaw_deg=0.0):
    rotation = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0, 0, 1), np.radians(yaw_deg)
    )
    return gymapi.Transform(p=gymapi.Vec3(x, y, z), r=rotation)


def local_xy_to_world(local_xy, origin_xy, yaw_deg):
    x, y = local_xy
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return (
        origin_xy[0] + c * x - s * y,
        origin_xy[1] + s * x + c * y,
    )


def radial_slot_direction_xy(bowl_world_xy, base_world_xy=(0.0, 0.0)):
    """로봇 베이스에서 그릇 중심을 향하는 단위 방향(수평면)."""
    delta = np.array(bowl_world_xy, dtype=np.float64) - np.array(
        base_world_xy, dtype=np.float64
    )
    return tuple(delta / np.linalg.norm(delta))


def set_actor_contact_properties(actor_handle, friction):
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for shape_prop in shape_props:
        shape_prop.friction = friction
        shape_prop.restitution = CONTACT_RESTITUTION
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)


def set_body_contact_properties(actor_handle, body_name, friction):
    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    if body_name not in body_names:
        raise RuntimeError(f"rigid body not found: {body_name}")
    body_index = body_names.index(body_name)
    shape_range = gym.get_actor_rigid_body_shape_indices(env, actor_handle)[body_index]
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for shape_index in range(shape_range.start, shape_range.start + shape_range.count):
        shape_props[shape_index].friction = friction
        shape_props[shape_index].restitution = CONTACT_RESTITUTION
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)

# ============================================================ [1] 시뮬
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(
    description="A0509 조리/준비 테이블 그릇 다중 위치 자동 배치 구조 검증",
    custom_parameters=[
        {
            "name": "--auto-place",
            "action": "store_true",
            "help": (
                "조리 테이블 그릇 1개 + 준비 테이블 두 뱅크 대표 그릇 1개씩, "
                "총 3곳에 순서대로 안전 위치까지 자동 접근한다(파지 없음)."
            ),
        },
    ],
)
sp = gymapi.SimParams()
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sp.dt = 1.0 / 60.0
sp.physx.solver_type = 1
sp.physx.use_gpu = True
sp.physx.num_position_iterations = 8
sp.physx.num_velocity_iterations = 1
sp.physx.contact_offset = CONTACT_OFFSET
sp.physx.rest_offset = REST_OFFSET
sp.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, pp)

# ============================================================ [2] 씬
env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0), gymapi.Vec3(1.5, 1.8, 2.2), 1)

fixture_opts = gymapi.AssetOptions(); fixture_opts.fix_base_link = True
cabinet_asset = gym.load_asset(sim, asset_root, ROBOT_CABINET_URDF, fixture_opts)
gym.create_actor(
    env,
    cabinet_asset,
    gymapi.Transform(p=gymapi.Vec3(0, 0, 0)),
    "robot_cabinetnplate",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)

compressor_asset = gym.load_asset(sim, asset_root, AIR_COMPRESSOR_URDF, fixture_opts)
controller_asset = gym.load_asset(sim, asset_root, DOOSAN_CONTROLLER_URDF, fixture_opts)
gym.create_actor(
    env,
    compressor_asset,
    pose(*AIR_COMPRESSOR_POS),
    "air_compressor_in_cabinet",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)
gym.create_actor(
    env,
    controller_asset,
    pose(*DOOSAN_CONTROLLER_POS),
    "doosan_controller_in_cabinet",
    0,
    CABINET_INTERNAL_COLLISION_FILTER,
)

fixed_opts = gymapi.AssetOptions(); fixed_opts.fix_base_link = True
bonitkit_asset = gym.load_asset(sim, asset_root, "urdf/bonitkit/bonitkit.urdf", fixed_opts)

table_opts = gymapi.AssetOptions()
table_opts.fix_base_link = True
complete_table_asset = gym.load_asset(
    sim, asset_root, COMPLETE_TABLE_URDF, table_opts
)
prepare_table_asset = gym.load_asset(
    sim, asset_root, PREPARE_TABLE_URDF, table_opts
)

bowl_opts = gymapi.AssetOptions()
bowl_opts.fix_base_link = False
bowl_opts.disable_gravity = False
bowl_asset = gym.load_asset(sim, asset_root, BOWL_URDF, bowl_opts)

gym.create_actor(env, bonitkit_asset, pose(*BONITKIT_POS), "bonitkit", 0, 0)
complete_table_handle = gym.create_actor(
    env,
    complete_table_asset,
    pose(*COMPLETE_TABLE_POS, yaw_deg=TABLE_YAW_DEG),
    "complete_table",
    0,
    0,
)
prepare_table_handle = gym.create_actor(
    env,
    prepare_table_asset,
    pose(*PREPARE_TABLE_POS, yaw_deg=TABLE_YAW_DEG),
    "prepare_table",
    0,
    0,
)

complete_bowl_xy = local_xy_to_world(
    COMPLETE_BOWL_LOCAL_XY, COMPLETE_TABLE_POS[:2], TABLE_YAW_DEG
)
cook_bowl_handle = gym.create_actor(
    env,
    bowl_asset,
    pose(*complete_bowl_xy, COOK_BOWL_Z, yaw_deg=TABLE_YAW_DEG),
    "stirfry_bowl_cook",
    0,
    0,
)
set_actor_contact_properties(cook_bowl_handle, BOWL_FRICTION)

# ingredient bowl handle/world 중심/scale을 index별로 보존한다.
# (run_stirfry.py의 동일 루프는 이 목록을 보존하지 않아 --auto-grasp가
#  항상 cook_bowl_handle만 쓴다 — 이 스크립트는 그 목록이 반드시 필요하다.)
ingredient_bowls_by_index = {}
for index, local_xy in enumerate(PREPARE_BOWL_LOCAL_XY, start=1):
    bowl_xy = local_xy_to_world(local_xy, PREPARE_TABLE_POS[:2], TABLE_YAW_DEG)
    bowl_handle = gym.create_actor(
        env,
        bowl_asset,
        pose(*bowl_xy, INGREDIENT_BOWL_Z, yaw_deg=TABLE_YAW_DEG),
        f"stirfry_bowl_ingredient_{index:02d}",
        0,
        0,
    )
    gym.set_actor_scale(env, bowl_handle, INGREDIENT_BOWL_SCALE)
    set_actor_contact_properties(bowl_handle, BOWL_FRICTION)
    ingredient_bowls_by_index[index] = (bowl_handle, bowl_xy, INGREDIENT_BOWL_SCALE)

set_actor_contact_properties(complete_table_handle, TABLE_FRICTION)
set_actor_contact_properties(prepare_table_handle, TABLE_FRICTION)

arm = DoosanController(
    gym, sim, env, asset_root,
    urdf=A0509_GRIPPER_URDF,
    ik_urdf=A0509_URDF,
    ee_link="link_6",
    fix_base=True,
    spawn_transform=gymapi.Transform(p=gymapi.Vec3(0, 0, BASE_Z)),
)
set_body_contact_properties(arm.actor, GRIPPER_BODY_NAME, GRIPPER_FRICTION)

# ============================================================ [3] 동역학 텐서(OSC)
gym.prepare_sim(sim)
arm.setup_osc()

print(f"""[GRASP PHYSICS READY]
bowl dynamic: {not bowl_opts.fix_base_link}
bowl gravity: {not bowl_opts.disable_gravity}
table fixed: {table_opts.fix_base_link}
robot fixture: A0509 mounted directly on cabinet top z={BASE_Z:.3f} m
cabinet equipment: air compressor + Doosan controller, centered side-by-side
robot control: {"AUTO PLACE (3곳 순차 접근)" if args.auto_place else "MANUAL"}""")

# ============================================================ [4] 뷰어 + 키 등록
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(
    viewer,
    env,
    gymapi.Vec3(2.4, -2.8, 2.4),
    gymapi.Vec3(0.0, 0.25, 0.80),
)

from doosan_arm_keyboard_teleop import DoosanArmKeyboardTeleop
from stirfry_arm_keyboard_teleop import StirfryArmKeyboardTeleop

if args.auto_place:
    from stirfry_auto_place_sequence import StirfryAutoPlaceSequence, PlaceTarget
    from stirfry_auto_sequence import StirfryAutoSequence

    cook_bowl_near_rim_local = StirfryAutoSequence.BOWL_NEAR_RIM_LOCAL
    ingredient_bowl_near_rim_local = (
        StirfryAutoSequence.BOWL_NEAR_RIM_LOCAL * INGREDIENT_BOWL_SCALE
    )

    bank1_handle, bank1_xy, _ = ingredient_bowls_by_index[PREPARE_BANK1_INDEX]
    bank2_handle, bank2_xy, _ = ingredient_bowls_by_index[PREPARE_BANK2_INDEX]

    targets = [
        PlaceTarget(
            "조리 테이블 그릇(complete table)",
            cook_bowl_handle,
            (-1.0, 0.0),
            cook_bowl_near_rim_local,
        ),
        PlaceTarget(
            f"준비 테이블 뱅크1 {PREPARE_BANK1_INDEX:02d}번 그릇",
            bank1_handle,
            radial_slot_direction_xy(bank1_xy),
            ingredient_bowl_near_rim_local,
        ),
        PlaceTarget(
            f"준비 테이블 뱅크2 {PREPARE_BANK2_INDEX:02d}번 그릇",
            bank2_handle,
            radial_slot_direction_xy(bank2_xy),
            ingredient_bowl_near_rim_local,
        ),
    ]
    auto_sequence = StirfryAutoPlaceSequence(
        gym, sim, env, arm, targets, base_z=BASE_Z, dt=sp.dt
    )
    teleop = None
else:
    teleop = DoosanArmKeyboardTeleop(gym, viewer, arm, base_z=BASE_Z)
    auto_sequence = None

while not gym.query_viewer_has_closed(viewer):
    if auto_sequence is not None:
        auto_sequence.update()
        if auto_sequence.handoff_ready:
            gravity_control = auto_sequence.auto_control
            teleop = StirfryArmKeyboardTeleop(
                gym,
                viewer,
                arm,
                base_z=BASE_Z,
                cart_step=0.001,
                joint_step=np.deg2rad(0.2),
                ori_step_deg=0.5,
                settle=0,
                home_on_start=False,
                gravity_control=gravity_control,
            )
            auto_sequence = None
            print("""
===== 배치 구조 검증 완료 -> 키보드 미세조작 전환 =====
 1) 2번: 현재 자세에서 안전 TSC 재동기화(선택)
 2) W/S: X±, A/D: Y±, Q/E: Z± (1 mm/프레임)
 3) T/G: pitch±, F/H: roll±, C/V: yaw± (0.5 deg/프레임)
 4) 필요하면 1번 JSC -> J/L 관절 선택 -> U/O 미세 회전
 5) 3번 OSC는 안전을 위해 비활성화
========================================================""")
    else:
        teleop.handle_and_apply()

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("종료.")
