import dataclasses as dc
import functools

import numpy as np

from pydrake.all import (
    AbstractValue,
    AddMultibodyPlant,
    ApplyVisualizationConfig,
    BasicVector,
    DiagramBuilder,
    EventStatus,
    GetScopedFrameByName,
    InverseDynamicsController,
    LeafSystem,
    MultibodyPlantConfig,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    SimulatorStatus,
    VisualizationConfig,
)

from drake_diff_ik_dual_arm import debug
from drake_diff_ik_dual_arm.diff_ik import MultiFrameDiffIk
from drake_diff_ik_dual_arm.basics import (
    GetActiveDof,
    GetAllRobotModelInstances,
    LoadMyDirectives,
    MakeCollisionChecker,
    MakeRobotDiagramBuilder,
    ProcessMyModelDirectives,
    ResolvePath,
)


class LatchPassive(LeafSystem):
    def __init__(self, active_dof):
        super().__init__()
        num_x_full = 2 * len(active_dof)
        num_x_active = 2 * np.sum(active_dof)
        self.active_dof = active_dof
        self.passive_dof = np.logical_not(active_dof)

        self.x_init_full_index = self.DeclareDiscreteState(num_x_full)
        self.state_measured_full = self.DeclareVectorInputPort(
            "state_measured_full", BasicVector(num_x_full)
        )
        self.state_desired_active = self.DeclareVectorInputPort(
            "state_desired_active", BasicVector(num_x_active)
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._discrete_init)
        self.state_desired_full = self.DeclareVectorOutputPort(
            "state_desired_full", num_x_full, self._calc_output
        )

    def _discrete_init(self, context, discrete_state):
        x_full = self.state_measured_full.Eval(context)
        discrete_state.set_value(self.x_init_full_index, x_full)

    def _calc_output(self, context, output):
        x_init_full = (
            context.get_discrete_state(self.x_init_full_index).get_value()
        )
        x_desired_active = self.state_desired_active.Eval(context)
        x_desired_full = x_init_full.copy()
        # Take mutable views.
        num_q_full = len(self.active_dof)
        num_q_active = np.sum(self.active_dof)
        # Position.
        q_desired_full = x_desired_full[:num_q_full]
        q_desired_active = x_desired_active[:num_q_active]
        q_desired_full[self.active_dof] = q_desired_active
        # Velocity.
        v_desired_full = x_desired_full[num_q_full:]
        v_desired_active = x_desired_active[num_q_active:]
        v_desired_full[self.active_dof] = v_desired_active
        output.set_value(x_desired_full)


class SimplePoseReferenceSystem(LeafSystem):

    @dc.dataclass
    class InitState:
        x_full: object = None
        X_BF_list: object = None

    def __init__(self, plant, frame_F_list, calc_desired_pose):
        super().__init__()
        self.plant = plant
        self.frame_B = self.plant.world_frame()
        self.frame_F_list = frame_F_list
        self.calc_desired_pose = calc_desired_pose

        # Scratch (cache).
        self.plant_context = self.plant.CreateDefaultContext()

        num_x_full = 2 * plant.num_positions()
        self.state_measured_full = self.DeclareVectorInputPort(
            "state_measured_full", BasicVector(num_x_full)
        )

        self.X_BF_desired_output_list = []
        for i in range(len(frame_F_list)):
            X_BF_desired_output = self.DeclareAbstractOutputPort(
                f"X_BF_desired_{i}",
                lambda: AbstractValue.Make(RigidTransform()),
                functools.partial(self._calc_X_BF_desired, i),
            )
            self.X_BF_desired_output_list.append(X_BF_desired_output)

        self.init_state_index = self.DeclareAbstractState(
            AbstractValue.Make(self.InitState())
        )
        self.DeclareInitializationUnrestrictedUpdateEvent(
            self._initialize_state
        )

    def _initialize_state(self, context, raw_state):
        abstract_state = raw_state.get_mutable_abstract_state(
            self.init_state_index
        )
        init = abstract_state.get_mutable_value()
        init.x_full = self.state_measured_full.Eval(context)
        init.X_BF_list = []
        self.plant.SetPositionsAndVelocities(self.plant_context, init.x_full)
        for i, frame_F in enumerate(self.frame_F_list):
            X_BF = self.plant.CalcRelativeTransform(
                self.plant_context, self.frame_B, frame_F
            )
            init.X_BF_list.append(X_BF)

    def _calc_X_BF_desired(self, i, context, output):
        init = context.get_abstract_state(self.init_state_index).get_value()
        X_BF = self.calc_desired_pose(
            context.get_time(), i, init.X_BF_list[i]
        )
        output.set_value(X_BF)


def calc_example_pose(t, i, X_BF_init):
    # Arbitrary trajectory for the two arms to drive them both towards each
    # other and towards the table.
    T = 1.0
    w = 2 * np.pi / T
    s = np.sin(w * t)
    dR = RollPitchYaw(
        np.deg2rad([
            -60.0 * s,
            45.0 * s,
            180.0 * s,
        ])
    )
    if i % 2 == 0:
        sign = -1
    else:
        sign = 1
    dp = np.array([
        0.1 * s,
        0.5 * s * sign,
        0.1 * s + 0.4 * t,
    ])
    dX = RigidTransform(dR, dp)
    X_BF = X_BF_init @ dX
    return X_BF


def monitor_stop_on_collision(plant, diagram_context):
    # Stop if simulation detects any collision.
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    contact_results = (
        plant.get_contact_results_output_port().Eval(plant_context)
    )
    if contact_results.num_point_pair_contacts() > 0:
        return EventStatus.ReachedTermination(
            plant, "Contact encountered! Stopping simulation"
        )


@debug.iex
def main():
    t_final = 3.0

    directives = LoadMyDirectives(
        "package://drake_diff_ik_dual_arm/models/add_dual_arm_pandas.yaml"
    )

    builder = DiagramBuilder()
    # Used for both simulation and control kinematics.
    plant_config = MultibodyPlantConfig(time_step=0.001)
    plant, scene_graph = AddMultibodyPlant(plant_config, builder)
    ProcessMyModelDirectives(directives, plant)
    plant.Finalize()
    ApplyVisualizationConfig(VisualizationConfig(), builder)

    model_instances = [
        plant.GetModelInstanceByName("right::panda"),
        plant.GetModelInstanceByName("left::panda"),
    ]
    frame_F_list = [
        GetScopedFrameByName(plant, "right::panda::panda_link8"),
        GetScopedFrameByName(plant, "left::panda::panda_link8"),
    ]

    # Create separate copy for use exclusively with collision checker - because
    # that's what the API wants :(
    robot_diagram_copy = (
        MakeRobotDiagramBuilder(directives, time_step=plant_config.time_step)
        .Build()
    )
    all_robot_model_instances = GetAllRobotModelInstances(
        plant, model_instances
    )
    collision_checker = MakeCollisionChecker(
        robot_diagram_copy, all_robot_model_instances
    )

    dt = 0.005

    active_dof = GetActiveDof(plant, model_instances)

    plant_context_init = plant.CreateDefaultContext()
    q_panda = np.deg2rad([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 0.0])
    for model in model_instances:
        plant.SetPositions(plant_context_init, model, q_panda)

    hand_instances = [
        plant.GetModelInstanceByName("right::panda_hand"),
        plant.GetModelInstanceByName("left::panda_hand"),
    ]
    q_hand = np.array([-0.01, 0.01])
    for model in hand_instances:
        plant.SetPositions(plant_context_init, model, q_hand)

    q0_full = plant.GetPositions(plant_context_init)
    # Choose a nominal configuration for posture control.
    q0_active = q0_full[active_dof]

    diff_ik = builder.AddSystem(
        MultiFrameDiffIk(
            plant=plant,
            collision_checker=collision_checker,
            frame_F_list=frame_F_list,
            period_sec=dt,
            active_dof=active_dof,
            q0_active=q0_active,
        )
    )
    # Measured full state for diff IK.
    builder.Connect(
        plant.get_state_output_port(),
        diff_ik.state_measured_full,
    )
    # Mux desired active DoF with latched version of passive DoF initial state.
    latch_passive = builder.AddSystem(
        LatchPassive(active_dof=active_dof),
    )
    builder.Connect(
        plant.get_state_output_port(),
        latch_passive.state_measured_full,
    )
    builder.Connect(
        diff_ik.state_desired_active,
        latch_passive.state_desired_active,
    )

    # Connect reference system.
    ref_sys = builder.AddSystem(
        SimplePoseReferenceSystem(
            plant, frame_F_list, calc_example_pose,
        )
    )
    builder.Connect(
        plant.get_state_output_port(),
        ref_sys.state_measured_full,
    )
    for i in range(len(frame_F_list)):
        builder.Connect(
            ref_sys.X_BF_desired_output_list[i],
            diff_ik.X_BF_desired_input_list[i],
        )

    num_q_full = plant.num_positions()
    # Gains must be high enough to handle deceleration.
    kp = 1000.0 * np.ones(num_q_full)
    kd = np.sqrt(2 * kp)
    ki = np.zeros(num_q_full)
    inv_dyn = builder.AddSystem(
        InverseDynamicsController(
            robot=plant,
            kp=kp,
            ki=ki,
            kd=kd,
            has_reference_acceleration=False,
        )
    )
    # Measured plant state to inverse dynamics joint controller.
    builder.Connect(
        plant.get_state_output_port(),
        inv_dyn.get_input_port_estimated_state(),
    )
    # Desired full coordinates to inverse dynamics reference signal.
    builder.Connect(
        latch_passive.state_desired_full,
        inv_dyn.get_input_port_desired_state(),
    )
    # Connect torques.
    builder.Connect(
       inv_dyn.get_output_port_control(),
       # TODO(eric.cousineau): Use actuation matrix + actuation port.
       plant.get_applied_generalized_force_input_port(),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    diagram_context = simulator.get_context()

    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant_context.SetStateAndParametersFrom(plant_context_init)

    simulator.Initialize()
    simulator.set_monitor(
        functools.partial(monitor_stop_on_collision, plant)
    )
    status = simulator.AdvanceTo(t_final)
    if status.reason() != SimulatorStatus.ReturnReason.kReachedBoundaryTime:
        diagram.ForcedPublish(diagram_context)
        print(f"Stopped due to {status.reason()}")
        print(f"  {status.message()}")
        q = plant.GetPositions(plant_context)
        print(f"  q = {q.tolist()}")


if __name__ == "__main__":
    main()
