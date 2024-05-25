import numpy as np

from pydrake.all import (
    AbstractValue,
    BasicVector,
    JacobianWrtVariable,
    MathematicalProgram,
    OsqpSolver,
    RigidTransform,
    SolverOptions,
    LeafSystem,
)

from drake_diff_ik_dual_arm.pose_util import se3_vector_minus
from drake_diff_ik_dual_arm.basics import (
    GetBodiesKinematicallyAffectedBy,
    CalcNullspace,
    JointLimits,
    MaybeUseActiveDistancesAndGradients,
    SolveOrDie,
)


class MultiFrameDiffIk(LeafSystem):
    """
    Loosely represents the math in `MultiFramePoseStream`.
    """

    def __init__(
        self,
        plant,
        collision_checker,
        frame_F_list,
        active_dof,
        period_sec,
        q0_active,
    ):
        super().__init__()
        self.plant = plant
        self.collision_checker = collision_checker
        self.dt = period_sec
        self.q0_active = q0_active

        self.frame_B = self.plant.world_frame()
        self.frame_F_list = frame_F_list
        self.joint_limits = (
            JointLimits.from_plant(self.plant).select(active_dof)
        )

        self.solver = OsqpSolver()
        self.solver_options = SolverOptions()

        self.active_dof = active_dof
        self.passive_dof = np.logical_not(self.active_dof)
        self.num_active_dof = np.sum(self.active_dof)
        self.active_bodies = GetBodiesKinematicallyAffectedBy(
            self.plant, self.active_dof
        )

        # Scratch (cache).
        self.plant_context = self.plant.CreateDefaultContext()

        # Ports.
        self.X_BF_desired_input_list = []
        for i, frame_F in enumerate(self.frame_F_list):
            X_BF_desired_input = self.DeclareAbstractInputPort(
                f"X_BF_desired_{i}", AbstractValue.Make(RigidTransform()),
            )
            self.X_BF_desired_input_list.append(X_BF_desired_input)

        num_x_full = 2 * self.plant.num_positions()
        self.state_measured_full = self.DeclareVectorInputPort(
            "state_measured_full", BasicVector(num_x_full)
        )
        # Note: This is for *active* DoF.
        self.x_desired_index = self.DeclareDiscreteState(
            2 * self.num_active_dof
        )
        self.state_desired_active = self.DeclareStateOutputPort(
            "state_desired_active", self.x_desired_index
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._discrete_init)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec,
            0.0,
            self._discrete_update,
        )

    def _discrete_init(self, context, discrete_state):
        # Initialize desired state to current configuration + zero velocity.
        x_full = self.state_measured_full.Eval(context)
        num_q = self.plant.num_positions()
        q_full = x_full[:num_q]
        q_desired = q_full[self.active_dof]
        v_desired = np.zeros(self.num_active_dof)
        x_desired = np.hstack([q_desired, v_desired])
        discrete_state.set_value(self.x_desired_index, x_desired)

    def _discrete_update(self, context, discrete_state):
        X_BF_desired_list = []
        for X_BF_desired_input in self.X_BF_desired_input_list:
            X_BF_desired = X_BF_desired_input.Eval(context)
            X_BF_desired_list.append(X_BF_desired)
        x_actual_full = self.state_measured_full.Eval(context)
        q_actual_full = x_actual_full[:self.plant.num_positions()]
        x_desired_prev = discrete_state.get_value(self.x_desired_index).copy()
        q_desired_prev = x_desired_prev[:self.num_active_dof]
        # Assemble full-state for open-loop integration (i.e., using previous
        # desired value).
        q_full = self.plant.GetPositions(self.plant_context).copy()
        q_full[self.active_dof] = q_desired_prev
        q_full[self.passive_dof] = q_actual_full[self.passive_dof]
        # Solve QP.
        v_desired = self._solve_for_desired_velocity(q_full, X_BF_desired_list)
        # Open-loop integration. Simple first-order Euler integration.
        q_desired = q_desired_prev + v_desired * self.dt
        x_desired = np.hstack([q_desired, v_desired])
        discrete_state.set_value(self.x_desired_index, x_desired)

    def _solve_for_desired_velocity(
        self, q_full, X_BF_desired_list
    ):
        # Update our context.
        self.plant.SetPositions(self.plant_context, q_full)

        prog = MathematicalProgram()
        num_v = np.sum(self.active_dof)
        v_next = prog.NewContinuousVariables(num_v, "v_next")

        J_concat = []
        V_concat = []

        for frame_F, X_BF_desired in zip(self.frame_F_list, X_BF_desired_list):
            X_BF = self.plant.CalcRelativeTransform(
                self.plant_context, self.frame_B, frame_F
            )
            Jv_WF_full = self.plant.CalcJacobianSpatialVelocity(
                self.plant_context,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=frame_F,
                p_BoBp_B=np.zeros(3),
                frame_A=self.frame_B,
                frame_E=self.frame_B,
            )
            Jv_WF = Jv_WF_full[:, self.active_dof]
            K_dXtoV = 1.0
            V_WF_des = -K_dXtoV * se3_vector_minus(X_BF, X_BF_desired)

            J_concat.append(Jv_WF)
            V_concat.append(V_WF_des)

        J_concat = np.concatenate(J_concat)
        V_concat = np.concatenate(V_concat)

        # Add tracking objective.
        #   weight*|V_desired - J*v_desired|^2
        # NOTE: This is really the main difference from Drake's current diff IK
        # formulation. In Drake, the decision variable is a scale, `alpha`,
        # meant to constrain `J*v_desired` to lie on the line from 0 to
        # `V_desired`. However, for collision avoidance, this causes "sticking".
        # This relaxes the constraint, which is good for "sliding" avoidance,
        # but can still create issue when certain other constraints are active,
        # e.g. if you rotate J7 on the Panda to its limit, then desire
        # rotation may be well beyond what the controller can realize, and that
        # "bleeds" over into translation.

        weight = 100.0
        prog.Add2NormSquaredCost(
            np.sqrt(weight) * J_concat,
            np.sqrt(weight) * V_concat,
            v_next,
        )

        # Add nullspace objectives.
        #   | P (v_posture - v_desired) |^2
        # where
        #   v_posture = -K_null * (q - q0)
        q_active = q_full[self.active_dof]
        P = CalcNullspace(J_concat)
        K_null = 1.0
        v_null_des = -K_null * (q_active - self.q0_active)
        prog.Add2NormSquaredCost(
            P,
            P @ v_null_des,
            v_next,
        )

        # add min-distance constraint
        influence_distance = 0.02
        safety_distance = 0.01
        robot_clearance = self.collision_checker.CalcRobotClearance(
            q_full, influence_distance
        )
        remove_inactive = True
        dist, ddist_dq = MaybeUseActiveDistancesAndGradients(
            self.active_dof,
            remove_inactive,
            self.active_bodies,
            robot_clearance,
        )
        if len(dist) > 0:
            dist_min = (safety_distance - dist) / self.dt
            dist_max = np.full_like(dist, np.inf)
            prog.AddLinearConstraint(ddist_dq, dist_min, dist_max, v_next)

        # Add position and velocity limits.
        prog.AddBoundingBoxConstraint(
            (self.joint_limits.position_lower - q_active) / self.dt,
            (self.joint_limits.position_upper - q_active) / self.dt,
            v_next,
        )
        prog.AddBoundingBoxConstraint(
            self.joint_limits.velocity_lower,
            self.joint_limits.velocity_upper,
            v_next,
        )

        result = SolveOrDie(self.solver, self.solver_options, prog, tol=1e-4)
        v = result.GetSolution(v_next)
        return v
