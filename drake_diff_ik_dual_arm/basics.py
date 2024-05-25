import dataclasses as dc
import os
from pathlib import Path

import numpy as np

from pydrake.all import (
    CollisionFilterDeclaration,
    CommonSolverOption,
    CoulombFriction,
    JointIndex,
    ModelDirectives,
    PackageMap,
    Parser,
    ProcessModelDirectives,
    ProximityProperties,
    RigidTransform,
    RobotCollisionType,
    RobotDiagramBuilder,
    RotationMatrix,
    SceneGraphCollisionChecker,
    Sphere,
    yaml_load_typed,
)

import drake_diff_ik_dual_arm as _me


def ResolvePath(path):
    """Expands a path, expanding variables, ~, and package URLs."""
    pkg_prefix = "package://"
    if path.startswith(pkg_prefix):
        return MakeMyDefaultPackageMap().ResolveUrl(path)
    else:
        path = os.path.expandvars(os.path.expanduser(path))
        return os.path.abspath(path)


def MakeMyDefaultPackageMap():
    project_dir = Path(_me.__file__).parent
    package_map = PackageMap()
    package_map.Add("drake_diff_ik_dual_arm", str(project_dir))
    return package_map


def ProcessMyModelDirectives(directives, plant):
    parser = Parser(plant)
    parser.package_map().AddMap(MakeMyDefaultPackageMap())
    added_models = ProcessModelDirectives(directives, plant, parser=parser)
    return added_models


def LoadMyDirectives(path):
    return yaml_load_typed(schema=ModelDirectives, filename=ResolvePath(path))


def MakeRobotDiagramBuilder(directives, time_step):
    robot_builder = RobotDiagramBuilder(time_step=time_step)
    # TODO(eric.cousineau): Use this once we have release (> 1.29.0) that has
    # SceneGraphConfig:
    # https://github.com/RobotLocomotion/drake/commit/16e6967
    # # Configure Scene Graph.
    # scene_graph_config = SceneGraphConfig()
    # scene_graph_config.default_proximity_properties.compliance_type = "compliant"  # noqa
    # robot_builder.scene_graph().set_config(scene_graph_config)
    # Load our plant.
    ProcessMyModelDirectives(directives, robot_builder.plant())
    return robot_builder


def configuration_distance(q1, q2):
    """A boring implementation of ConfigurationDistanceFunction."""
    # TODO(eric.cousineau): Is there a way to ignore this for just a config
    # distance function?
    # What about a C++ function, without Python indirection, for better speed?
    return np.linalg.norm(q1 - q2)


def MakeCollisionChecker(robot_diagram, model_instances):
    return SceneGraphCollisionChecker(
        model=robot_diagram,
        configuration_distance_function=configuration_distance,
        edge_step_size=0.05,
        env_collision_padding=0.0,
        self_collision_padding=0.0,
        robot_model_instances=model_instances,
    )


def SolveOrDie(solver, solver_options, prog, *, tol, x0=None):
    """
    Solves a program; if it does not report success, or if solution
    constraints violate beyond a specified tolerance, it will re-solve the
    problem with some additional debug information enabled.
    """
    result = solver.Solve(
        prog, solver_options=solver_options, initial_guess=x0
    )
    infeasible = result.GetInfeasibleConstraintNames(prog, tol=tol)
    if not result.is_success() or len(infeasible) > 0:
        # TODO(eric.cousineau): Print out max violation.
        print(f"Infeasible constraints per Drake for tol={tol}:")
        print("\n".join(infeasible))
        print()
        print("Re-solving with verbose output")
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, True)
        result = solver.Solve(prog, solver_options=solver_options)
        print(result.get_solution_result())
        raise RuntimeError("Solver reports failure")
    return result


def CalcNullspace(J):
    n = J.shape[1]
    eye = np.eye(n)
    Jpinv = np.linalg.pinv(J)
    N = eye - Jpinv @ J
    return N


def GetJoints(plant, model_instances = None):
    joints = []
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        if model_instances is None or joint.model_instance() in model_instances:
            joints.append(joint)
    return joints


def GetActiveDof(plant, model_instances):
    assert plant.num_positions() == plant.num_velocities()
    num_q = plant.num_positions()
    active_dof = np.zeros(num_q, dtype=bool)
    joints = GetJoints(plant, model_instances)
    for joint in joints:
        start = joint.position_start()
        end = start + joint.num_positions()
        active_dof[start:end] = True
    return active_dof


def GetDofJoints(plant, active_dof):
    joints = []
    for joint in GetJoints(plant):
        start = joint.position_start()
        if joint.num_positions() != 1:
            continue
        end = start + joint.num_positions()
        for i in range(start, end):
            if active_dof[i]:
                joints.append(joint)
                break
    return joints


def GetBodiesKinematicallyAffectedBy(plant, active_dof):
    joints = GetDofJoints(plant, active_dof)
    joint_indices = [joint.index() for joint in joints]
    body_indices = plant.GetBodiesKinematicallyAffectedBy(joint_indices)
    return body_indices


def GetAllRobotModelInstances(plant, model_instances):
    # Get all model instances kinematically affected by a given set of model
    # instances.
    active_dof = GetActiveDof(plant, model_instances)
    body_indices = GetBodiesKinematicallyAffectedBy(plant, active_dof)
    all_model_instances = set()
    for body_index in body_indices:
        body = plant.get_body(body_index)
        all_model_instances.add(body.model_instance())
    return list(all_model_instances)


def IsSelfCollision(type: RobotCollisionType) -> bool:
    if type == RobotCollisionType.kSelfCollision:
        return True
    elif type == RobotCollisionType.kEnvironmentCollision:
        return False
    elif type == RobotCollisionType.kEnvironmentAndSelfCollision:
        raise RuntimeError("This case needs to be filled out")
    else:
        assert False, "Unreachable code execution"


def MaybeUseActiveDistancesAndGradients(
    active_dof,
    remove_inactive,
    active_bodies_for_collision_avoidance,
    robot_clearance,
):
    # WARNING! This is slow in Python.
    num_active_dof = np.sum(active_dof)

    all_dist = robot_clearance.distances()
    all_ddist_dq_full = robot_clearance.jacobians()
    all_collision_types = robot_clearance.collision_types()
    all_robot_indices = robot_clearance.robot_indices()
    all_other_indices = robot_clearance.other_indices()

    total = len(all_dist)
    dist_out = np.zeros(total)
    ddist_dq_out = np.zeros((total, num_active_dof))

    if not remove_inactive:
        dist_out[:] = all_dist
        ddist_dq_out[:] = all_ddist_dq_full[:, active_dof]
        return

    index = 0
    for all_index in range(total):
        robot_index = all_robot_indices[all_index]
        other_index = all_other_indices[all_index]
        collision_type = all_collision_types[all_index]
        is_self_collision = IsSelfCollision(collision_type)

        row_matches_active_body = (
            robot_index in active_bodies_for_collision_avoidance or
            (is_self_collision and other_index in active_bodies_for_collision_avoidance)
        )

        if row_matches_active_body:
            dist_out[index] = all_dist[all_index]
            ddist_dq_out[index, :] = all_ddist_dq_full[all_index, active_dof]
            index += 1

    return dist_out, ddist_dq_out


@dc.dataclass
class JointLimits:
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity_lower: np.ndarray
    velocity_upper: np.ndarray

    @staticmethod
    def from_plant(plant):
        return JointLimits(
            position_lower=plant.GetPositionLowerLimits(),
            position_upper=plant.GetPositionUpperLimits(),
            velocity_lower=plant.GetVelocityLowerLimits(),
            velocity_upper=plant.GetVelocityUpperLimits(),
        )

    def select(self, dof):
        return JointLimits(
            position_lower=self.position_lower[dof],
            position_upper=self.position_upper[dof],
            velocity_lower=self.velocity_lower[dof],
            velocity_upper=self.velocity_upper[dof],
        )
