import numpy as np

from pydrake.all import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)

from drake_diff_ik_dual_arm.basics import (
    GetActiveDof,
    GetAllRobotModelInstances,
    GetBodiesKinematicallyAffectedBy,
    LoadMyDirectives,
    MakeCollisionChecker,
    MakeRobotDiagramBuilder,
    MaybeUseActiveDistancesAndGradients,
)


def main():
    q = np.array(
        [-0.026876758365793365, -0.12254021249865074, -0.04811051330828734, -1.728093155433115, -0.03319111222948067, 1.8167875954294601, -0.3060053541065822, -0.01, 0.01, 0.04490423483755359, -0.1218547056500335, 0.043701697986823995, -1.7276672492590839, -0.0583164629312108, 1.8158546069486317, -0.13826765722663548, -0.01, 0.01]
    )

    directives = LoadMyDirectives(
        "package://drake_diff_ik_dual_arm/models/add_dual_arm_pandas.yaml"
    )
    robot_builder = MakeRobotDiagramBuilder(directives, time_step=0.0)
    builder = robot_builder.builder()
    plant = robot_builder.plant()
    plant.Finalize()
    ApplyVisualizationConfig(VisualizationConfig(), builder)
    diagram = robot_builder.Build()

    model_instances = [
        plant.GetModelInstanceByName("right::panda"),
        plant.GetModelInstanceByName("left::panda"),
    ]
    all_robot_model_instances = GetAllRobotModelInstances(
        plant, model_instances
    )
    collision_checker = MakeCollisionChecker(
        diagram, all_robot_model_instances
    )

    influence_distance = 0.02
    robot_clearance = collision_checker.CalcRobotClearance(
        q, influence_distance
    )
    print(robot_clearance.distances().min())

    active_dof = GetActiveDof(plant, model_instances)
    active_bodies = GetBodiesKinematicallyAffectedBy(
        plant, active_dof
    )
    remove_inactive = True
    dist, ddist_dq = MaybeUseActiveDistancesAndGradients(
        active_dof,
        remove_inactive,
        active_bodies,
        robot_clearance,
    )
    print(dist.min())

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)


if __name__ == "__main__":
    main()
