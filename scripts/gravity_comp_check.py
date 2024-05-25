import numpy as np

from pydrake.all import (
    AddMultibodyPlant,
    ApplyVisualizationConfig,
    DiagramBuilder,
    InverseDynamics,
    MultibodyPlantConfig,
    Simulator,
    VisualizationConfig,
)

from drake_diff_ik_dual_arm import debug
from drake_diff_ik_dual_arm.basics import (
    LoadMyDirectives,
    ProcessMyModelDirectives,
)


@debug.iex
def main():
    directives = LoadMyDirectives(
        "package://drake_diff_ik_dual_arm/models/add_dual_arm_pandas.yaml"
    )

    builder = DiagramBuilder()
    # Used for both simulation and control kinematics.
    plant_config = MultibodyPlantConfig(time_step=0.0)
    plant, scene_graph = AddMultibodyPlant(plant_config, builder)
    ProcessMyModelDirectives(directives, plant)
    plant.Finalize()
    ApplyVisualizationConfig(VisualizationConfig(), builder)

    plant_context_init = plant.CreateDefaultContext()
    q_panda = np.deg2rad([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 0.0])
    model_instances = [
        plant.GetModelInstanceByName("right::panda"),
        plant.GetModelInstanceByName("left::panda"),
    ]
    for model in model_instances:
        plant.SetPositions(plant_context_init, model, q_panda)

    hand_instances = [
        plant.GetModelInstanceByName("right::panda_hand"),
        plant.GetModelInstanceByName("left::panda_hand"),
    ]
    q_hand = np.array([-0.01, 0.01])
    for model in hand_instances:
        plant.SetPositions(plant_context_init, model, q_hand)

    gravity_comp = builder.AddSystem(
        InverseDynamics(
            plant=plant,
            mode=InverseDynamics.InverseDynamicsMode.kGravityCompensation,
        ),
    )
    builder.Connect(
        plant.get_state_output_port(),
        gravity_comp.get_input_port_estimated_state(),
    )
    # print(plant.MakeActuationMatrix())
    builder.Connect(
       gravity_comp.get_output_port_generalized_force(),
       plant.get_applied_generalized_force_input_port(),
        # plant.get_actuation_input_port(),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    diagram_context = simulator.get_context()

    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant_context.SetStateAndParametersFrom(plant_context_init)

    simulator.Initialize()
    simulator.AdvanceTo(1.0)


if __name__ == "__main__":
    main()
