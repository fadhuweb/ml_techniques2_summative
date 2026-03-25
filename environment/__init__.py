from environment.custom_env import NigerianWildlifeConservationEnv, register_env
from environment.world_model import (
    ZONES, NUM_ZONES, ACTIONS, NUM_ACTIONS, ACTION_COSTS,
    ACTION_DEFINITIONS, ACTION_ECOSYSTEM_AFFINITY,
    ActionDefinition, get_action_detail, validate_action_precondition,
    get_effective_action_effects,
    ClimateDynamics, EcologicalModel, RewardCalculator,
)