"""
World Model: Nigerian Wildlife Conservation Environment
========================================================

This module defines the ecological world model for the RL environment.
It encodes real Nigerian conservation zones, endemic species, climate
dynamics, and the interplay between agent actions and ecosystem health.

The agent acts as a conservation resource manager allocating a limited
budget across 6 real Nigerian wildlife zones, making monthly decisions
over a 10-year horizon (120 timesteps) to maximize biodiversity and
ecosystem health under stochastic climate change.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ZONE DEFINITIONS — Real Nigerian Conservation Areas
@dataclass
class WildlifeZone:
    """Represents a real Nigerian conservation zone with ecological parameters."""
    name: str
    zone_id: int
    ecosystem_type: str          # e.g., "savanna", "rainforest", "wetland"
    area_km2: float              # approximate area
    base_temperature: float      # baseline mean annual temp (°C)
    base_rainfall: float         # baseline mean annual rainfall (mm/month)
    base_vegetation_index: float # baseline NDVI (0-1)
    key_species: List[str]       # flagship/indicator species
    base_wildlife_pop: float     # normalized population index (0-1)
    base_poaching_threat: float  # baseline threat level (0-1)
    base_habitat_integrity: float # baseline integrity (0-1)
    climate_vulnerability: float # how sensitive to climate shifts (0-1)
    latitude: float
    longitude: float


# 6 zones spanning Nigeria's major ecological regions
ZONES = [
    WildlifeZone(
        name="Yankari",
        zone_id=0,
        ecosystem_type="guinea_savanna",
        area_km2=2244,
        base_temperature=28.0,
        base_rainfall=90.0,
        base_vegetation_index=0.55,
        key_species=["African Elephant", "West African Lion", "Roan Antelope", "Hippopotamus"],
        base_wildlife_pop=0.65,
        base_poaching_threat=0.40,
        base_habitat_integrity=0.60,
        climate_vulnerability=0.55,
        latitude=9.75,
        longitude=10.51,
    ),
    WildlifeZone(
        name="Cross River",
        zone_id=1,
        ecosystem_type="tropical_rainforest",
        area_km2=4000,
        base_temperature=26.5,
        base_rainfall=200.0,
        base_vegetation_index=0.82,
        key_species=["Cross River Gorilla", "Nigeria-Cameroon Chimpanzee", "Forest Elephant", "African Grey Parrot"],
        base_wildlife_pop=0.45,
        base_poaching_threat=0.55,
        base_habitat_integrity=0.50,
        climate_vulnerability=0.70,
        latitude=5.83,
        longitude=8.75,
    ),
    WildlifeZone(
        name="Chad Basin",
        zone_id=2,
        ecosystem_type="sahel_wetland",
        area_km2=2258,
        base_temperature=30.5,
        base_rainfall=40.0,
        base_vegetation_index=0.30,
        key_species=["Migratory Palearctic Birds", "Nile Crocodile", "Marsh Mongoose", "Dama Gazelle"],
        base_wildlife_pop=0.40,
        base_poaching_threat=0.60,
        base_habitat_integrity=0.35,
        climate_vulnerability=0.85,
        latitude=12.80,
        longitude=14.10,
    ),
    WildlifeZone(
        name="Okomu",
        zone_id=3,
        ecosystem_type="lowland_rainforest",
        area_km2=181,
        base_temperature=27.0,
        base_rainfall=180.0,
        base_vegetation_index=0.78,
        key_species=["White-throated Guenon", "Forest Elephant", "Red-capped Mangabey", "Pangolin"],
        base_wildlife_pop=0.50,
        base_poaching_threat=0.50,
        base_habitat_integrity=0.55,
        climate_vulnerability=0.60,
        latitude=6.33,
        longitude=5.43,
    ),
    WildlifeZone(
        name="Gashaka Gumti",
        zone_id=4,
        ecosystem_type="montane_forest_savanna",
        area_km2=6731,
        base_temperature=24.0,
        base_rainfall=150.0,
        base_vegetation_index=0.70,
        key_species=["Nigeria-Cameroon Chimpanzee", "African Wild Dog", "Buffalo", "Leopard"],
        base_wildlife_pop=0.55,
        base_poaching_threat=0.45,
        base_habitat_integrity=0.58,
        climate_vulnerability=0.50,
        latitude=7.35,
        longitude=11.55,
    ),
    WildlifeZone(
        name="Hadejia-Nguru Wetlands",
        zone_id=5,
        ecosystem_type="floodplain_wetland",
        area_km2=3000,
        base_temperature=29.0,
        base_rainfall=55.0,
        base_vegetation_index=0.40,
        key_species=["Migratory Waterfowl", "Hippopotamus", "Fish Eagle", "Sitatunga"],
        base_wildlife_pop=0.50,
        base_poaching_threat=0.35,
        base_habitat_integrity=0.45,
        climate_vulnerability=0.80,
        latitude=12.45,
        longitude=10.45,
    ),
]

NUM_ZONES = len(ZONES)


# ACTION SPACE — Conservation Interventions
ACTIONS = {
    0: "no_action",                 # Do nothing (save budget)
    1: "anti_poaching_patrol",      # Deploy rangers to reduce poaching
    2: "habitat_restoration",       # Reforestation / wetland restoration
    3: "water_provision",           # Build/maintain water points
    4: "species_relocation",        # Relocate species to safer zones
    5: "community_engagement",      # Fund local communities for conservation
    6: "wildlife_monitoring",       # Deploy sensors / survey teams
    7: "emergency_intervention",    # Emergency response (fire, flood, disease)
}

NUM_ACTIONS = len(ACTIONS)

# Cost per action (fraction of monthly budget)
ACTION_COSTS = {
    0: 0.00,   # no_action
    1: 0.20,   # anti_poaching_patrol
    2: 0.25,   # habitat_restoration
    3: 0.15,   # water_provision
    4: 0.30,   # species_relocation
    5: 0.10,   # community_engagement
    6: 0.12,   # wildlife_monitoring
    7: 0.35,   # emergency_intervention
}



# CLIMATE DYNAMICS
@dataclass
class ClimateDynamics:
    """
    Stochastic climate model for the simulation.
    
    Models gradual warming trend + seasonal cycles + random weather events.
    Each zone responds differently based on its climate_vulnerability.
    """
    
    # Global warming trend: degrees C per timestep (month)
    warming_rate: float = 0.005
    
    # Seasonal amplitude for temperature (°C)
    temp_seasonal_amplitude: float = 3.0
    
    # Seasonal amplitude for rainfall (fraction of base)
    rain_seasonal_amplitude: float = 0.4
    
    # Noise standard deviations
    temp_noise_std: float = 1.5
    rain_noise_std: float = 0.25
    
    # Extreme event probability per zone per timestep
    drought_probability: float = 0.03
    flood_probability: float = 0.02
    wildfire_probability: float = 0.02
    disease_outbreak_probability: float = 0.015
    
    def get_climate_state(
        self, 
        zone: WildlifeZone, 
        timestep: int, 
        rng: np.random.Generator
    ) -> Dict[str, float]:
        """
        Compute climate variables for a zone at a given timestep.
        
        Returns dict with temperature_deviation, rainfall_factor, and
        any active extreme events.
        """
        # Seasonal cycle (12-month period)
        season = np.sin(2 * np.pi * timestep / 12)
        
        # Temperature: trend + seasonal + noise
        trend = self.warming_rate * timestep * zone.climate_vulnerability
        seasonal_temp = self.temp_seasonal_amplitude * season
        noise_temp = rng.normal(0, self.temp_noise_std)
        temp_deviation = trend + seasonal_temp + noise_temp
        
        # Rainfall: seasonal + noise (multiplicative)
        # Dry season vs wet season effect
        seasonal_rain = 1.0 + self.rain_seasonal_amplitude * season
        noise_rain = rng.normal(0, self.rain_noise_std)
        # Climate change tends to make dry areas drier, wet areas more variable
        climate_rain_effect = 1.0 - 0.002 * timestep * zone.climate_vulnerability
        rainfall_factor = max(0.1, seasonal_rain + noise_rain) * climate_rain_effect
        
        # Extreme events (stochastic)
        events = {
            "drought": rng.random() < self.drought_probability * (1 + 0.3 * zone.climate_vulnerability),
            "flood": rng.random() < self.flood_probability * rainfall_factor,
            "wildfire": rng.random() < self.wildfire_probability * (1 if season > 0.3 else 0.3),
            "disease_outbreak": rng.random() < self.disease_outbreak_probability,
        }
        
        return {
            "temp_deviation": temp_deviation,
            "rainfall_factor": rainfall_factor,
            "drought": events["drought"],
            "flood": events["flood"],
            "wildfire": events["wildfire"],
            "disease_outbreak": events["disease_outbreak"],
        }


# ─────────────────────────────────────────────────────────────
# ECOLOGICAL DYNAMICS — How state variables interact
# ─────────────────────────────────────────────────────────────

class EcologicalModel:
    """
    Models how climate, actions, and current state interact to produce
    the next ecological state for each zone.
    
    Key dynamics:
    - Vegetation responds to rainfall and temperature
    - Wildlife depends on vegetation, water, and poaching pressure
    - Habitat integrity degrades with climate stress, recovers with restoration
    - Poaching pressure is stochastic but reducible by patrols & community work
    """
    
    # --- Vegetation dynamics ---
    VEGETATION_RAINFALL_SENSITIVITY = 0.15    # How much rainfall affects NDVI
    VEGETATION_TEMP_SENSITIVITY = -0.02       # High temp slightly reduces vegetation
    VEGETATION_RECOVERY_RATE = 0.03           # Natural monthly recovery rate
    VEGETATION_DEGRADATION_RATE = 0.02        # Natural monthly degradation rate
    
    # --- Wildlife population dynamics ---
    WILDLIFE_GROWTH_RATE = 0.02               # Natural monthly growth when conditions good
    WILDLIFE_DECLINE_RATE = 0.03              # Decline when conditions bad
    WILDLIFE_POACHING_IMPACT = -0.05          # Population loss per unit poaching
    WILDLIFE_VEGETATION_DEPENDENCY = 0.3      # How much wildlife depends on vegetation
    WILDLIFE_WATER_DEPENDENCY = 0.2           # How much wildlife depends on water
    
    # --- Habitat integrity dynamics ---
    HABITAT_NATURAL_DEGRADATION = 0.005       # Monthly natural degradation
    HABITAT_CLIMATE_STRESS_FACTOR = 0.01      # Additional degradation from climate
    
    # --- Poaching dynamics ---
    POACHING_RANDOM_DRIFT = 0.05              # Monthly random change in threat
    POACHING_BASE_INCREASE = 0.01             # Poaching tends to increase over time
    
    # --- Action effects ---
    ACTION_EFFECTS = {
        0: {},  # no_action
        1: {"poaching_threat": -0.12, "wildlife_pop": 0.01},           # anti_poaching
        2: {"habitat_integrity": 0.08, "vegetation_index": 0.05},      # habitat_restoration
        3: {"wildlife_pop": 0.03, "vegetation_index": 0.02},           # water_provision
        4: {"wildlife_pop": 0.04},                                     # species_relocation
        5: {"poaching_threat": -0.08, "habitat_integrity": 0.03},      # community_engagement
        6: {"poaching_threat": -0.03, "wildlife_pop": 0.01},           # wildlife_monitoring
        7: {"wildlife_pop": 0.06, "habitat_integrity": 0.05,           # emergency_intervention
            "vegetation_index": 0.04},
    }
    
    # --- Extreme event impacts ---
    EVENT_IMPACTS = {
        "drought": {
            "vegetation_index": -0.15,
            "wildlife_pop": -0.08,
            "habitat_integrity": -0.05,
        },
        "flood": {
            "vegetation_index": -0.05,
            "wildlife_pop": -0.06,
            "habitat_integrity": -0.08,
        },
        "wildfire": {
            "vegetation_index": -0.20,
            "wildlife_pop": -0.10,
            "habitat_integrity": -0.12,
        },
        "disease_outbreak": {
            "wildlife_pop": -0.15,
        },
    }
    
    @staticmethod
    def compute_next_state(
        current_state: Dict[str, float],
        zone: WildlifeZone,
        climate: Dict[str, float],
        action: int,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        Transition function: compute the next ecological state for a zone.
        
        Parameters
        ----------
        current_state : dict with keys:
            temperature, rainfall, vegetation_index, wildlife_pop,
            poaching_threat, habitat_integrity
        zone : WildlifeZone metadata
        climate : output from ClimateDynamics.get_climate_state()
        action : int, action taken in this zone
        rng : numpy random generator
        
        Returns
        -------
        next_state : dict with same keys as current_state
        events_occurred : list of event names that fired this step
        """
        model = EcologicalModel
        
        veg = current_state["vegetation_index"]
        pop = current_state["wildlife_pop"]
        poach = current_state["poaching_threat"]
        hab = current_state["habitat_integrity"]
        
        # --- 1. Climate effects on temperature & rainfall ---
        new_temp = zone.base_temperature + climate["temp_deviation"]
        new_rain = zone.base_rainfall * climate["rainfall_factor"]
        
        # --- 2. Vegetation dynamics ---
        # Rainfall effect (positive when rain is near optimal, negative when too low/high)
        rain_ratio = new_rain / zone.base_rainfall
        rain_effect = model.VEGETATION_RAINFALL_SENSITIVITY * (rain_ratio - 1.0)
        rain_effect = np.clip(rain_effect, -0.10, 0.10)
        
        # Temperature effect (slight negative when warmer than baseline)
        temp_effect = model.VEGETATION_TEMP_SENSITIVITY * climate["temp_deviation"]
        
        # Natural recovery/degradation
        if veg < zone.base_vegetation_index:
            veg_natural = model.VEGETATION_RECOVERY_RATE * (zone.base_vegetation_index - veg)
        else:
            veg_natural = -model.VEGETATION_DEGRADATION_RATE * (veg - zone.base_vegetation_index)
        
        new_veg = veg + rain_effect + temp_effect + veg_natural
        
        # --- 3. Wildlife population dynamics ---
        # Carrying capacity depends on vegetation and habitat
        carrying_capacity = 0.5 * veg + 0.3 * hab + 0.2 * (rain_ratio)
        carrying_capacity = np.clip(carrying_capacity, 0.1, 1.0)
        
        # Logistic growth toward carrying capacity
        if pop < carrying_capacity:
            pop_growth = model.WILDLIFE_GROWTH_RATE * pop * (1 - pop / carrying_capacity)
        else:
            pop_growth = -model.WILDLIFE_DECLINE_RATE * (pop - carrying_capacity)
        
        # Poaching impact
        poaching_loss = model.WILDLIFE_POACHING_IMPACT * poach
        
        new_pop = pop + pop_growth + poaching_loss
        
        # --- 4. Habitat integrity dynamics ---
        climate_stress = model.HABITAT_CLIMATE_STRESS_FACTOR * abs(climate["temp_deviation"])
        new_hab = hab - model.HABITAT_NATURAL_DEGRADATION - climate_stress
        
        # --- 5. Poaching dynamics (stochastic) ---
        poach_drift = rng.normal(0, model.POACHING_RANDOM_DRIFT)
        new_poach = poach + model.POACHING_BASE_INCREASE + poach_drift
        
        # --- 6. Apply action effects ---
        action_effects = model.ACTION_EFFECTS.get(action, {})
        new_veg += action_effects.get("vegetation_index", 0)
        new_pop += action_effects.get("wildlife_pop", 0)
        new_poach += action_effects.get("poaching_threat", 0)
        new_hab += action_effects.get("habitat_integrity", 0)
        
        # --- 7. Apply extreme event impacts ---
        events_occurred = []
        for event_name in ["drought", "flood", "wildfire", "disease_outbreak"]:
            if climate.get(event_name, False):
                events_occurred.append(event_name)
                impacts = model.EVENT_IMPACTS.get(event_name, {})
                new_veg += impacts.get("vegetation_index", 0)
                new_pop += impacts.get("wildlife_pop", 0)
                new_hab += impacts.get("habitat_integrity", 0)
        
        # --- 8. Clip all values to valid ranges ---
        new_state = {
            "temperature": np.clip(new_temp, 15.0, 45.0),
            "rainfall": np.clip(new_rain, 0.0, 500.0),
            "vegetation_index": np.clip(new_veg, 0.0, 1.0),
            "wildlife_pop": np.clip(new_pop, 0.0, 1.0),
            "poaching_threat": np.clip(new_poach, 0.0, 1.0),
            "habitat_integrity": np.clip(new_hab, 0.0, 1.0),
        }
        
        return new_state, events_occurred



# REWARD FUNCTION
class RewardCalculator:
    """
    Composite reward function for the conservation agent.
    
    Components:
    - Biodiversity index: weighted sum of wildlife populations across zones
    - Habitat health: mean habitat integrity
    - Population stability: penalizes large drops in wildlife
    - Budget efficiency: rewards achieving goals with less spending
    - Extinction penalty: massive penalty if any zone's pop hits 0
    - Poaching incidents: penalty proportional to poaching pressure
    """
    
    # Reward component weights
    W_BIODIVERSITY = 3.0
    W_HABITAT = 2.0
    W_STABILITY = 1.5
    W_BUDGET_EFFICIENCY = 0.5
    W_EXTINCTION_PENALTY = -50.0
    W_POACHING_PENALTY = -1.0
    W_EXTREME_EVENT_RESPONSE = 2.0
    
    @staticmethod
    def compute_reward(
        prev_states: List[Dict[str, float]],
        curr_states: List[Dict[str, float]],
        actions: List[int],
        budget_remaining: float,
        total_budget: float,
        events: List[List[str]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the total reward and its components.
        
        Parameters
        ----------
        prev_states, curr_states : list of zone state dicts
        actions : list of actions taken per zone
        budget_remaining : remaining budget after actions
        total_budget : total budget for the episode
        events : list of event lists per zone
        
        Returns
        -------
        total_reward : float
        reward_breakdown : dict of component values
        """
        calc = RewardCalculator
        num_zones = len(curr_states)
        
        # --- Biodiversity index (mean wildlife pop) ---
        biodiversity = np.mean([s["wildlife_pop"] for s in curr_states])
        biodiversity_reward = calc.W_BIODIVERSITY * biodiversity
        
        # --- Habitat health ---
        habitat_health = np.mean([s["habitat_integrity"] for s in curr_states])
        habitat_reward = calc.W_HABITAT * habitat_health
        
        # --- Population stability (penalize drops) ---
        pop_changes = [
            curr_states[i]["wildlife_pop"] - prev_states[i]["wildlife_pop"]
            for i in range(num_zones)
        ]
        # Only penalize decreases, don't reward increases here (handled by biodiversity)
        stability_penalty = calc.W_STABILITY * sum(min(0, pc) for pc in pop_changes)
        
        # --- Budget efficiency ---
        budget_ratio = budget_remaining / max(total_budget, 1e-6)
        efficiency_reward = calc.W_BUDGET_EFFICIENCY * budget_ratio * biodiversity
        
        # --- Extinction penalty ---
        extinction_penalty = 0.0
        for s in curr_states:
            if s["wildlife_pop"] <= 0.02:  # Near-extinction threshold
                extinction_penalty += calc.W_EXTINCTION_PENALTY
        
        # --- Poaching penalty ---
        poaching_penalty = calc.W_POACHING_PENALTY * np.mean(
            [s["poaching_threat"] for s in curr_states]
        )
        
        # --- Extreme event response bonus ---
        event_response_bonus = 0.0
        for i in range(num_zones):
            if events[i] and actions[i] == 7:  # Emergency intervention during event
                event_response_bonus += calc.W_EXTREME_EVENT_RESPONSE
        
        # --- Total ---
        total = (
            biodiversity_reward
            + habitat_reward
            + stability_penalty
            + efficiency_reward
            + extinction_penalty
            + poaching_penalty
            + event_response_bonus
        )
        
        breakdown = {
            "biodiversity": biodiversity_reward,
            "habitat_health": habitat_reward,
            "stability": stability_penalty,
            "budget_efficiency": efficiency_reward,
            "extinction_penalty": extinction_penalty,
            "poaching_penalty": poaching_penalty,
            "event_response": event_response_bonus,
            "total": total,
        }
        
        return total, breakdown
