import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, List, Tuple, Any

from environment.world_model import (
    ZONES,
    NUM_ZONES,
    ACTIONS,
    NUM_ACTIONS,
    ACTION_COSTS,
    WildlifeZone,
    ClimateDynamics,
    EcologicalModel,
    RewardCalculator,
)


class NigerianWildlifeConservationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}
    
    # State features per zone
    ZONE_FEATURES = [
        "temperature",          # normalized temp deviation from baseline
        "rainfall",             # normalized rainfall level
        "vegetation_index",     # NDVI proxy (0-1)
        "wildlife_pop",         # wildlife population index (0-1)
        "poaching_threat",      # poaching pressure (0-1)
        "habitat_integrity",    # habitat health (0-1)
        "last_action",          # last action taken in this zone (one-hot encoded → single normalized)
        "months_since_action",  # how many months since last intervention here
        "has_active_event",     # is there an extreme event active (0 or 1)
    ]
    NUM_ZONE_FEATURES = len(ZONE_FEATURES)  # 9 per zone
    NUM_GLOBAL_FEATURES = 5  # budget, timestep, active_events, mean_pop_trend, season
    
    def __init__(
        self,
        max_timesteps: int = 120,
        initial_budget: float = 100.0,
        monthly_budget_income: float = 5.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: str = "normal",  # "easy", "normal", "hard"
    ):
        super().__init__()
        
        self.max_timesteps = max_timesteps
        self.initial_budget = initial_budget
        self.monthly_budget_income = monthly_budget_income
        self.render_mode = render_mode
        self.difficulty = difficulty
        
        # Climate dynamics
        self.climate = ClimateDynamics()
        if difficulty == "hard":
            self.climate.warming_rate = 0.008
            self.climate.drought_probability = 0.05
            self.climate.wildfire_probability = 0.03
        elif difficulty == "easy":
            self.climate.warming_rate = 0.002
            self.climate.drought_probability = 0.01
            self.climate.wildfire_probability = 0.01
        
        # Random generator
        self._rng = np.random.default_rng(seed)
        
        # --- Define observation space ---
        obs_dim = NUM_ZONES * self.NUM_ZONE_FEATURES + self.NUM_GLOBAL_FEATURES
        # All features normalized to roughly [0, 1] except temperature
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        
        # --- Define action space ---
        self.action_space = spaces.Discrete(NUM_ZONES * NUM_ACTIONS)
        
        # --- Internal state ---
        self.zone_states: List[Dict[str, float]] = []
        self.budget = 0.0
        self.timestep = 0
        self.episode_events: List[List[str]] = []
        self.cumulative_reward = 0.0
        self.episode_history: List[Dict] = []
        
        # Resource allocation tracking (per zone)
        self.last_actions: np.ndarray = np.zeros(NUM_ZONES, dtype=np.float32)
        self.months_since_action: np.ndarray = np.zeros(NUM_ZONES, dtype=np.float32)
        self.prev_wildlife_pops: np.ndarray = np.zeros(NUM_ZONES, dtype=np.float32)
        
        # Rendering
        self._renderer = None
    
    def _get_initial_zone_state(self, zone: WildlifeZone) -> Dict[str, float]:
        """Generate the initial state for a zone with slight randomness."""
        noise = self._rng.normal(0, 0.02)
        return {
            "temperature": zone.base_temperature,
            "rainfall": zone.base_rainfall,
            "vegetation_index": np.clip(zone.base_vegetation_index + noise, 0, 1),
            "wildlife_pop": np.clip(zone.base_wildlife_pop + noise, 0, 1),
            "poaching_threat": np.clip(zone.base_poaching_threat + noise, 0, 1),
            "habitat_integrity": np.clip(zone.base_habitat_integrity + noise, 0, 1),
        }
    
    def _state_to_observation(self) -> np.ndarray:
        obs = []
        
        for i, zone in enumerate(ZONES):
            state = self.zone_states[i]
            
            # Core ecological features (6)
            obs.append((state["temperature"] - 15.0) / 30.0)  # 15-45°C → [0, 1]
            obs.append(state["rainfall"] / 500.0)               # 0-500mm → [0, 1]
            obs.append(state["vegetation_index"])                # already [0, 1]
            obs.append(state["wildlife_pop"])                    # already [0, 1]
            obs.append(state["poaching_threat"])                 # already [0, 1]
            obs.append(state["habitat_integrity"])               # already [0, 1]
            
            # Resource allocation features (3)
            obs.append(self.last_actions[i])                     # last action normalized [0, 1]
            obs.append(min(self.months_since_action[i] / 12.0, 1.0))  # capped at 12 months
            
            # Extreme event flag
            has_event = 1.0 if (self.episode_events and len(self.episode_events[i]) > 0) else 0.0
            obs.append(has_event)
        
        # Global features (5)
        obs.append(min(self.budget / self.initial_budget, 1.0))  # budget ratio
        obs.append(self.timestep / self.max_timesteps)           # time progress
        
        # Active events count (normalized)
        total_events = sum(len(e) for e in self.episode_events) if self.episode_events else 0
        obs.append(min(total_events / 10.0, 1.0))
        
        # Mean population trend (positive = growing, negative = declining)
        if self.timestep > 0:
            current_pops = np.array([s["wildlife_pop"] for s in self.zone_states])
            pop_trend = np.mean(current_pops - self.prev_wildlife_pops)
            obs.append(np.clip(pop_trend + 0.5, 0.0, 1.0))  # center at 0.5
        else:
            obs.append(0.5)  # neutral at start
        
        # Season indicator (cyclical encoding using sin)
        season = (np.sin(2 * np.pi * self.timestep / 12) + 1.0) / 2.0  # [0, 1]
        obs.append(season)
        
        return np.array(obs, dtype=np.float32)
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode flat action into (zone_index, action_type)."""
        zone_idx = action // NUM_ACTIONS
        action_type = action % NUM_ACTIONS
        return zone_idx, action_type
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Initialize zone states
        self.zone_states = [
            self._get_initial_zone_state(zone) for zone in ZONES
        ]
        
        # Initialize budget and time
        self.budget = self.initial_budget
        self.timestep = 0
        self.cumulative_reward = 0.0
        self.episode_events = [[] for _ in range(NUM_ZONES)]
        self.episode_history = []
        
        # Reset resource allocation tracking
        self.last_actions = np.zeros(NUM_ZONES, dtype=np.float32)
        self.months_since_action = np.zeros(NUM_ZONES, dtype=np.float32)
        self.prev_wildlife_pops = np.array(
            [s["wildlife_pop"] for s in self.zone_states], dtype=np.float32
        )
        
        obs = self._state_to_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep (one month of conservation management).
        
        The agent selects one zone to intervene in with a specific action.
        All other zones evolve naturally (action=0).
        """
        zone_idx, action_type = self._decode_action(action)
        
        # Build per-zone action list (only selected zone gets the action)
        zone_actions = [0] * NUM_ZONES
        
        # Check if we can afford the action
        cost = ACTION_COSTS[action_type] * self.initial_budget * 0.1
        if self.budget >= cost:
            zone_actions[zone_idx] = action_type
            self.budget -= cost
        else:
            # Can't afford → no action taken
            action_type = 0
        
        # Monthly budget income
        self.budget += self.monthly_budget_income
        self.budget = min(self.budget, self.initial_budget * 1.5)  # Cap budget
        
        # Save previous states
        prev_states = [dict(s) for s in self.zone_states]
        
        # Evolve each zone
        all_events = []
        invalid_actions = []
        for i, zone in enumerate(ZONES):
            # Get climate for this zone
            climate = self.climate.get_climate_state(zone, self.timestep, self._rng)
            
            # Compute next state (ecosystem-aware with cooldown)
            new_state, events, action_valid = EcologicalModel.compute_next_state(
                current_state=self.zone_states[i],
                zone=zone,
                climate=climate,
                action=zone_actions[i],
                rng=self._rng,
                months_since_last_action=float(self.months_since_action[i]),
            )
            
            self.zone_states[i] = new_state
            all_events.append(events)
            if not action_valid and zone_actions[i] != 0:
                invalid_actions.append((i, zone_actions[i]))
        
        self.episode_events = all_events
        self.timestep += 1
        
        # Update resource allocation tracking
        self.months_since_action += 1  # increment all zones
        if action_type != 0:
            self.last_actions[zone_idx] = action_type / (NUM_ACTIONS - 1)  # normalize to [0,1]
            self.months_since_action[zone_idx] = 0  # reset for acted zone
        
        # Track population trend
        self.prev_wildlife_pops = np.array(
            [s["wildlife_pop"] for s in self.zone_states], dtype=np.float32
        )
        
        # Compute reward
        reward, reward_breakdown = RewardCalculator.compute_reward(
            prev_states=prev_states,
            curr_states=self.zone_states,
            actions=zone_actions,
            budget_remaining=self.budget,
            total_budget=self.initial_budget,
            events=all_events,
            invalid_actions=invalid_actions,
        )
        self.cumulative_reward += reward
        
        # Check terminal conditions
        terminated = False
        truncated = False
        termination_reason = None
        
        # Extinction check (with cascading detection)
        # Single zone going critical is penalized via reward but doesn't end episode
        # Episode only terminates if 2+ zones go critical (cascading failure)
        critical_zones = []
        for i, state in enumerate(self.zone_states):
            if state["wildlife_pop"] <= 0.01:  # true extinction threshold
                critical_zones.append(ZONES[i].name)
        
        if len(critical_zones) >= 2:
            terminated = True
            termination_reason = f"cascading_extinction_in_{'+'.join(critical_zones)}"
        elif len(critical_zones) == 1:
            # Single extinction — heavy penalty in reward but episode continues
            # (agent can still try to save remaining zones)
            pass
        
        # Budget depletion (cannot afford even the cheapest non-free action)
        min_action_cost = min(c for c in ACTION_COSTS.values() if c > 0) * self.initial_budget * 0.1
        if self.budget < min_action_cost and not terminated:
            terminated = True
            termination_reason = "budget_depleted"
        
        # Critical ecosystem collapse (weighted by conservation priority)
        from environment.world_model import ZONE_CONSERVATION_PRIORITY
        weighted_health = np.mean([
            (0.4 * s["wildlife_pop"] + 0.3 * s["habitat_integrity"] + 0.3 * s["vegetation_index"])
            * ZONE_CONSERVATION_PRIORITY.get(i, 1.0)
            for i, s in enumerate(self.zone_states)
        ])
        if weighted_health < 0.1 and not terminated:
            terminated = True
            termination_reason = "ecosystem_collapse"
        
        # Time limit (truncation, not termination — important for RL)
        if self.timestep >= self.max_timesteps:
            truncated = True
            termination_reason = "time_limit"
        
        # Store history for analysis
        self.episode_history.append({
            "timestep": self.timestep,
            "action_zone": zone_idx,
            "action_type": action_type,
            "action_name": ACTIONS[action_type],
            "zone_name": ZONES[zone_idx].name,
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "budget": self.budget,
            "events": all_events,
            "zone_states": [dict(s) for s in self.zone_states],
        })
        
        obs = self._state_to_observation()
        info = self._get_info()
        info["reward_breakdown"] = reward_breakdown
        info["events"] = all_events
        info["termination_reason"] = termination_reason
        info["invalid_actions"] = invalid_actions  # actions that failed preconditions
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info dict."""
        zone_summaries = {}
        for i, zone in enumerate(ZONES):
            s = self.zone_states[i]
            zone_summaries[zone.name] = {
                "wildlife_pop": round(s["wildlife_pop"], 3),
                "habitat_integrity": round(s["habitat_integrity"], 3),
                "vegetation_index": round(s["vegetation_index"], 3),
                "poaching_threat": round(s["poaching_threat"], 3),
            }
        
        return {
            "timestep": self.timestep,
            "budget": round(self.budget, 2),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "zone_summaries": zone_summaries,
            "mean_wildlife_pop": round(
                np.mean([s["wildlife_pop"] for s in self.zone_states]), 3
            ),
            "mean_habitat_integrity": round(
                np.mean([s["habitat_integrity"] for s in self.zone_states]), 3
            ),
        }
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ("human", "rgb_array"):
            return self._render_visual()
        return None
    
    def _render_ansi(self) -> str:
        """Text-based rendering for terminal output."""
        # Season name from timestep
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        year = self.timestep // 12 + 1
        month = month_names[self.timestep % 12]
        
        # Last action taken (from history)
        last_action_str = "None"
        if self.episode_history:
            last = self.episode_history[-1]
            last_action_str = f"{last['action_name']} → {last['zone_name']}"
        
        lines = [
            f"\n{'='*80}",
            f"  NIGERIAN WILDLIFE CONSERVATION — Year {year}, {month} (Step {self.timestep}/{self.max_timesteps})",
            f"  Budget: {self.budget:.1f}/{self.initial_budget:.0f} | Reward: {self.cumulative_reward:+.2f} | Action: {last_action_str}",
            f"{'='*80}",
            f"  {'Zone':25s} | {'Pop':5s} | {'Hab':5s} | {'Poach':5s} | {'Veg':5s} | {'Events':20s}",
            f"  {'-'*74}",
        ]
        
        for i, zone in enumerate(ZONES):
            s = self.zone_states[i]
            events = self.episode_events[i] if self.episode_events else []
            event_str = ", ".join(events) if events else "-"
            
            # Color-code population status
            pop_indicator = "OK" if s["wildlife_pop"] > 0.3 else ("LOW" if s["wildlife_pop"] > 0.1 else "CRIT")
            
            lines.append(
                f"  {zone.name:25s} | {s['wildlife_pop']:.2f} | {s['habitat_integrity']:.2f} "
                f"| {s['poaching_threat']:.2f}  | {s['vegetation_index']:.2f} "
                f"| {event_str:20s} [{pop_indicator}]"
            )
        
        # Show reward breakdown if available
        if self.episode_history:
            rb = self.episode_history[-1]["reward_breakdown"]
            lines.append(f"  {'-'*74}")
            components = [f"{k}={v:+.2f}" for k, v in rb.items() if k != "total"]
            lines.append(f"  Reward: {', '.join(components)}")
        
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)
    
    def _render_visual(self):
        """Render using the Arcade (OpenGL-based) dashboard."""
        try:
            from environment.rendering import ArcadeRenderer
            if self._renderer is None:
                self._renderer = ArcadeRenderer(self)
            return self._renderer.render(
                self.zone_states, self.episode_events,
                self.budget, self.timestep, self.cumulative_reward,
            )
        except ImportError:
            return self._render_ansi()
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
    
    def get_zone_names(self) -> List[str]:
        """Return list of zone names for display."""
        return [z.name for z in ZONES]
    
    def get_action_name(self, action: int) -> str:
        """Return human-readable action description."""
        zone_idx, action_type = self._decode_action(action)
        return f"{ACTIONS[action_type]} → {ZONES[zone_idx].name}"
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive episode summary for post-training analysis.
        
        Returns a dict with:
        - overall metrics (total reward, episode length, termination reason)
        - per-zone final states and trajectories
        - action distribution (which actions were used, how often, where)
        - event log (all extreme events that occurred)
        - reward component breakdown over time
        """
        if not self.episode_history:
            return {"error": "No episode history available. Run an episode first."}
        
        num_steps = len(self.episode_history)
        last = self.episode_history[-1]
        
        # Action distribution
        action_counts = {}
        zone_action_counts = {z.name: {} for z in ZONES}
        for entry in self.episode_history:
            a_name = entry["action_name"]
            z_name = entry["zone_name"]
            action_counts[a_name] = action_counts.get(a_name, 0) + 1
            zone_action_counts[z_name][a_name] = zone_action_counts[z_name].get(a_name, 0) + 1
        
        # Event log
        all_events = []
        for entry in self.episode_history:
            for i, zone_events in enumerate(entry["events"]):
                for event in zone_events:
                    all_events.append({
                        "timestep": entry["timestep"],
                        "zone": ZONES[i].name,
                        "event": event,
                    })
        
        # Reward trajectory
        reward_trajectory = [e["reward"] for e in self.episode_history]
        reward_components_over_time = [e["reward_breakdown"] for e in self.episode_history]
        
        # Per-zone population trajectories
        zone_pop_trajectories = {z.name: [] for z in ZONES}
        zone_hab_trajectories = {z.name: [] for z in ZONES}
        for entry in self.episode_history:
            for i, zone in enumerate(ZONES):
                zone_pop_trajectories[zone.name].append(
                    entry["zone_states"][i]["wildlife_pop"]
                )
                zone_hab_trajectories[zone.name].append(
                    entry["zone_states"][i]["habitat_integrity"]
                )
        
        # Final state summary
        final_states = {}
        for i, zone in enumerate(ZONES):
            s = last["zone_states"][i]
            final_states[zone.name] = {
                "wildlife_pop": round(s["wildlife_pop"], 4),
                "habitat_integrity": round(s["habitat_integrity"], 4),
                "vegetation_index": round(s["vegetation_index"], 4),
                "poaching_threat": round(s["poaching_threat"], 4),
            }
        
        return {
            "episode_length": num_steps,
            "total_reward": round(self.cumulative_reward, 4),
            "mean_reward_per_step": round(self.cumulative_reward / max(num_steps, 1), 4),
            "termination_reason": last.get("termination_reason", "unknown"),
            "final_budget": round(self.budget, 2),
            "action_distribution": action_counts,
            "zone_action_distribution": zone_action_counts,
            "total_extreme_events": len(all_events),
            "event_log": all_events,
            "reward_trajectory": reward_trajectory,
            "reward_components": reward_components_over_time,
            "zone_population_trajectories": zone_pop_trajectories,
            "zone_habitat_trajectories": zone_hab_trajectories,
            "final_zone_states": final_states,
        }



# ENVIRONMENT REGISTRATION
def register_env():
    """Register the environment with Gymnasium."""
    gym.register(
        id="NigerianWildlifeConservation-v0",
        entry_point="environment.custom_env:NigerianWildlifeConservationEnv",
        max_episode_steps=120,
    )



# QUICK TEST
if __name__ == "__main__":
    print("Testing NigerianWildlifeConservationEnv...")
    
    env = NigerianWildlifeConservationEnv(render_mode="ansi", seed=42)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    # Run a few random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_desc = env.get_action_name(action)
        print(f"\nStep {step+1}: Action = {action_desc}")
        print(f"  Reward: {reward:.3f} | Budget: {info['budget']}")
        print(env.render())
        
        if terminated or truncated:
            print(f"Episode ended: {info.get('termination_reason', 'unknown')}")
            break
    
    print(f"\nTotal reward over {step+1} steps: {total_reward:.3f}")
    env.close()
    print("✓ Environment test passed!")