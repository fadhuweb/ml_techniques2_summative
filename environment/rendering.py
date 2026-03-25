import arcade
import math
import numpy as np
from typing import List, Dict, Optional, Tuple


# CONSTANTS
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 780
SCREEN_TITLE = "Nigerian Wildlife Conservation — RL Agent Dashboard"

# Color palette
COLOR_BG = (18, 22, 28)
COLOR_BG_PANEL = (28, 33, 42)
COLOR_BG_CARD = (38, 44, 56)
COLOR_TEXT = (220, 225, 235)
COLOR_TEXT_DIM = (140, 148, 165)
COLOR_TEXT_ACCENT = (100, 200, 160)
COLOR_BORDER = (55, 62, 78)
COLOR_GRID = (30, 36, 48)

# Zone health color gradient (green → yellow → orange → red)
HEALTH_COLORS = [
    (220, 50, 50),     # 0.0 - critical (red)
    (240, 130, 40),    # 0.25 - poor (orange)
    (240, 200, 50),    # 0.50 - fair (yellow)
    (100, 200, 100),   # 0.75 - good (green)
    (50, 180, 130),    # 1.0 - excellent (teal-green)
]

# Ecosystem type colors (for zone background tint)
ECOSYSTEM_COLORS = {
    "guinea_savanna": (180, 160, 80, 40),
    "tropical_rainforest": (40, 140, 60, 40),
    "sahel_wetland": (160, 140, 100, 40),
    "lowland_rainforest": (60, 120, 50, 40),
    "montane_forest_savanna": (100, 140, 80, 40),
    "floodplain_wetland": (60, 120, 160, 40),
}

# Action colors
ACTION_COLORS = {
    0: (80, 80, 80),       # no_action — gray
    1: (220, 80, 80),      # anti_poaching — red
    2: (80, 180, 80),      # habitat_restoration — green
    3: (80, 140, 220),     # water_provision — blue
    4: (200, 160, 60),     # species_relocation — gold
    5: (180, 100, 200),    # community_engagement — purple
    6: (100, 180, 180),    # wildlife_monitoring — teal
    7: (240, 120, 40),     # emergency — orange
}

ACTION_DISPLAY_NAMES = {
    0: "No Action",
    1: "Anti-Poaching",
    2: "Habitat Restore",
    3: "Water Provision",
    4: "Species Relocate",
    5: "Community Engage",
    6: "Wildlife Monitor",
    7: "Emergency",
}

# Approximate zone positions on the Nigeria map (normalized 0-1, then scaled)
ZONE_MAP_POSITIONS = {
    "Yankari":               (0.65, 0.58),
    "Cross River":           (0.75, 0.30),
    "Chad Basin":            (0.70, 0.82),
    "Okomu":                 (0.38, 0.28),
    "Gashaka Gumti":         (0.80, 0.48),
    "Hadejia-Nguru Wetlands": (0.62, 0.75),
}

# Nigeria outline (simplified polygon, normalized 0-1)
NIGERIA_OUTLINE = [
    (0.22, 0.40), (0.25, 0.50), (0.20, 0.60), (0.22, 0.70),
    (0.28, 0.78), (0.35, 0.85), (0.45, 0.90), (0.55, 0.92),
    (0.65, 0.90), (0.75, 0.88), (0.82, 0.82), (0.88, 0.72),
    (0.90, 0.60), (0.88, 0.48), (0.85, 0.38), (0.80, 0.28),
    (0.72, 0.20), (0.60, 0.18), (0.48, 0.18), (0.38, 0.20),
    (0.30, 0.25), (0.24, 0.32), (0.22, 0.40),
]



# HELPER FUNCTIONS
def health_to_color(health: float) -> Tuple[int, int, int]:
    """Convert a 0-1 health value to an RGB color along the gradient."""
    health = max(0.0, min(1.0, health))
    idx = health * (len(HEALTH_COLORS) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(HEALTH_COLORS) - 1)
    frac = idx - lower
    
    c1 = HEALTH_COLORS[lower]
    c2 = HEALTH_COLORS[upper]
    return (
        int(c1[0] + (c2[0] - c1[0]) * frac),
        int(c1[1] + (c2[1] - c1[1]) * frac),
        int(c1[2] + (c2[2] - c1[2]) * frac),
    )


def draw_bar(x, y, width, height, value, max_val, color, bg_color=(50, 55, 65)):
    """Draw a horizontal progress bar."""
    arcade.draw_lrtb_rectangle_filled(x, x + width, y + height, y, bg_color)
    fill_w = max(0, min(width, width * (value / max(max_val, 1e-6))))
    if fill_w > 0:
        arcade.draw_lrtb_rectangle_filled(x, x + fill_w, y + height, y, color)
    arcade.draw_lrtb_rectangle_outline(x, x + width, y + height, y, COLOR_BORDER, 1)



# MAIN VISUALIZATION WINDOW
class ConservationDashboard(arcade.Window):
    def __init__(self, env=None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=False)
        arcade.set_background_color(COLOR_BG)
        
        self.env = env
        
        # State cache
        self.zone_states: List[Dict] = []
        self.zone_names: List[str] = []
        self.events: List[List[str]] = [[] for _ in range(6)]
        self.budget: float = 100.0
        self.initial_budget: float = 100.0
        self.timestep: int = 0
        self.max_timesteps: int = 120
        self.cumulative_reward: float = 0.0
        self.last_action_zone: int = -1
        self.last_action_type: int = 0
        self.last_action_name: str = "—"
        self.reward_history: List[float] = []
        self.pop_history: List[float] = []
        
        # Animation timers
        self.pulse_timer: float = 0.0
        self.event_flash_timer: float = 0.0
        
        # Layout
        self.map_x = 40
        self.map_y = 140
        self.map_w = 620
        self.map_h = 600
        
        self.panel_x = 680
        self.panel_y = 140
        self.panel_w = 560
        self.panel_h = 600
        
        # ESC to close
        self.should_close = False
    
    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.should_close = True
    
    def update_state(
        self,
        zone_states: List[Dict],
        zone_names: List[str],
        events: List[List[str]],
        budget: float,
        timestep: int,
        max_timesteps: int,
        cumulative_reward: float,
        last_action_zone: int = -1,
        last_action_type: int = 0,
        last_action_name: str = "—",
        initial_budget: float = 100.0,
    ):
        """Update the dashboard state."""
        self.zone_states = zone_states
        self.zone_names = zone_names
        self.events = events
        self.budget = budget
        self.initial_budget = initial_budget
        self.timestep = timestep
        self.max_timesteps = max_timesteps
        self.cumulative_reward = cumulative_reward
        self.last_action_zone = last_action_zone
        self.last_action_type = last_action_type
        self.last_action_name = last_action_name
        
        if zone_states:
            mean_pop = np.mean([s.get("wildlife_pop", 0) for s in zone_states])
            self.pop_history.append(mean_pop)
            self.reward_history.append(cumulative_reward)
    
    def on_update(self, delta_time):
        self.pulse_timer += delta_time
        self.event_flash_timer += delta_time
    
    def on_draw(self):
        self.clear()
        self._draw_title_bar()
        self._draw_nigeria_map()
        self._draw_zone_panel()
        self._draw_bottom_hud()
    
    # TITLE BAR 
    def _draw_title_bar(self):
        """Draw the top title bar."""
        arcade.draw_lrtb_rectangle_filled(0, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT - 35, COLOR_BG_PANEL)
        arcade.draw_line(0, SCREEN_HEIGHT - 35, SCREEN_WIDTH, SCREEN_HEIGHT - 35, COLOR_BORDER, 1)
        
        arcade.draw_text(
            "NIGERIAN WILDLIFE CONSERVATION — RL AGENT DASHBOARD",
            SCREEN_WIDTH // 2, SCREEN_HEIGHT - 18,
            COLOR_TEXT_ACCENT, 13,
            anchor_x="center", anchor_y="center", bold=True,
        )
        
        # Model indicator (right side)
        arcade.draw_text(
            "OpenGL / Arcade Engine",
            SCREEN_WIDTH - 20, SCREEN_HEIGHT - 18,
            COLOR_TEXT_DIM, 9,
            anchor_x="right", anchor_y="center",
        )
    
    # MAP SECTION 
    
    def _draw_nigeria_map(self):
        """Draw the Nigeria outline map with zone markers."""
        mx, my, mw, mh = self.map_x, self.map_y, self.map_w, self.map_h
        
        # Map background
        arcade.draw_lrtb_rectangle_filled(mx, mx + mw, my + mh, my, COLOR_BG_PANEL)
        arcade.draw_lrtb_rectangle_outline(mx, mx + mw, my + mh, my, COLOR_BORDER, 1)
        
        # Section label
        arcade.draw_text(
            "Conservation Zone Map", mx + 12, my + mh - 18,
            COLOR_TEXT, 12, bold=True,
        )
        
        # Draw Nigeria outline
        outline_points = []
        for ox, oy in NIGERIA_OUTLINE:
            px = mx + 30 + ox * (mw - 60)
            py = my + 20 + oy * (mh - 60)
            outline_points.append((px, py))
        
        if len(outline_points) >= 3:
            # Filled polygon via triangle fan
            for i in range(1, len(outline_points) - 1):
                arcade.draw_triangle_filled(
                    outline_points[0][0], outline_points[0][1],
                    outline_points[i][0], outline_points[i][1],
                    outline_points[i+1][0], outline_points[i+1][1],
                    (30, 48, 35),
                )
            # Outline border
            for i in range(len(outline_points)):
                x1, y1 = outline_points[i]
                x2, y2 = outline_points[(i + 1) % len(outline_points)]
                arcade.draw_line(x1, y1, x2, y2, (70, 110, 75), 1.5)
        
        # Draw zone markers
        for i, z_name in enumerate(self.zone_names):
            if i >= len(self.zone_states):
                break
            
            pos = ZONE_MAP_POSITIONS.get(z_name, (0.5, 0.5))
            zx = mx + 30 + pos[0] * (mw - 60)
            zy = my + 20 + pos[1] * (mh - 60)
            
            state = self.zone_states[i]
            pop = state.get("wildlife_pop", 0.5)
            hab = state.get("habitat_integrity", 0.5)
            health = (pop + hab) / 2
            color = health_to_color(health)
            
            radius = 16 + int(pop * 22)
            
            # Pulse ring if this zone was acted on
            if i == self.last_action_zone:
                pulse = 5 * abs(math.sin(self.pulse_timer * 3))
                ac = ACTION_COLORS.get(self.last_action_type, (255, 255, 255))
                arcade.draw_circle_filled(zx, zy, radius + pulse + 5, (*ac, 50))
                arcade.draw_circle_outline(zx, zy, radius + pulse + 5, (*ac, 120), 2)
            
            # Event warning ring
            zone_events = self.events[i] if i < len(self.events) else []
            if zone_events:
                flash = 0.5 + 0.5 * abs(math.sin(self.event_flash_timer * 5))
                arcade.draw_circle_filled(zx, zy, radius + 8, (240, 50, 50, int(70 * flash)))
            
            # Main zone circle
            arcade.draw_circle_filled(zx, zy, radius, color)
            arcade.draw_circle_outline(zx, zy, radius, (255, 255, 255, 60), 1.5)
            
            # Population percentage inside
            arcade.draw_text(
                f"{pop:.0%}", zx, zy + 2,
                (255, 255, 255), 11,
                anchor_x="center", anchor_y="center", bold=True,
            )
            
            # Zone name below circle
            arcade.draw_text(
                z_name, zx, zy - radius - 13,
                COLOR_TEXT_DIM, 9,
                anchor_x="center", anchor_y="center",
            )
            
            # Event text above circle
            if zone_events:
                arcade.draw_text(
                    ", ".join(zone_events), zx, zy + radius + 14,
                    (240, 90, 60), 9,
                    anchor_x="center", anchor_y="center", bold=True,
                )
        
        # Population sparkline at bottom of map
        self._draw_sparkline(mx + 12, my + 12, mw - 24, 50)
    
    def _draw_sparkline(self, x, y, w, h):
        """Draw a mini sparkline of mean population over time."""
        points = self.pop_history[-120:]
        if len(points) < 2:
            return
        
        # Background
        arcade.draw_lrtb_rectangle_filled(x, x + w, y + h, y, (25, 30, 38, 180))
        
        min_v = max(0, min(points) - 0.05)
        max_v = min(1, max(points) + 0.05)
        rng = max(max_v - min_v, 0.01)
        
        for i in range(1, len(points)):
            x1 = x + (i - 1) / max(len(points) - 1, 1) * w
            x2 = x + i / max(len(points) - 1, 1) * w
            y1 = y + ((points[i-1] - min_v) / rng) * h
            y2 = y + ((points[i] - min_v) / rng) * h
            
            color = health_to_color(points[i])
            arcade.draw_line(x1, y1, x2, y2, (*color, 200), 1.5)
        
        arcade.draw_text(
            f"Avg Wildlife Pop: {points[-1]:.1%}", x + 4, y + h + 3,
            COLOR_TEXT_DIM, 8,
        )
    
    # ZONE DETAIL PANEL
    
    def _draw_zone_panel(self):
        """Draw detailed zone status cards."""
        px, py, pw, ph = self.panel_x, self.panel_y, self.panel_w, self.panel_h
        
        arcade.draw_lrtb_rectangle_filled(px, px + pw, py + ph, py, COLOR_BG_PANEL)
        arcade.draw_lrtb_rectangle_outline(px, px + pw, py + ph, py, COLOR_BORDER, 1)
        
        arcade.draw_text(
            "Zone Status Detail", px + 12, py + ph - 18,
            COLOR_TEXT, 12, bold=True,
        )
        
        # 2 columns x 3 rows of zone cards
        card_w = (pw - 30) // 2
        card_h = (ph - 50) // 3 - 6
        
        for i, z_name in enumerate(self.zone_names):
            if i >= len(self.zone_states):
                break
            
            col = i % 2
            row = 2 - i // 2
            
            cx = px + 10 + col * (card_w + 10)
            cy = py + 10 + row * (card_h + 6)
            
            self._draw_zone_card(cx, cy, card_w, card_h, z_name, self.zone_states[i], i)
    
    def _draw_zone_card(self, x, y, w, h, name, state, zone_idx):
        """Draw a single zone status card with bars and info."""
        pop = state.get("wildlife_pop", 0)
        hab = state.get("habitat_integrity", 0)
        veg = state.get("vegetation_index", 0)
        poach = state.get("poaching_threat", 0)
        temp = state.get("temperature", 28)
        rain = state.get("rainfall", 100)
        
        health = (pop + hab) / 2
        border_color = health_to_color(health)
        
        is_active = (zone_idx == self.last_action_zone)
        bg = (45, 52, 65) if is_active else COLOR_BG_CARD
        
        # Card body
        arcade.draw_lrtb_rectangle_filled(x, x + w, y + h, y, bg)
        
        # Left accent bar
        arcade.draw_lrtb_rectangle_filled(x, x + 4, y + h, y, border_color)
        
        # Status indicator dot
        status_color = (50, 200, 100) if pop > 0.3 else ((240, 180, 40) if pop > 0.1 else (220, 50, 50))
        arcade.draw_circle_filled(x + w - 14, y + h - 14, 5, status_color)
        
        # Zone name
        arcade.draw_text(name, x + 14, y + h - 16, COLOR_TEXT, 11, bold=True)
        
        # Current action indicator
        if is_active:
            ac = ACTION_COLORS.get(self.last_action_type, (200, 200, 200))
            action_label = ACTION_DISPLAY_NAMES.get(self.last_action_type, "?")
            arcade.draw_text(
                f">> {action_label}",
                x + 14, y + h - 32, ac, 9, bold=True,
            )
        
        # Stat bars
        bar_x = x + 14
        bar_w = w - 28
        bar_h = 8
        start_y = y + h - 50
        spacing = 26
        
        stats = [
            ("Wildlife Pop", pop, health_to_color(pop)),
            ("Habitat",      hab, health_to_color(hab)),
            ("Vegetation",   veg, (80, 170, 80)),
            ("Poaching",     poach, (200, 70, 70)),
        ]
        
        for j, (label, val, color) in enumerate(stats):
            by = start_y - j * spacing
            arcade.draw_text(f"{label}", bar_x, by + bar_h + 2, COLOR_TEXT_DIM, 8)
            arcade.draw_text(f"{val:.0%}", bar_x + bar_w - 1, by + bar_h + 2,
                             COLOR_TEXT_DIM, 8, anchor_x="right")
            draw_bar(bar_x, by, bar_w, bar_h, val, 1.0, color)
        
        # Climate info
        climate_y = start_y - len(stats) * spacing
        arcade.draw_text(
            f"Temp: {temp:.1f}C  |  Rain: {rain:.0f}mm",
            bar_x, climate_y, COLOR_TEXT_DIM, 8,
        )
        
        # Events
        zone_events = self.events[zone_idx] if zone_idx < len(self.events) else []
        if zone_events:
            arcade.draw_text(
                "! " + ", ".join(zone_events),
                bar_x, climate_y - 14, (240, 100, 60), 9, bold=True,
            )
    
    # BOTTOM HUD 
    
    def _draw_bottom_hud(self):
        """Draw the bottom status bar."""
        hx, hy, hw, hh = 40, 15, SCREEN_WIDTH - 80, 108
        
        arcade.draw_lrtb_rectangle_filled(hx, hx + hw, hy + hh, hy, COLOR_BG_PANEL)
        arcade.draw_lrtb_rectangle_outline(hx, hx + hw, hy + hh, hy, COLOR_BORDER, 1)
        
        # Time
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        year = self.timestep // 12 + 1
        month = month_names[self.timestep % 12] if self.timestep < self.max_timesteps else "END"
        
        sections = [
            ("TIME",   f"Y{year} {month}", f"Step {self.timestep}/{self.max_timesteps}", COLOR_TEXT_ACCENT),
            ("BUDGET", f"{self.budget:.1f}", f"of {self.initial_budget:.0f}", (100, 180, 240)),
            ("REWARD", f"{self.cumulative_reward:+.1f}", "cumulative", (220, 180, 60)),
            ("LAST ACTION", self.last_action_name,
             self.zone_names[self.last_action_zone] if 0 <= self.last_action_zone < len(self.zone_names) else "—",
             ACTION_COLORS.get(self.last_action_type, COLOR_TEXT)),
            ("AVG POP", f"{self.pop_history[-1]:.1%}" if self.pop_history else "—",
             "mean wildlife", health_to_color(self.pop_history[-1]) if self.pop_history else COLOR_TEXT),
        ]
        
        section_w = hw // len(sections)
        for i, (label, value, sub, color) in enumerate(sections):
            sx = hx + i * section_w + section_w // 2
            
            arcade.draw_text(label, sx, hy + hh - 16, COLOR_TEXT_DIM, 8,
                             anchor_x="center", anchor_y="center")
            arcade.draw_text(str(value), sx, hy + hh // 2 + 4, color, 16,
                             anchor_x="center", anchor_y="center", bold=True)
            arcade.draw_text(sub, sx, hy + 20, COLOR_TEXT_DIM, 8,
                             anchor_x="center", anchor_y="center")
            
            if i < len(sections) - 1:
                sep_x = hx + (i + 1) * section_w
                arcade.draw_line(sep_x, hy + 8, sep_x, hy + hh - 8, COLOR_BORDER, 1)
        
        # Budget bar
        budget_ratio = self.budget / max(self.initial_budget, 1)
        bx = hx + 4
        bw = hw - 8
        arcade.draw_lrtb_rectangle_filled(bx, bx + bw, hy + 6, hy + 2, (40, 45, 55))
        fill_w = bw * min(1, max(0, budget_ratio))
        bar_color = (100, 180, 240) if budget_ratio > 0.3 else (220, 80, 80)
        arcade.draw_lrtb_rectangle_filled(bx, bx + fill_w, hy + 6, hy + 2, bar_color)


# RENDERER WRAPPER (called from custom_env.py)
class ArcadeRenderer:
    def __init__(self, env):
        self.env = env
        self.window: Optional[ConservationDashboard] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        if not self._initialized:
            self.window = ConservationDashboard(self.env)
            self._initialized = True
    
    def render(
        self,
        zone_states: List[Dict],
        events: List[List[str]],
        budget: float,
        timestep: int,
        cumulative_reward: float,
    ):
        """Update dashboard and render one frame."""
        self._ensure_initialized()
        
        from environment.world_model import ZONES, ACTIONS
        
        zone_names = [z.name for z in ZONES]
        
        # Get last action from env history
        last_zone = -1
        last_action_type = 0
        last_action_name = "—"
        if hasattr(self.env, 'episode_history') and self.env.episode_history:
            last = self.env.episode_history[-1]
            last_zone = last.get("action_zone", -1)
            last_action_type = last.get("action_type", 0)
            last_action_name = last.get("action_name", "—")
        
        self.window.update_state(
            zone_states=zone_states,
            zone_names=zone_names,
            events=events,
            budget=budget,
            timestep=timestep,
            max_timesteps=self.env.max_timesteps,
            cumulative_reward=cumulative_reward,
            last_action_zone=last_zone,
            last_action_type=last_action_type,
            last_action_name=last_action_name,
            initial_budget=self.env.initial_budget,
        )
        
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.on_draw()
        self.window.flip()
        
        import time
        time.sleep(0.08)
        
        return self.window.should_close
    
    def close(self):
        if self.window:
            self.window.close()
            self.window = None
            self._initialized = False



# STANDALONE RANDOM AGENT DEMO
def run_random_demo():
    import sys
    sys.path.insert(0, ".")
    
    from environment.custom_env import NigerianWildlifeConservationEnv
    
    print("=" * 60)
    print("  RANDOM AGENT DEMO — Nigerian Wildlife Conservation")
    print("  Visualization: Arcade (OpenGL-based)")
    print("  Close window or press ESC to stop")
    print("=" * 60)
    
    env = NigerianWildlifeConservationEnv(
        render_mode="human",
        seed=42,
        max_timesteps=120,
    )
    
    renderer = ArcadeRenderer(env)
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        should_close = renderer.render(
            zone_states=env.zone_states,
            events=env.episode_events,
            budget=env.budget,
            timestep=env.timestep,
            cumulative_reward=env.cumulative_reward,
        )
        
        if should_close:
            break
        
        step_count += 1
        done = terminated or truncated
        
        # Terminal verbose output (required by rubric)
        action_name = env.get_action_name(action)
        print(f"Step {step_count:3d} | {action_name:40s} | "
              f"R: {reward:+.2f} | Budget: {env.budget:.1f} | "
              f"Pop: {info['mean_wildlife_pop']:.3f} | "
              f"Hab: {info['mean_habitat_integrity']:.3f}")
        
        if info.get("events"):
            for i, ze in enumerate(info["events"]):
                if ze:
                    from environment.world_model import ZONES
                    print(f"         !! {ZONES[i].name}: {', '.join(ze)}")
    
    # Episode summary
    summary = env.get_episode_summary()
    print(f"\n{'='*60}")
    print(f"  Episode Complete: {summary.get('termination_reason', '?')}")
    print(f"  Steps: {summary['episode_length']} | "
          f"Reward: {summary['total_reward']:.2f} | "
          f"Budget Left: {summary['final_budget']:.1f}")
    print(f"  Actions: {summary['action_distribution']}")
    print(f"  Events: {summary['total_extreme_events']} total")
    print(f"{'='*60}")
    
    renderer.close()
    env.close()


if __name__ == "__main__":
    run_random_demo()