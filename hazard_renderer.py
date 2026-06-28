"""
Pure PIL renderer for PointHazardEnv — no OpenGL, no MuJoCo.

Draws a top-down 2D view:
  - Light gray arena background with subtle grid lines
  - Red filled circles for hazards (lava)
  - Green circle for the goal
  - Red circle (with body radius) for the agent
  - Blue line trail of agent positions
  - Optional velocity arrow
  - Optional info-text overlay (top-left)

This file mirrors the structure of `maze_renderer.py` for the old maze env,
but the world is just a square arena defined by `arena_half`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class HazardRenderer:
    def __init__(
        self,
        arena_half: float,
        agent_radius: float,
        goal_radius: float,
        *,
        img_size: int = 480,
        bg_color: tuple[int, int, int] = (245, 245, 245),
        grid_color: tuple[int, int, int] = (220, 220, 220),
        wall_color: tuple[int, int, int] = (90, 90, 90),
        hazard_fill: tuple[int, int, int] = (220, 60, 60),
        hazard_outline: tuple[int, int, int] = (140, 20, 20),
        agent_color: tuple[int, int, int] = (40, 90, 200),
        goal_color: tuple[int, int, int] = (50, 180, 50),
        trail_color: tuple[int, int, int] = (90, 130, 220),
        vel_color: tuple[int, int, int] = (30, 30, 120),
        semantic_fill: tuple[int, int, int, int] = (250, 205, 50, 80),  # translucent amber
        semantic_outline: tuple[int, int, int] = (180, 130, 0),         # dark amber
        semantic_style: str = "restricted",  # "restricted" (amber X) | "water"
        water_fill: tuple[int, int, int, int] = (90, 170, 205, 95),     # translucent teal
        water_outline: tuple[int, int, int] = (40, 110, 150),           # deep teal
        water_ripple: tuple[int, int, int, int] = (240, 250, 255, 175), # light ripples
    ):
        self.arena_half = float(arena_half)
        self.agent_radius = float(agent_radius)
        self.goal_radius = float(goal_radius)
        self.img_size = int(img_size)

        # Add a small visual margin around the arena so the wall is visible.
        margin = 0.4
        self.world_x_min = -self.arena_half - margin
        self.world_x_max = +self.arena_half + margin
        self.world_y_min = -self.arena_half - margin
        self.world_y_max = +self.arena_half + margin

        self.bg_color = bg_color
        self.grid_color = grid_color
        self.wall_color = wall_color
        self.hazard_fill = hazard_fill
        self.hazard_outline = hazard_outline
        self.agent_color = agent_color
        self.goal_color = goal_color
        self.trail_color = trail_color
        self.vel_color = vel_color
        self.semantic_fill = semantic_fill
        self.semantic_outline = semantic_outline
        self.semantic_style = semantic_style
        self.water_fill = water_fill
        self.water_outline = water_outline
        self.water_ripple = water_ripple

    @classmethod
    def from_env(cls, env, **kwargs) -> "HazardRenderer":
        """Construct from a PointHazardEnv (reads geometry from env.cfg)."""
        cfg = env.cfg
        return cls(
            arena_half=cfg.arena_half,
            agent_radius=cfg.agent_radius,
            goal_radius=cfg.goal_radius,
            img_size=cfg.render_size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # World <-> pixel mapping
    # ------------------------------------------------------------------

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        frac_x = (x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        frac_y = (y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        px = int(frac_x * self.img_size)
        py = int((1.0 - frac_y) * self.img_size)  # flip y so up is up
        return (int(np.clip(px, 0, self.img_size - 1)),
                int(np.clip(py, 0, self.img_size - 1)))

    def world_scale(self, d: float) -> int:
        frac = d / (self.world_x_max - self.world_x_min)
        return max(1, int(frac * self.img_size))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        agent_xy: np.ndarray,
        goal_xy: np.ndarray,
        hazards: np.ndarray,
        vel_xy: np.ndarray | None = None,
        trail: Sequence[np.ndarray] | None = None,
        info_text: str | None = None,
        semantic_zones: np.ndarray | None = None,
    ) -> np.ndarray:
        img = Image.new("RGB", (self.img_size, self.img_size), self.bg_color)
        draw = ImageDraw.Draw(img)

        # ---- arena bounding rectangle ---------------------------------
        x0, y0 = self.world_to_pixel(-self.arena_half, +self.arena_half)
        x1, y1 = self.world_to_pixel(+self.arena_half, -self.arena_half)
        lx, rx = min(x0, x1), max(x0, x1)
        ty, by = min(y0, y1), max(y0, y1)
        draw.rectangle([lx, ty, rx, by], outline=self.wall_color, width=3)

        # ---- subtle grid lines (every 1 unit) -------------------------
        step_world = 1.0
        n_lines = int(round(2 * self.arena_half / step_world))
        for i in range(n_lines + 1):
            wx = -self.arena_half + i * step_world
            px0, py0 = self.world_to_pixel(wx, -self.arena_half)
            px1, py1 = self.world_to_pixel(wx, +self.arena_half)
            draw.line([(px0, py0), (px1, py1)], fill=self.grid_color, width=1)
            wy = -self.arena_half + i * step_world
            px0, py0 = self.world_to_pixel(-self.arena_half, wy)
            px1, py1 = self.world_to_pixel(+self.arena_half, wy)
            draw.line([(px0, py0), (px1, py1)], fill=self.grid_color, width=1)

        # ---- semantic keep-out zones (off-limits regions) -------------
        # Drawn UNDER the hazards/agent so those stay readable, via a see-through
        # RGBA overlay.  Two appearances:
        #   "restricted" — a translucent amber disk + bold "X": an abstract
        #       "do not enter" symbol the prompt names explicitly (instruction-
        #       following test).
        #   "water" — a translucent teal disk with wavy ripple lines: terrain
        #       that *looks* unsafe to drive across, with NO symbol or label, so
        #       the VLM must recognise it from appearance (commonsense test).
        # Both are visually unlike the solid red hazards.
        if semantic_zones is not None and len(semantic_zones) > 0:
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            for zx, zy, zr in np.asarray(semantic_zones):
                cx, cy = self.world_to_pixel(float(zx), float(zy))
                pr = self.world_scale(float(zr))
                if self.semantic_style == "water":
                    odraw.ellipse(
                        [cx - pr, cy - pr, cx + pr, cy + pr],
                        fill=self.water_fill,
                        outline=self.water_outline + (255,),
                        width=3,
                    )
                    # Wavy ripple lines read as a water surface — no symbol, no
                    # label; the meaning lives entirely in the appearance.
                    for frac in (-0.4, -0.1, 0.2, 0.5):
                        yy = cy + frac * pr
                        half = 0.78 * pr
                        pts = []
                        for i in range(25):
                            t = i / 24.0
                            xx = cx - half + 2.0 * half * t
                            yo = yy + 0.13 * pr * np.sin(t * 4.0 * np.pi)
                            if (xx - cx) ** 2 + (yo - cy) ** 2 <= (0.92 * pr) ** 2:
                                pts.append((int(xx), int(yo)))
                        if len(pts) >= 2:
                            odraw.line(pts, fill=self.water_ripple, width=2, joint="curve")
                else:  # "restricted": amber disk + bold X
                    odraw.ellipse(
                        [cx - pr, cy - pr, cx + pr, cy + pr],
                        fill=self.semantic_fill,
                        outline=self.semantic_outline + (255,),
                        width=3,
                    )
                    # "X" from two chords that stay inside the circle (0.65*r corners)
                    k = int(0.65 * pr)
                    xline = self.semantic_outline + (255,)
                    odraw.line([(cx - k, cy - k), (cx + k, cy + k)], fill=xline, width=3)
                    odraw.line([(cx - k, cy + k), (cx + k, cy - k)], fill=xline, width=3)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

        # ---- hazards (lava circles) -----------------------------------
        if hazards is not None and len(hazards) > 0:
            for hx, hy, hr in np.asarray(hazards):
                cx, cy = self.world_to_pixel(float(hx), float(hy))
                pr = self.world_scale(float(hr))
                draw.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=self.hazard_fill,
                    outline=self.hazard_outline,
                    width=2,
                )

        # ---- trail ----------------------------------------------------
        if trail and len(trail) >= 2:
            for i in range(1, len(trail)):
                p0 = self.world_to_pixel(float(trail[i - 1][0]), float(trail[i - 1][1]))
                p1 = self.world_to_pixel(float(trail[i][0]), float(trail[i][1]))
                draw.line([p0, p1], fill=self.trail_color, width=2)

        # ---- goal -----------------------------------------------------
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        gpx, gpy = self.world_to_pixel(gx, gy)
        gpr = self.world_scale(self.goal_radius)
        draw.ellipse(
            [gpx - gpr, gpy - gpr, gpx + gpr, gpy + gpr],
            fill=self.goal_color,
            outline=(30, 130, 30),
            width=2,
        )
        draw.text((gpx - 4, gpy - 6), "G", fill=(255, 255, 255))

        # ---- agent ----------------------------------------------------
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        apx, apy = self.world_to_pixel(ax, ay)
        apr = self.world_scale(self.agent_radius)
        draw.ellipse(
            [apx - apr, apy - apr, apx + apr, apy + apr],
            fill=self.agent_color,
            outline=(20, 50, 130),
            width=2,
        )

        # ---- velocity arrow ------------------------------------------
        if vel_xy is not None:
            vx, vy = float(vel_xy[0]), float(vel_xy[1])
            speed = float(np.sqrt(vx * vx + vy * vy))
            if speed > 0.05:
                arrow_len_world = 0.6 * min(speed / 2.0, 1.0)
                ex = ax + (vx / speed) * arrow_len_world
                ey = ay + (vy / speed) * arrow_len_world
                epx, epy = self.world_to_pixel(ex, ey)
                draw.line([(apx, apy), (epx, epy)], fill=self.vel_color, width=3)
                draw.ellipse([epx - 3, epy - 3, epx + 3, epy + 3], fill=self.vel_color)

        # ---- info text ------------------------------------------------
        if info_text:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
                )
            except (OSError, IOError):
                font = ImageFont.load_default()
            y_off = 5
            for line in info_text.split("\n"):
                bbox = draw.textbbox((5, y_off), line, font=font)
                draw.rectangle(
                    [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                    fill=(255, 255, 255),
                )
                draw.text((5, y_off), line, fill=(0, 0, 0), font=font)
                y_off += bbox[3] - bbox[1] + 3

        return np.array(img, dtype=np.uint8)

    def render_to_pil(self, *args, **kwargs) -> Image.Image:
        return Image.fromarray(self.render(*args, **kwargs))
