"""
Pure PIL renderer for PointPushHazardEnv.

Draws a top-down 2D view:
  - Light gray square arena with a subtle grid
  - Red filled circles for hazards
  - Green circle for the box goal
  - Blue circle for the point agent
  - Orange square for the pushed box
  - Optional trails and velocity arrows for both bodies
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class PointPushHazardRenderer:
    def __init__(
        self,
        arena_half: float,
        agent_radius: float,
        box_half_size: float,
        goal_radius: float,
        *,
        img_size: int = 480,
        bg_color: tuple[int, int, int] = (246, 246, 244),
        grid_color: tuple[int, int, int] = (220, 220, 218),
        wall_color: tuple[int, int, int] = (85, 85, 85),
        hazard_fill: tuple[int, int, int] = (220, 60, 60),
        hazard_outline: tuple[int, int, int] = (140, 20, 20),
        agent_color: tuple[int, int, int] = (40, 90, 200),
        box_color: tuple[int, int, int] = (236, 157, 42),
        box_outline: tuple[int, int, int] = (130, 82, 18),
        goal_color: tuple[int, int, int] = (50, 180, 70),
        agent_trail_color: tuple[int, int, int] = (90, 130, 220),
        box_trail_color: tuple[int, int, int] = (210, 130, 45),
        agent_vel_color: tuple[int, int, int] = (20, 40, 130),
        box_vel_color: tuple[int, int, int] = (120, 70, 15),
    ):
        self.arena_half = float(arena_half)
        self.agent_radius = float(agent_radius)
        self.box_half_size = float(box_half_size)
        self.goal_radius = float(goal_radius)
        self.img_size = int(img_size)

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
        self.box_color = box_color
        self.box_outline = box_outline
        self.goal_color = goal_color
        self.agent_trail_color = agent_trail_color
        self.box_trail_color = box_trail_color
        self.agent_vel_color = agent_vel_color
        self.box_vel_color = box_vel_color

    @classmethod
    def from_env(cls, env, **kwargs) -> "PointPushHazardRenderer":
        cfg = env.cfg
        return cls(
            arena_half=cfg.arena_half,
            agent_radius=cfg.agent_radius,
            box_half_size=cfg.box_half_size,
            goal_radius=cfg.goal_radius,
            img_size=cfg.render_size,
            **kwargs,
        )

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        frac_x = (x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        frac_y = (y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        px = int(frac_x * self.img_size)
        py = int((1.0 - frac_y) * self.img_size)
        return (
            int(np.clip(px, 0, self.img_size - 1)),
            int(np.clip(py, 0, self.img_size - 1)),
        )

    def world_scale(self, d: float) -> int:
        frac = float(d) / (self.world_x_max - self.world_x_min)
        return max(1, int(frac * self.img_size))

    def _draw_trail(
        self,
        draw: ImageDraw.ImageDraw,
        trail: Sequence[np.ndarray] | None,
        color: tuple[int, int, int],
        *,
        width: int,
    ) -> None:
        if trail and len(trail) >= 2:
            for i in range(1, len(trail)):
                p0 = self.world_to_pixel(float(trail[i - 1][0]), float(trail[i - 1][1]))
                p1 = self.world_to_pixel(float(trail[i][0]), float(trail[i][1]))
                draw.line([p0, p1], fill=color, width=width)

    def _draw_velocity_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        xy: np.ndarray,
        vel_xy: np.ndarray | None,
        color: tuple[int, int, int],
        *,
        speed_scale: float,
    ) -> None:
        if vel_xy is None:
            return
        x, y = float(xy[0]), float(xy[1])
        vx, vy = float(vel_xy[0]), float(vel_xy[1])
        speed = float(np.hypot(vx, vy))
        if speed <= 0.05:
            return
        arrow_len_world = float(speed_scale) * min(speed / 2.0, 1.0)
        ex = x + (vx / speed) * arrow_len_world
        ey = y + (vy / speed) * arrow_len_world
        px, py = self.world_to_pixel(x, y)
        epx, epy = self.world_to_pixel(ex, ey)
        draw.line([(px, py), (epx, epy)], fill=color, width=3)
        draw.ellipse([epx - 3, epy - 3, epx + 3, epy + 3], fill=color)

    def render(
        self,
        agent_xy: np.ndarray,
        box_xy: np.ndarray,
        goal_xy: np.ndarray,
        hazards: np.ndarray,
        *,
        agent_vel_xy: np.ndarray | None = None,
        box_vel_xy: np.ndarray | None = None,
        agent_trail: Sequence[np.ndarray] | None = None,
        box_trail: Sequence[np.ndarray] | None = None,
        info_text: str | None = None,
    ) -> np.ndarray:
        img = Image.new("RGB", (self.img_size, self.img_size), self.bg_color)
        draw = ImageDraw.Draw(img)

        x0, y0 = self.world_to_pixel(-self.arena_half, +self.arena_half)
        x1, y1 = self.world_to_pixel(+self.arena_half, -self.arena_half)
        lx, rx = min(x0, x1), max(x0, x1)
        ty, by = min(y0, y1), max(y0, y1)
        draw.rectangle([lx, ty, rx, by], outline=self.wall_color, width=3)

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

        if hazards is not None and len(hazards) > 0:
            for hx, hy, hr in np.asarray(hazards, dtype=np.float32):
                cx, cy = self.world_to_pixel(float(hx), float(hy))
                pr = self.world_scale(float(hr))
                draw.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=self.hazard_fill,
                    outline=self.hazard_outline,
                    width=2,
                )

        self._draw_trail(draw, agent_trail, self.agent_trail_color, width=2)
        self._draw_trail(draw, box_trail, self.box_trail_color, width=3)

        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        gpx, gpy = self.world_to_pixel(gx, gy)
        gpr = self.world_scale(self.goal_radius)
        draw.ellipse(
            [gpx - gpr, gpy - gpr, gpx + gpr, gpy + gpr],
            fill=self.goal_color,
            outline=(30, 130, 45),
            width=2,
        )
        draw.text((gpx - 4, gpy - 6), "G", fill=(255, 255, 255))

        bx, by_ = float(box_xy[0]), float(box_xy[1])
        bpr = self.world_scale(self.box_half_size)
        bpx, bpy = self.world_to_pixel(bx, by_)
        draw.rectangle(
            [bpx - bpr, bpy - bpr, bpx + bpr, bpy + bpr],
            fill=self.box_color,
            outline=self.box_outline,
            width=2,
        )
        draw.text((bpx - 4, bpy - 6), "B", fill=(255, 255, 255))

        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        apx, apy = self.world_to_pixel(ax, ay)
        apr = self.world_scale(self.agent_radius)
        draw.ellipse(
            [apx - apr, apy - apr, apx + apr, apy + apr],
            fill=self.agent_color,
            outline=(20, 50, 130),
            width=2,
        )

        self._draw_velocity_arrow(
            draw, agent_xy, agent_vel_xy, self.agent_vel_color, speed_scale=0.6
        )
        self._draw_velocity_arrow(
            draw, box_xy, box_vel_xy, self.box_vel_color, speed_scale=0.5
        )

        if info_text:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
                )
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
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

        return np.asarray(img, dtype=np.uint8)

    def render_to_pil(self, *args, **kwargs) -> Image.Image:
        return Image.fromarray(self.render(*args, **kwargs))
