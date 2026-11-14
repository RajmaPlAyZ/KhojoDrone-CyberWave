#!/usr/bin/env python3
"""
Khoj-o-Drone: autonomous disaster-zone search & rescue quadcopter simulation.

The mission logic (planner, perception, localization, navigation, PID flight
control, comms, supervisor, evaluator) runs locally and is fully seeded and
reproducible. The drone's pose is mirrored to a Cyberwave twin so you can watch
it in the Cyberwave environment viewer.

Usage
-----
  pip install cyberwave numpy matplotlib

  # 1) Offline dry run (no Cyberwave account needed):
  python khoj_o_drone.py --seed 7

  # 2) Batch of 10 seeded missions + aggregate pass rate:
  python khoj_o_drone.py --runs 10 --seed 100 --no-plot

  # 3) Mirror the drone into Cyberwave (free Playground):
  export CYBERWAVE_API_KEY="..."
  python khoj_o_drone.py --cyberwave --speed 4

Agents (each a class below)
---------------------------
  MissionPlanner   lawnmower coverage x2 + priority "investigate" detours
  Perception       downward camera model, occlusion, temporal confirmation
  Localization     pixel/pose -> world (x, y, z), fusion, de-duplication
  Navigator        potential-field obstacle avoidance + stuck recovery
  FlightController cascaded PID (position -> velocity -> acceleration)
  Communication    ground-station link with ack/retry, critical-first
  Supervisor       state machine + evaluator + report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np

G = 9.81


# ----------------------------------------------------------------------------
# Configuration (everything tunable lives here; override via CLI or edit)
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    # arena
    arena_x: float = 20.0
    arena_y: float = 20.0
    ceiling: float = 4.0
    n_survivors: int = 6
    n_critical: int = 2
    n_obstacles: int = 10
    mission_time: float = 480.0          # 8 min
    seed: int = 7
    pad_xy: tuple = (1.5, 1.5)           # launch pad
    station_xy: tuple = (0.5, 0.5)       # RESCUE TEAM ground station

    # drone / physics
    dt: float = 0.05
    vmax: float = 1.5                    # m/s horizontal
    amax: float = 4.0                    # m/s^2 commanded accel limit
    drone_radius: float = 0.15
    cruise_alt: float = 2.0
    investigate_alt: float = 1.5
    drag: float = 0.3
    wind_sigma: float = 0.3              # m/s^2 disturbance (OU process)

    # PID gains (position -> velocity, velocity -> accel)
    kp_pos_xy: float = 1.1
    kp_pos_z: float = 1.5
    kp_vel: float = 3.0
    ki_vel: float = 0.6
    kd_vel: float = 0.0

    # camera / perception
    fov_deg: float = 70.0
    camera_hz: float = 5.0
    p_detect: float = 0.90               # at zero occlusion
    p_class_correct: float = 0.92
    p_false_positive: float = 0.02       # per frame
    confirm_hits: int = 3                # temporal confirmation (N detections)
    meas_sigma_base: float = 0.10
    meas_sigma_per_m: float = 0.06       # grows with altitude
    dedupe_radius: float = 1.0

    # navigation
    avoid_influence: float = 1.5
    avoid_gain: float = 0.30
    stuck_timeout: float = 4.0

    # comms
    link_loss: float = 0.20
    link_latency: tuple = (0.05, 0.30)
    retry_backoff: float = 0.5
    max_attempts: int = 8

    # battery (% of pack)
    battery_hover_pct_s: float = 0.14
    battery_speed_pct_s: float = 0.02


# ----------------------------------------------------------------------------
# World (ground truth - used ONLY for physics, sensing and scoring)
# ----------------------------------------------------------------------------
@dataclass
class Survivor:
    sid: int
    pos: np.ndarray
    cls: str                 # "critical" | "stable"
    occlusion: float         # 0..1 fraction hidden by rubble


@dataclass
class World:
    cfg: Cfg
    obstacles: list = field(default_factory=list)   # (cx, cy, r)
    survivors: list = field(default_factory=list)

    @staticmethod
    def generate(cfg: Cfg, rng: np.random.Generator) -> "World":
        w = World(cfg)
        pad = np.array(cfg.pad_xy)
        tries = 0
        while len(w.obstacles) < cfg.n_obstacles and tries < 5000:
            tries += 1
            r = rng.uniform(0.6, 1.4)
            c = rng.uniform([3.0, 3.0], [cfg.arena_x - 3.0, cfg.arena_y - 3.0])
            if np.linalg.norm(c - pad) < r + 3.0:
                continue
            if all(np.linalg.norm(c - np.array(o[:2])) - r - o[2] > 1.0 for o in w.obstacles):
                w.obstacles.append((float(c[0]), float(c[1]), float(r)))
        # survivors sit just beside/behind rubble (partially occluded)
        tries = 0
        while len(w.survivors) < cfg.n_survivors and tries < 5000 and w.obstacles:
            tries += 1
            cx, cy, r = w.obstacles[rng.integers(len(w.obstacles))]
            ang = rng.uniform(0, 2 * math.pi)
            d = r + rng.uniform(0.5, 1.0)
            p = np.array([cx + d * math.cos(ang), cy + d * math.sin(ang)])
            if not (1.0 < p[0] < cfg.arena_x - 1 and 1.0 < p[1] < cfg.arena_y - 1):
                continue
            if any(np.linalg.norm(p - np.array(o[:2])) - o[2] < 0.3 for o in w.obstacles):
                continue
            if any(np.linalg.norm(p - s.pos[:2]) < 2.5 for s in w.survivors):
                continue
            if np.linalg.norm(p - pad) < 2.5:
                continue
            sid = len(w.survivors)
            cls = "critical" if sid < cfg.n_critical else "stable"
            occ = rng.uniform(0.3, 0.6) if cls == "critical" else rng.uniform(0.0, 0.3)
            w.survivors.append(Survivor(sid, np.array([p[0], p[1], 0.0]), cls, float(occ)))
        return w


# ----------------------------------------------------------------------------
# Flight controller agent: cascaded PID
# ----------------------------------------------------------------------------
class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, limit=1e9, i_limit=1.0):
        self.kp, self.ki, self.kd, self.limit, self.i_limit = kp, ki, kd, limit, i_limit
        self.i = 0.0
        self.prev = None

    def step(self, err, dt):
        self.i = float(np.clip(self.i + err * dt, -self.i_limit, self.i_limit))
        d = 0.0 if self.prev is None else (err - self.prev) / dt
        self.prev = err
        return float(np.clip(self.kp * err + self.ki * self.i + self.kd * d, -self.limit, self.limit))


class FlightController:
    """Position PID -> desired velocity; velocity PID -> acceleration command."""

    def __init__(self, cfg: Cfg, start):
        self.cfg = cfg
        self.p = np.array(start, float)
        self.v = np.zeros(3)
        self.wind = np.zeros(3)
        self.pos_pid = [PID(cfg.kp_pos_xy, limit=cfg.vmax), PID(cfg.kp_pos_xy, limit=cfg.vmax),
                        PID(cfg.kp_pos_z, limit=1.0)]
        self.vel_pid = [PID(cfg.kp_vel, cfg.ki_vel, cfg.kd_vel, limit=cfg.amax) for _ in range(3)]

    def step(self, target, v_extra, rng):
        c, dt = self.cfg, self.cfg.dt
        est_p = self.p + rng.normal(0, 0.03, 3)              # noisy pose estimate
        err = np.asarray(target, float) - est_p
        v_des = np.array([self.pos_pid[i].step(err[i], dt) for i in range(3)])
        v_des[:2] += v_extra
        n = np.linalg.norm(v_des[:2])
        if n > c.vmax:
            v_des[:2] *= c.vmax / n
        v_meas = self.v + rng.normal(0, 0.02, 3)
        a_cmd = np.array([self.vel_pid[i].step(v_des[i] - v_meas[i], dt) for i in range(3)])
        sat = bool(np.any(np.abs(a_cmd) >= c.amax - 1e-9))
        # disturbance: Ornstein-Uhlenbeck wind gusts
        self.wind += -0.5 * self.wind * dt + c.wind_sigma * math.sqrt(dt) * rng.normal(0, 1, 3)
        a = a_cmd + self.wind - c.drag * self.v
        self.v += a * dt
        self.p += self.v * dt
        if self.p[2] < 0:
            self.p[2], self.v[2] = 0.0, max(self.v[2], 0.0)
        tilt = math.degrees(math.atan2(float(np.linalg.norm(a[:2])), G))
        return v_des, sat, tilt


# ----------------------------------------------------------------------------
# Navigation & safety agent
# ----------------------------------------------------------------------------
class Navigator:
    def __init__(self, cfg: Cfg, obstacles):
        self.cfg, self.obstacles = cfg, obstacles

    def avoid(self, p_xy, goal_dir):
        """Repulsive + tangential potential field. Returns (v_extra, min_clearance)."""
        c = self.cfg
        rep = np.zeros(2)
        min_clear = float("inf")
        for cx, cy, r in self.obstacles:
            d_vec = p_xy - np.array([cx, cy])
            dist = float(np.linalg.norm(d_vec))
            clear = dist - r - c.drone_radius
            min_clear = min(min_clear, clear)
            if clear < c.avoid_influence:
                n = d_vec / max(dist, 1e-6)
                mag = min(c.avoid_gain * (1.0 / max(clear, 0.05) - 1.0 / c.avoid_influence), 2 * c.vmax)
                tang = np.array([-n[1], n[0]])
                if np.dot(tang, goal_dir) < 0:        # slide around on the goal side
                    tang = -tang
                rep += n * mag + tang * mag * 0.8
        return rep, min_clear

    def resolve_collision(self, p):
        """Hard safety: push the drone out of an obstacle and report a collision."""
        hit = False
        for cx, cy, r in self.obstacles:
            d_vec = p[:2] - np.array([cx, cy])
            dist = float(np.linalg.norm(d_vec))
            if dist < r + self.cfg.drone_radius:
                hit = True
                n = d_vec / max(dist, 1e-6)
                p[:2] = np.array([cx, cy]) + n * (r + self.cfg.drone_radius)
        return hit


# ----------------------------------------------------------------------------
# Mission planner agent
# ----------------------------------------------------------------------------
@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    kind: str = "lane"       # lane | investigate | takeoff | return
    dwell: float = 0.0
    tol: float = 0.5
    timeout: float = 25.0

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


class MissionPlanner:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.queue: deque[Waypoint] = deque()
        self.pass_idx = 0

    def _lanes(self, vertical: bool, alt: float):
        c = self.cfg
        half = alt * math.tan(math.radians(c.fov_deg / 2))
        margin = 1.2
        span = (c.arena_x if vertical else c.arena_y) - 2 * margin
        n = int(math.ceil(span / (1.7 * half))) + 1
        lanes = np.linspace(margin, (c.arena_x if vertical else c.arena_y) - margin, n)
        if vertical:
            lanes = lanes[::-1]
        end_lo, end_hi = margin, (c.arena_y if vertical else c.arena_x) - margin
        wps = []
        for i, l in enumerate(lanes):
            a, b = (end_lo, end_hi) if i % 2 == 0 else (end_hi, end_lo)
            if vertical:
                wps += [Waypoint(l, a, alt), Waypoint(l, b, alt)]
            else:
                wps += [Waypoint(a, l, alt), Waypoint(b, l, alt)]
        return wps

    def next_pass(self) -> bool:
        """Pass 0: horizontal lanes @cruise alt. Pass 1: vertical lanes, lower & closer."""
        if self.pass_idx == 0:
            self.queue.extend(self._lanes(False, self.cfg.cruise_alt))
        elif self.pass_idx == 1:
            self.queue.extend(self._lanes(True, self.cfg.cruise_alt - 0.4))
        else:
            return False
        self.pass_idx += 1
        return True

    def investigate(self, est_xyz):
        c = self.cfg
        self.queue.appendleft(Waypoint(est_xyz[0], est_xyz[1], c.investigate_alt,
                                       kind="investigate", dwell=1.2, tol=0.8, timeout=10.0))


# ----------------------------------------------------------------------------
# Perception + localization agents
# ----------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    sum_xyz: np.ndarray
    n: int = 0
    votes: dict = field(default_factory=lambda: {"critical": 0, "stable": 0})
    first_t: float = 0.0
    last_t: float = 0.0
    confirmed: bool = False
    confirmed_t: float = -1.0
    investigated: bool = False
    truth_sid: int | None = None          # evaluator-only; never used for decisions
    sent_est: np.ndarray | None = None

    @property
    def est(self):
        return self.sum_xyz / max(self.n, 1)

    @property
    def cls(self):
        return "critical" if self.votes["critical"] >= self.votes["stable"] else "stable"

    @property
    def confidence(self):
        return min(0.99, 1 - 0.5 ** self.n)


class Perception:
    """Downward RGB camera model. Emits noisy (class, pixel-offset) detections."""

    def __init__(self, cfg: Cfg, rng):
        self.cfg, self.rng = cfg, rng

    def half_footprint(self, z):
        return max(z, 0.1) * math.tan(math.radians(self.cfg.fov_deg / 2))

    def frame(self, drone_p, survivors):
        c, rng = self.cfg, self.rng
        half = self.half_footprint(drone_p[2])
        dets = []
        for s in survivors:
            off = s.pos[:2] - drone_p[:2]
            if abs(off[0]) < half and abs(off[1]) < half:
                if rng.random() < c.p_detect * (1.0 - s.occlusion):
                    cls = s.cls if rng.random() < c.p_class_correct else \
                        ("stable" if s.cls == "critical" else "critical")
                    dets.append((cls, off / half, "true"))   # normalized pixel offset in [-1, 1]
        if rng.random() < c.p_false_positive:                # clutter that looks like a person
            dets.append((("critical", "stable")[rng.integers(2)], rng.uniform(-1, 1, 2), "false"))
        return dets


class Localization:
    """Pixel + drone pose -> world xyz; fuse repeated observations; de-duplicate."""

    def __init__(self, cfg: Cfg, rng, perception: Perception):
        self.cfg, self.rng, self.perc = cfg, rng, perception
        self.tracks: list[Track] = []

    def to_world(self, norm_px, drone_p):
        c = self.cfg
        half = self.perc.half_footprint(drone_p[2])
        sigma = c.meas_sigma_base + c.meas_sigma_per_m * drone_p[2]
        xy = drone_p[:2] + np.asarray(norm_px) * half + self.rng.normal(0, sigma, 2)
        return np.array([xy[0], xy[1], 0.0])

    def update(self, detections, drone_p, t):
        newly_confirmed = []
        for cls, px, _ in detections:
            w = self.to_world(px, drone_p)
            best, bd = None, 1e9
            for tr in self.tracks:
                d = float(np.linalg.norm(tr.est[:2] - w[:2]))
                if d < bd:
                    best, bd = tr, d
            if best is None or bd > self.cfg.dedupe_radius:
                best = Track(len(self.tracks), np.zeros(3), first_t=t)
                self.tracks.append(best)
            best.sum_xyz += w
            best.n += 1
            best.votes[cls] += 1
            best.last_t = t
            if not best.confirmed and best.n >= self.cfg.confirm_hits:
                best.confirmed, best.confirmed_t = True, t
                newly_confirmed.append(best)
        return newly_confirmed

    def prune(self, t):
        self.tracks = [tr for tr in self.tracks if tr.confirmed or t - tr.last_t < 3.0]


# ----------------------------------------------------------------------------
# Communication agent + ground station
# ----------------------------------------------------------------------------
class GroundStation:
    def __init__(self):
        self.received: dict[int, dict] = {}
        self.log: list[dict] = []

    def receive(self, msg, t_recv):
        self.received[msg["id"]] = {**msg, "t_recv": t_recv}
        self.log.append({"id": msg["id"], "class": msg["class"], "t_recv": round(t_recv, 2)})


class Communication:
    def __init__(self, cfg: Cfg, rng, station: GroundStation):
        self.cfg, self.rng, self.station = cfg, rng, station
        self.outbox: list[dict] = []
        self.sent = self.acked = self.failed = 0
        self.latencies: list[float] = []

    def enqueue(self, tr: Track, t):
        # supersede any pending message for the same survivor
        self.outbox = [m for m in self.outbox if m["id"] != tr.tid]
        e = tr.est
        self.outbox.append({"id": tr.tid, "class": tr.cls, "xyz": [round(float(v), 3) for v in e],
                            "confidence": round(tr.confidence, 3), "t_created": t,
                            "attempts": 0, "next_try": t})
        tr.sent_est = e.copy()

    def pump(self, t):
        due = [m for m in self.outbox if m["next_try"] <= t]
        if not due:
            return
        due.sort(key=lambda m: (m["class"] != "critical", m["t_created"]))   # critical first
        m = due[0]
        m["attempts"] += 1
        self.sent += 1
        if self.rng.random() > self.cfg.link_loss:
            lat = float(self.rng.uniform(*self.cfg.link_latency))
            self.station.receive({k: m[k] for k in ("id", "class", "xyz", "confidence", "t_created")}, t + lat)
            self.latencies.append(t + lat - m["t_created"])
            self.acked += 1
            self.outbox.remove(m)
        elif m["attempts"] >= self.cfg.max_attempts:
            self.failed += 1
            self.outbox.remove(m)
        else:
            m["next_try"] = t + self.cfg.retry_backoff


# ----------------------------------------------------------------------------
# Optional Cyberwave bridge (mirrors the drone pose into a Cyberwave twin)
# ----------------------------------------------------------------------------
class CyberwaveBridge:
    """
    Mirrors the simulated drone into a Cyberwave twin using SDK calls that are
    documented: cw.twin(), cw.affect(), twin.edit_position(), cw.assets.search().
    takeoff()/land() are aerial-twin helpers; they are optional (--flight-helpers).
    """

    def __init__(self, asset=None, mode="playground", origin=(0.0, 0.0, 0.0), flight_helpers=False,
                 twin_id=None, environment_id=None):
        from cyberwave import Cyberwave          # imported lazily so offline runs work
        self.cw = Cyberwave()                    # reads CYBERWAVE_API_KEY
        self.environment_id = environment_id
        self.build_scene_on_start = True
        self.detail = "medium"
        self.show_truth = True
        twin_id = twin_id or os.getenv("CYBERWAVE_TWIN_ID")
        asset = asset or os.getenv("CYBERWAVE_DRONE_ASSET")
        self.twin = None
        errors = []
        if twin_id:                              # twin already placed in your environment (dashboard)
            try:
                self.twin = self.cw.twin(twin_id=twin_id)
            except Exception as e:
                errors.append(f"twin_id={twin_id}: {e}")
        if self.twin is None and asset:
            for key in self._candidate_keys(asset):
                try:
                    self.twin = self.cw.twin(key)
                    print(f"[cyberwave] using asset key: {key}")
                    break
                except Exception as e:
                    errors.append(f"{key}: {e}")
        if self.twin is None:
            for e in errors:
                print("[cyberwave]", e)
            raise RuntimeError(
                "Could not create/find a drone twin.\n"
                "Fix: open your environment in the Cyberwave dashboard, add a drone from the catalog, "
                "copy that twin's ID/slug, then run with --twin-id <id>.\n"
                "Run with --list-assets to see what the SDK returns for your catalog.")
        self.cw.affect(mode)                     # "playground" (free) | "simulation" | "live"
        self.origin = np.array(origin, float)
        self.flight_helpers = flight_helpers
        self._last_push = 0.0

    def _candidate_keys(self, asset):
        keys = [asset]
        try:
            for a in self.cw.assets.search(asset.split("/")[-1]):
                if getattr(a, "registry_id", None) == asset:
                    for attr in ("slug", "uuid"):
                        v = getattr(a, attr, None)
                        if v and v not in keys:
                            keys.append(str(v))
        except Exception:
            pass
        return keys

    @staticmethod
    def list_assets(query="drone"):
        from cyberwave import Cyberwave
        cw = Cyberwave()
        res = list(cw.assets.search(query))
        print(f"{len(res)} result(s) for '{query}'")
        for a in res:
            fields = {k: getattr(a, k, None) for k in ("registry_id", "name", "slug", "uuid")}
            print("  ", fields)
        if res:
            print("attributes of first result:", [k for k in dir(res[0]) if not k.startswith("_")][:40])

    # ---- scene: make the "map" visible inside Cyberwave --------------------------------
    def _env_id(self):
        for attr in ("environment_id", "environment_uuid", "environment"):
            v = getattr(self.twin, attr, None)
            if v:
                return str(getattr(v, "uuid", v))
        return self.environment_id or os.getenv("CYBERWAVE_ENVIRONMENT_ID")

    def _waypoint(self, env, wid, name, pos, meta):
        q = np.asarray(pos, float) + self.origin
        try:
            try:
                self.cw.environments.delete_waypoint(env, wid)   # clean up leftovers from a previous run
            except Exception:
                pass
            self.cw.environments.create_waypoint(
                env, waypoint_id=wid, name=name,
                position={"x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                rotation={"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}, metadata=meta)
            return True
        except Exception as e:
            print(f"[cyberwave] waypoint '{wid}' failed: {e}")
            return False

    # ---- primitives / spawning ----------------------------------------------------------
    def _prims(self):
        if getattr(self, "_prim_cache", None) is not None:
            return self._prim_cache
        try:
            prims = list(self.cw.assets.list_primitives())
        except Exception as e:
            print(f"[cyberwave] list_primitives failed: {e}")
            prims = []
        names = [(str(getattr(x, "registry_id_alias", "")).lower(), x) for x in prims]
        print("[cyberwave] primitives available:", [n for n, _ in names])

        def find(*keys):
            for k in keys:
                for n, x in names:
                    if k in n:
                        return x
            return None
        box, cyl, sph = find("cube", "box"), find("cylinder"), find("sphere")
        self._prim_cache = {"box": box or cyl, "cyl": cyl or box, "sph": sph or box or cyl}
        return self._prim_cache

    def _spawn(self, env, prim, name, pos, scale, yaw=0.0, pitch=0.0):
        created = self.cw.twins.create(asset_id=prim.uuid, environment_id=env, name=name)
        tid = getattr(created, "uuid", None) or created["uuid"]
        t = self.cw.twin(twin_id=str(tid))
        t.edit_position(x=float(pos[0] + self.origin[0]), y=float(pos[1] + self.origin[1]),
                        z=float(pos[2] + self.origin[2]))
        t.edit_scale(x=float(scale[0]), y=float(scale[1]), z=float(scale[2]))
        if yaw or pitch:
            try:
                t.edit_rotation(yaw=float(yaw), pitch=float(pitch))
            except Exception:
                t.edit_rotation(yaw=float(yaw))
        return t

    def clear_scene(self):
        """Delete objects created by earlier runs (names start with 'khoj-' or 'rubble-')."""
        try:
            twins = list(self.cw.twins.list())
        except Exception as e:
            print(f"[cyberwave] could not list twins for cleanup: {e}")
            return 0
        n = 0
        for t in twins:
            name = str(getattr(t, "name", "") or "")
            uid = getattr(t, "uuid", None)
            if uid and name.startswith(("khoj-", "rubble-")):
                try:
                    self.cw.twin(twin_id=str(uid)).delete()
                    n += 1
                except Exception:
                    try:
                        self.cw.twins.delete(str(uid))
                        n += 1
                    except Exception as e:
                        print(f"[cyberwave] could not delete {name}: {e}")
        print(f"[cyberwave] removed {n} old scene object(s)")
        return n

    def build_scene(self, world, cfg, show_truth=True):
        """Disaster-zone arena: floor, border walls, rubble piles with debris, station, survivors."""
        env = self._env_id()
        if not env:
            print("[cyberwave] scene skipped: unknown environment. Pass --environment-id <uuid>.")
            return
        self._env = env
        tag = f"khoj-s{cfg.seed}"
        self.clear_scene()
        print(f"[cyberwave] building scene in environment {env} (detail={self.detail}) ...")
        # labelled markers (always work, even if shape spawning is unavailable)
        self._waypoint(env, f"{tag}-station", "RESCUE TEAM (ground station)",
                       [cfg.station_xy[0], cfg.station_xy[1], 0.0], {"kind": "station"})
        self._waypoint(env, f"{tag}-pad", "LAUNCH PAD", [cfg.pad_xy[0], cfg.pad_xy[1], 0.0], {"kind": "pad"})
        if show_truth:
            for sv in world.survivors:
                self._waypoint(env, f"{tag}-survivor-{sv.sid}",
                               f"SURVIVOR {sv.sid} ({sv.cls.upper()}) [hidden from drone]",
                               sv.pos, {"kind": "survivor_truth", "class": sv.cls})

        pr = self._prims()
        rng = np.random.default_rng(cfg.seed + 999)
        state = {"ok": pr["box"] is not None, "n": 0}

        def sp(kind, name, pos, scale, yaw=0.0, pitch=0.0):
            if not state["ok"]:
                return False
            try:
                self._spawn(env, pr[kind], name, pos, scale, yaw, pitch)
                state["n"] += 1
                if state["n"] % 10 == 0:
                    print(f"[cyberwave]   ...{state['n']} objects placed")
                return True
            except Exception as e:
                print(f"[cyberwave] spawning shapes failed ({e}); using markers only")
                state["ok"] = False
                return False

        W, H = cfg.arena_x, cfg.arena_y
        # ground slab + low border walls so the arena has a clear boundary
        sp("box", "khoj-floor", [W / 2, H / 2, -0.03], [W, H, 0.06])
        wall_h, wall_t = 0.5, 0.2
        sp("box", "khoj-wall-s", [W / 2, -wall_t / 2, wall_h / 2], [W + 2 * wall_t, wall_t, wall_h])
        sp("box", "khoj-wall-n", [W / 2, H + wall_t / 2, wall_h / 2], [W + 2 * wall_t, wall_t, wall_h])
        sp("box", "khoj-wall-w", [-wall_t / 2, H / 2, wall_h / 2], [wall_t, H, wall_h])
        sp("box", "khoj-wall-e", [W + wall_t / 2, H / 2, wall_h / 2], [wall_t, H, wall_h])
        # rescue team building + launch pad
        sp("box", "khoj-station", [cfg.station_xy[0], cfg.station_xy[1], 0.7], [1.4, 1.0, 1.4])
        sp("cyl", "khoj-pad", [cfg.pad_xy[0], cfg.pad_xy[1], 0.02], [1.2, 1.2, 0.04])

        # rubble piles: main mound + debris (all kept inside the avoidance radius)
        n_ring = {"low": 0, "medium": 3, "high": 5}[self.detail]
        for i, (cx, cy, r) in enumerate(world.obstacles):
            h = float(rng.uniform(0.6, 1.4))
            sp("cyl", f"khoj-rubble-{i}", [cx, cy, h / 2], [2 * r, 2 * r, h], yaw=float(rng.uniform(0, 90)))
            if self.detail == "low":
                continue
            for k in range(n_ring):                                   # broken blocks around the base
                ang = rng.uniform(0, 2 * math.pi)
                d = r * rng.uniform(0.85, 1.05)
                sz = rng.uniform(0.25, 0.55)
                sp("box", f"khoj-debris-{i}-{k}", [cx + d * math.cos(ang), cy + d * math.sin(ang), sz * 0.4],
                   [sz * rng.uniform(0.8, 1.6), sz * rng.uniform(0.6, 1.2), sz * 0.8],
                   yaw=float(rng.uniform(0, 180)))
            for k in range(2 if self.detail == "high" else 1):        # rocks on top of the pile
                ang = rng.uniform(0, 2 * math.pi)
                d = r * rng.uniform(0.0, 0.45)
                sz = rng.uniform(0.3, 0.6)
                sp("box", f"khoj-rock-{i}-{k}", [cx + d * math.cos(ang), cy + d * math.sin(ang), h + sz * 0.4],
                   [sz, sz * rng.uniform(0.7, 1.3), sz * 0.8], yaw=float(rng.uniform(0, 180)))
            if self.detail == "high":                                  # leaning slab (collapsed wall)
                ang = rng.uniform(0, 2 * math.pi)
                sp("box", f"khoj-slab-{i}", [cx + 0.3 * r * math.cos(ang), cy + 0.3 * r * math.sin(ang), h + 0.35],
                   [r * 1.6, 0.15, 0.9], yaw=math.degrees(ang) + 90, pitch=float(rng.uniform(15, 35)))
            if not state["ok"]:
                break

        # survivors: shape tells the class (lying = critical, standing = stable); drone never sees these
        if show_truth:
            for sv in world.survivors:
                if sv.cls == "critical":
                    sp("box", f"khoj-survivor-{sv.sid}", [sv.pos[0], sv.pos[1], 0.12], [1.7, 0.5, 0.24],
                       yaw=float(rng.uniform(0, 180)))
                else:
                    sp("cyl", f"khoj-survivor-{sv.sid}", [sv.pos[0], sv.pos[1], 0.5], [0.45, 0.45, 1.0])

        if not state["ok"]:                                           # fallback: markers for rubble
            for i, (cx, cy, r) in enumerate(world.obstacles):
                self._waypoint(env, f"{tag}-rubble-{i}", f"RUBBLE {i} (r={r:.1f} m)", [cx, cy, 0.0],
                               {"kind": "rubble", "radius": r})
        print(f"[cyberwave] scene ready: {state['n']} shapes placed. Refresh the environment page.")

    def mark_detection(self, tr, seed):
        env = getattr(self, "_env", None)
        if not env:
            return
        e = tr.est
        self._waypoint(env, f"khoj-s{seed}-detected-{tr.tid}", f"DETECTED #{tr.tid} ({tr.cls})",
                       [e[0], e[1], 0.0], {"kind": "detected", "class": tr.cls})

    def _call(self, name):
        if not self.flight_helpers:
            return
        try:
            getattr(self.twin, name)()
        except Exception as e:                   # helper not available on this twin
            print(f"[cyberwave] {name}() unavailable: {e}")

    def takeoff(self):
        self._call("takeoff")

    def land(self):
        self._call("land")

    def push(self, p, force=False):
        now = time.time()
        if not force and now - self._last_push < 0.2:       # <= 5 Hz to the cloud
            return
        self._last_push = now
        q = p + self.origin
        try:
            self.twin.edit_position(x=float(q[0]), y=float(q[1]), z=float(q[2]))
        except Exception as e:
            print(f"[cyberwave] edit_position failed: {e}")

    def close(self):
        try:
            self.cw.disconnect()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Supervisor / evaluator
# ----------------------------------------------------------------------------
def run_mission(cfg: Cfg, bridge: CyberwaveBridge | None = None, speed: float = 1.0, verbose=True):
    rng = np.random.default_rng(cfg.seed)
    world = World.generate(cfg, rng)
    if bridge and bridge.build_scene_on_start:
        bridge.build_scene(world, cfg, bridge.show_truth)
    pad3 = np.array([cfg.pad_xy[0], cfg.pad_xy[1], 0.0])

    fc = FlightController(cfg, pad3)
    nav = Navigator(cfg, world.obstacles)
    planner = MissionPlanner(cfg)
    perc = Perception(cfg, rng)
    loc = Localization(cfg, rng, perc)
    station = GroundStation()
    comms = Communication(cfg, rng, station)

    cell = 0.5
    cov = np.zeros((int(cfg.arena_y / cell), int(cfg.arena_x / cell)), bool)

    state, t = "IDLE", 0.0
    log_state = [(0.0, "IDLE")]
    path = [pad3.copy()]
    cov_series = []
    wp: Waypoint | None = None
    wp_id = None
    wp_t0 = progress_t = dwell_t0 = None
    best_dist = 1e9
    camera_period, next_cam = 1.0 / cfg.camera_hz, 0.0
    report_t0 = None

    battery = 100.0
    path_len = 0.0
    collisions = recoveries = 0
    min_clear = float("inf")
    vel_err_sq, n_steps = 0.0, 0
    max_tilt, sat_steps = 0.0, 0
    first_crit_t = all_found_t = None
    matched: set[int] = set()

    fixed_wp = None          # waypoint held for the duration of a non-EXPLORE state
    report_flushed = False

    def set_state(s):
        nonlocal state, fixed_wp
        if s != state:
            state = s
            fixed_wp = None
            log_state.append((round(t, 2), s))
            if verbose:
                print(f"  t={t:6.1f}s  -> {s}")

    def time_to_home_ok():
        d = float(np.linalg.norm(fc.p[:2] - np.array(cfg.pad_xy)))
        return (cfg.mission_time - t) > d / cfg.vmax + 12.0

    set_state("TAKEOFF")
    if bridge:
        bridge.takeoff()

    while t < cfg.mission_time and state != "DONE":
        # ---------------- decide current waypoint ----------------
        if state == "TAKEOFF":
            fixed_wp = fixed_wp or Waypoint(pad3[0], pad3[1], cfg.cruise_alt, "takeoff", tol=0.3)
            cur = fixed_wp
        elif state in ("EXPLORE", "LOCALIZE"):
            if not planner.queue and not planner.next_pass():
                set_state("REPORT")
                report_t0 = t
            elif not time_to_home_ok() or battery < 15.0:
                set_state("REPORT")
                report_t0 = t
            if state in ("EXPLORE", "LOCALIZE"):
                cur = planner.queue[0]
                set_state("LOCALIZE" if cur.kind == "investigate" else "EXPLORE")
        if state == "REPORT":
            fixed_wp = fixed_wp or Waypoint(fc.p[0], fc.p[1], fc.p[2], "hold", tol=9.0)
            cur = fixed_wp
            if not report_flushed:
                report_flushed = True
                for tr in loc.tracks:
                    if tr.confirmed:
                        comms.enqueue(tr, t)               # final refined estimates
            if not comms.outbox or t - report_t0 > 8.0:
                set_state("RETURN")
        if state == "RETURN":
            fixed_wp = fixed_wp or Waypoint(pad3[0], pad3[1], cfg.cruise_alt, "return", tol=0.4)
            cur = fixed_wp
        elif state == "LAND":
            fixed_wp = fixed_wp or Waypoint(pad3[0], pad3[1], 0.0, "land", tol=0.3)
            cur = fixed_wp
        elif state == "DONE":
            cur = Waypoint(fc.p[0], fc.p[1], fc.p[2], "hold")

        # new waypoint bookkeeping
        if id(cur) != wp_id:
            wp, wp_id = cur, id(cur)
            wp_t0 = progress_t = t
            best_dist = 1e9
            dwell_t0 = None
        d_h = float(np.linalg.norm(wp.pos[:2] - fc.p[:2]))
        d_z = abs(wp.z - fc.p[2])

        # ---------------- navigation + control ----------------
        goal_dir = (wp.pos[:2] - fc.p[:2]) / max(d_h, 1e-6)
        v_avoid, clear = nav.avoid(fc.p[:2], goal_dir) if fc.p[2] > 0.5 else (np.zeros(2), 9.0)
        min_clear = min(min_clear, clear) if fc.p[2] > 0.5 else min_clear
        v_des, sat, tilt = fc.step(wp.pos, v_avoid, rng)
        if fc.p[2] > 0.5 and nav.resolve_collision(fc.p):
            collisions += 1
        prev = path[-1]
        path_len += float(np.linalg.norm(fc.p - prev))
        if n_steps % 4 == 0:
            path.append(fc.p.copy())
        vel_err_sq += float(np.sum((v_des - fc.v) ** 2))
        n_steps += 1
        max_tilt = max(max_tilt, tilt)
        sat_steps += int(sat)
        speed_now = float(np.linalg.norm(fc.v))
        if state != "IDLE":
            battery -= (cfg.battery_hover_pct_s + cfg.battery_speed_pct_s * speed_now) * cfg.dt

        # ---------------- waypoint completion / stuck recovery ----------------
        reached = d_h < wp.tol and d_z < 0.3 and wp.kind != "hold"
        if reached and wp.dwell > 0:
            dwell_t0 = t if dwell_t0 is None else dwell_t0
            reached = (t - dwell_t0) >= wp.dwell
        if d_h < best_dist - 0.1:
            best_dist, progress_t = d_h, t
        stuck = (t - progress_t) > cfg.stuck_timeout and d_h > wp.tol
        timed_out = (t - wp_t0) > wp.timeout
        if state in ("EXPLORE", "LOCALIZE") and planner.queue and (stuck or timed_out) and not reached:
            recoveries += 1
            planner.queue.popleft()                 # recovery: skip an unreachable waypoint
        elif reached:
            if state == "TAKEOFF":
                set_state("EXPLORE")
            elif state in ("EXPLORE", "LOCALIZE") and planner.queue:
                planner.queue.popleft()
            elif state == "RETURN":
                set_state("LAND")
            elif state == "LAND" and fc.p[2] < 0.08:
                set_state("DONE")
                if bridge:
                    bridge.land()
        if state == "LAND" and fc.p[2] < 0.08:
            set_state("DONE")

        # ---------------- perception / localization ----------------
        if t >= next_cam and fc.p[2] > 0.5:
            next_cam += camera_period
            half = perc.half_footprint(fc.p[2])
            x0 = max(int((fc.p[0] - half) / cell), 0)
            x1 = min(int((fc.p[0] + half) / cell) + 1, cov.shape[1])
            y0 = max(int((fc.p[1] - half) / cell), 0)
            y1 = min(int((fc.p[1] + half) / cell) + 1, cov.shape[0])
            cov[y0:y1, x0:x1] = True
            dets = perc.frame(fc.p, world.survivors)
            for tr in loc.update(dets, fc.p, t):
                # evaluator-only ground-truth matching (never feeds decisions)
                best, bd = None, 1.5
                for s in world.survivors:
                    d = float(np.linalg.norm(s.pos[:2] - tr.est[:2]))
                    if s.sid not in matched and d < bd:
                        best, bd = s, d
                if best is not None:
                    tr.truth_sid = best.sid
                    matched.add(best.sid)
                    if best.cls == "critical" and first_crit_t is None:
                        first_crit_t = t
                    if len(matched) == len(world.survivors) and all_found_t is None:
                        all_found_t = t
                comms.enqueue(tr, t)
                if bridge:
                    bridge.mark_detection(tr, cfg.seed)
                if verbose:
                    print(f"  t={t:6.1f}s  survivor confirmed: id={tr.tid} class={tr.cls} "
                          f"est=({tr.est[0]:.2f},{tr.est[1]:.2f})")
            # priority logic: detour to refine the position of likely-critical survivors
            for tr in loc.tracks:
                if (not tr.investigated and tr.n >= 2 and tr.cls == "critical"
                        and state in ("EXPLORE", "LOCALIZE")):
                    tr.investigated = True
                    planner.investigate(tr.est)
                    break
            loc.prune(t)
            if int(t / 10) != int((t - camera_period) / 10):
                cov_series.append((round(t, 1), round(100.0 * float(cov.mean()), 1)))

        comms.pump(t)

        # ---------------- Cyberwave mirror ----------------
        if bridge:
            bridge.push(fc.p)
            time.sleep(cfg.dt / max(speed, 1e-6))

        t += cfg.dt

    if bridge:
        bridge.push(fc.p, force=True)

    # ---------------- evaluation ----------------
    delivered = station.received
    confirmed = [tr for tr in loc.tracks if tr.confirmed]
    tp = [tr for tr in confirmed if tr.truth_sid is not None]
    fp = [tr for tr in confirmed if tr.truth_sid is None]
    errs, cls_ok = [], 0
    for tr in tp:
        s = world.survivors[tr.truth_sid]
        errs.append(float(np.linalg.norm(s.pos[:2] - tr.est[:2])))
        cls_ok += int(tr.cls == s.cls)
    deliv_tp = [tr for tr in tp if tr.tid in delivered]
    reported_err = []
    for tr in deliv_tp:
        s = world.survivors[tr.truth_sid]
        reported_err.append(float(np.linalg.norm(s.pos[:2] - np.array(delivered[tr.tid]["xyz"][:2]))))
    crit_found = {world.survivors[tr.truth_sid].cls for tr in tp}
    n_total = len(world.survivors)
    t_recv = sorted((delivered[tr.tid]["t_recv"], world.survivors[tr.truth_sid].cls) for tr in deliv_tp)
    crit_first = bool(t_recv) and t_recv[0][1] == "critical" if t_recv else False
    crit_times = [x[0] for x in t_recv if x[1] == "critical"]
    stab_times = [x[0] for x in t_recv if x[1] == "stable"]
    crit_before_stable_pct = None
    if crit_times and stab_times:
        crit_before_stable_pct = round(100.0 * float(np.mean(
            [ct <= min(stab_times) for ct in crit_times])), 1)

    reasons = []
    if len(tp) < n_total:
        reasons.append(f"only {len(tp)}/{n_total} survivors detected")
    if len(deliv_tp) < len(tp):
        reasons.append("some detected survivors never reached the ground station")
    if any(tr.tid not in delivered for tr in confirmed):
        reasons.append("some confirmed tracks (incl. false positives) were not delivered")
    want_classes = {s.cls for s in world.survivors}
    if not want_classes.issubset(crit_found):
        reasons.append("did not identify both critical and stable survivors")
    if collisions:
        reasons.append(f"{collisions} collision step(s)")
    if state != "DONE":
        reasons.append("mission time ended before landing")
    passed = not [r for r in reasons if "landing" not in r] and len(tp) == n_total

    report = {
        "seed": cfg.seed,
        "result": "PASS" if passed else "FAIL",
        "fail_reasons": [] if passed else reasons,
        "notes": reasons if passed else [],
        "mission_time_s": round(t, 1),
        "state_log": log_state,
        "survivors": {
            "total": n_total, "detected": len(tp), "false_positives": len(fp),
            "false_negatives": n_total - len(tp),
            "class_correct": cls_ok,
            "delivered_to_station": len(deliv_tp),
        },
        "localization_error_m": {
            "per_survivor": [round(e, 3) for e in errs],
            "mean": round(float(np.mean(errs)), 3) if errs else None,
            "reported_mean": round(float(np.mean(reported_err)), 3) if reported_err else None,
        },
        "timing_s": {
            "first_critical": round(first_crit_t, 1) if first_crit_t is not None else None,
            "all_survivors": round(all_found_t, 1) if all_found_t is not None else None,
            "critical_reported_before_stable_pct": crit_before_stable_pct,
        },
        "coverage_pct": {"final": round(100.0 * float(cov.mean()), 1), "series": cov_series},
        "safety": {"collisions": collisions, "min_clearance_m": round(min_clear, 3) if min_clear < 1e8 else None,
                   "stuck_recoveries": recoveries},
        "stability": {"rms_velocity_error_mps": round(math.sqrt(vel_err_sq / max(n_steps, 1)), 3),
                      "max_tilt_deg": round(max_tilt, 1),
                      "accel_saturation_pct": round(100.0 * sat_steps / max(n_steps, 1), 2)},
        "energy": {"battery_used_pct": round(100.0 - battery, 1), "path_length_m": round(path_len, 1)},
        "comms": {"sent": comms.sent, "acked": comms.acked, "failed": comms.failed,
                  "success_rate_pct": round(100.0 * comms.acked / max(comms.sent, 1), 1),
                  "mean_latency_s": round(float(np.mean(comms.latencies)), 3) if comms.latencies else None},
        "station_log": station.log,
    }
    return report, world, loc, np.array(path), cov


# ----------------------------------------------------------------------------
# Visualization (optional)
# ----------------------------------------------------------------------------
def save_map(cfg, world, loc, path, cov, report, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception:
        print("matplotlib not installed; skipping map")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cov, origin="lower", extent=[0, cfg.arena_x, 0, cfg.arena_y], cmap="Greens", alpha=0.25)
    for cx, cy, r in world.obstacles:
        ax.add_patch(Circle((cx, cy), r, color="#8b6f5a", alpha=0.85))
    ax.plot(path[:, 0], path[:, 1], lw=0.8, color="#1f77b4", label="drone path")
    for s in world.survivors:
        ax.scatter(*s.pos[:2], marker="o", s=90, facecolors="none",
                   edgecolors="red" if s.cls == "critical" else "green", linewidths=2)
    for tr in loc.tracks:
        if tr.confirmed:
            ax.scatter(*tr.est[:2], marker="x", s=60, color="red" if tr.cls == "critical" else "green")
    ax.scatter(*cfg.station_xy, marker="s", s=100, color="black", label="ground station")
    ax.set_xlim(0, cfg.arena_x)
    ax.set_ylim(0, cfg.arena_y)
    ax.set_aspect("equal")
    ax.set_title(f"Khoj-o-Drone seed={cfg.seed}  {report['result']}  "
                 f"({report['survivors']['detected']}/{report['survivors']['total']} found)\n"
                 "o = ground truth, x = estimate, red = critical, green = stable")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Khoj-o-Drone search & rescue simulation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs", type=int, default=1, help="number of seeded missions (seed, seed+1, ...)")
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--critical", type=int, default=2)
    ap.add_argument("--obstacles", type=int, default=10)
    ap.add_argument("--arena", type=float, nargs=2, default=[20.0, 20.0], metavar=("X", "Y"))
    ap.add_argument("--mission-time", type=float, default=480.0)
    ap.add_argument("--out", default="khoj_output")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    # Cyberwave
    ap.add_argument("--cyberwave", action="store_true", help="mirror the drone into a Cyberwave twin")
    ap.add_argument("--asset", default=None, help="Cyberwave drone asset id, e.g. vendor/model")
    ap.add_argument("--twin-id", default=None, help="UUID/slug of a drone twin already in your environment")
    ap.add_argument("--environment-id", default=None, help="Cyberwave environment UUID/slug (for scene markers)")
    ap.add_argument("--detail", choices=["low", "medium", "high"], default="medium",
                    help="amount of rubble/debris to place in Cyberwave (more = slower startup)")
    ap.add_argument("--clear-scene", action="store_true", help="delete old khoj-/rubble- objects and exit")
    ap.add_argument("--no-scene", action="store_true", help="do not create rubble/survivor markers in Cyberwave")
    ap.add_argument("--hide-truth", action="store_true", help="do not mark true survivor positions")
    ap.add_argument("--list-assets", nargs="?", const="drone", default=None, metavar="QUERY",
                    help="print catalog search results (id/slug/uuid) and exit")
    ap.add_argument("--mode", default="playground", choices=["playground", "simulation", "live"])
    ap.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="offset added to positions when pushed to Cyberwave")
    ap.add_argument("--speed", type=float, default=4.0, help="sim speed-up when mirroring (1 = real time)")
    ap.add_argument("--flight-helpers", action="store_true", help="also call twin.takeoff()/land()")
    a = ap.parse_args()

    if a.list_assets is not None:
        CyberwaveBridge.list_assets(a.list_assets)
        return
    os.makedirs(a.out, exist_ok=True)
    bridge = None
    if a.cyberwave:
        if a.runs > 1:
            sys.exit("--cyberwave supports a single run; drop --runs.")
        bridge = CyberwaveBridge(a.asset, a.mode, tuple(a.origin), a.flight_helpers, a.twin_id,
                                 a.environment_id)
        bridge.build_scene_on_start = not a.no_scene
        bridge.show_truth = not a.hide_truth
        bridge.detail = a.detail
        if a.clear_scene:
            bridge.clear_scene()
            bridge.close()
            return

    reports = []
    try:
        for i in range(a.runs):
            cfg = Cfg(seed=a.seed + i, n_survivors=a.survivors, n_critical=a.critical,
                      n_obstacles=a.obstacles, arena_x=a.arena[0], arena_y=a.arena[1],
                      mission_time=a.mission_time)
            print(f"\n=== Mission {i + 1}/{a.runs}  seed={cfg.seed} ===")
            rep, world, loc, path, cov = run_mission(cfg, bridge, a.speed, verbose=not a.quiet)
            reports.append(rep)
            with open(os.path.join(a.out, f"mission_report_seed{cfg.seed}.json"), "w") as f:
                json.dump(rep, f, indent=2)
            if not a.no_plot:
                save_map(cfg, world, loc, path, cov, rep, os.path.join(a.out, f"mission_map_seed{cfg.seed}.png"))
            s = rep["survivors"]
            print(f"  {rep['result']}: found {s['detected']}/{s['total']}, FP={s['false_positives']}, "
                  f"delivered={s['delivered_to_station']}, loc err={rep['localization_error_m']['mean']} m, "
                  f"coverage={rep['coverage_pct']['final']}%, collisions={rep['safety']['collisions']}, "
                  f"t={rep['mission_time_s']}s")
            if rep["fail_reasons"]:
                print("  reasons:", "; ".join(rep["fail_reasons"]))
    finally:
        if bridge:
            bridge.close()

    if len(reports) > 1:
        def mean(vals):
            vals = [v for v in vals if v is not None]
            return round(float(np.mean(vals)), 3) if vals else None
        agg = {
            "runs": len(reports),
            "pass_rate_pct": round(100.0 * sum(r["result"] == "PASS" for r in reports) / len(reports), 1),
            "mean_detected_frac": mean([r["survivors"]["detected"] / r["survivors"]["total"] for r in reports]),
            "mean_false_positives": mean([r["survivors"]["false_positives"] for r in reports]),
            "mean_loc_error_m": mean([r["localization_error_m"]["mean"] for r in reports]),
            "mean_coverage_pct": mean([r["coverage_pct"]["final"] for r in reports]),
            "total_collisions": sum(r["safety"]["collisions"] for r in reports),
            "mean_time_first_critical_s": mean([r["timing_s"]["first_critical"] for r in reports]),
            "mean_time_all_found_s": mean([r["timing_s"]["all_survivors"] for r in reports]),
            "mean_comms_success_pct": mean([r["comms"]["success_rate_pct"] for r in reports]),
        }
        with open(os.path.join(a.out, "aggregate_report.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print("\n=== AGGREGATE ===")
        print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()