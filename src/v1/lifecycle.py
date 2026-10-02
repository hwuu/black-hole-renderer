from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import numpy as np
from src.core.constants import FILAMENT_BIRTH_FADE_DUR, FILAMENT_DEATH_THRESHOLD, FILAMENT_MAX_LIFETIME, FILAMENT_SHEAR_ALPHA, FILAMENT_TAU_COOL
from src.v1.renderer import TaichiRenderer
from src.v1.texture import _spawn_single_filament, _spawn_single_hotspot, _spawn_single_rt_spike


@dataclass
class EntityInstance:
    """Single entity instance in the lifecycle system.

    Stores entity contributions and lifecycle parameters. Non-filament entities
    use pre-computed phi_density/phi_temp arrays; filaments use blob parameters
    and compute their profile each frame from physics.

    Args:
        row_indices: affected radial row indices, shape (n_affected,).
        phi_density: density contribution, shape (n_affected, n_phi).
            Pre-computed for hotspot/RT spike; empty for filaments (blob mode).
        phi_temp: temperature contribution, shape (n_affected, n_phi).
            Pre-computed for hotspot/RT spike; empty for filaments (blob mode).
        omega: Keplerian angular velocity at entity center (rad/s).
        birth_time: wall-clock time when entity was spawned (seconds).
        lifetime: total alive duration excluding fade periods (seconds).
        fade_in: fade-in duration (seconds). Non-filament only.
        fade_out: fade-out duration (seconds). Non-filament only.
        entity_type: 'filament', 'hotspot', or 'rt_spike'.
        source_phi: filament source azimuthal position (rad).
        alpha_shear: precomputed turbulent shear rate (rad/s).
        tau_cool: radiative cooling timescale (seconds).
        blob_base_r: filament radial center in r_norm units.
        blob_sigma_r: filament radial Gaussian width (r_norm units).
        blob_sigma_phi0: filament initial azimuthal Gaussian width (rad).
        blob_peak_density: filament initial density peak value.
        blob_peak_temp: filament initial temperature peak value.

    Physical Meaning:
        Represents a localized structure in the accretion disk. Filaments are
        born as circular 2D Gaussian blobs that get naturally sheared into arcs
        by differential Keplerian rotation. Non-filament entities use fixed-timer
        fade with pre-computed profiles.
    """
    row_indices: np.ndarray
    phi_density: np.ndarray
    phi_temp: np.ndarray
    omega: float
    birth_time: float
    lifetime: float
    fade_in: float
    fade_out: float
    fade_noise: np.ndarray
    entity_type: str = 'generic'
    source_phi: float = 0.0
    total_extent: float = 0.0
    alpha_shear: float = 0.0
    tau_cool: float = FILAMENT_TAU_COOL
    blob_base_r: float = 0.0
    blob_sigma_r: float = 0.0
    blob_sigma_phi0: float = 0.0
    blob_peak_density: float = 0.0
    blob_peak_temp: float = 0.0

    @property
    def total_duration(self) -> float:
        """Total time from birth to fully faded out."""
        return self.fade_in + self.lifetime + self.fade_out

    def density_factor(self, age: float) -> float:
        """Compute density decay factor for filament blob model.

        Combines turbulent shear dilution (blob stretching, mass conservation)
        with radiative cooling (exponential energy loss).

        Args:
            age: time since birth in seconds, must be >= 0.

        Returns:
            float in (0, 1]: 1.0 at birth, monotonically decreasing.

        Formula:
            sigma_phi_t = blob_sigma_phi0 + alpha_shear * age
            factor = (blob_sigma_phi0 / sigma_phi_t) * exp(-age / tau_cool)
        """
        s0 = max(self.blob_sigma_phi0, 1e-6)
        sigma_phi_t = s0 + self.alpha_shear * age
        shear_term = s0 / sigma_phi_t
        cool_term = math.exp(-age / self.tau_cool) if self.tau_cool > 0 else 1.0
        return shear_term * cool_term

    def is_dead(self, now: float) -> bool:
        """Whether the entity should be recycled.

        Filaments die when density_factor drops below FILAMENT_DEATH_THRESHOLD
        or exceeds FILAMENT_MAX_LIFETIME. Other entity types use fixed-timer
        lifecycle (fade_in + lifetime + fade_out).
        """
        age = now - self.birth_time
        if self.entity_type == 'filament':
            if age >= FILAMENT_MAX_LIFETIME:
                return True
            return age >= 0 and self.density_factor(age) < FILAMENT_DEATH_THRESHOLD
        return age >= self.total_duration

    def fade_factor(self, now: float) -> float:
        """Compute current fade alpha based on age (non-filament entities only).

        Returns:
            alpha in [0, 1]: 0 during pre-birth or post-death,
            linear ramp during fade-in/out, 1.0 during alive phase.

        Formula:
            age = now - birth_time
            if age < fade_in:       alpha = age / fade_in
            elif age < fade_in + lifetime: alpha = 1.0
            elif age < total_duration:     alpha = 1.0 - (age - fade_in - lifetime) / fade_out
            else:                          alpha = 0.0
        """
        age = now - self.birth_time
        if age < 0:
            return 0.0
        if age < self.fade_in:
            return age / self.fade_in if self.fade_in > 0 else 1.0
        age_after_fade_in = age - self.fade_in
        if age_after_fade_in < self.lifetime:
            return 1.0
        age_in_fade_out = age_after_fade_in - self.lifetime
        if age_in_fade_out < self.fade_out:
            return 1.0 - age_in_fade_out / self.fade_out if self.fade_out > 0 else 0.0
        return 0.0


class EntityFactory:
    """Manages lifecycle of entity instances — spawning, aging, and recycling.

    Maintains a pool of alive entities, spawning new ones at a controlled rate
    to maintain a target count. Dead entities are automatically removed.

    Args:
        spawn_fn: callable(rng, n_r, n_phi, r_norm_all, omega_all) -> (row_indices, phi_density, phi_temp, omega).
            The single-instance generation function (e.g. _spawn_single_filament).
        target_count: desired number of alive entities at steady state.
        lifetime_range: (min_seconds, max_seconds) for entity lifetime.
        fade_in: fade-in duration in seconds.
        fade_out: fade-out duration in seconds.
        n_r: radial resolution.
        n_phi: azimuthal resolution.
        r_norm_all: normalized radial positions, shape (n_r,).
        omega_all: Keplerian angular velocity per row, shape (n_r,).
        seed: random seed for reproducibility.

    Physical Meaning:
        Models the continuous birth and death of transient structures in the
        accretion disk. The target_count and lifetime_range determine the
        visual density and turnover rate of structures.
    """

    def __init__(self, spawn_fn, target_count: int,
                 lifetime_range: Tuple[float, float],
                 fade_in: float, fade_out: float,
                 n_r: int, n_phi: int,
                 r_norm_all: np.ndarray, omega_all: np.ndarray,
                 seed: int = 0, entity_type: str = 'generic'):
        self.spawn_fn = spawn_fn
        self.target_count = target_count
        self.lifetime_range = lifetime_range
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.n_r = n_r
        self.n_phi = n_phi
        self.r_norm_all = r_norm_all
        self.omega_all = omega_all
        self.rng = np.random.default_rng(seed)
        self.entities: List[EntityInstance] = []
        self._spawn_debt = 0.0
        self.entity_type = entity_type

    def _spawn_one(self, now: float) -> EntityInstance:
        """Spawn a single entity at the current time.

        For filaments, spawn_fn returns 6 values (row_indices, phi_density,
        phi_temp, omega, source_phi, total_extent). For other types, it
        returns 4 values (row_indices, phi_density, phi_temp, omega).
        """
        result = self.spawn_fn(
            self.rng, self.n_r, self.n_phi, self.r_norm_all, self.omega_all)
        lifetime = float(self.rng.uniform(*self.lifetime_range))

        if self.entity_type == 'filament':
            (row_indices, phi_density, phi_temp, omega, source_phi,
             total_extent, sigma_r, sigma_phi0, peak_density,
             peak_temp, base_r) = result
            return EntityInstance(
                row_indices=row_indices,
                phi_density=phi_density,
                phi_temp=phi_temp,
                omega=omega,
                birth_time=now,
                lifetime=lifetime,
                fade_in=self.fade_in,
                fade_out=self.fade_out,
                fade_noise=self._make_fade_noise(),
                entity_type='filament',
                source_phi=source_phi,
                total_extent=total_extent,
                alpha_shear=FILAMENT_SHEAR_ALPHA * omega,
                tau_cool=FILAMENT_TAU_COOL,
                blob_base_r=base_r,
                blob_sigma_r=sigma_r,
                blob_sigma_phi0=sigma_phi0,
                blob_peak_density=peak_density,
                blob_peak_temp=peak_temp,
            )
        else:
            row_indices, phi_density, phi_temp, omega = result
            return EntityInstance(
                row_indices=row_indices,
                phi_density=phi_density,
                phi_temp=phi_temp,
                omega=omega,
                birth_time=now,
                lifetime=lifetime,
                fade_in=self.fade_in,
                fade_out=self.fade_out,
                fade_noise=self._make_fade_noise(),
                entity_type=self.entity_type,
            )

    def _make_fade_noise(self) -> np.ndarray:
        """Generate smooth 1D dissolve noise along phi, range [0, 1].

        Uses 2-3 sinusoidal components for low-frequency spatial variation,
        ensuring the dissolve front is smooth (cloud-like), not pixelated.
        """
        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        freq1 = int(self.rng.integers(3, 8))
        freq2 = int(self.rng.integers(8, 16))
        p1 = float(self.rng.uniform(0, 2 * np.pi))
        p2 = float(self.rng.uniform(0, 2 * np.pi))
        noise = (0.6 * np.sin(phi * freq1 + p1)
                 + 0.4 * np.sin(phi * freq2 + p2))
        noise = np.clip(noise * 0.5 + 0.5, 0, 1)
        return noise.astype(np.float32)

    def seed_initial(self, now: float) -> None:
        """Pre-populate with target_count entities at staggered ages.

        Distributes entities uniformly across their lifecycle so that the
        visual result is immediately at steady state, avoiding a "cold start"
        where all entities fade in simultaneously.

        Args:
            now: current wall-clock time in seconds
        """
        for i in range(self.target_count):
            entity = self._spawn_one(now)
            if entity.entity_type == 'filament':
                death_age = self._filament_death_age(entity)
                min_age = FILAMENT_BIRTH_FADE_DUR
                age_range = max(death_age - min_age, 1.0)
                stagger = min_age + age_range * (i / max(self.target_count, 1))
            else:
                max_age = entity.fade_in + entity.lifetime
                stagger = max_age * (i / max(self.target_count, 1))
            entity.birth_time = now - stagger
            self.entities.append(entity)

    @staticmethod
    def _filament_death_age(entity) -> float:
        """Find the age at which a filament's density_factor drops below threshold."""
        for t in range(1, int(FILAMENT_MAX_LIFETIME) + 1):
            if entity.density_factor(float(t)) < FILAMENT_DEATH_THRESHOLD:
                return float(t)
        return FILAMENT_MAX_LIFETIME

    def tick(self, now: float, dt: float) -> None:
        """Advance the factory by one frame: remove dead, spawn replacements.

        Args:
            now: current wall-clock time in seconds
            dt: time elapsed since last frame (seconds)
        """
        self.entities = [e for e in self.entities if not e.is_dead(now)]

        deficit = self.target_count - len(self.entities)
        if deficit <= 0:
            return

        avg_lifetime = sum(self.lifetime_range) / 2.0
        spawn_rate = self.target_count / avg_lifetime
        self._spawn_debt += spawn_rate * dt
        n_spawn = min(int(self._spawn_debt), deficit)
        self._spawn_debt -= n_spawn

        for _ in range(n_spawn):
            self.entities.append(self._spawn_one(now))

    @property
    def alive_entities(self) -> List['EntityInstance']:
        """Return list of currently alive (not fully dead) entities."""
        return self.entities


def _init_lifecycle_system(renderer: TaichiRenderer, n_r: int, n_phi: int,
                          seed: int = 42) -> dict:
    """Initialize the entity lifecycle system for disk texture generation.

    Sets up background layer (GPU noise) and entity factories (filaments,
    hotspots, RT spikes), pre-populates entities at staggered ages, generates
    the first frame, and computes initial normalization stats.

    Args:
        renderer: TaichiRenderer instance (must have r_disk_inner/r_disk_outer set)
        n_r: radial resolution of disk texture
        n_phi: azimuthal resolution of disk texture
        seed: random seed for reproducibility

    Returns:
        dict with keys 'filament', 'hotspot', 'rt_spike', each an EntityFactory
    """
    renderer.init_background_layer(n_r=n_r, n_phi=n_phi, seed=seed)

    r_norm_all = np.linspace(0, 1, n_r)
    r_vals = renderer.r_disk_inner + (renderer.r_disk_outer - renderer.r_disk_inner) * r_norm_all
    omega_all = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)

    factories = {
        'filament': EntityFactory(
            _spawn_single_filament, target_count=200,
            lifetime_range=(15.0, 60.0), fade_in=0.0, fade_out=0.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 100,
            entity_type='filament'),
        'hotspot': EntityFactory(
            _spawn_single_hotspot, target_count=30,
            lifetime_range=(15.0, 30.0), fade_in=4.0, fade_out=4.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 200,
            entity_type='hotspot'),
        'rt_spike': EntityFactory(
            _spawn_single_rt_spike, target_count=15,
            lifetime_range=(15.0, 30.0), fade_in=3.0, fade_out=3.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 300,
            entity_type='rt_spike'),
    }
    for f in factories.values():
        f.seed_initial(now=0.0)

    renderer.generate_background(t=0.0)
    renderer.accumulate_entity_layer(factories, now=0.0)
    renderer.recompute_interactive_stats()
    renderer.compose_interactive_texture()

    return factories


def _advance_lifecycle_frame(renderer: TaichiRenderer, factories: dict,
                             t: float, dt: float,
                             recompute_stats: bool = False,
                             solo_idx: int = -1) -> None:
    """Advance the lifecycle system by one frame and compose disk texture.

    Args:
        renderer: TaichiRenderer instance with lifecycle system initialized
        factories: dict of EntityFactory instances
        t: current simulation time in seconds
        dt: time step since last frame (seconds)
        recompute_stats: whether to recompute normalization stats this frame
        solo_idx: -1 for all components, >= 0 to solo a single component
    """
    for f in factories.values():
        f.tick(now=t, dt=dt)
    renderer.generate_background(t=t)
    renderer.accumulate_entity_layer(factories, now=t)
    if recompute_stats:
        renderer.recompute_interactive_stats()
    renderer.compose_interactive_texture(solo_idx=solo_idx)
