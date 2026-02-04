"""
Kinematic utilities for MadGraphMLProducer.

Provides four-vector operations and common kinematic calculations.
"""

import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass


@dataclass
class FourVector:
    """
    Relativistic four-vector representation.

    Uses (E, px, py, pz) convention.
    """

    e: float
    px: float
    py: float
    pz: float

    @classmethod
    def from_pt_eta_phi_m(cls, pt: float, eta: float, phi: float, m: float) -> "FourVector":
        """Create from (pT, eta, phi, mass) coordinates"""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(pt**2 * np.cosh(eta) ** 2 + m**2)
        return cls(e=e, px=px, py=py, pz=pz)

    @classmethod
    def from_pt_eta_phi_e(cls, pt: float, eta: float, phi: float, e: float) -> "FourVector":
        """Create from (pT, eta, phi, E) coordinates"""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        return cls(e=e, px=px, py=py, pz=pz)

    @property
    def pt(self) -> float:
        """Transverse momentum"""
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def p(self) -> float:
        """Total momentum magnitude"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def eta(self) -> float:
        """Pseudorapidity"""
        p = self.p
        if p == 0:
            return 0.0
        return np.arctanh(np.clip(self.pz / p, -0.9999999, 0.9999999))

    @property
    def phi(self) -> float:
        """Azimuthal angle in [-pi, pi]"""
        return np.arctan2(self.py, self.px)

    @property
    def mass(self) -> float:
        """Invariant mass"""
        m2 = self.e**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(max(0, m2))

    @property
    def rapidity(self) -> float:
        """Rapidity"""
        if self.e == abs(self.pz):
            return np.sign(self.pz) * float("inf")
        return 0.5 * np.log((self.e + self.pz) / (self.e - self.pz))

    def __add__(self, other: "FourVector") -> "FourVector":
        """Add two four-vectors"""
        return FourVector(
            e=self.e + other.e,
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
        )

    def __sub__(self, other: "FourVector") -> "FourVector":
        """Subtract two four-vectors"""
        return FourVector(
            e=self.e - other.e,
            px=self.px - other.px,
            py=self.py - other.py,
            pz=self.pz - other.pz,
        )

    def __mul__(self, scalar: float) -> "FourVector":
        """Multiply by a scalar"""
        return FourVector(
            e=self.e * scalar,
            px=self.px * scalar,
            py=self.py * scalar,
            pz=self.pz * scalar,
        )

    def __rmul__(self, scalar: float) -> "FourVector":
        """Right multiply by a scalar"""
        return self.__mul__(scalar)

    def dot(self, other: "FourVector") -> float:
        """Minkowski dot product (metric: +---)"""
        return self.e * other.e - self.px * other.px - self.py * other.py - self.pz * other.pz

    def delta_r(self, other: "FourVector") -> float:
        """Angular distance in eta-phi space"""
        return delta_r(self.eta, self.phi, other.eta, other.phi)

    def boost(self, beta_x: float, beta_y: float, beta_z: float) -> "FourVector":
        """Apply Lorentz boost"""
        beta2 = beta_x**2 + beta_y**2 + beta_z**2
        if beta2 >= 1:
            raise ValueError("Boost velocity must be less than c")
        gamma = 1 / np.sqrt(1 - beta2)

        # Boost components
        bp = beta_x * self.px + beta_y * self.py + beta_z * self.pz
        gamma2 = (gamma - 1) / beta2 if beta2 > 0 else 0

        e_new = gamma * (self.e - bp)
        px_new = self.px + beta_x * (gamma2 * bp - gamma * self.e)
        py_new = self.py + beta_y * (gamma2 * bp - gamma * self.e)
        pz_new = self.pz + beta_z * (gamma2 * bp - gamma * self.e)

        return FourVector(e=e_new, px=px_new, py=py_new, pz=pz_new)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [E, px, py, pz]"""
        return np.array([self.e, self.px, self.py, self.pz])

    def to_pt_eta_phi_m(self) -> Tuple[float, float, float, float]:
        """Convert to (pT, eta, phi, mass) tuple"""
        return (self.pt, self.eta, self.phi, self.mass)


def delta_phi(phi1: float, phi2: float) -> float:
    """
    Calculate delta phi in [-pi, pi] range.

    Args:
        phi1: First azimuthal angle
        phi2: Second azimuthal angle

    Returns:
        Difference phi1 - phi2, wrapped to [-pi, pi]
    """
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi


def delta_r(
    eta1: float, phi1: float, eta2: float, phi2: float
) -> float:
    """
    Calculate angular distance in eta-phi space.

    Args:
        eta1, phi1: Pseudorapidity and azimuth of first object
        eta2, phi2: Pseudorapidity and azimuth of second object

    Returns:
        Delta R = sqrt(delta_eta^2 + delta_phi^2)
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def invariant_mass(particles: list[FourVector]) -> float:
    """
    Calculate invariant mass of a system of particles.

    Args:
        particles: List of four-vectors

    Returns:
        Invariant mass of the system
    """
    if not particles:
        return 0.0

    total = particles[0]
    for p in particles[1:]:
        total = total + p
    return total.mass


def transverse_mass(
    pt1: float, phi1: float, pt2: float, phi2: float
) -> float:
    """
    Calculate transverse mass of two-body system.

    Useful for W mass reconstruction with MET.

    Args:
        pt1, phi1: pT and phi of first object
        pt2, phi2: pT and phi of second object (e.g., MET)

    Returns:
        Transverse mass
    """
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(2 * pt1 * pt2 * (1 - np.cos(dphi)))


def ht(pts: Union[list[float], np.ndarray]) -> float:
    """
    Calculate scalar sum of transverse momenta (HT).

    Args:
        pts: List or array of pT values

    Returns:
        HT = sum of pT values
    """
    return float(np.sum(pts))
