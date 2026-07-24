#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Denavit-Hartenberg kinematics for the UR10/UR10e.

The public interface uses millimetres for length and degrees for joint angles.
The module also provides the fixed tool transform used for the UR10e with an
OnRobot RG2-FT gripper.
"""

import numpy as np
from dataclasses import dataclass
from typing import Iterable


def _rot_x(alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.eye(4, dtype=float)
    T[1, 1], T[1, 2], T[2, 1], T[2, 2] = ca, -sa, sa, ca
    return T


def _rot_z(theta):
    ct, st = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=float)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = ct, -st, st, ct
    return T


def _tx(a):
    T = np.eye(4, dtype=float)
    T[0, 3] = a
    return T


def _tz(d):
    T = np.eye(4, dtype=float)
    T[2, 3] = d
    return T


def dh_transform(a, alpha, d, theta):
    """Return one standard-DH homogeneous transform."""
    return _rot_z(theta) @ _tz(d) @ _tx(a) @ _rot_x(alpha)


def fk_chain(a_mm, alpha_deg, d_mm, theta_deg):
    """Return the base-to-end homogeneous transform for a DH chain."""
    a = np.asarray(a_mm, dtype=float)
    alpha = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    d = np.asarray(d_mm, dtype=float)
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    T = np.eye(4, dtype=float)
    for i in range(len(a)):
        T = T @ dh_transform(a[i], alpha[i], d[i], th[i])
    return T


def rpy_zyx_from_R(R):
    """Convert a rotation matrix to ZYX roll-pitch-yaw angles in degrees."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-9:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.rad2deg([roll, pitch, yaw]).astype(float)


@dataclass
class SerialDH:
    a_mm: np.ndarray
    alpha_deg: np.ndarray
    d_mm: np.ndarray
    theta_offset_deg: np.ndarray

    @classmethod
    def from_degrees(cls, a_mm: Iterable[float], alpha_deg: Iterable[float],
                     d_mm: Iterable[float], theta_offset_deg: Iterable[float] = None):
        a = np.asarray(a_mm, dtype=float)
        alpha = np.asarray(alpha_deg, dtype=float)
        d = np.asarray(d_mm, dtype=float)
        th_off = np.zeros_like(a) if theta_offset_deg is None else np.asarray(theta_offset_deg, dtype=float)
        return cls(a, alpha, d, th_off)

    def fk(self, q_deg: Iterable[float]) -> np.ndarray:
        """Return the base-to-tool0 transform for joint angles in degrees."""
        q = np.asarray(q_deg, dtype=float) + self.theta_offset_deg
        return fk_chain(self.a_mm, self.alpha_deg, self.d_mm, q)

    def fk_xyz_rpy(self, q_deg: Iterable[float]):
        """Return tool0 position and ZYX orientation."""
        T = self.fk(q_deg)
        xyz = T[:3, 3].copy()
        rpy = rpy_zyx_from_R(T[:3, :3])
        return xyz, rpy

    # Numerical translation Jacobian
    def jacobian_xyz(self, q_deg: Iterable[float], step_deg: float = 1e-3) -> np.ndarray:
        """
        Return the translational Jacobian using central differences.

        The output has shape (3, n) and units of mm/deg.
        """
        q = np.asarray(q_deg, dtype=float)
        n = q.size
        J = np.zeros((3, n), dtype=float)

        x0, _ = self.fk_xyz_rpy(q)

        for j in range(n):
            dq = np.zeros_like(q)
            dq[j] = step_deg
            x_plus, _  = self.fk_xyz_rpy(q + dq)
            x_minus, _ = self.fk_xyz_rpy(q - dq)
            J[:, j] = (x_plus - x_minus) / (2.0 * step_deg)

        return J

    # Numerical TCP translation Jacobian
    def jacobian_xyz_with_tool(
        self,
        q_deg: Iterable[float],
        T_tool_tcp: np.ndarray,
        step_deg: float = 1e-3
    ) -> np.ndarray:
        """
        Return the TCP translational Jacobian using central differences.

        ``T_tool_tcp`` is the fixed transform from tool0 to the TCP. The
        output has shape (3, n) and units of mm/deg.
        """
        q = np.asarray(q_deg, dtype=float)
        n = q.size
        J = np.zeros((3, n), dtype=float)

        x0, _ = self.fk_xyz_rpy_with_tool(q, T_tool_tcp)

        for j in range(n):
            dq = np.zeros_like(q)
            dq[j] = step_deg
            x_plus, _  = self.fk_xyz_rpy_with_tool(q + dq, T_tool_tcp)
            x_minus, _ = self.fk_xyz_rpy_with_tool(q - dq, T_tool_tcp)
            J[:, j] = (x_plus - x_minus) / (2.0 * step_deg)

        return J

    # Tool-aware forward kinematics
    def fk_with_tool(self, q_deg: Iterable[float], T_tool_tcp: np.ndarray) -> np.ndarray:
        """Return the base-to-TCP transform for a fixed rigid tool."""
        T_base_tool = self.fk(q_deg)
        return T_base_tool @ T_tool_tcp

    def fk_xyz_rpy_with_tool(self, q_deg: Iterable[float], T_tool_tcp: np.ndarray):
        """Return TCP position and ZYX orientation."""
        T = self.fk_with_tool(q_deg, T_tool_tcp)
        xyz = T[:3, 3].copy()
        rpy = rpy_zyx_from_R(T[:3, :3])
        return xyz, rpy


# UR10 and UR10e models

UR10_DH = SerialDH.from_degrees(
    a_mm=[0, -612, -572.3, 0, 0, 0],
    alpha_deg=[90, 0, 0, 90, -90, 0],
    d_mm=[127.3, 0, 0, 163.941, 115.7, 92.2],
    theta_offset_deg=[0, 0, 0, 0, 0, 0],
)

UR10e_DH = SerialDH.from_degrees(
    a_mm=[0, -612.7, -571.55, 0, 0, 0],
    alpha_deg=[90, 0, 0, 90, -90, 0],
    d_mm=[180.7, 0, 0, 174.15, 119.85, 116.55],
    theta_offset_deg=[0, 0, 0, 0, 0, 0],  # Add encoder offsets here if required.
)

# UR10e with RG2-FT TCP

# TCP configured on the UR teach pendant: [0, 0, 233.6] mm, zero rotation.
RG2FT_TCP_Z_MM = 233.6

T_UR10e_TOOL_TO_RG2FT_TCP = np.eye(4, dtype=float)
T_UR10e_TOOL_TO_RG2FT_TCP[2, 3] = RG2FT_TCP_Z_MM


def ur10e_rg2ft_fk(q_deg: Iterable[float]) -> np.ndarray:
    """Return the base-to-RG2-FT-TCP transform."""
    return UR10e_DH.fk_with_tool(q_deg, T_UR10e_TOOL_TO_RG2FT_TCP)


def ur10e_rg2ft_fk_xyz_rpy(q_deg: Iterable[float]):
    """Return RG2-FT TCP position and ZYX orientation."""
    return UR10e_DH.fk_xyz_rpy_with_tool(q_deg, T_UR10e_TOOL_TO_RG2FT_TCP)
