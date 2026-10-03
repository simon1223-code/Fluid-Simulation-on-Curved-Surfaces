#!/usr/bin/env python3
"""Two-dimensional incompressible Euler prototype with face Clebsch fields.

Both ``(mu_1, lambda_1)`` and ``(mu_2, lambda_2)`` are dual 0-cochains stored
on triangle faces.  Their dual-edge differences reconstruct a raw normal edge
flux, which is projected to be divergence-free with impermeable boundaries.
The centroid-distance conductance is a local two-point (TPFA) approximation;
on a nonorthogonal barycentric dual it is not generally affine-exact.

A resettable local chart generates velocity while a never-reset material copy
and a passive vertex dye are advected by the same SSPRK3 stages.  Optional graph
viscosity is deliberately dissipative.  Centered face transport preserves its
scalar mass and quadratic invariant for fixed divergence-free flux, but exact
kinetic-energy conservation is not claimed.  Use ``--run-tests`` for diagnostics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr, splu
from scipy.spatial import Delaunay


DOMAIN_LENGTH = 2.0 * np.pi
TAYLOR_GREEN = "taylor-green"
PERTURBED = "perturbed"
INITIAL_CONDITIONS = (TAYLOR_GREEN, PERTURBED)
FIELD_NAMES = ("mu_1", "mu_2", "lambda_1", "lambda_2", "dye")
MU1, MU2, LAMBDA1, LAMBDA2 = range(4)
CLEBSCH_FIELD_COUNT = 4
DYE_INDEX = 4


@dataclass(frozen=True)
class Settings:
    resolution: int = 10
    final_time: float = 10.0
    frames: int = 401
    cfl: float = 0.25
    maximum_time_step: float = 0.02
    rechart_cutoff: float = 1.5
    rechart_every_frames: int = 5
    determinant_floor: float = 0.05
    initial_condition: str = PERTURBED
    perturbation_amplitude: float = 0.25
    stabilization: str = "centered"
    graph_viscosity: float = 0.0
    jitter: float = 0.12
    random_seed: int = 7
    arrow_stride: int = 3
    playback_fps: float = 20.0


def make_delaunay_mesh(settings: Settings):
    """Create a deterministic, mildly irregular Delaunay mesh of the square.

    Boundary samples are not jittered.  Because the domain is convex, their
    convex-hull edges are retained by scipy.spatial.Delaunay.  For nonconvex
    domains this should be replaced by a constrained Delaunay mesher.
    """
    n = settings.resolution
    if n < 4:
        raise ValueError("resolution must be at least 4")
    grid = np.linspace(0.0, DOMAIN_LENGTH, n + 1)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    rng = np.random.default_rng(settings.random_seed)
    h = DOMAIN_LENGTH / n
    interior = (
        (xy[:, 0] > 0.0)
        & (xy[:, 0] < DOMAIN_LENGTH)
        & (xy[:, 1] > 0.0)
        & (xy[:, 1] < DOMAIN_LENGTH)
    )
    xy[interior] += settings.jitter * h * rng.uniform(-1.0, 1.0, (interior.sum(), 2))
    triangulation = Delaunay(xy)
    faces = triangulation.simplices.copy()

    first_edge = xy[faces[:, 1]] - xy[faces[:, 0]]
    second_edge = xy[faces[:, 2]] - xy[faces[:, 0]]
    signed_double_area = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    clockwise = signed_double_area < 0.0
    faces[clockwise, 1], faces[clockwise, 2] = (
        faces[clockwise, 2].copy(),
        faces[clockwise, 1].copy(),
    )
    vertices = np.column_stack((xy, np.zeros(len(xy))))
    return vertices, faces.astype(np.int64), triangulation


def triangle_quality(points):
    lengths = np.stack(
        (
            np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
        ),
        axis=1,
    )
    first_edge = points[:, 1] - points[:, 0]
    second_edge = points[:, 2] - points[:, 0]
    double_areas = np.abs(first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0])
    areas = 0.5 * double_areas
    altitudes = double_areas[:, None] / np.maximum(lengths, 1e-15)
    minimum_altitudes = np.min(altitudes, axis=1)
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    angles = np.column_stack(
        (
            np.arccos(np.clip((a * a + c * c - b * b) / np.maximum(2.0 * a * c, 1e-15), -1.0, 1.0)),
            np.arccos(np.clip((a * a + b * b - c * c) / np.maximum(2.0 * a * b, 1e-15), -1.0, 1.0)),
        )
    )
    third = np.pi - angles[:, 0] - angles[:, 1]
    minimum_angles = np.min(np.column_stack((angles, third)), axis=1)
    aspect_ratio = np.max(lengths, axis=1) / np.maximum(minimum_altitudes, 1e-15)
    return areas, minimum_altitudes, np.degrees(minimum_angles), aspect_ratio


def build_operators(settings: Settings):
    vertices, faces, point_locator = make_delaunay_mesh(settings)
    number_of_vertices = len(vertices)
    number_of_faces = len(faces)
    local_edges = np.stack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1
    ).reshape(-1, 2)
    edges, local_edge_indices = np.unique(
        np.sort(local_edges, axis=1), axis=0, return_inverse=True
    )
    number_of_edges = len(edges)
    face_indices = np.repeat(np.arange(number_of_faces), 3)
    local_orientation = np.where(local_edges[:, 0] < local_edges[:, 1], 1.0, -1.0)

    d0 = sp.csr_matrix(
        (
            np.tile((-1.0, 1.0), number_of_edges),
            (np.repeat(np.arange(number_of_edges), 2), edges.ravel()),
        ),
        shape=(number_of_edges, number_of_vertices),
    )
    d1 = sp.csr_matrix(
        (local_orientation, (face_indices, local_edge_indices)),
        shape=(number_of_faces, number_of_edges),
    )
    points3 = vertices[faces]
    points = points3[:, :, :2]
    face_areas, minimum_altitudes, minimum_angles, aspect_ratios = triangle_quality(points)
    if np.any(face_areas <= 1e-14):
        raise ValueError("Delaunay generated a degenerate triangle")
    if np.min(minimum_angles) < 15.0:
        raise ValueError(
            f"mesh quality check failed: minimum angle is {np.min(minimum_angles):.2f} degrees"
        )
    face_centers = points3.mean(axis=1)
    edge_vectors = vertices[edges[:, 1], :2] - vertices[edges[:, 0], :2]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    tangents = edge_vectors / edge_lengths[:, None]
    right_normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))

    star0 = np.zeros(number_of_vertices)
    np.add.at(star0, faces.ravel(), np.repeat(face_areas / 3.0, 3))
    incident_face_count = np.bincount(local_edge_indices, minlength=number_of_edges)
    boundary_edges = incident_face_count == 1
    face_edge_indices = local_edge_indices.reshape(number_of_faces, 3)
    face_edge_signs = local_orientation.reshape(number_of_faces, 3)
    edge_faces = [[] for _ in range(number_of_edges)]
    for face_index, edge_index in zip(face_indices, local_edge_indices):
        edge_faces[edge_index].append(face_index)
    face_edge_average = (
        sp.diags(1.0 / incident_face_count) @ abs(d1).T
    ).tocsr()
    interior_edges = np.flatnonzero(~boundary_edges)
    interior_edge_faces = np.asarray(
        [edge_faces[edge_index] for edge_index in interior_edges], dtype=np.int64
    )

    edge01 = points[:, 1] - points[:, 0]
    edge02 = points[:, 2] - points[:, 0]
    denominator = edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0]
    gradients = np.empty((number_of_faces, 3, 2))
    gradients[:, 0] = np.column_stack(
        (points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0])
    ) / denominator[:, None]
    gradients[:, 1] = np.column_stack(
        (points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0])
    ) / denominator[:, None]
    gradients[:, 2] = np.column_stack(
        (points[:, 0, 1] - points[:, 1, 1], points[:, 1, 0] - points[:, 0, 0])
    ) / denominator[:, None]

    # Constant face vector reconstructed from three outward integrated fluxes.
    flux_to_face_vector = np.empty((number_of_faces, 2, 3))
    for face_index in range(number_of_faces):
        local_ids = face_edge_indices[face_index]
        outward_flux_rows = (
            face_edge_signs[face_index, :, None]
            * right_normals[local_ids]
            * edge_lengths[local_ids, None]
        )
        flux_to_face_vector[face_index] = np.linalg.pinv(outward_flux_rows)

    # Map dual-edge differences to primal-edge normal flux with a local TPFA
    # conductance.  Centroid connectors are not generally edge-orthogonal, so
    # this stable local approximation is not an affine-exact gradient.
    conductance = np.zeros(number_of_edges)
    for edge_index in interior_edges:
        first_face, second_face = edge_faces[edge_index]
        dual_length = np.linalg.norm(
            face_centers[second_face, :2] - face_centers[first_face, :2]
        )
        conductance[edge_index] = edge_lengths[edge_index] / max(dual_length, 1e-15)
    pressure_gradient = (sp.diags(conductance) @ d1.T).tocsr()
    dual_gradient_matrix = (-pressure_gradient).tocsr()

    # Local least-squares gradients are used only for face diagnostics and
    # chart-distortion tests.  Clebsch reconstruction uses the TPFA map above.
    # Grow a small face-neighbor stencil
    # until it spans the plane; this also gives boundary triangles an affine-
    # exact gradient even when they have only one immediate interior neighbor.
    face_neighbors = [set() for _ in range(number_of_faces)]
    for adjacent_faces in edge_faces:
        if len(adjacent_faces) == 2:
            first_face, second_face = adjacent_faces
            face_neighbors[first_face].add(second_face)
            face_neighbors[second_face].add(first_face)
    gradient_rows = ([], [])
    gradient_columns = ([], [])
    gradient_values = ([], [])
    for face_index in range(number_of_faces):
        stencil = set(face_neighbors[face_index])
        frontier = set(stencil)
        while True:
            if stencil:
                stencil_indices = np.fromiter(sorted(stencil), dtype=np.int64)
                offsets = (
                    face_centers[stencil_indices, :2]
                    - face_centers[face_index, :2]
                )
                if len(stencil) >= 4 and np.linalg.matrix_rank(offsets) == 2:
                    break
            expanded = set()
            for neighbor in frontier:
                expanded.update(face_neighbors[neighbor])
            expanded.discard(face_index)
            expanded.difference_update(stencil)
            if not expanded:
                raise ValueError("could not build a full-rank face-gradient stencil")
            stencil.update(expanded)
            frontier = expanded
        squared_distance = np.sum(offsets * offsets, axis=1)
        weights = 1.0 / np.maximum(squared_distance, 1e-30)
        normal_matrix = offsets.T @ (weights[:, None] * offsets)
        coefficients = np.linalg.solve(
            normal_matrix, offsets.T * weights[None, :]
        )
        for component in range(2):
            component_coefficients = coefficients[component]
            gradient_rows[component].extend(
                np.full(len(stencil_indices), face_index, dtype=np.int64)
            )
            gradient_columns[component].extend(stencil_indices)
            gradient_values[component].extend(component_coefficients)
            gradient_rows[component].append(face_index)
            gradient_columns[component].append(face_index)
            gradient_values[component].append(-np.sum(component_coefficients))
    face_gradient_x = sp.csr_matrix(
        (gradient_values[0], (gradient_rows[0], gradient_columns[0])),
        shape=(number_of_faces, number_of_faces),
    )
    face_gradient_y = sp.csr_matrix(
        (gradient_values[1], (gradient_rows[1], gradient_columns[1])),
        shape=(number_of_faces, number_of_faces),
    )

    # Identity-label reconstruction matrix used by the rechart fit.
    coordinate_x = face_centers[:, 0]
    coordinate_y = face_centers[:, 1]
    coordinate_gradients = np.asarray(
        dual_gradient_matrix @ np.column_stack((coordinate_x, coordinate_y))
    )
    coordinate_grad_x = coordinate_gradients[:, 0]
    coordinate_grad_y = coordinate_gradients[:, 1]
    identity_chart_flux_matrix = sp.hstack(
        (
            sp.diags(coordinate_grad_x) @ face_edge_average,
            sp.diags(coordinate_grad_y) @ face_edge_average,
        ),
        format="csr",
    )

    # Pin one pressure per connected face component and factor the scalar
    # Neumann problem once.  Projection and label reconstruction intentionally
    # use the same conductance/Hodge map.
    neumann_laplacian = (d1 @ pressure_gradient).tocsc()
    adjacency = neumann_laplacian.copy()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    number_of_components, labels = connected_components(adjacency, directed=False)
    pinned_faces = np.array(
        [np.flatnonzero(labels == label)[0] for label in range(number_of_components)]
    )
    free_faces = np.setdiff1d(np.arange(number_of_faces), pinned_faces)
    poisson_factorization = splu(
        neumann_laplacian[free_faces][:, free_faces].tocsc()
    )
    exactness = (d1 @ d0).tocsr()
    exactness_error = 0.0 if exactness.nnz == 0 else float(np.max(np.abs(exactness.data)))

    return {
        "vertices": vertices,
        "faces": faces,
        "point_locator": point_locator,
        "edges": edges,
        "face_centers": face_centers,
        "face_areas": face_areas,
        "minimum_altitudes": minimum_altitudes,
        "minimum_angles": minimum_angles,
        "aspect_ratios": aspect_ratios,
        "face_edge_indices": face_edge_indices,
        "face_edge_signs": face_edge_signs,
        "flux_to_face_vector": flux_to_face_vector,
        "edge_lengths": edge_lengths,
        "right_normals": right_normals,
        "gradients": gradients,
        "d0": d0,
        "d1": d1,
        "star0": star0,
        # Mass/Hodge for point-valued dual 0-cochains.  This is deliberately
        # not named star2: primal integrated 2-cochains use the reciprocal-area
        # convention, while lambda and mu here are values at dual vertices.
        "dual_star0": face_areas,
        "boundary_edges": boundary_edges,
        "interior_edges": interior_edges,
        "interior_edge_faces": interior_edge_faces,
        "face_edge_average": face_edge_average,
        "conductance": conductance,
        "dual_gradient_matrix": dual_gradient_matrix,
        "pressure_gradient": pressure_gradient,
        "face_gradient_x": face_gradient_x,
        "face_gradient_y": face_gradient_y,
        "identity_chart_flux_matrix": identity_chart_flux_matrix,
        "poisson_factorization": poisson_factorization,
        "free_faces": free_faces,
        "number_of_projection_components": number_of_components,
        "exactness_error": exactness_error,
    }


def edge_flux_to_face_vectors(edge_flux, operators):
    outward_flux = (
        operators["face_edge_signs"]
        * np.asarray(edge_flux)[operators["face_edge_indices"]]
    )
    return np.einsum("fij,fj->fi", operators["flux_to_face_vector"], outward_flux)


def face_fields_to_vertex_fields(face_fields, operators):
    """Area-average scalar or vector face fields onto mesh vertices."""
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    if face_fields.ndim != 2 or face_fields.shape[0] != len(operators["faces"]):
        raise ValueError("face fields must have one row per face")

    accumulated = np.zeros((len(operators["vertices"]), face_fields.shape[1]))
    weighted = operators["face_areas"][:, None] * face_fields
    for local_vertex in range(3):
        np.add.at(accumulated, operators["faces"][:, local_vertex], weighted)
    vertex_fields = accumulated / np.maximum(
        operators["star0"][:, None] * 3.0, 1e-15
    )
    return vertex_fields[:, 0] if scalar_input else vertex_fields


def face_vectors_to_vertex_vectors(face_vectors, operators):
    return face_fields_to_vertex_fields(face_vectors, operators)


def dual_gradient(face_fields, operators):
    """Map face dual-0 values to local normal-flux gradient approximations."""
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    if face_fields.shape[0] != len(operators["faces"]):
        raise ValueError("dual-gradient fields must have one row per face")
    gradient = np.asarray(operators["dual_gradient_matrix"] @ face_fields)
    return gradient[:, 0] if scalar_input else gradient


def reconstruct_and_project(chart_fields, operators):
    """Return ``(p, u_e)`` for ``u_e = P sum_a(mu_a grad(lambda_a))``.

    ``p`` is the auxiliary face projection potential and ``u_e`` is an array
    of integrated normal edge fluxes.  ``chart_fields`` stores one dual
    0-cochain per column, hence it must have one row per primal face.  The
    label gradient is the local barycentric dual derivative
    ``-B C d1.T lambda`` and mu is centered from the two incident faces to the
    corresponding primal edge.
    """
    chart_fields = np.asarray(chart_fields)
    expected_shape = (len(operators["faces"]), CLEBSCH_FIELD_COUNT)
    if chart_fields.shape != expected_shape:
        raise ValueError(
            f"chart_fields must contain face-based Clebsch data with shape "
            f"{expected_shape}, got {chart_fields.shape}"
        )
    edge_mu_mean = np.asarray(
        operators["face_edge_average"] @ chart_fields[:, MU1 : MU2 + 1]
    )
    lambda_normal_integrals = dual_gradient(
        chart_fields[:, LAMBDA1 : LAMBDA2 + 1], operators
    )
    raw_edge_flux = np.sum(edge_mu_mean * lambda_normal_integrals, axis=1)
    return project_edge_flux(raw_edge_flux, operators)


def project_edge_flux(raw_edge_flux, operators):
    """Project ``u_raw`` to ``d1 u_e=0`` with zero boundary flux."""
    raw_edge_flux = np.asarray(raw_edge_flux).ravel()
    if raw_edge_flux.shape != (len(operators["edges"]),):
        raise ValueError("raw_edge_flux must have one value per edge")
    masked_edge_flux = raw_edge_flux.copy()
    masked_edge_flux[operators["boundary_edges"]] = 0.0
    free_faces = operators["free_faces"]
    divergence = np.asarray(operators["d1"] @ masked_edge_flux).ravel()
    projection_potential = np.zeros(len(operators["faces"]))
    projection_potential[free_faces] = operators["poisson_factorization"].solve(
        divergence[free_faces]
    )
    correction_flux = np.asarray(
        operators["pressure_gradient"] @ projection_potential
    ).ravel()
    projected_edge_flux = masked_edge_flux - correction_flux
    projected_edge_flux[operators["boundary_edges"]] = 0.0
    return projection_potential, projected_edge_flux


def centered_advection_matrix(edge_flux, operators):
    """Return the face/dual-vertex centered transport generator.

    For a face scalar f, the finite-volume weak rate is

        dual_star0 f_dot = -d1 diag(u_e) A_f->e f,

    where ``A_f->e`` averages the values of the incident faces.  With a
    divergence-free impermeable flux, this operator preserves constants,
    face-area-weighted mass, and the face-area-weighted quadratic invariant.
    """
    weak_operator = (
        -operators["d1"]
        @ sp.diags(np.asarray(edge_flux))
        @ operators["face_edge_average"]
    ).tocsr()
    return (sp.diags(1.0 / operators["dual_star0"]) @ weak_operator).tocsr()


def initial_velocity(x, y, settings: Settings):
    if settings.initial_condition == TAYLOR_GREEN:
        epsilon = 0.0
    elif settings.initial_condition == PERTURBED:
        epsilon = settings.perturbation_amplitude
    else:
        raise ValueError(
            f"unknown initial condition {settings.initial_condition!r}; "
            f"expected one of {INITIAL_CONDITIONS}"
        )
    ux = np.sin(x) * np.cos(y) + 3.0 * epsilon * np.sin(2.0 * x) * np.cos(3.0 * y)
    uy = -np.cos(x) * np.sin(y) - 2.0 * epsilon * np.cos(2.0 * x) * np.sin(3.0 * y)
    return np.column_stack((ux, uy))


def initial_condition_label(settings: Settings):
    if settings.initial_condition == TAYLOR_GREEN:
        return "Taylor-Green vortex"
    if settings.initial_condition == PERTURBED:
        return (
            "Perturbed Taylor-Green vortex "
            f"(epsilon={settings.perturbation_amplitude:g})"
        )
    raise ValueError(f"unknown initial condition {settings.initial_condition!r}")


def initialize_fields(operators, settings: Settings):
    """Return ``(face_clebsch_fields, vertex_dye)`` initial data."""
    face_x = operators["face_centers"][:, 0]
    face_y = operators["face_centers"][:, 1]
    face_velocity = initial_velocity(face_x, face_y, settings)
    vertex_x = operators["vertices"][:, 0]
    vertex_y = operators["vertices"][:, 1]
    dye = np.exp(
        -(
            (vertex_x - 0.72 * np.pi) ** 2
            + (vertex_y - 1.18 * np.pi) ** 2
        )
        / (0.23 * np.pi) ** 2
    )
    face_clebsch_fields = np.column_stack(
        (face_velocity[:, 0], face_velocity[:, 1], face_x, face_y)
    )
    return face_clebsch_fields, dye


def advection_rhs(transported_fields, edge_flux, operators, settings: Settings):
    """Advect face-based Clebsch dual 0-cochains by finite-volume fluxes."""
    transported_fields = np.asarray(transported_fields)
    if transported_fields.ndim == 1:
        transported_fields = transported_fields[:, None]
        squeeze = True
    else:
        squeeze = False
    if transported_fields.shape[0] != len(operators["faces"]):
        raise ValueError("face-transported fields must have one row per face")

    edge_values = operators["face_edge_average"] @ transported_fields
    weak_rhs = np.asarray(
        -operators["d1"] @ (np.asarray(edge_flux)[:, None] * edge_values)
    )
    if settings.stabilization == "graph" and settings.graph_viscosity > 0.0:
        edge_faces = operators["interior_edge_faces"]
        diffusion_strength = (
            settings.graph_viscosity
            * np.abs(np.asarray(edge_flux)[operators["interior_edges"]])
        )
        difference = (
            transported_fields[edge_faces[:, 1]]
            - transported_fields[edge_faces[:, 0]]
        )
        contribution = diffusion_strength[:, None] * difference
        np.add.at(weak_rhs, edge_faces[:, 0], contribution)
        np.add.at(weak_rhs, edge_faces[:, 1], -contribution)
    result = weak_rhs / operators["dual_star0"][:, None]
    return result[:, 0] if squeeze else result


def vertex_advection_rhs(transported_fields, edge_flux, operators, settings: Settings):
    """Advect the passive vertex dye with the original centered P1 scheme."""
    transported_fields = np.asarray(transported_fields)
    if transported_fields.ndim == 1:
        transported_fields = transported_fields[:, None]
        squeeze = True
    else:
        squeeze = False
    if transported_fields.shape[0] != len(operators["vertices"]):
        raise ValueError("vertex-transported fields must have one row per vertex")
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    velocity_dot_grad_basis = np.einsum(
        "fij,fj->fi", operators["gradients"], face_velocity
    )
    face_field_mean = np.mean(transported_fields[operators["faces"]], axis=1)
    face_weak_rhs = (
        operators["face_areas"][:, None, None]
        * velocity_dot_grad_basis[:, :, None]
        * face_field_mean[:, None, :]
    )
    weak_rhs = np.zeros_like(transported_fields)
    np.add.at(
        weak_rhs,
        operators["faces"].ravel(),
        face_weak_rhs.reshape(-1, transported_fields.shape[1]),
    )

    if settings.stabilization == "graph" and settings.graph_viscosity > 0.0:
        # Conservative element-graph viscosity.  It is an intentionally simple
        # first-order stabilizer, useful for distinguishing instability from
        # under-resolution; set --stabilization centered to disable it.
        diffusion_strength = (
            settings.graph_viscosity
            * operators["face_areas"]
            * np.max(np.abs(velocity_dot_grad_basis), axis=1)
            / 3.0
        )
        face_values = transported_fields[operators["faces"]]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            difference = face_values[:, j] - face_values[:, i]
            contribution = diffusion_strength[:, None] * difference
            np.add.at(weak_rhs, operators["faces"][:, i], contribution)
            np.add.at(weak_rhs, operators["faces"][:, j], -contribution)
    result = weak_rhs / operators["star0"][:, None]
    return result[:, 0] if squeeze else result


def stage_rhs(face_state, dye, operators, settings: Settings):
    """Advect ``[local chart | material copy]`` with the chart's velocity."""
    chart = face_state[:, :CLEBSCH_FIELD_COUNT]
    _, edge_flux = reconstruct_and_project(chart, operators)
    return (
        advection_rhs(face_state, edge_flux, operators, settings),
        vertex_advection_rhs(dye, edge_flux, operators, settings),
    )


def ssprk3_step(
    chart_fields, material_fields, dye, time_step, operators, settings: Settings
):
    """Advance face ``(C, M)`` and vertex dye with feedback from chart ``C``."""
    face_state = np.hstack((chart_fields, material_fields))
    face_rate, dye_rate = stage_rhs(face_state, dye, operators, settings)
    face_stage1 = face_state + time_step * face_rate
    dye_stage1 = dye + time_step * dye_rate

    face_rate, dye_rate = stage_rhs(face_stage1, dye_stage1, operators, settings)
    face_stage2 = 0.75 * face_state + 0.25 * (
        face_stage1 + time_step * face_rate
    )
    dye_stage2 = 0.75 * dye + 0.25 * (dye_stage1 + time_step * dye_rate)

    face_rate, dye_rate = stage_rhs(face_stage2, dye_stage2, operators, settings)
    next_face_state = (1.0 / 3.0) * face_state + (2.0 / 3.0) * (
        face_stage2 + time_step * face_rate
    )
    next_chart, next_material = np.hsplit(next_face_state, 2)
    next_dye = (1.0 / 3.0) * dye + (2.0 / 3.0) * (
        dye_stage2 + time_step * dye_rate
    )
    projection_potential, edge_flux = reconstruct_and_project(next_chart, operators)
    return next_chart, next_material, next_dye, projection_potential, edge_flux


def face_field_gradients(fields, operators):
    """Recover affine-exact planar gradients from face-centered samples."""
    fields = np.asarray(fields)
    if fields.ndim == 1:
        return np.column_stack(
            (
                operators["face_gradient_x"] @ fields,
                operators["face_gradient_y"] @ fields,
            )
        )
    return np.stack(
        (
            operators["face_gradient_x"] @ fields,
            operators["face_gradient_y"] @ fields,
        ),
        axis=2,
    )


def chart_metrics(chart_fields, operators):
    gradients = face_field_gradients(chart_fields, operators)
    jacobian = np.stack((gradients[:, LAMBDA1], gradients[:, LAMBDA2]), axis=1)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, 1]
    determinant = np.linalg.det(jacobian)
    log_sigma_max = np.log(np.maximum(sigma_max, 1e-15))
    log_condition = np.log(np.maximum(sigma_max / np.maximum(sigma_min, 1e-15), 1.0))
    return log_sigma_max, log_condition, determinant, sigma_min


def interpolate_planar_face_fields(query_points, face_fields, operators):
    """Sample face fields with a local linear face-centered reconstruction."""
    query = np.asarray(query_points, dtype=np.float64).copy()
    query[:, 0] = np.clip(query[:, 0], 0.0, DOMAIN_LENGTH)
    query[:, 1] = np.clip(query[:, 1], 0.0, DOMAIN_LENGTH)
    triangulation = operators["point_locator"]
    simplex = triangulation.find_simplex(query, tol=1e-10)
    if np.any(simplex < 0):
        count = int(np.count_nonzero(simplex < 0))
        raise FloatingPointError(
            f"could not locate {count} back-map samples in the reference mesh"
        )
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    gradients = face_field_gradients(face_fields, operators)
    offsets = query - operators["face_centers"][simplex, :2]
    samples = face_fields[simplex] + np.einsum(
        "nsi,ni->ns", gradients[simplex], offsets
    )
    return samples[:, 0] if scalar_input else samples


def compose_planar_backmap(local_lambda, reference_to_initial, operators):
    """Compose the current incremental chart with all prior chart transitions."""
    return interpolate_planar_face_fields(
        local_lambda, reference_to_initial, operators
    )


def clebsch_vorticity(chart_fields, operators):
    """Evaluate sum_a det(grad(mu_a), grad(lambda_a)) on each face."""
    gradients = face_field_gradients(chart_fields, operators)
    result = np.zeros(len(operators["faces"]))
    for mu_index, lambda_index in ((MU1, LAMBDA1), (MU2, LAMBDA2)):
        grad_mu = gradients[:, mu_index]
        grad_lambda = gradients[:, lambda_index]
        result += grad_mu[:, 0] * grad_lambda[:, 1] - grad_mu[:, 1] * grad_lambda[:, 0]
    return result


def recovered_velocity_vorticity(edge_flux, operators):
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    vertex_velocity = face_vectors_to_vertex_vectors(face_velocity, operators)
    gradients = np.einsum(
        "fvi,fvc->fci", operators["gradients"], vertex_velocity[operators["faces"]]
    )
    return gradients[:, 1, 0] - gradients[:, 0, 1]


def relative_divergence(edge_flux, operators):
    residual = np.asarray(operators["d1"] @ edge_flux).ravel()
    scale = np.asarray(abs(operators["d1"]) @ np.abs(edge_flux)).ravel()
    return np.linalg.norm(residual, np.inf) / max(np.linalg.norm(scale, np.inf), 1e-15)


def kinetic_energy(edge_flux, operators):
    vectors = edge_flux_to_face_vectors(edge_flux, operators)
    return 0.5 * np.sum(operators["face_areas"] * np.sum(vectors * vectors, axis=1))


def relative_vorticity_l2_change(vorticity, initial_vorticity, face_areas):
    """Return the area-weighted L2 change from the initial vorticity, in percent."""
    difference = np.asarray(vorticity) - np.asarray(initial_vorticity)
    numerator = np.sqrt(np.sum(face_areas * difference * difference, axis=-1))
    initial_norm = np.sqrt(
        np.sum(face_areas * np.asarray(initial_vorticity) ** 2)
    )
    return 100.0 * numerator / max(float(initial_norm), 1e-15)


def scalar_invariants(transported_fields, operators):
    """Return face-area-weighted invariants for face Clebsch fields."""
    mass = operators["dual_star0"] @ transported_fields
    l2 = np.sqrt(
        np.sum(
            operators["dual_star0"][:, None] * transported_fields * transported_fields,
            axis=0,
        )
    )
    return mass, l2


def vertex_scalar_invariants(transported_field, operators):
    """Return barycentric-dual vertex invariants for the passive dye."""
    mass = float(operators["star0"] @ transported_field)
    l2 = float(np.sqrt(operators["star0"] @ (transported_field * transported_field)))
    return mass, l2


def stable_time_step(edge_flux, operators, settings: Settings):
    """Return a CFL-limited step from the current projected edge flux."""
    speed = np.linalg.norm(edge_flux_to_face_vectors(edge_flux, operators), axis=1)
    rates = speed / np.maximum(operators["minimum_altitudes"], 1e-15)
    maximum_rate = max(float(np.max(rates)), 1e-15)
    return min(settings.maximum_time_step, settings.cfl / maximum_rate), maximum_rate


def rechart(chart_fields, edge_flux, reference_to_initial, operators):
    """Fit a fresh identity-label chart to the current projected edge flux."""
    edge_flux_before = edge_flux.copy()
    global_labels_before = compose_planar_backmap(
        chart_fields[:, LAMBDA1:LAMBDA2 + 1], reference_to_initial, operators
    )
    reset_chart = chart_fields.copy()
    interior = operators["interior_edges"]
    recovery_matrix = operators["identity_chart_flux_matrix"][interior]
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    baseline_coefficients = np.concatenate(
        (face_velocity[:, 0], face_velocity[:, 1])
    )
    residual_target = (
        edge_flux[interior] - recovery_matrix @ baseline_coefficients
    )
    row_scale = np.sqrt(
        np.mean(np.asarray(recovery_matrix.multiply(recovery_matrix).sum(axis=1)).ravel())
    )
    fit = lsqr(
        # The face system has more coefficient unknowns than interior-edge
        # targets.  A light minimum-change damping selects a stable solution
        # without sacrificing the flux transfer at every rechart.
        recovery_matrix, residual_target, damp=0.01 * max(row_scale, 1e-15),
        atol=1e-11, btol=1e-11, iter_lim=4 * recovery_matrix.shape[1],
    )
    recovered_coefficients = baseline_coefficients + fit[0]
    if not np.all(np.isfinite(recovered_coefficients)):
        raise FloatingPointError("non-finite Clebsch rechart fit")

    fitted_raw_flux = np.asarray(
        recovery_matrix @ recovered_coefficients
    ).ravel()
    fit_relative_residual = np.linalg.norm(
        fitted_raw_flux - edge_flux[interior]
    ) / max(np.linalg.norm(edge_flux[interior]), 1e-15)
    number_of_faces = len(operators["faces"])
    reset_chart[:, MU1] = recovered_coefficients[:number_of_faces]
    reset_chart[:, MU2] = recovered_coefficients[number_of_faces:]
    reset_chart[:, LAMBDA1] = operators["face_centers"][:, 0]
    reset_chart[:, LAMBDA2] = operators["face_centers"][:, 1]
    projection_potential, edge_flux_after = reconstruct_and_project(
        reset_chart, operators
    )
    new_reference_to_initial = global_labels_before.copy()
    global_labels_after = compose_planar_backmap(
        reset_chart[:, LAMBDA1:LAMBDA2 + 1], new_reference_to_initial, operators
    )
    relative_velocity_jump = np.linalg.norm(
        edge_flux_after - edge_flux_before
    ) / max(np.linalg.norm(edge_flux_before), 1e-15)
    energy_before = kinetic_energy(edge_flux_before, operators)
    relative_energy_jump = (
        kinetic_energy(edge_flux_after, operators) - energy_before
    ) / max(energy_before, 1e-15)
    transition = {
        "velocity_jump": relative_velocity_jump,
        "energy_jump": relative_energy_jump,
        "fit_relative_residual": float(fit_relative_residual),
        "global_label_jump": float(
            np.max(
                np.linalg.norm(global_labels_after - global_labels_before, axis=1),
                initial=0.0,
            )
        ),
    }
    return (
        reset_chart,
        projection_potential,
        edge_flux_after,
        new_reference_to_initial,
        transition,
    )


def simulate(operators, settings: Settings):
    material_fields, dye = initialize_fields(operators, settings)
    chart_fields = material_fields.copy()
    projection_potential, edge_flux = reconstruct_and_project(
        chart_fields, operators
    )
    reference_to_initial = operators["face_centers"][:, :2].copy()
    output_times = np.linspace(0.0, settings.final_time, settings.frames)
    histories = {
        "times": [], "material_fields": [], "dye": [],
        "projection_potential": [], "edge_fluxes": [], "composition_error": [],
        "clebsch_vorticity": [], "velocity_vorticity": [],
        "log_sigma_max": [], "log_condition": [], "determinant": [],
        "divergence": [], "boundary_error": [], "energy": [],
        "mass": [], "l2": [], "maximum_speed": [], "step_cfl": [],
        "chart": [], "chart_age": [],
    }
    reset_events = []
    current_time = 0.0
    next_output = 0
    step = 0
    chart_index = 0
    chart_start = 0.0
    last_rechart_time = -np.inf
    maximum_start_state_cfl_since_output = 0.0

    def store():
        nonlocal maximum_start_state_cfl_since_output
        log_sigma, log_condition, determinant, _ = chart_metrics(
            chart_fields, operators
        )
        composed_labels = compose_planar_backmap(
            chart_fields[:, LAMBDA1:LAMBDA2 + 1],
            reference_to_initial,
            operators,
        )
        material_labels = material_fields[:, LAMBDA1:LAMBDA2 + 1]
        omega_c = clebsch_vorticity(chart_fields, operators)
        omega_u = recovered_velocity_vorticity(edge_flux, operators)
        clebsch_mass, clebsch_l2 = scalar_invariants(material_fields, operators)
        dye_mass, dye_l2 = vertex_scalar_invariants(dye, operators)
        mass = np.concatenate((clebsch_mass, (dye_mass,)))
        l2 = np.concatenate((clebsch_l2, (dye_l2,)))
        face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
        histories["times"].append(current_time)
        histories["material_fields"].append(material_fields.copy())
        histories["dye"].append(dye.copy())
        histories["composition_error"].append(
            np.linalg.norm(composed_labels - material_labels, axis=1)
        )
        histories["projection_potential"].append(projection_potential.copy())
        histories["edge_fluxes"].append(edge_flux.copy())
        histories["clebsch_vorticity"].append(omega_c)
        histories["velocity_vorticity"].append(omega_u)
        histories["log_sigma_max"].append(log_sigma)
        histories["log_condition"].append(log_condition)
        histories["determinant"].append(determinant)
        histories["divergence"].append(relative_divergence(edge_flux, operators))
        histories["boundary_error"].append(
            float(
                np.max(
                    np.abs(edge_flux[operators["boundary_edges"]]), initial=0.0
                )
            )
        )
        histories["energy"].append(kinetic_energy(edge_flux, operators))
        histories["mass"].append(mass)
        histories["l2"].append(l2)
        histories["maximum_speed"].append(float(np.max(np.linalg.norm(face_velocity, axis=1))))
        histories["step_cfl"].append(maximum_start_state_cfl_since_output)
        histories["chart"].append(chart_index)
        histories["chart_age"].append(current_time - chart_start)
        maximum_start_state_cfl_since_output = 0.0

    def apply_rechart(reason, log_sigma, determinant):
        nonlocal chart_fields, projection_potential, edge_flux
        nonlocal reference_to_initial, chart_index, chart_start, last_rechart_time
        (
            chart_fields,
            projection_potential,
            edge_flux,
            reference_to_initial,
            transition,
        ) = rechart(
            chart_fields,
            edge_flux,
            reference_to_initial,
            operators,
        )
        chart_index += 1
        chart_start = current_time
        last_rechart_time = current_time
        reset_events.append(
            {
                "time": current_time,
                "chart": chart_index,
                "reason": reason,
                "pre_log_sigma_max": float(np.max(log_sigma)),
                "pre_min_determinant": float(np.min(determinant)),
                **transition,
            }
        )

    store()
    next_output = 1
    tolerance = 1e-12 * max(1.0, settings.final_time)
    while current_time < settings.final_time - tolerance:
        proposed_dt, start_state_rate = stable_time_step(
            edge_flux, operators, settings
        )
        target = output_times[next_output] if next_output < len(output_times) else settings.final_time
        dt = min(proposed_dt, target - current_time, settings.final_time - current_time)
        if not np.isfinite(dt) or dt <= tolerance:
            maximum_speed = float(
                np.max(
                    np.linalg.norm(
                        edge_flux_to_face_vectors(edge_flux, operators), axis=1
                    )
                )
            )
            raise FloatingPointError(
                "CFL step collapsed before the next output: "
                f"t={current_time:.12g}, dt={dt:.3e}, "
                f"proposed_dt={proposed_dt:.3e}, max_speed={maximum_speed:.3e}"
            )
        (
            chart_fields,
            material_fields,
            dye,
            projection_potential,
            edge_flux,
        ) = ssprk3_step(
            chart_fields, material_fields, dye, dt, operators, settings
        )
        current_time += dt
        step += 1
        maximum_start_state_cfl_since_output = max(
            maximum_start_state_cfl_since_output, dt * start_state_rate
        )
        finite_state = (
            np.all(np.isfinite(chart_fields))
            and np.all(np.isfinite(material_fields))
            and np.all(np.isfinite(dye))
            and np.all(np.isfinite(projection_potential))
            and np.all(np.isfinite(edge_flux))
        )
        if not finite_state:
            raise FloatingPointError(f"solution became non-finite at step {step}, t={current_time:.6g}")

        log_sigma, _, determinant, sigma_min = chart_metrics(
            chart_fields, operators
        )
        trigger_reasons = []
        if float(np.max(log_sigma)) > settings.rechart_cutoff:
            trigger_reasons.append("sigma_max")
        if float(np.min(determinant)) <= settings.determinant_floor:
            trigger_reasons.append("determinant")
        if float(np.min(sigma_min)) <= 1e-8:
            trigger_reasons.append("sigma_min")
        if trigger_reasons:
            apply_rechart("+".join(trigger_reasons), log_sigma, determinant)

        if next_output < len(output_times) and current_time >= output_times[next_output] - tolerance:
            current_time = output_times[next_output]
            scheduled_rechart = (
                settings.rechart_every_frames > 0
                and next_output % settings.rechart_every_frames == 0
                and abs(current_time - last_rechart_time) > tolerance
            )
            if scheduled_rechart:
                log_sigma, _, determinant, _ = chart_metrics(
                    chart_fields, operators
                )
                apply_rechart("frame interval", log_sigma, determinant)
            store()
            next_output += 1

    result = {key: np.asarray(value) for key, value in histories.items()}
    result["velocity_vorticity_change"] = relative_vorticity_l2_change(
        result["velocity_vorticity"],
        result["velocity_vorticity"][0],
        operators["face_areas"],
    )
    result["clebsch_vorticity_change"] = relative_vorticity_l2_change(
        result["clebsch_vorticity"],
        result["clebsch_vorticity"][0],
        operators["face_areas"],
    )
    for key in (
        "material_fields", "dye", "composition_error", "projection_potential",
        "edge_fluxes", "clebsch_vorticity", "velocity_vorticity",
        "log_sigma_max", "log_condition", "determinant",
    ):
        result[key] = result[key].astype(np.float32)
    result["reset_events"] = reset_events
    result["number_of_steps"] = step
    return result


def percentage_change(values, frame=-1):
    return 100.0 * (values[frame] / max(abs(values[0]), 1e-15) - 1.0)


def diagnostic_time_series(result):
    """Return the scalar histories used by the diagnostic time-series plot.

    This is deliberately a post-processing helper: it does not alter any
    state used by the simulation or visualization.  The vorticity entries are
    the area-weighted relative L2 differences already computed by
    :func:`simulate`, not changes in the scalar vorticity norm.
    """
    energy = np.asarray(result["energy"], dtype=float)
    return {
        "time": np.asarray(result["times"], dtype=float),
        "relative_divergence": np.asarray(result["divergence"], dtype=float),
        "energy_change_percent": 100.0 * (
            energy / max(abs(float(energy[0])), 1e-15) - 1.0
        ),
        "velocity_vorticity_change_percent": np.asarray(
            result["velocity_vorticity_change"], dtype=float
        ),
        "clebsch_vorticity_change_percent": np.asarray(
            result["clebsch_vorticity_change"], dtype=float
        ),
    }


def plot_diagnostic_histories(result, settings: Settings, output_path=None, show=False):
    """Plot divergence, energy, and vorticity diagnostics versus physical time.

    ``output_path`` is normalized to a PNG path and its parent directory is
    created if necessary.  Matplotlib is imported only when this optional
    function is requested, so it is not a dependency of ordinary solver runs.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "diagnostic plotting requires matplotlib; install it or omit "
            "--diagnostics-plot/--show-diagnostic-plots"
        ) from error

    series = diagnostic_time_series(result)
    times = series["time"]
    figure, axes = plt.subplots(
        3, 1, figsize=(8.0, 8.5), sharex=True, constrained_layout=True
    )

    # A log axis cannot display exact zeros.  Clipping is only a display
    # operation; the stored divergence history remains unchanged.
    log_floor = 1e-18
    plotted_divergence = np.maximum(
        np.abs(series["relative_divergence"]), log_floor
    )
    axes[0].semilogy(
        times, plotted_divergence, color="tab:blue", linewidth=1.8,
        label=r"relative divergence $\epsilon_{\mathrm{div}}$",
    )
    axes[0].axhline(
        log_floor, color="0.55", linewidth=0.8, linestyle=":",
        label=rf"display floor $10^{{{int(np.log10(log_floor))}}}$",
    )
    axes[0].set_ylabel("Relative divergence")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        times, series["energy_change_percent"], color="tab:green", linewidth=1.8
    )
    axes[1].axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axes[1].set_ylabel("Relative energy change [%]")

    axes[2].plot(
        times, series["velocity_vorticity_change_percent"],
        color="tab:red", linewidth=1.8,
        label="from projected edge velocity",
    )
    axes[2].plot(
        times, series["clebsch_vorticity_change_percent"],
        color="tab:purple", linewidth=1.8, linestyle="--",
        label="from Clebsch gradients",
    )
    axes[2].axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axes[2].set_ylabel(r"Relative vorticity $L^2$ difference [\%]")
    axes[2].set_xlabel("Physical time")
    axes[2].legend(loc="best", fontsize=8)

    for event_index, event in enumerate(result.get("reset_events", [])):
        label = "rechart" if event_index == 0 else None
        for axis in axes:
            axis.axvline(
                float(event["time"]), color="0.35", linewidth=0.7,
                linestyle="--", alpha=0.35, label=label,
            )
    if result.get("reset_events"):
        axes[0].legend(loc="best", fontsize=8)

    for axis in axes:
        axis.grid(True, which="both", linewidth=0.45, alpha=0.3)
        axis.margins(x=0.0)
    figure.suptitle(
        f"2D Clebsch Euler diagnostics: {initial_condition_label(settings)}"
    )

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path).expanduser()
        if saved_path.suffix.lower() != ".png":
            saved_path = saved_path.with_suffix(".png")
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=200, bbox_inches="tight", format="png")
        print(f"Diagnostic time-series plot:    {saved_path.resolve()}")
    if show:
        plt.show()
    plt.close(figure)
    return saved_path


def visualization_limits(values, signed):
    magnitude = max(float(np.percentile(np.abs(values), 99.5)), 1e-12)
    return (-magnitude, magnitude) if signed else (float(np.min(values)), float(np.percentile(values, 99.5)))


def visualize(result, operators, settings: Settings):
    import polyscope as ps
    import polyscope.imgui as psim

    face_velocities = np.asarray(
        [edge_flux_to_face_vectors(flux, operators) for flux in result["edge_fluxes"]]
    )
    vertex_velocities = np.asarray(
        [face_vectors_to_vertex_vectors(velocity, operators) for velocity in face_velocities]
    )
    vertex_material_fields = np.asarray(
        [
            face_fields_to_vertex_fields(material_fields, operators)
            for material_fields in result["material_fields"]
        ]
    )
    face_speed = np.linalg.norm(face_velocities, axis=2)
    vertex_speed = np.linalg.norm(vertex_velocities, axis=2)
    face_velocity_change = face_velocities - face_velocities[0]
    vertex_velocity_change = vertex_velocities - vertex_velocities[0]
    determinant_error = result["determinant"] - 1.0
    names = (
        "Speed |u|", "Local-chart Clebsch vorticity", "Recovered velocity vorticity",
        "Velocity change", "log(sigma_max)", "log(condition number)",
        "det(Dlambda_local)-1", "mu_1 (material, face-based)",
        "mu_2 (material, face-based)", "lambda_1 = Psi_0 X (face-based)",
        "lambda_2 = Psi_0 Y (face-based)", "Persistent dye (vertex-based)",
        "Projection potential Phi", "Chart-composition label error",
    )
    speed_view = 0
    velocity_change_view = 3
    dye_view = 11
    material_views = {7: MU1, 8: MU2, 9: LAMBDA1, 10: LAMBDA2}
    scalar_data = (
        face_speed, result["clebsch_vorticity"], result["velocity_vorticity"],
        np.linalg.norm(face_velocity_change, axis=2),
        result["log_sigma_max"], result["log_condition"],
        determinant_error, result["material_fields"][:, :, MU1],
        result["material_fields"][:, :, MU2],
        result["material_fields"][:, :, LAMBDA1],
        result["material_fields"][:, :, LAMBDA2],
        result["dye"], result["projection_potential"],
        result["composition_error"],
    )
    vertex_fields = {dye_view}
    sequential_fields = {0, velocity_change_view, 5, dye_view, 13}
    defined_on = tuple("vertices" if index in vertex_fields else "faces" for index in range(len(names)))
    color_maps = tuple("viridis" if index in sequential_fields else "coolwarm" for index in range(len(names)))
    limits = [visualization_limits(data, index not in sequential_fields) for index, data in enumerate(scalar_data)]
    limits[speed_view] = (
        0.0,
        max(
            float(np.percentile(face_speed, 99.5)),
            float(np.percentile(vertex_speed, 99.5)),
            1e-12,
        ),
    )
    limits[velocity_change_view] = (
        0.0,
        max(
            float(np.percentile(np.linalg.norm(face_velocity_change, axis=2), 99.5)),
            float(np.percentile(np.linalg.norm(vertex_velocity_change, axis=2), 99.5)),
            1e-12,
        ),
    )
    for view, component in material_views.items():
        magnitude = max(
            float(np.percentile(np.abs(scalar_data[view]), 99.5)),
            float(np.percentile(np.abs(vertex_material_fields[:, :, component]), 99.5)),
            1e-12,
        )
        limits[view] = (-magnitude, magnitude)

    arrow_faces = np.arange(0, len(operators["faces"]), max(settings.arrow_stride**2, 1))
    arrow_points = operators["face_centers"][arrow_faces].copy()
    arrow_points[:, 2] = 0.012
    target_length = 0.8 * settings.arrow_stride * DOMAIN_LENGTH / settings.resolution
    velocity_scale = target_length / max(float(np.percentile(face_speed[:, arrow_faces], 95)), 1e-15)
    change_scale = target_length / max(
        float(np.percentile(np.linalg.norm(face_velocity_change[:, arrow_faces], axis=2), 95)), 1e-15
    )

    ps.set_use_prefs_file(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.init()
    ps.set_ground_plane_mode("none")
    surface = ps.register_surface_mesh(
        "Delaunay Clebsch mesh", operators["vertices"], operators["faces"],
        edge_width=0.7, smooth_shade=False,
    )
    arrows = ps.register_point_cloud("Velocity samples", arrow_points, radius=0.0008)
    frame = 0
    selected = 0
    current_time = 0.0
    playing = False
    loop_playback = True
    show_arrows = True
    show_edges = True
    smooth_display = True
    playback_rate = settings.playback_fps
    accumulator = 0.0
    previous_clock = time.perf_counter()

    def remove_quantities():
        surface.remove_quantity(names[selected])
        arrows.remove_quantity("Vectors")

    def add_quantities():
        if smooth_display and selected == speed_view:
            displayed_scalars = vertex_speed[frame]
            scalar_location = "vertices"
        elif smooth_display and selected == velocity_change_view:
            displayed_scalars = np.linalg.norm(vertex_velocity_change[frame], axis=1)
            scalar_location = "vertices"
        elif smooth_display and selected in material_views:
            displayed_scalars = vertex_material_fields[
                frame, :, material_views[selected]
            ]
            scalar_location = "vertices"
        else:
            displayed_scalars = scalar_data[selected][frame]
            scalar_location = defined_on[selected]
        surface.add_scalar_quantity(
            names[selected], displayed_scalars, defined_on=scalar_location,
            enabled=True, cmap=color_maps[selected], vminmax=limits[selected],
        )
        if selected == velocity_change_view:
            vector_field = vertex_velocity_change if smooth_display else face_velocity_change
            scale, color = change_scale, (0.95, 0.10, 0.05)
        else:
            vector_field = vertex_velocities if smooth_display else face_velocities
            scale, color = velocity_scale, (0.05, 0.25, 1.0)
        if smooth_display:
            vectors = vector_field[frame][operators["faces"][arrow_faces]].mean(axis=1)
        else:
            vectors = vector_field[frame, arrow_faces]
        arrows.add_vector_quantity(
            "Vectors", scale * np.column_stack((vectors, np.zeros(len(vectors)))),
            enabled=show_arrows, vectortype="ambient", color=color, radius=0.003,
        )
        arrows.set_enabled(show_arrows)

    add_quantities()

    def callback():
        nonlocal frame, selected, current_time, playing, loop_playback
        nonlocal show_arrows, show_edges, smooth_display
        nonlocal playback_rate, accumulator, previous_clock
        clock = time.perf_counter()
        elapsed = min(clock - previous_clock, 0.1)
        previous_clock = clock
        if playing:
            accumulator += elapsed * playback_rate
            advance = int(accumulator)
            if advance:
                accumulator -= advance
                remove_quantities()
                candidate = frame + advance
                if candidate >= len(result["times"]):
                    if loop_playback:
                        frame = candidate % len(result["times"])
                    else:
                        frame = len(result["times"]) - 1
                        playing = False
                else:
                    frame = candidate
                current_time = float(result["times"][frame])
                add_quantities()

        scale = ps.get_ui_scale() if hasattr(ps, "get_ui_scale") else 1.0
        width, _ = ps.get_window_size()
        psim.SetNextWindowPos((width - 12.0 * scale, 12.0 * scale), pivot=(1.0, 0.0))
        psim.SetNextWindowBgAlpha(0.94)
        flags = (
            psim.ImGuiWindowFlags_AlwaysAutoResize | psim.ImGuiWindowFlags_NoCollapse
            | psim.ImGuiWindowFlags_NoMove | psim.ImGuiWindowFlags_NoSavedSettings
        )
        begin = psim.Begin("Two-pair Clebsch Euler", True, flags=flags)
        visible = begin[0] if isinstance(begin, tuple) else begin
        if visible:
            psim.PushItemWidth(240.0 * scale)
            changed_view, new_selected = psim.Combo("Field", selected, names)
            changed_time, new_time = psim.SliderFloat("Time", current_time, 0.0, settings.final_time)
            if psim.Button("Pause" if playing else "Play"):
                playing = not playing
                accumulator = 0.0
            psim.SameLine()
            previous = psim.Button("Previous")
            psim.SameLine()
            following = psim.Button("Next")
            changed_rate, new_rate = psim.SliderFloat("Playback frames/s", playback_rate, 1.0, 60.0)
            changed_loop, new_loop = psim.Checkbox("Loop", loop_playback)
            changed_arrows, new_arrows = psim.Checkbox("Velocity arrows", show_arrows)
            changed_smoothing, new_smoothing = psim.Checkbox(
                "Smooth velocity/Clebsch display", smooth_display
            )
            changed_edges, new_edges = psim.Checkbox("Show triangle edges", show_edges)
            new_frame = int(np.argmin(np.abs(result["times"] - new_time)))
            if previous:
                playing, new_frame, changed_time = False, max(frame - 1, 0), True
            elif following:
                playing, new_frame, changed_time = False, min(frame + 1, len(result["times"]) - 1), True
            if changed_rate:
                playback_rate = new_rate
            if changed_loop:
                loop_playback = new_loop
            if changed_arrows:
                show_arrows = new_arrows
                arrows.set_enabled(show_arrows)
            if changed_edges:
                show_edges = new_edges
                surface.set_edge_width(0.7 if show_edges else 0.0)
            if changed_view or changed_smoothing or (changed_time and new_frame != frame):
                remove_quantities()
                selected, frame = new_selected, new_frame
                smooth_display = new_smoothing
                current_time = float(result["times"][frame])
                add_quantities()

            psim.Separator()
            psim.TextUnformatted(
                f"Initial condition: {initial_condition_label(settings)}"
            )
            psim.TextUnformatted(f"SSPRK3; stabilization: {settings.stabilization}")
            psim.TextUnformatted(
                "Phi is an auxiliary face projection potential; it is not transported"
            )
            psim.TextUnformatted(f"Frame: {frame + 1}/{len(result['times'])}; t={result['times'][frame]:.5f}")
            psim.TextUnformatted(f"Chart: {int(result['chart'][frame])}; age={result['chart_age'][frame]:.4f}")
            psim.TextUnformatted(
                "Start-state CFL since previous frame: "
                f"{result['step_cfl'][frame]:.3e}"
            )
            psim.TextUnformatted(f"Max speed: {result['maximum_speed'][frame]:.3e}")
            psim.TextUnformatted(f"Max log(sigma_max): {np.max(result['log_sigma_max'][frame]):.3e}")
            psim.TextUnformatted(f"Min det(Dlambda): {np.min(result['determinant'][frame]):.3e}")
            psim.TextUnformatted(f"Relative divergence: {result['divergence'][frame]:.3e}")
            psim.TextUnformatted(f"Boundary-normal error: {result['boundary_error'][frame]:.3e}")
            psim.TextUnformatted(f"Energy change: {percentage_change(result['energy'], frame):+.3e}%")
            psim.TextUnformatted(
                "Recovered-u vorticity L2 change: "
                f"{result['velocity_vorticity_change'][frame]:.3e}%"
            )
            psim.TextUnformatted(
                "Clebsch vorticity L2 change: "
                f"{result['clebsch_vorticity_change'][frame]:.3e}%"
            )
            psim.TextUnformatted(f"Dye mass change: {100.0 * (result['mass'][frame, DYE_INDEX] - result['mass'][0, DYE_INDEX]) / max(abs(result['mass'][0, DYE_INDEX]), 1e-15):+.3e}%")
            dye_l2_loss = 100.0 * (
                1.0
                - result["l2"][frame, DYE_INDEX]
                / max(result["l2"][0, DYE_INDEX], 1e-15)
            )
            psim.TextUnformatted(f"Dye L2 norm loss: {dye_l2_loss:+.3e}%")
            global_label_l2_loss = 100.0 * (
                1.0
                - result["l2"][frame, LAMBDA1:LAMBDA2 + 1]
                / np.maximum(result["l2"][0, LAMBDA1:LAMBDA2 + 1], 1e-15)
            )
            psim.TextUnformatted(
                "Global-label L2 loss X/Y: "
                f"{global_label_l2_loss[0]:+.3e}% / {global_label_l2_loss[1]:+.3e}%"
            )
            material_mu_l2_loss = 100.0 * (
                1.0
                - result["l2"][frame, MU1:MU2 + 1]
                / np.maximum(result["l2"][0, MU1:MU2 + 1], 1e-15)
            )
            psim.TextUnformatted(
                "Global-mu L2 loss 1/2: "
                f"{material_mu_l2_loss[0]:+.3e}% / {material_mu_l2_loss[1]:+.3e}%"
            )
            psim.TextUnformatted(f"Recharts so far: {int(result['chart'][frame])}")
            psim.TextUnformatted(f"Scheduled rechart interval: {settings.rechart_every_frames} frames")
            psim.TextUnformatted(
                "Display smoothing only; face Clebsch state and DEC flux are unchanged"
                if smooth_display
                else "Showing raw piecewise-constant face fields and DEC velocity"
            )
            psim.TextUnformatted("Stored face mu/lambda and vertex dye are never reset")
            psim.TextUnformatted("Only the hidden local reconstruction chart is reinitialized")
            psim.PopItemWidth()
        psim.End()

    ps.set_user_callback(callback)
    center = 0.5 * DOMAIN_LENGTH
    ps.look_at((center, center, 11.0), (center, center, 0.0))
    ps.show()


def print_report(result, operators, settings: Settings):
    initial_face_velocity = edge_flux_to_face_vectors(result["edge_fluxes"][0], operators)
    centers = operators["face_centers"]
    analytic = initial_velocity(centers[:, 0], centers[:, 1], settings)
    initial_error = np.sqrt(
        np.sum(operators["face_areas"] * np.sum((initial_face_velocity - analytic) ** 2, axis=1))
        / max(np.sum(operators["face_areas"] * np.sum(analytic ** 2, axis=1)), 1e-15)
    )
    print(f"Initial condition:              {initial_condition_label(settings)}")
    print(f"Vertices / edges / triangles:   {len(operators['vertices']):,} / {len(operators['edges']):,} / {len(operators['faces']):,}")
    print(
        "Mesh min angle / max aspect:   "
        f"{np.min(operators['minimum_angles']):.3f} deg / "
        f"{np.max(operators['aspect_ratios']):.3f}"
    )
    print(f"SSPRK3 accepted steps:          {result['number_of_steps']:,}")
    print(f"Maximum accepted start CFL:     {np.max(result['step_cfl']):.6e}")
    if settings.stabilization == "graph":
        print(f"Graph-viscosity coefficient:    {settings.graph_viscosity:.6e}")
    print(f"Scheduled rechart interval:     {settings.rechart_every_frames} frames")
    print(f"Initial projected-velocity err: {initial_error:.6e}")
    print(f"Maximum relative divergence:    {np.max(result['divergence']):.6e}")
    print(f"Maximum boundary-normal error:  {np.max(result['boundary_error']):.6e}")
    print(f"Kinetic-energy change:          {percentage_change(result['energy']):+.6e}%")
    print(
        "Recovered-u vorticity L2 change: "
        f"{result['velocity_vorticity_change'][-1]:.6e}%"
    )
    print(
        "Clebsch vorticity L2 change:     "
        f"{result['clebsch_vorticity_change'][-1]:.6e}%"
    )
    clebsch_initial_l1 = np.sum(
        operators["dual_star0"][:, None] * np.abs(result["material_fields"][0]),
        axis=0,
    )
    dye_initial_l1 = operators["star0"] @ np.abs(result["dye"][0])
    initial_l1 = np.concatenate((clebsch_initial_l1, (dye_initial_l1,)))
    mass_errors = 100.0 * (result["mass"][-1] - result["mass"][0]) / np.maximum(initial_l1, 1e-15)
    l2_changes = 100.0 * (result["l2"][-1] / np.maximum(result["l2"][0], 1e-15) - 1.0)
    for index, name in enumerate(FIELD_NAMES):
        descriptor = f"face material {name}" if index < DYE_INDEX else "vertex dye"
        print(
            f"  {descriptor:24s} mass/L2 change: "
            f"{mass_errors[index]:+.3e}% / {l2_changes[index]:+.3e}%"
        )
    print(f"Number of recharts:             {len(result['reset_events'])}")
    if result["reset_events"]:
        maximum_velocity_jump = max(event["velocity_jump"] for event in result["reset_events"])
        maximum_energy_jump = max(abs(event["energy_jump"]) for event in result["reset_events"])
        maximum_label_jump = max(event["global_label_jump"] for event in result["reset_events"])
        maximum_fit_residual = max(
            event["fit_relative_residual"] for event in result["reset_events"]
        )
        print(f"Maximum reset velocity jump:    {maximum_velocity_jump:.6e}")
        print(f"Maximum reset energy jump:      {maximum_energy_jump:.6e}")
        print(f"Maximum global-label jump:      {maximum_label_jump:.6e}")
        print(f"Maximum raw-flux fit residual:  {maximum_fit_residual:.6e}")


def run_diagnostic_tests():
    """Run focused algebra, transport, rechart, and nonlinear regressions."""
    settings = Settings(
        resolution=8,
        final_time=0.1,
        frames=11,
        cfl=0.1,
        maximum_time_step=0.005,
        rechart_cutoff=99.0,
        rechart_every_frames=0,
        determinant_floor=0.0,
        initial_condition=TAYLOR_GREEN,
        perturbation_amplitude=0.0,
        stabilization="centered",
        graph_viscosity=0.0,
        jitter=0.08,
        random_seed=19,
    )
    operators = build_operators(settings)
    clebsch, dye = initialize_fields(operators, settings)
    _, edge_flux = reconstruct_and_project(clebsch, operators)
    rng = np.random.default_rng(12345)
    results = []

    def check(name, error, tolerance):
        passed = bool(np.isfinite(error) and error <= tolerance)
        results.append(passed)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name:43s} "
            f"error={error:.3e}  tolerance={tolerance:.3e}"
        )

    face_count = len(operators["faces"])
    vertex_count = len(operators["vertices"])
    sample_x = np.array((0.21, 1.07, 2.38))
    sample_y = np.array((0.43, 1.61, 4.22))
    expected_tgv = np.column_stack(
        (
            np.sin(sample_x) * np.cos(sample_y),
            -np.cos(sample_x) * np.sin(sample_y),
        )
    )
    check(
        "Taylor--Green initializer matches formula",
        np.max(np.abs(initial_velocity(sample_x, sample_y, settings) - expected_tgv)),
        2e-15,
    )
    perturbed_settings = replace(
        settings, initial_condition=PERTURBED, perturbation_amplitude=0.25
    )
    expected_perturbation = 0.25 * np.column_stack(
        (
            3.0 * np.sin(2.0 * sample_x) * np.cos(3.0 * sample_y),
            -2.0 * np.cos(2.0 * sample_x) * np.sin(3.0 * sample_y),
        )
    )
    check(
        "perturbed initializer adds prescribed mode",
        np.max(
            np.abs(
                initial_velocity(sample_x, sample_y, perturbed_settings)
                - expected_tgv
                - expected_perturbation
            )
        ),
        2e-15,
    )
    try:
        initial_velocity(
            sample_x, sample_y,
            replace(settings, initial_condition="not-a-valid-condition"),
        )
    except ValueError:
        invalid_condition_error = 0.0
    else:
        invalid_condition_error = 1.0
    check(
        "initializer rejects unknown condition names",
        invalid_condition_error,
        0.0,
    )
    check(
        "lambda/mu arrays are face-based",
        float(clebsch.shape != (face_count, CLEBSCH_FIELD_COUNT)),
        0.0,
    )
    check("passive dye remains vertex-based", float(dye.shape != (vertex_count,)), 0.0)
    constant_velocity = np.tile((1.25, -0.75), (face_count, 1))
    smoothed_velocity = face_vectors_to_vertex_vectors(
        constant_velocity, operators
    )
    check(
        "display projection preserves constant velocity",
        np.max(np.abs(smoothed_velocity - (1.25, -0.75))),
        2e-14,
    )
    constant_clebsch = np.tile((0.25, -0.5, 1.5, 2.0), (face_count, 1))
    smoothed_clebsch = face_fields_to_vertex_fields(constant_clebsch, operators)
    check(
        "display projection preserves Clebsch fields",
        np.max(np.abs(smoothed_clebsch - (0.25, -0.5, 1.5, 2.0))),
        2e-14,
    )
    test_vorticity = np.array((1.0, -2.0, 0.5))
    test_areas = np.array((0.25, 0.5, 1.5))
    test_changes = relative_vorticity_l2_change(
        np.vstack((test_vorticity, 1.25 * test_vorticity)),
        test_vorticity,
        test_areas,
    )
    check(
        "vorticity metric is relative weighted L2",
        np.max(np.abs(test_changes - (0.0, 25.0))),
        2e-14,
    )
    check("incidence exactness ||d1 d0||", operators["exactness_error"], 1e-14)
    vertex_xy = operators["vertices"][:, :2]
    stream_function = np.sin(vertex_xy[:, 0]) * np.sin(vertex_xy[:, 1])
    exact_tgv_flux = np.asarray(operators["d0"] @ stream_function).ravel()
    exact_tgv_flux[operators["boundary_edges"]] = 0.0
    check(
        "exact Taylor--Green edge flux obeys constraints",
        max(
            np.linalg.norm(operators["d1"] @ exact_tgv_flux, np.inf),
            np.max(
                np.abs(exact_tgv_flux[operators["boundary_edges"]]),
                initial=0.0,
            ),
        ),
        2e-13,
    )
    check(
        "dual gradient annihilates constants",
        np.linalg.norm(dual_gradient(np.ones(face_count), operators), np.inf),
        1e-14,
    )
    check(
        "interior conductances are positive",
        float(np.any(operators["conductance"][operators["interior_edges"]] <= 0.0)),
        0.0,
    )

    raw_flux = rng.standard_normal(len(operators["edges"]))
    _, projected = project_edge_flux(raw_flux, operators)
    _, projected_twice = project_edge_flux(projected, operators)
    check("projection is divergence-free", relative_divergence(projected, operators), 2e-12)
    check(
        "projection enforces impermeable boundary",
        np.max(np.abs(projected[operators["boundary_edges"]]), initial=0.0),
        1e-14,
    )
    check(
        "projection is idempotent",
        np.linalg.norm(projected_twice - projected)
        / max(np.linalg.norm(projected), 1e-15),
        2e-12,
    )
    gradient_flux = dual_gradient(rng.standard_normal(face_count), operators)
    _, projected_gradient = project_edge_flux(gradient_flux, operators)
    check(
        "projection removes its discrete gradients",
        np.linalg.norm(projected_gradient)
        / max(np.linalg.norm(gradient_flux), 1e-15),
        2e-11,
    )

    transport = centered_advection_matrix(edge_flux, operators)
    face_mass = sp.diags(operators["dual_star0"])
    weak_transport = (face_mass @ transport).tocsr()
    ones = np.ones(face_count)
    operator_scale = max(np.linalg.norm(transport.data, np.inf), 1.0)
    weak_scale = max(np.linalg.norm(weak_transport.data), 1e-15)
    check(
        "face transport preserves constants",
        np.linalg.norm(transport @ ones, np.inf) / operator_scale,
        2e-12,
    )
    check(
        "face transport preserves scalar mass",
        np.linalg.norm(np.asarray(ones @ weak_transport).ravel(), np.inf)
        / weak_scale,
        2e-12,
    )
    check(
        "centered face transport is weighted-skew",
        np.linalg.norm((weak_transport + weak_transport.T).data) / weak_scale,
        2e-12,
    )
    check(
        "matrix and face-RHS implementations agree",
        np.linalg.norm(
            advection_rhs(clebsch, edge_flux, operators, settings)
            - transport @ clebsch,
            np.inf,
        )
        / max(np.linalg.norm(transport @ clebsch, np.inf), 1e-15),
        2e-12,
    )

    dt, rate = stable_time_step(edge_flux, operators, settings)
    check("adaptive step obeys requested CFL", max(0.0, dt * rate - settings.cfl), 2e-14)
    evolved_chart, evolved_material, evolved_dye, _, evolved_flux = (
        ssprk3_step(
            clebsch, clebsch.copy(), dye, min(dt, 0.001), operators, settings
        )
    )
    initial_mass = np.concatenate(
        (
            scalar_invariants(clebsch, operators)[0],
            (vertex_scalar_invariants(dye, operators)[0],),
        )
    )
    evolved_mass = np.concatenate(
        (
            scalar_invariants(evolved_material, operators)[0],
            (vertex_scalar_invariants(evolved_dye, operators)[0],),
        )
    )
    check(
        "coupled SSPRK3 preserves transported mass",
        np.linalg.norm(evolved_mass - initial_mass, np.inf)
        / max(np.linalg.norm(initial_mass, np.inf), 1e-15),
        2e-12,
    )

    reference_map = operators["face_centers"][:, :2].copy()
    reset_chart, _, reset_flux, _, transition = rechart(
        evolved_chart, evolved_flux, reference_map, operators
    )
    identity_error = np.max(
        np.abs(
            reset_chart[:, LAMBDA1:LAMBDA2 + 1]
            - operators["face_centers"][:, :2]
        )
    )
    check("rechart restores identity face labels", identity_error, 1e-14)
    check("rechart preserves composed labels", transition["global_label_jump"], 2e-12)
    check("rechart keeps velocity transfer small", transition["velocity_jump"], 2e-3)
    check(
        "recharted flux remains divergence-free",
        relative_divergence(reset_flux, operators),
        2e-12,
    )

    schedule_settings = replace(
        settings,
        resolution=6,
        final_time=0.02,
        frames=11,
        maximum_time_step=0.002,
        rechart_every_frames=5,
        jitter=0.05,
        random_seed=31,
    )
    schedule_operators = build_operators(schedule_settings)
    scheduled = simulate(schedule_operators, schedule_settings)
    expected_times = np.linspace(0.0, schedule_settings.final_time, schedule_settings.frames)
    scheduled_events = [
        event for event in scheduled["reset_events"]
        if event["reason"] == "frame interval"
    ]
    event_times = np.asarray([event["time"] for event in scheduled_events])
    schedule_error = (
        np.max(np.abs(event_times - np.array((0.01, 0.02))))
        if len(event_times) == 2
        else np.inf
    )
    check(
        "simulation stores every requested output",
        np.max(np.abs(scheduled["times"] - expected_times)),
        2e-14,
    )
    check("scheduled recharts occur once per interval", schedule_error, 2e-14)
    check(
        "stored simulation fluxes satisfy constraints",
        max(np.max(scheduled["divergence"]), np.max(scheduled["boundary_error"])),
        2e-12,
    )
    check(
        "stored vorticity changes start at zero",
        max(
            abs(scheduled["velocity_vorticity_change"][0]),
            abs(scheduled["clebsch_vorticity_change"][0]),
        ),
        1e-14,
    )

    # Cross the old full-metric failure time to catch both the energy
    # explosion and the former fake-output-time freeze.
    stability_settings = replace(
        settings,
        final_time=2.0,
        frames=81,
        cfl=0.25,
        maximum_time_step=0.02,
        rechart_cutoff=1.5,
        rechart_every_frames=5,
        determinant_floor=0.05,
        initial_condition=PERTURBED,
        perturbation_amplitude=0.25,
    )
    stability_operators = build_operators(stability_settings)
    stable = simulate(stability_operators, stability_settings)
    energy_ratio = stable["energy"] / stable["energy"][0]
    check(
        "nonlinear feedback stays finite",
        float(
            not np.all(np.isfinite(stable["edge_fluxes"]))
            or np.max(stable["maximum_speed"]) >= 10.0
        ),
        0.0,
    )
    check(
        "nonlinear kinetic energy stays bounded",
        max(0.0, np.max(np.abs(energy_ratio - 1.0)) - 0.5),
        0.0,
    )
    check(
        "nonlinear run reaches final time by steps",
        abs(stable["times"][-1] - stability_settings.final_time)
        + float(stable["number_of_steps"] <= 0),
        2e-14,
    )
    print(
        "Nonlinear T=2 energy range: "
        f"[{np.min(energy_ratio):.6f}, {np.max(energy_ratio):.6f}] E0; "
        f"steps={stable['number_of_steps']}"
    )

    passed = all(results)
    print("-" * 82)
    print(f"Mandatory tests: {sum(results)}/{len(results)} passed")
    return passed


def parse_arguments(*, fixed_initial_condition=None, description=None):
    """Parse the shared solver CLI, optionally locking one experiment IC."""
    if fixed_initial_condition is not None and fixed_initial_condition not in INITIAL_CONDITIONS:
        raise ValueError(
            f"fixed_initial_condition must be None or one of {INITIAL_CONDITIONS}"
        )
    defaults = Settings()
    if fixed_initial_condition is not None:
        defaults = replace(
            defaults,
            initial_condition=fixed_initial_condition,
            perturbation_amplitude=(
                0.0
                if fixed_initial_condition == TAYLOR_GREEN
                else defaults.perturbation_amplitude
            ),
        )
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resolution", type=int, default=defaults.resolution)
    parser.add_argument("--final-time", type=float, default=defaults.final_time)
    parser.add_argument("--frames", type=int, default=defaults.frames)
    parser.add_argument("--cfl", type=float, default=defaults.cfl)
    parser.add_argument("--maximum-time-step", type=float, default=defaults.maximum_time_step)
    parser.add_argument("--rechart-cutoff", type=float, default=defaults.rechart_cutoff)
    parser.add_argument(
        "--rechart-every-frames", type=int, default=defaults.rechart_every_frames,
        help="reset the local Clebsch chart every N stored-frame intervals; use 0 to disable",
    )
    parser.add_argument("--determinant-floor", type=float, default=defaults.determinant_floor)
    if fixed_initial_condition is None:
        parser.add_argument(
            "--initial-condition", choices=INITIAL_CONDITIONS,
            default=defaults.initial_condition,
        )
    else:
        parser.set_defaults(initial_condition=fixed_initial_condition)
    if fixed_initial_condition != TAYLOR_GREEN:
        parser.add_argument(
            "--perturbation-amplitude", type=float,
            default=defaults.perturbation_amplitude,
            help="epsilon in the perturbed Taylor--Green streamfunction",
        )
    else:
        parser.set_defaults(perturbation_amplitude=0.0)
    parser.add_argument(
        "--stabilization", choices=("centered", "graph"),
        default=defaults.stabilization,
    )
    parser.add_argument("--graph-viscosity", type=float, default=defaults.graph_viscosity)
    parser.add_argument("--jitter", type=float, default=defaults.jitter)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--arrow-stride", type=int, default=defaults.arrow_stride)
    parser.add_argument("--playback-fps", type=float, default=defaults.playback_fps)
    parser.add_argument("--headless", action="store_true", help="run diagnostics without opening Polyscope")
    parser.add_argument(
        "--diagnostics-plot", metavar="PATH",
        help=(
            "save a PNG time-series plot of relative divergence, kinetic-energy "
            "change, and the two relative vorticity L2 differences"
        ),
    )
    parser.add_argument(
        "--show-diagnostic-plots", action="store_true",
        help="show the diagnostic time-series plot in a Matplotlib window",
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="run the shared solver diagnostics for both initial conditions and exit",
    )
    arguments = parser.parse_args()
    if (
        arguments.final_time <= 0.0
        or arguments.frames < 2
        or not (0.0 < arguments.cfl <= 1.0)
        or arguments.maximum_time_step <= 0.0
        or arguments.rechart_every_frames < 0
        or arguments.graph_viscosity < 0.0
    ):
        parser.error(
            "require final-time > 0, frames >= 2, 0 < cfl <= 1, "
            "maximum-time-step > 0, rechart-every-frames >= 0, "
            "and graph-viscosity >= 0"
        )
    values = vars(arguments)
    headless = values.pop("headless")
    diagnostics_plot = values.pop("diagnostics_plot")
    show_diagnostic_plots = values.pop("show_diagnostic_plots")
    run_tests = values.pop("run_tests")
    return (
        Settings(**values), headless, run_tests,
        diagnostics_plot, show_diagnostic_plots,
    )


def main(*, fixed_initial_condition=None, description=None):
    (
        settings, headless, run_tests,
        diagnostics_plot, show_diagnostic_plots,
    ) = parse_arguments(
        fixed_initial_condition=fixed_initial_condition,
        description=description,
    )
    if run_tests:
        if not run_diagnostic_tests():
            raise SystemExit(1)
        return
    operators = build_operators(settings)
    result = simulate(operators, settings)
    print_report(result, operators, settings)
    if diagnostics_plot is not None or show_diagnostic_plots:
        plot_diagnostic_histories(
            result, settings,
            output_path=diagnostics_plot,
            show=show_diagnostic_plots,
        )
    if not headless:
        visualize(result, operators, settings)


if __name__ == "__main__":
    main()
