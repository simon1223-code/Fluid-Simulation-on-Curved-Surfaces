#!/usr/bin/env python3
"""Two-dimensional incompressible Euler prototype with face Clebsch fields.

Both ``(mu_1, lambda_1)`` and ``(mu_2, lambda_2)`` are point-valued dual
0-cochains stored on triangle faces.  One affine-exact least-squares gradient
reconstructs the physical Weber field, which is mapped to integrated normal
edge fluxes and RT0-mass Hodge-projected to enforce incompressibility and
impermeable walls.

The local Clebsch chart is reinitialized only by adaptive distortion criteria.
A constrained identity-chart KKT solve preserves the projected edge flux
exactly while matching the pre-reset Clebsch curl as closely as possible.
The primary vorticity diagnostic is the enstrophy of the weak curl recovered
from the actual projected velocity.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
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
    resolution: int = 80
    final_time: float = 10.0
    frames: int = 401
    cfl: float = 0.25
    maximum_time_step: float = 0.02
    rechart_cutoff: float = 1.5
    rechart_every_frames: int = 0
    determinant_floor: float = 0.05
    initial_condition: str = PERTURBED
    perturbation_amplitude: float = 0.25
    stabilization: str = "muscl"
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
    boundary_edge_indices = np.flatnonzero(boundary_edges)
    boundary_edge_faces = np.asarray(
        [edge_faces[edge_index][0] for edge_index in boundary_edge_indices],
        dtype=np.int64,
    )
    boundary_edge_signs = np.asarray(
        [d1[face_index, edge_index] for face_index, edge_index in
         zip(boundary_edge_faces, boundary_edge_indices)],
        dtype=float,
    )
    boundary_ccw_tangents = (
        boundary_edge_signs[:, None] * tangents[boundary_edge_indices]
    )

    edge01 = points[:, 1] - points[:, 0]
    edge02 = points[:, 2] - points[:, 0]
    denominator = edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0]
    p1_basis_derivatives = np.empty((number_of_faces, 3, 2))
    p1_basis_derivatives[:, 0] = np.column_stack(
        (points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0])
    ) / denominator[:, None]
    p1_basis_derivatives[:, 1] = np.column_stack(
        (points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0])
    ) / denominator[:, None]
    p1_basis_derivatives[:, 2] = np.column_stack(
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

    # Canonical physical gradient G_F for every face scalar. Grow each
    # weighted least-squares stencil until it spans the plane; this keeps the
    # reconstruction affine-exact, including on boundary triangles.
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
    gradient_x = sp.csr_matrix(
        (gradient_values[0], (gradient_rows[0], gradient_columns[0])),
        shape=(number_of_faces, number_of_faces),
    )
    gradient_y = sp.csr_matrix(
        (gradient_values[1], (gradient_rows[1], gradient_columns[1])),
        shape=(number_of_faces, number_of_faces),
    )
    face_gradient_operator = sp.vstack((gradient_x, gradient_y), format="csr")
    coordinate_samples = face_centers[:, :2]
    coordinate_gradients = np.asarray(
        face_gradient_operator @ coordinate_samples
    ).reshape(2, number_of_faces, 2).transpose(1, 2, 0)
    affine_gradient_error = float(
        np.max(np.abs(coordinate_gradients - np.eye(2)[None, :, :]))
    )

    # Affine-exact reconstruction R_{F->E}.  A plain arithmetic average of
    # neighboring face-centroid values is not consistent on a general
    # triangular mesh: for an affine scalar it need not equal the value at the
    # edge midpoint.  Reconstruct each incident face linearly to the midpoint
    # with G_F, then average the one-sided values.  Thus R reproduces every
    # affine scalar exactly at every edge midpoint.
    edge_midpoints = 0.5 * (
        vertices[edges[:, 0], :2] + vertices[edges[:, 1], :2]
    )
    face_edge_midpoint_offsets = (
        edge_midpoints[face_edge_indices] - face_centers[:, None, :2]
    )
    occurrence_faces = np.repeat(np.arange(number_of_faces), 3)
    occurrence_edges = face_edge_indices.ravel()
    occurrence_offsets = face_edge_midpoint_offsets.reshape(-1, 2)
    inverse_incidence_count = 1.0 / incident_face_count[occurrence_edges]
    edge_offset_x = sp.csr_matrix(
        (
            occurrence_offsets[:, 0] * inverse_incidence_count,
            (occurrence_edges, occurrence_faces),
        ),
        shape=(number_of_edges, number_of_faces),
    )
    edge_offset_y = sp.csr_matrix(
        (
            occurrence_offsets[:, 1] * inverse_incidence_count,
            (occurrence_edges, occurrence_faces),
        ),
        shape=(number_of_edges, number_of_faces),
    )
    face_to_edge_reconstruction = (
        face_edge_average + edge_offset_x @ gradient_x + edge_offset_y @ gradient_y
    ).tocsr()
    reconstructed_coordinates = np.asarray(
        face_to_edge_reconstruction @ coordinate_samples
    )
    affine_edge_reconstruction_error = float(
        np.max(np.abs(reconstructed_coordinates - edge_midpoints))
    )

    # N maps face-vector samples to globally oriented integrated normal edge
    # fluxes.  The vector components are first reconstructed to the edge
    # midpoint by the same affine-exact scalar operator R_{F->E}.
    normal_x = (
        sp.diags(edge_lengths * right_normals[:, 0])
        @ face_to_edge_reconstruction
    )
    normal_y = (
        sp.diags(edge_lengths * right_normals[:, 1])
        @ face_to_edge_reconstruction
    )
    face_vector_to_edge_flux = sp.hstack((normal_x, normal_y), format="csr")

    # With identity labels, affine exactness gives w_F=(mu_1,mu_2), so the
    # rechart recovery matrix is exactly the same face-vector-to-flux map N.
    identity_chart_flux_matrix = face_vector_to_edge_flux

    # With identity labels lambda_1=x and lambda_2=y, the Clebsch field is
    # w=(mu_1,mu_2), so its scalar curl is d_x mu_2 - d_y mu_1.  This
    # linear operator is used only by the optional vorticity-aware rechart fit.
    identity_chart_vorticity_matrix = sp.hstack(
        (-gradient_y, gradient_x), format="csr"
    )

    # Assemble the RT0 mass matrix for globally oriented integrated edge
    # fluxes. This supplies the physical metric for the constrained Hodge
    # projection and keeps that projection distinct from G_F.
    mass_rows = []
    mass_columns = []
    mass_values = []
    opposite_vertices = np.array((2, 0, 1))
    barycentric_quadrature = np.array(
        ((2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
         (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
         (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0))
    )
    for face_index in range(number_of_faces):
        quadrature_points = barycentric_quadrature @ points[face_index]
        basis_values = (
            quadrature_points[:, None, :]
            - points[face_index, opposite_vertices][None, :, :]
        ) / (2.0 * face_areas[face_index])
        local_mass = (
            face_areas[face_index]
            / 3.0
            * np.einsum("qic,qjc->ij", basis_values, basis_values)
        )
        signs = face_edge_signs[face_index]
        local_mass *= signs[:, None] * signs[None, :]
        local_ids = face_edge_indices[face_index]
        mass_rows.extend(np.repeat(local_ids, 3))
        mass_columns.extend(np.tile(local_ids, 3))
        mass_values.extend(local_mass.ravel())
    edge_flux_mass = sp.coo_matrix(
        (mass_values, (mass_rows, mass_columns)),
        shape=(number_of_edges, number_of_edges),
    ).tocsr()

    # Eliminate boundary fluxes, remove one dependent divergence equation per
    # connected face component, and factor the mixed RT0-P0 Hodge system once.
    incidence = abs(d1[:, interior_edges])
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    number_of_components, labels = connected_components(adjacency, directed=False)
    pinned_faces = np.array(
        [np.flatnonzero(labels == label)[0] for label in range(number_of_components)]
    )
    free_faces = np.setdiff1d(np.arange(number_of_faces), pinned_faces)
    divergence_constraints = d1[free_faces][:, interior_edges].tocsr()
    interior_flux_mass = edge_flux_mass[interior_edges][:, interior_edges].tocsc()
    zero_pressure_block = sp.csc_matrix((len(free_faces), len(free_faces)))
    hodge_system = sp.bmat(
        (
            (interior_flux_mass, divergence_constraints.T),
            (divergence_constraints, zero_pressure_block),
        ),
        format="csc",
    )
    hodge_factorization = splu(hodge_system)
    # Constrained rechart system. The new identity-chart coefficients c are
    # chosen to match the current edge flux exactly while minimizing a weighted
    # curl mismatch plus a tiny minimum-change regularizer.
    rechart_curl = identity_chart_vorticity_matrix.tocsr()
    rechart_flux = identity_chart_flux_matrix.tocsr()
    rechart_face_mass = sp.diags(face_areas)
    rechart_curl_normal = (
        rechart_curl.T @ rechart_face_mass @ rechart_curl
    ).tocsc()
    positive_diagonal = np.asarray(rechart_curl_normal.diagonal())
    positive_diagonal = positive_diagonal[positive_diagonal > 0.0]
    rechart_scale = float(np.mean(positive_diagonal)) if len(positive_diagonal) else 1.0
    rechart_beta = 1e-4 * max(rechart_scale, 1e-15)
    rechart_hessian = rechart_curl_normal + rechart_beta * sp.eye(
        2 * number_of_faces, format="csc"
    )
    rechart_system = sp.bmat(
        (
            (rechart_hessian, rechart_flux.T),
            (rechart_flux, sp.csc_matrix((number_of_edges, number_of_edges))),
        ),
        format="csc",
    )
    rechart_factorization = splu(rechart_system)
    rechart_curl_rhs = (rechart_curl.T @ rechart_face_mass).tocsr()

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
        "p1_basis_derivatives": p1_basis_derivatives,
        "d0": d0,
        "d1": d1,
        "star0": star0,
        # Mass/Hodge for point-valued dual 0-cochains.  This is deliberately
        # not named star2: primal integrated 2-cochains use the reciprocal-area
        # convention, while lambda and mu here are values at dual vertices.
        "dual_star0": face_areas,
        "boundary_edges": boundary_edges,
        "boundary_edge_indices": boundary_edge_indices,
        "boundary_edge_faces": boundary_edge_faces,
        "boundary_ccw_tangents": boundary_ccw_tangents,
        "interior_edges": interior_edges,
        "interior_edge_faces": interior_edge_faces,
        "face_edge_average": face_edge_average,
        "face_edge_midpoint_offsets": face_edge_midpoint_offsets,
        "face_to_edge_reconstruction": face_to_edge_reconstruction,
        "face_gradient": face_gradient_operator,
        "face_vector_to_edge_flux": face_vector_to_edge_flux,
        "edge_flux_mass": edge_flux_mass,
        "identity_chart_flux_matrix": identity_chart_flux_matrix,
        "identity_chart_vorticity_matrix": identity_chart_vorticity_matrix,
        "rechart_beta": rechart_beta,
        "rechart_curl_rhs": rechart_curl_rhs,
        "rechart_factorization": rechart_factorization,
        "hodge_factorization": hodge_factorization,
        "projection_free_faces": free_faces,
        "number_of_projection_components": number_of_components,
        "affine_gradient_error": affine_gradient_error,
        "affine_edge_reconstruction_error": affine_edge_reconstruction_error,
        "exactness_error": exactness_error,
    }


def edge_flux_to_face_vectors(edge_flux, operators):
    """Recover the constant face velocity represented by projected fluxes."""
    outward_flux = (
        operators["face_edge_signs"]
        * np.asarray(edge_flux)[operators["face_edge_indices"]]
    )
    return np.einsum("fij,fj->fi", operators["flux_to_face_vector"], outward_flux)


def face_vectors_to_edge_flux(face_vectors, operators):
    """Apply N: centered face vectors to integrated global-normal edge flux."""
    face_vectors = np.asarray(face_vectors)
    expected_shape = (len(operators["faces"]), 2)
    if face_vectors.shape != expected_shape:
        raise ValueError(
            f"face vectors must have shape {expected_shape}, got {face_vectors.shape}"
        )
    coefficients = np.concatenate((face_vectors[:, 0], face_vectors[:, 1]))
    return np.asarray(
        operators["face_vector_to_edge_flux"] @ coefficients
    ).ravel()


def face_gradient(face_fields, operators):
    """Apply the affine-exact face gradient G_F to one or more scalars."""
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    if face_fields.ndim != 2 or face_fields.shape[0] != len(operators["faces"]):
        raise ValueError("face fields must have one row per face")
    number_of_faces, number_of_fields = face_fields.shape
    derivatives = np.asarray(operators["face_gradient"] @ face_fields)
    gradients = derivatives.reshape(2, number_of_faces, number_of_fields)
    gradients = gradients.transpose(1, 2, 0)
    return gradients[:, 0] if scalar_input else gradients


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


def clebsch_face_velocity(chart_fields, operators):
    """Return w_F = sum_a mu_a G_F(lambda_a) on each face."""
    chart_fields = np.asarray(chart_fields)
    expected_shape = (len(operators["faces"]), CLEBSCH_FIELD_COUNT)
    if chart_fields.shape != expected_shape:
        raise ValueError(
            f"chart_fields must contain face-based Clebsch data with shape "
            f"{expected_shape}, got {chart_fields.shape}"
        )
    lambda_gradients = face_gradient(
        chart_fields[:, LAMBDA1 : LAMBDA2 + 1], operators
    )
    raw_face_velocity = np.sum(
        chart_fields[:, MU1 : MU2 + 1, None] * lambda_gradients,
        axis=1,
    )
    return raw_face_velocity


def reconstruct_and_project(chart_fields, operators):
    """Reconstruct face Clebsch velocity, map it to flux, and project it."""
    raw_face_velocity = clebsch_face_velocity(chart_fields, operators)
    raw_edge_flux = face_vectors_to_edge_flux(raw_face_velocity, operators)
    return project_edge_flux(raw_edge_flux, operators)


def project_edge_flux(raw_edge_flux, operators):
    """RT0-mass Hodge projection onto divergence-free, wall-tangent fluxes."""
    raw_edge_flux = np.asarray(raw_edge_flux).ravel()
    if raw_edge_flux.shape != (len(operators["edges"]),):
        raise ValueError("raw_edge_flux must have one value per edge")
    interior_edges = operators["interior_edges"]
    free_faces = operators["projection_free_faces"]
    weighted_raw_flux = np.asarray(
        operators["edge_flux_mass"] @ raw_edge_flux
    ).ravel()[interior_edges]
    right_hand_side = np.concatenate(
        (weighted_raw_flux, np.zeros(len(free_faces)))
    )
    solution = operators["hodge_factorization"].solve(right_hand_side)
    projected_edge_flux = np.zeros(len(operators["edges"]))
    projected_edge_flux[interior_edges] = solution[: len(interior_edges)]
    projection_multiplier = np.zeros(len(operators["faces"]))
    projection_multiplier[free_faces] = solution[len(interior_edges) :]
    return projection_multiplier, projected_edge_flux


def centered_advection_matrix(edge_flux, operators):
    """Return the affine-exact centered transport generator for face scalars."""
    weak_operator = (
        -operators["d1"]
        @ sp.diags(np.asarray(edge_flux))
        @ operators["face_to_edge_reconstruction"]
    ).tocsr()
    return (sp.diags(1.0 / operators["dual_star0"]) @ weak_operator).tocsr()


def initial_stream_function(x, y, settings: Settings):
    """Streamfunction whose rotated gradient gives the built-in initial velocity."""
    if settings.initial_condition == TAYLOR_GREEN:
        epsilon = 0.0
    elif settings.initial_condition == PERTURBED:
        epsilon = settings.perturbation_amplitude
    else:
        raise ValueError(
            f"unknown initial condition {settings.initial_condition!r}; "
            f"expected one of {INITIAL_CONDITIONS}"
        )
    return np.sin(x) * np.sin(y) + epsilon * np.sin(2.0 * x) * np.sin(3.0 * y)


def exact_initial_edge_flux(operators, settings: Settings):
    """Exact integrated normal flux of the built-in streamfunction velocity.

    With globally oriented tangent t and right normal n=(t_y,-t_x),
    u=(psi_y,-psi_x) satisfies int_e u.n ds = psi(v_1)-psi(v_0).
    Thus d0 applied to vertex streamfunction samples gives a divergence-free
    wall-tangent initial edge flux up to roundoff.
    """
    xy = operators["vertices"][:, :2]
    psi = initial_stream_function(xy[:, 0], xy[:, 1], settings)
    flux = np.asarray(operators["d0"] @ psi).ravel()
    flux[operators["boundary_edges"]] = 0.0
    return flux


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


def initial_vorticity(x, y, settings: Settings):
    """Analytic initial scalar vorticity curl(u_0) for the built-in tests."""
    if settings.initial_condition == TAYLOR_GREEN:
        epsilon = 0.0
    elif settings.initial_condition == PERTURBED:
        epsilon = settings.perturbation_amplitude
    else:
        raise ValueError(
            f"unknown initial condition {settings.initial_condition!r}; "
            f"expected one of {INITIAL_CONDITIONS}"
        )
    return (
        2.0 * np.sin(x) * np.sin(y)
        + 13.0 * epsilon * np.sin(2.0 * x) * np.sin(3.0 * y)
    )


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


def muscl_upwind_edge_values(transported_fields, edge_flux, operators):
    """Slope-limited second-order upwind values at primal-edge midpoints.

    Face values are treated as centroid samples.  The same affine-exact G_F
    used by Clebsch reconstruction provides slopes.  A Barth--Jespersen-style
    scalar limiter keeps every reconstructed edge value inside the range of
    the current face and its edge-neighbors, suppressing the grid-scale
    oscillations produced by the formerly centered nonlinear transport.
    """
    transported_fields = np.asarray(transported_fields)
    number_of_faces, number_of_fields = transported_fields.shape
    gradients = face_gradient(transported_fields, operators)
    local_increments = np.einsum(
        "fsi,fei->fes", gradients, operators["face_edge_midpoint_offsets"]
    )

    local_minimum = transported_fields.copy()
    local_maximum = transported_fields.copy()
    edge_faces = operators["interior_edge_faces"]
    first_faces = edge_faces[:, 0]
    second_faces = edge_faces[:, 1]
    np.minimum.at(local_minimum, first_faces, transported_fields[second_faces])
    np.minimum.at(local_minimum, second_faces, transported_fields[first_faces])
    np.maximum.at(local_maximum, first_faces, transported_fields[second_faces])
    np.maximum.at(local_maximum, second_faces, transported_fields[first_faces])

    limiter = np.ones((number_of_faces, number_of_fields))
    tolerance = 1e-14
    for local_edge in range(3):
        increment = local_increments[:, local_edge, :]
        ratio = np.ones_like(increment)
        positive = increment > tolerance
        negative = increment < -tolerance
        ratio[positive] = (
            (local_maximum - transported_fields)[positive] / increment[positive]
        )
        ratio[negative] = (
            (local_minimum - transported_fields)[negative] / increment[negative]
        )
        limiter = np.minimum(limiter, np.clip(ratio, 0.0, 1.0))

    local_edge_values = (
        transported_fields[:, None, :]
        + limiter[:, None, :] * local_increments
    )
    local_edge_indices = operators["face_edge_indices"]
    local_edge_signs = operators["face_edge_signs"]
    local_outward_flux = np.asarray(edge_flux)[local_edge_indices] * local_edge_signs

    edge_values = np.zeros((len(operators["edges"]), number_of_fields))
    selected_count = np.zeros(len(operators["edges"]))
    upwind = local_outward_flux > 0.0
    np.add.at(edge_values, local_edge_indices[upwind], local_edge_values[upwind])
    np.add.at(selected_count, local_edge_indices[upwind], 1.0)

    # Projected wall fluxes and exact stagnation edges carry no advective flux,
    # so their scalar value is immaterial.  Fill them with the affine-exact
    # centered value to keep the returned array finite and useful diagnostically.
    unselected = selected_count == 0.0
    if np.any(unselected):
        centered_values = (
            operators["face_to_edge_reconstruction"] @ transported_fields
        )
        edge_values[unselected] = centered_values[unselected]
    return edge_values


def face_advection_rhs(
    transported_fields, edge_flux, operators, *, stabilization, graph_viscosity
):
    """Conservative finite-volume advection for one or more face scalars."""
    transported_fields = np.asarray(transported_fields)
    if transported_fields.ndim == 1:
        transported_fields = transported_fields[:, None]
        squeeze = True
    else:
        squeeze = False
    if transported_fields.shape[0] != len(operators["faces"]):
        raise ValueError("face-transported fields must have one row per face")

    flux = np.asarray(edge_flux).ravel()
    if stabilization == "muscl":
        edge_values = muscl_upwind_edge_values(
            transported_fields, flux, operators
        )
        weak_rhs = np.asarray(
            -operators["d1"] @ (flux[:, None] * edge_values)
        )
    elif stabilization == "skew":
        # Affine-consistent skew-adjoint split form.  Let
        #   B = -D diag(F) R,
        # where R reconstructs face values to edge midpoints.  For a projected
        # flux D F = 0 with F=0 on the boundary, the continuum transport
        # operator is skew-adjoint.  We enforce the same algebraic property by
        # using K=(B-B^T)/2.  Hence q^T K q=0 and, because K 1=0, both
        # face-mass and the semidiscrete quadratic invariant are preserved.
        reconstruction = operators["face_to_edge_reconstruction"]
        incidence = operators["d1"]
        edge_values = reconstruction @ transported_fields
        forward = np.asarray(-incidence @ (flux[:, None] * edge_values))
        adjoint = np.asarray(
            -reconstruction.T @ (flux[:, None] * (incidence.T @ transported_fields))
        )
        weak_rhs = 0.5 * (forward - adjoint)
    else:
        edge_values = operators["face_to_edge_reconstruction"] @ transported_fields
        weak_rhs = np.asarray(
            -operators["d1"] @ (flux[:, None] * edge_values)
        )
    if stabilization == "graph" and graph_viscosity > 0.0:
        edge_faces = operators["interior_edge_faces"]
        diffusion_strength = (
            graph_viscosity
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


def advection_rhs(transported_fields, edge_flux, operators, settings: Settings):
    """Advect the face Clebsch state with the selected solver stabilization."""
    return face_advection_rhs(
        transported_fields, edge_flux, operators,
        stabilization=settings.stabilization,
        graph_viscosity=settings.graph_viscosity,
    )


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
        "fij,fj->fi", operators["p1_basis_derivatives"], face_velocity
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
    """Advect chart/material fields and passive dye with the chart velocity."""
    chart = face_state[:, :CLEBSCH_FIELD_COUNT]
    _, edge_flux = reconstruct_and_project(chart, operators)
    return (
        advection_rhs(face_state, edge_flux, operators, settings),
        vertex_advection_rhs(dye, edge_flux, operators, settings),
    )


def ssprk3_step(
    chart_fields, material_fields, dye, time_step, operators, settings: Settings
):
    """Advance Clebsch state and passive dye with SSPRK3 stage velocities."""
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
    projection_multiplier, edge_flux = reconstruct_and_project(next_chart, operators)
    return next_chart, next_material, next_dye, projection_multiplier, edge_flux

def chart_metrics(chart_fields, operators):
    gradients = face_gradient(chart_fields, operators)
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
    gradients = face_gradient(face_fields, operators)
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
    gradients = face_gradient(chart_fields, operators)
    result = np.zeros(len(operators["faces"]))
    for mu_index, lambda_index in ((MU1, LAMBDA1), (MU2, LAMBDA2)):
        grad_mu = gradients[:, mu_index]
        grad_lambda = gradients[:, lambda_index]
        result += grad_mu[:, 0] * grad_lambda[:, 1] - grad_mu[:, 1] * grad_lambda[:, 0]
    return result


def weak_rt0_vertex_vorticity(edge_flux, operators):
    """Mass-lumped P1 weak curl of the projected RT0/H(div) velocity.

    A divergence-free RT0 field is constant inside each triangle, so taking a
    strong facewise curl would return zero and differentiating a separately
    smoothed face reconstruction amplifies O(h) velocity error.  Instead use
    the distributional identity

        int omega phi = int (u_x phi_y - u_y phi_x)
                        + int_boundary phi (u . t),

    project it to vertex P1 values with the barycentric lumped mass, and average
    those vertex values back to triangle centroids for facewise diagnostics.
    """
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    face_areas = operators["face_areas"]
    basis_gradients = operators["p1_basis_derivatives"]
    faces = operators["faces"]

    face_weak_curl = face_areas[:, None] * (
        face_velocity[:, 0, None] * basis_gradients[:, :, 1]
        - face_velocity[:, 1, None] * basis_gradients[:, :, 0]
    )
    vertex_rhs = np.zeros(len(operators["vertices"]))
    np.add.at(vertex_rhs, faces.ravel(), face_weak_curl.ravel())

    boundary_edges = operators["boundary_edge_indices"]
    if len(boundary_edges):
        boundary_velocity = face_velocity[operators["boundary_edge_faces"]]
        boundary_circulation = (
            operators["edge_lengths"][boundary_edges]
            * np.sum(
                boundary_velocity * operators["boundary_ccw_tangents"], axis=1
            )
        )
        endpoints = operators["edges"][boundary_edges]
        np.add.at(vertex_rhs, endpoints[:, 0], 0.5 * boundary_circulation)
        np.add.at(vertex_rhs, endpoints[:, 1], 0.5 * boundary_circulation)

    return vertex_rhs / np.maximum(operators["star0"], 1e-15)


def recovered_velocity_vorticity(edge_flux, operators):
    """Face samples of the weak RT0 curl, used only for visualization."""
    vertex_vorticity = weak_rt0_vertex_vorticity(edge_flux, operators)
    return np.mean(vertex_vorticity[operators["faces"]], axis=1)


def physical_enstrophy(edge_flux, operators):
    """Discrete enstrophy 1/2 * omega^T M_lumped omega from projected velocity."""
    omega = weak_rt0_vertex_vorticity(edge_flux, operators)
    return 0.5 * float(operators["star0"] @ (omega * omega))


def relative_divergence(edge_flux, operators):
    residual = np.asarray(operators["d1"] @ edge_flux).ravel()
    scale = np.asarray(abs(operators["d1"]) @ np.abs(edge_flux)).ravel()
    return np.linalg.norm(residual, np.inf) / max(np.linalg.norm(scale, np.inf), 1e-15)


def kinetic_energy(edge_flux, operators):
    edge_flux = np.asarray(edge_flux)
    return 0.5 * float(edge_flux @ (operators["edge_flux_mass"] @ edge_flux))


def relative_vorticity_l2_error(vorticity_history, face_areas):
    """Eulerian field displacement, not a 2D Euler conservation error.

    Returns 100 ||omega(.,t)-omega(.,0)||_L2 / ||omega(.,0)||_L2.  A general
    incompressible Euler solution advects vorticity, so this quantity may be
    large even for an exact solution.
    """
    vorticity_history = np.asarray(vorticity_history)
    initial_field = vorticity_history[0]
    delta_vorticity = vorticity_history - initial_field
    delta_norm = np.sqrt(
        np.sum(face_areas * delta_vorticity * delta_vorticity, axis=-1)
    )
    initial_norm = np.sqrt(np.sum(face_areas * initial_field**2))
    return 100.0 * delta_norm / max(float(initial_norm), 1e-15)


def relative_enstrophy_change(vorticity_history, face_areas):
    """Percent change of integral omega^2, a genuine 2D Euler Casimir."""
    vorticity_history = np.asarray(vorticity_history)
    enstrophy = np.sum(face_areas * vorticity_history * vorticity_history, axis=-1)
    return 100.0 * (enstrophy / max(float(enstrophy[0]), 1e-15) - 1.0)


def material_vorticity_relative_l2_error(
    vorticity_history, material_fields_history, face_areas, settings: Settings
):
    """Compare omega(x,t) with the advected target omega_0(A(x,t))."""
    vorticity_history = np.asarray(vorticity_history)
    material_fields_history = np.asarray(material_fields_history)
    labels = material_fields_history[:, :, LAMBDA1:LAMBDA2 + 1]
    target = initial_vorticity(labels[:, :, 0], labels[:, :, 1], settings)
    difference = vorticity_history - target
    numerator = np.sqrt(np.sum(face_areas * difference * difference, axis=-1))
    denominator = np.sqrt(np.sum(face_areas * target * target, axis=-1))
    return 100.0 * numerator / np.maximum(denominator, 1e-15)


def relative_l2_to_target(field_history, target_history, face_areas):
    """Return 100 ||field-target||_L2 / ||target||_L2 at each stored time."""
    field_history = np.asarray(field_history)
    target_history = np.asarray(target_history)
    difference = field_history - target_history
    numerator = np.sqrt(np.sum(face_areas * difference * difference, axis=-1))
    denominator = np.sqrt(np.sum(face_areas * target_history * target_history, axis=-1))
    return 100.0 * numerator / np.maximum(denominator, 1e-15)


def relative_l2_between_histories(first, second, face_areas):
    """Return 100 ||first-second||_L2 / ||second||_L2 at each stored time."""
    return relative_l2_to_target(first, second, face_areas)


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


def rechart(chart_fields, edge_flux, reference_to_initial, operators, settings: Settings):
    """Constrained identity-chart reparameterization.

    The new labels are x,y. Their coefficient vector c=(mu_1,mu_2) is chosen
    by a fixed sparse KKT solve that enforces N c = F exactly, where F is the
    current projected edge flux, while minimizing the discrepancy between the
    new identity-chart curl C c and the pre-reset Clebsch curl. A tiny
    minimum-change term selects a stable representative.
    """
    edge_flux_before = np.asarray(edge_flux).copy()
    global_labels_before = compose_planar_backmap(
        chart_fields[:, LAMBDA1:LAMBDA2 + 1], reference_to_initial, operators
    )
    target_curl = clebsch_vorticity(chart_fields, operators)
    face_velocity = edge_flux_to_face_vectors(edge_flux_before, operators)
    baseline = np.concatenate((face_velocity[:, 0], face_velocity[:, 1]))

    coefficient_rhs = np.asarray(
        operators["rechart_curl_rhs"] @ target_curl
    ).ravel() + operators["rechart_beta"] * baseline
    rhs = np.concatenate((coefficient_rhs, edge_flux_before))
    solution = operators["rechart_factorization"].solve(rhs)
    number_of_faces = len(operators["faces"])
    coefficients = solution[: 2 * number_of_faces]
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("non-finite constrained Clebsch rechart")

    reset_chart = chart_fields.copy()
    reset_chart[:, MU1] = coefficients[:number_of_faces]
    reset_chart[:, MU2] = coefficients[number_of_faces:]
    reset_chart[:, LAMBDA1] = operators["face_centers"][:, 0]
    reset_chart[:, LAMBDA2] = operators["face_centers"][:, 1]

    projection_multiplier, edge_flux_after = reconstruct_and_project(
        reset_chart, operators
    )
    new_reference_to_initial = global_labels_before.copy()
    global_labels_after = compose_planar_backmap(
        reset_chart[:, LAMBDA1:LAMBDA2 + 1], new_reference_to_initial, operators
    )

    raw_flux_after = face_vectors_to_edge_flux(
        reset_chart[:, MU1:MU2 + 1], operators
    )
    raw_flux_residual = np.linalg.norm(
        raw_flux_after - edge_flux_before
    ) / max(np.linalg.norm(edge_flux_before), 1e-15)
    relative_velocity_jump = np.linalg.norm(
        edge_flux_after - edge_flux_before
    ) / max(np.linalg.norm(edge_flux_before), 1e-15)
    energy_before = kinetic_energy(edge_flux_before, operators)
    relative_energy_jump = (
        kinetic_energy(edge_flux_after, operators) - energy_before
    ) / max(energy_before, 1e-15)

    curl_after = clebsch_vorticity(reset_chart, operators)
    face_areas = operators["face_areas"]
    curl_scale = np.sqrt(np.sum(face_areas * target_curl**2))
    clebsch_curl_jump = np.sqrt(
        np.sum(face_areas * (curl_after - target_curl)**2)
    ) / max(float(curl_scale), 1e-15)

    transition = {
        "velocity_jump": float(relative_velocity_jump),
        "energy_jump": float(relative_energy_jump),
        "raw_flux_residual": float(raw_flux_residual),
        "clebsch_curl_jump": float(clebsch_curl_jump),
        "global_label_jump": float(
            np.max(
                np.linalg.norm(global_labels_after - global_labels_before, axis=1),
                initial=0.0,
            )
        ),
    }
    return (
        reset_chart, projection_multiplier, edge_flux_after,
        new_reference_to_initial, transition,
    )

def simulate(operators, settings: Settings):
    material_fields, dye = initialize_fields(operators, settings)
    chart_fields = material_fields.copy()
    centers = operators["face_centers"]
    reference_to_initial = centers[:, :2].copy()

    # Start from the analytic identity Clebsch chart itself.  Do not perform an
    # LSQR calibration to the exact edge flux: that made the projected velocity
    # nearly exact while substantially corrupting the hidden Clebsch curl.
    projection_multiplier, edge_flux = reconstruct_and_project(chart_fields, operators)
    exact_flux = exact_initial_edge_flux(operators, settings)
    initial_flux_transfer_error = np.linalg.norm(
        edge_flux - exact_flux
    ) / max(np.linalg.norm(exact_flux), 1e-15)
    initial_transition = None

    output_times = np.linspace(0.0, settings.final_time, settings.frames)
    histories = {
        "times": [], "material_fields": [], "dye": [],
        "projection_multiplier": [], "edge_fluxes": [],
        "composition_error": [], "local_clebsch_vorticity": [],
        "velocity_vorticity": [],
        "log_sigma_max": [], "log_condition": [], "determinant": [],
        "material_max_log_condition": [], "divergence": [], "boundary_error": [],
        "energy": [], "physical_enstrophy": [], "mass": [], "l2": [],
        "maximum_speed": [], "step_cfl": [],
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
        _, material_log_condition, _, _ = chart_metrics(material_fields, operators)
        composed_labels = compose_planar_backmap(
            chart_fields[:, LAMBDA1:LAMBDA2 + 1],
            reference_to_initial, operators,
        )
        material_labels = material_fields[:, LAMBDA1:LAMBDA2 + 1]
        omega_local = clebsch_vorticity(chart_fields, operators)
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
        histories["projection_multiplier"].append(projection_multiplier.copy())
        histories["edge_fluxes"].append(edge_flux.copy())
        histories["local_clebsch_vorticity"].append(omega_local)
        histories["velocity_vorticity"].append(omega_u)
        histories["log_sigma_max"].append(log_sigma)
        histories["log_condition"].append(log_condition)
        histories["determinant"].append(determinant)
        histories["material_max_log_condition"].append(float(np.max(material_log_condition)))
        histories["divergence"].append(relative_divergence(edge_flux, operators))
        histories["boundary_error"].append(float(
            np.max(np.abs(edge_flux[operators["boundary_edges"]]), initial=0.0)
        ))
        histories["energy"].append(kinetic_energy(edge_flux, operators))
        histories["physical_enstrophy"].append(physical_enstrophy(edge_flux, operators))
        histories["mass"].append(mass)
        histories["l2"].append(l2)
        histories["maximum_speed"].append(float(np.max(np.linalg.norm(face_velocity, axis=1))))
        histories["step_cfl"].append(maximum_start_state_cfl_since_output)
        histories["chart"].append(chart_index)
        histories["chart_age"].append(current_time - chart_start)
        maximum_start_state_cfl_since_output = 0.0

    def apply_rechart(reason, log_sigma, determinant):
        nonlocal chart_fields, projection_multiplier, edge_flux
        nonlocal reference_to_initial, chart_index, chart_start, last_rechart_time
        (
            chart_fields, projection_multiplier, edge_flux,
            reference_to_initial, transition,
        ) = rechart(
            chart_fields, edge_flux, reference_to_initial, operators, settings
        )
        chart_index += 1
        chart_start = current_time
        last_rechart_time = current_time
        reset_events.append({
            "time": current_time, "chart": chart_index, "reason": reason,
            "pre_log_sigma_max": float(np.max(log_sigma)),
            "pre_min_determinant": float(np.min(determinant)),
            **transition,
        })

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
            maximum_speed = float(np.max(np.linalg.norm(
                edge_flux_to_face_vectors(edge_flux, operators), axis=1
            )))
            raise FloatingPointError(
                "CFL step collapsed before the next output: "
                f"t={current_time:.12g}, dt={dt:.3e}, "
                f"proposed_dt={proposed_dt:.3e}, max_speed={maximum_speed:.3e}"
            )
        (
            chart_fields, material_fields, dye, projection_multiplier, edge_flux,
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
            and np.all(np.isfinite(projection_multiplier))
            and np.all(np.isfinite(edge_flux))
        )
        if not finite_state:
            raise FloatingPointError(
                f"solution became non-finite at step {step}, t={current_time:.6g}"
            )

        log_sigma, _, determinant, sigma_min = chart_metrics(chart_fields, operators)
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
                log_sigma, _, determinant, _ = chart_metrics(chart_fields, operators)
                apply_rechart("frame interval", log_sigma, determinant)
            store()
            next_output += 1

    result = {key: np.asarray(value) for key, value in histories.items()}
    areas = operators["face_areas"]
    result["physical_enstrophy_change"] = 100.0 * (
        result["physical_enstrophy"]
        / max(float(result["physical_enstrophy"][0]), 1e-15)
        - 1.0
    )
    result["initial_flux_transfer_error"] = float(initial_flux_transfer_error)
    result["initial_rechart_transition"] = initial_transition
    for key in (
        "material_fields", "dye", "composition_error",
        "projection_multiplier", "edge_fluxes", "local_clebsch_vorticity",
        "velocity_vorticity", "log_sigma_max", "log_condition", "determinant",
    ):
        result[key] = result[key].astype(np.float32)
    result["reset_events"] = reset_events
    result["number_of_steps"] = step
    return result

def percentage_change(values, frame=-1):
    return 100.0 * (values[frame] / max(abs(values[0]), 1e-15) - 1.0)


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
        "Speed |u|",
        "Physical vorticity (weak RT0 curl)",
        "Local-chart Clebsch vorticity (debug)",
        "Velocity change",
        "log(sigma_max)",
        "log(condition number)",
        "det(Dlambda_local)-1",
        "mu_1 (material, face-based)",
        "mu_2 (material, face-based)",
        "lambda_1 = Psi_0 X (face-based)",
        "lambda_2 = Psi_0 Y (face-based)",
        "Persistent dye (vertex-based)",
        "Hodge constraint multiplier",
        "Chart-composition label error",
    )
    speed_view = 0
    velocity_change_view = 3
    dye_view = 11
    material_views = {7: MU1, 8: MU2, 9: LAMBDA1, 10: LAMBDA2}
    scalar_data = (
        face_speed,
        result["velocity_vorticity"],
        result["local_clebsch_vorticity"],
        np.linalg.norm(face_velocity_change, axis=2),
        result["log_sigma_max"], result["log_condition"], determinant_error,
        result["material_fields"][:, :, MU1],
        result["material_fields"][:, :, MU2],
        result["material_fields"][:, :, LAMBDA1],
        result["material_fields"][:, :, LAMBDA2],
        result["dye"], result["projection_multiplier"], result["composition_error"],
    )
    vertex_fields = {dye_view}
    sequential_fields = {0, velocity_change_view, 5, dye_view, 13}
    defined_on = tuple(
        "vertices" if index in vertex_fields else "faces"
        for index in range(len(names))
    )
    color_maps = tuple(
        "viridis" if index in sequential_fields else "coolwarm"
        for index in range(len(names))
    )
    limits = [
        visualization_limits(data, index not in sequential_fields)
        for index, data in enumerate(scalar_data)
    ]
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
            psim.TextUnformatted(f"SSPRK3; Clebsch stabilization: {settings.stabilization}")
            psim.TextUnformatted(
                "The Hodge multiplier enforces divergence; it is not a physical gradient"
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
                "Physical enstrophy change: "
                f"{result['physical_enstrophy_change'][frame]:+.3e}%"
            )
            psim.TextUnformatted(
                "Max log(condition(Dlambda_material)): "
                f"{result['material_max_log_condition'][frame]:.3e}"
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
            psim.TextUnformatted("Rechart mode: adaptive only" if settings.rechart_every_frames == 0 else f"Scheduled rechart interval: {settings.rechart_every_frames} frames")
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
    centers = operators["face_centers"]
    analytic = initial_velocity(centers[:, 0], centers[:, 1], settings)
    initial_material, _ = initialize_fields(operators, settings)
    raw_velocity = clebsch_face_velocity(initial_material, operators)
    projected_velocity = edge_flux_to_face_vectors(result["edge_fluxes"][0], operators)
    analytic_norm_squared = np.sum(
        operators["face_areas"] * np.sum(analytic * analytic, axis=1)
    )
    raw_error = np.sqrt(np.sum(
        operators["face_areas"] * np.sum((raw_velocity - analytic) ** 2, axis=1)
    ) / max(analytic_norm_squared, 1e-15))
    projected_error = np.sqrt(np.sum(
        operators["face_areas"] * np.sum((projected_velocity - analytic) ** 2, axis=1)
    ) / max(analytic_norm_squared, 1e-15))

    print(f"Initial condition:              {initial_condition_label(settings)}")
    print(f"Vertices / edges / triangles:   {len(operators['vertices']):,} / {len(operators['edges']):,} / {len(operators['faces']):,}")
    print(
        "Mesh min angle / max aspect:   "
        f"{np.min(operators['minimum_angles']):.3f} deg / "
        f"{np.max(operators['aspect_ratios']):.3f}"
    )
    print(f"G_F coordinate affine error:    {operators['affine_gradient_error']:.6e}")
    print(f"R_F->E affine edge error:       {operators['affine_edge_reconstruction_error']:.6e}")
    print(f"SSPRK3 accepted steps:          {result['number_of_steps']:,}")
    print(f"Maximum accepted start CFL:     {np.max(result['step_cfl']):.6e}")
    print(f"Clebsch stabilization:          {settings.stabilization}")
    print(f"Scheduled rechart interval:     {settings.rechart_every_frames} frames")
    print(f"Initial raw Clebsch error:       {raw_error:.6e}")
    print(f"Initial projected-velocity err: {projected_error:.6e}")
    print(f"Initial edge-flux transfer err: {result['initial_flux_transfer_error']:.6e}")
    print(f"Maximum relative divergence:    {np.max(result['divergence']):.6e}")
    print(f"Maximum boundary-normal error:  {np.max(result['boundary_error']):.6e}")
    print(f"Kinetic-energy change:          {percentage_change(result['energy']):+.6e}%")
    print(
        "Physical enstrophy change:            "
        f"{result['physical_enstrophy_change'][-1]:+.6e}%"
    )
    print(
        "Max persistent-map log condition:    "
        f"{np.max(result['material_max_log_condition']):.6e}"
    )

    clebsch_initial_l1 = np.sum(
        operators["dual_star0"][:, None] * np.abs(result["material_fields"][0]), axis=0
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
        maximum_raw_flux_residual = max(event["raw_flux_residual"] for event in result["reset_events"])
        maximum_curl_jump = max(event["clebsch_curl_jump"] for event in result["reset_events"])
        print(f"Maximum reset velocity jump:    {maximum_velocity_jump:.6e}")
        print(f"Maximum reset energy jump:      {maximum_energy_jump:.6e}")
        print(f"Maximum global-label jump:      {maximum_label_jump:.6e}")
        print(f"Maximum reset raw-flux residual:{maximum_raw_flux_residual:.6e}")
        print(f"Maximum reset Clebsch-curl jump:{maximum_curl_jump:.6e}")

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
        "--stabilization", choices=("skew", "muscl", "centered", "graph"),
        default=defaults.stabilization,
    )
    parser.add_argument("--graph-viscosity", type=float, default=defaults.graph_viscosity)
    parser.add_argument("--jitter", type=float, default=defaults.jitter)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--arrow-stride", type=int, default=defaults.arrow_stride)
    parser.add_argument("--playback-fps", type=float, default=defaults.playback_fps)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening the interactive Polyscope viewer",
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
    return Settings(**values), headless


def main(*, fixed_initial_condition=None, description=None):
    settings, headless = parse_arguments(
        fixed_initial_condition=fixed_initial_condition,
        description=description,
    )
    operators = build_operators(settings)
    result = simulate(operators, settings)
    print_report(result, operators, settings)
    if not headless:
        visualize(result, operators, settings)


if __name__ == "__main__":
    main()
