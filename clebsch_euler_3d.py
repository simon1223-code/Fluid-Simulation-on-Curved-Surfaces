#!/usr/bin/env python3
"""Incompressible Clebsch flow on a closed triangle surface embedded in R^3.

The three pairs ``(mu_i, lambda_i)``, i=1,2,3, are dual 0-cochains stored
on faces.  Their dual-edge differences reconstruct an integrated conormal
edge flux, which is projected to satisfy exact finite-volume divergence.
The three initial labels are the ambient embedding coordinates ``(x,y,z)``;
this avoids the singularities of any global two-coordinate chart on the
Stanford bunny.

This is surface Euler on a two-dimensional manifold embedded in 3D, not
volumetric three-dimensional Euler.  The positive centroid-dual conductance
is the curved-surface analogue of the TPFA operator in ``clebsch_euler_2d``.
A resettable local chart reconstructs velocity while a never-reset material
copy and passive vertex dye are transported by the same SSPRK3 stages.
Display smoothing never feeds back into the simulation or diagnostics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr, splu
from scipy.spatial import cKDTree


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_BUNNY = PROJECT_DIRECTORY / "data_meshes" / "bunny.obj"
FIELD_NAMES = (
    "mu_1", "mu_2", "mu_3", "lambda_1", "lambda_2", "lambda_3", "dye"
)
MU1, MU2, MU3, LAMBDA1, LAMBDA2, LAMBDA3 = range(6)
MU = slice(MU1, MU3 + 1)
LAMBDA = slice(LAMBDA1, LAMBDA3 + 1)
CLEBSCH_FIELD_COUNT = 6
DYE_INDEX = 6


@dataclass(frozen=True)
class Settings:
    mesh_path: Path = DEFAULT_BUNNY
    final_time: float = 10
    frames: int = 101
    cfl: float = 0.20
    maximum_time_step: float = 0.005
    target_speed: float = 0.55
    rechart_cutoff: float = 1.5
    rechart_every_frames: int = 10
    singular_value_floor: float = 0.05
    dye_width_fraction: float = 0.10
    stabilization: str = "centered"
    graph_viscosity: float = 0.0
    arrow_stride: int = 10
    playback_fps: float = 20.0


def load_surface_mesh(path):
    """Load, merge, center, and normalize a closed triangular surface."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"surface mesh not found: {path}")

    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"mesh scene contains no triangle geometry: {path}")
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported mesh object in {path}")

    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh must contain only triangles")
    if not np.all(np.isfinite(vertices)) or not len(vertices) or not len(faces):
        raise ValueError("surface mesh must contain finite nonempty geometry")

    vertices = vertices.copy()
    vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if diagonal <= 1e-15:
        raise ValueError("surface mesh has zero spatial extent")
    vertices /= diagonal
    return vertices, faces, path


def triangle_quality(points):
    edge_vectors = np.stack(
        (points[:, 1] - points[:, 0], points[:, 2] - points[:, 1],
         points[:, 0] - points[:, 2]),
        axis=1,
    )
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    double_areas = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    )
    areas = 0.5 * double_areas
    altitudes = double_areas[:, None] / np.maximum(edge_lengths, 1e-15)
    minimum_altitudes = np.min(altitudes, axis=1)
    a, b, c = edge_lengths[:, 0], edge_lengths[:, 1], edge_lengths[:, 2]
    angles = np.column_stack(
        (
            np.arccos(np.clip((a * a + c * c - b * b) /
                              np.maximum(2.0 * a * c, 1e-15), -1.0, 1.0)),
            np.arccos(np.clip((a * a + b * b - c * c) /
                              np.maximum(2.0 * a * b, 1e-15), -1.0, 1.0)),
        )
    )
    third_angle = np.pi - angles[:, 0] - angles[:, 1]
    minimum_angles = np.degrees(
        np.min(np.column_stack((angles, third_angle)), axis=1)
    )
    aspect_ratios = np.max(edge_lengths, axis=1) / np.maximum(
        minimum_altitudes, 1e-15
    )
    return areas, minimum_altitudes, minimum_angles, aspect_ratios


def _face_gradient_matrices(face_centers, tangent_1, tangent_2, edge_faces):
    """Build intrinsic least-squares gradients for face-centered diagnostics."""
    face_count = len(face_centers)
    neighbors = [set() for _ in range(face_count)]
    for first_face, second_face in edge_faces:
        neighbors[first_face].add(second_face)
        neighbors[second_face].add(first_face)

    rows = ([], [])
    columns = ([], [])
    values = ([], [])
    for face_index in range(face_count):
        stencil = set(neighbors[face_index])
        frontier = set(stencil)
        while True:
            indices = np.fromiter(sorted(stencil), dtype=np.int64)
            offsets_3d = face_centers[indices] - face_centers[face_index]
            offsets = np.column_stack(
                (offsets_3d @ tangent_1[face_index], offsets_3d @ tangent_2[face_index])
            )
            if len(stencil) >= 3 and np.linalg.matrix_rank(offsets) == 2:
                break
            expanded = set()
            for neighbor in frontier:
                expanded.update(neighbors[neighbor])
            expanded.discard(face_index)
            expanded.difference_update(stencil)
            if not expanded:
                raise ValueError("could not build a full-rank surface gradient stencil")
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
            rows[component].extend(
                np.full(len(indices), face_index, dtype=np.int64)
            )
            columns[component].extend(indices)
            values[component].extend(component_coefficients)
            rows[component].append(face_index)
            columns[component].append(face_index)
            values[component].append(-np.sum(component_coefficients))

    matrices = []
    for component in range(2):
        matrices.append(
            sp.csr_matrix(
                (values[component], (rows[component], columns[component])),
                shape=(face_count, face_count),
            )
        )
    return tuple(matrices)


def build_operators(vertices, faces):
    """Construct the oriented surface complex and face-dual DEC operators."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_count = len(vertices)
    face_count = len(faces)
    points = vertices[faces]
    area_vectors = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    double_areas = np.linalg.norm(area_vectors, axis=1)
    face_areas, minimum_altitudes, minimum_angles, aspect_ratios = triangle_quality(points)
    if np.any(double_areas <= 1e-15):
        raise ValueError("surface mesh contains a degenerate triangle")
    face_normals = area_vectors / double_areas[:, None]
    face_centers = points.mean(axis=1)

    local_edges = np.stack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1
    ).reshape(-1, 2)
    edges, local_edge_indices = np.unique(
        np.sort(local_edges, axis=1), axis=0, return_inverse=True
    )
    edge_count = len(edges)
    face_indices = np.repeat(np.arange(face_count), 3)
    local_signs = np.where(local_edges[:, 0] < local_edges[:, 1], 1.0, -1.0)
    face_edge_indices = local_edge_indices.reshape(face_count, 3)
    face_edge_signs = local_signs.reshape(face_count, 3)

    d0 = sp.csr_matrix(
        (
            np.tile((-1.0, 1.0), edge_count),
            (np.repeat(np.arange(edge_count), 2), edges.ravel()),
        ),
        shape=(edge_count, vertex_count),
    )
    d1 = sp.csr_matrix(
        (local_signs, (face_indices, local_edge_indices)),
        shape=(face_count, edge_count),
    )
    incidence_count = np.bincount(local_edge_indices, minlength=edge_count)
    orientation_sum = np.bincount(
        local_edge_indices, weights=local_signs, minlength=edge_count
    )
    if np.any(incidence_count != 2):
        raise ValueError("surface solver requires a closed two-manifold mesh")
    if np.any(np.abs(orientation_sum) > 0.0):
        raise ValueError("surface faces must have consistent winding")

    face_graph = (abs(d1) @ abs(d1).T).tocsr()
    face_graph.setdiag(0.0)
    face_graph.eliminate_zeros()
    component_count = connected_components(face_graph, directed=False, return_labels=False)
    if component_count != 1:
        raise ValueError("surface solver currently requires one connected component")
    euler_characteristic = vertex_count - edge_count + face_count
    genus = (2 - euler_characteristic) // 2
    exactness = (d1 @ d0).tocsr()
    exactness_error = (
        0.0 if exactness.nnz == 0 else float(np.max(np.abs(exactness.data)))
    )

    positive_faces = np.full(edge_count, -1, dtype=np.int64)
    negative_faces = np.full(edge_count, -1, dtype=np.int64)
    positive = local_signs > 0.0
    negative = ~positive
    positive_faces[local_edge_indices[positive]] = face_indices[positive]
    negative_faces[local_edge_indices[negative]] = face_indices[negative]
    edge_faces = np.column_stack((positive_faces, negative_faces))

    edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    edge_tangents = edge_vectors / edge_lengths[:, None]
    edge_midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    dual_lengths = (
        np.linalg.norm(face_centers[positive_faces] - edge_midpoints, axis=1)
        + np.linalg.norm(face_centers[negative_faces] - edge_midpoints, axis=1)
    )
    conductance = edge_lengths / np.maximum(dual_lengths, 1e-15)
    face_edge_average = (0.5 * abs(d1).T).tocsr()
    pressure_gradient = (sp.diags(conductance) @ d1.T).tocsr()
    dual_gradient_matrix = (-pressure_gradient).tocsr()
    neumann_laplacian = (d1 @ pressure_gradient).tocsc()
    free_faces = np.arange(1, face_count)
    poisson_factorization = splu(
        neumann_laplacian[free_faces][:, free_faces].tocsc()
    )

    outward_conormals = (
        face_edge_signs[:, :, None]
        * np.cross(
            edge_tangents[face_edge_indices], face_normals[:, None, :]
        )
    )
    integrated_conormals = (
        outward_conormals * edge_lengths[face_edge_indices, None]
    )
    flux_to_face_vector = np.linalg.pinv(integrated_conormals)

    vertex_mass = np.zeros(vertex_count)
    np.add.at(vertex_mass, faces.ravel(), np.repeat(face_areas / 3.0, 3))
    vertex_normals = np.zeros_like(vertices)
    for local_vertex in range(3):
        np.add.at(vertex_normals, faces[:, local_vertex], area_vectors)
    vertex_normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(vertex_normal_lengths <= 1e-15):
        raise ValueError("cannot construct vertex normals")
    vertex_normals /= vertex_normal_lengths[:, None]

    p1_gradients = np.stack(
        (
            np.cross(face_normals, points[:, 2] - points[:, 1]),
            np.cross(face_normals, points[:, 0] - points[:, 2]),
            np.cross(face_normals, points[:, 1] - points[:, 0]),
        ),
        axis=1,
    ) / double_areas[:, None, None]
    tangent_1 = points[:, 1] - points[:, 0]
    tangent_1 /= np.linalg.norm(tangent_1, axis=1)[:, None]
    tangent_2 = np.cross(face_normals, tangent_1)
    face_gradient_1, face_gradient_2 = _face_gradient_matrices(
        face_centers, tangent_1, tangent_2, edge_faces
    )

    identity_first = np.asarray(face_gradient_1 @ face_centers)
    identity_second = np.asarray(face_gradient_2 @ face_centers)
    identity_jacobian = np.stack((identity_first, identity_second), axis=2)
    identity_metric = np.einsum(
        "fai,faj->fij", identity_jacobian, identity_jacobian
    )
    metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(identity_metric)
    if np.min(metric_eigenvalues) <= 1e-12:
        raise ValueError("embedding-coordinate chart is locally rank deficient")
    chart_metric_normalizer = np.einsum(
        "fij,fj,fkj->fik",
        metric_eigenvectors,
        1.0 / np.sqrt(metric_eigenvalues),
        metric_eigenvectors,
    )

    coordinate_fluxes = np.asarray(dual_gradient_matrix @ face_centers)
    identity_chart_flux_matrix = sp.hstack(
        tuple(
            sp.diags(coordinate_fluxes[:, component]) @ face_edge_average
            for component in range(3)
        ),
        format="csr",
    )

    return {
        "vertices": vertices,
        "faces": faces,
        "points": points,
        "edges": edges,
        "face_edges": face_edge_indices,
        "face_edge_signs": face_edge_signs,
        "edge_faces": edge_faces,
        "d0": d0,
        "d1": d1,
        "face_areas": face_areas,
        "dual_star0": face_areas,
        "star0": vertex_mass,
        "face_normals": face_normals,
        "vertex_normals": vertex_normals,
        "face_centers": face_centers,
        "edge_midpoints": edge_midpoints,
        "edge_lengths": edge_lengths,
        "dual_lengths": dual_lengths,
        "minimum_altitudes": minimum_altitudes,
        "minimum_angles": minimum_angles,
        "aspect_ratios": aspect_ratios,
        "p1_gradients": p1_gradients,
        "tangent_1": tangent_1,
        "tangent_2": tangent_2,
        "face_gradient_1": face_gradient_1,
        "face_gradient_2": face_gradient_2,
        "chart_metric_normalizer": chart_metric_normalizer,
        "face_edge_average": face_edge_average,
        "conductance": conductance,
        "dual_gradient_matrix": dual_gradient_matrix,
        "pressure_gradient": pressure_gradient,
        "free_faces": free_faces,
        "poisson_factorization": poisson_factorization,
        "integrated_conormals": integrated_conormals,
        "flux_to_face_vector": flux_to_face_vector,
        "identity_chart_flux_matrix": identity_chart_flux_matrix,
        "reference_triangles": points.copy(),
        "reference_face_tree": cKDTree(face_centers),
        "euler_characteristic": euler_characteristic,
        "genus": genus,
        "exactness_error": exactness_error,
    }


def edge_flux_to_face_vectors(edge_flux, operators):
    outward_flux = (
        operators["face_edge_signs"]
        * np.asarray(edge_flux)[operators["face_edges"]]
    )
    return np.einsum(
        "fij,fj->fi", operators["flux_to_face_vector"], outward_flux
    )


def face_fields_to_vertex_fields(face_fields, operators):
    """Area-average scalar or ambient-vector face data onto vertices."""
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
        3.0 * operators["star0"][:, None], 1e-15
    )
    return vertex_fields[:, 0] if scalar_input else vertex_fields


def face_vectors_to_vertex_vectors(face_vectors, operators):
    vectors = face_fields_to_vertex_fields(face_vectors, operators)
    normals = operators["vertex_normals"]
    return vectors - np.sum(vectors * normals, axis=1)[:, None] * normals


def dual_gradient(face_fields, operators):
    """Map face dual-0 values to integrated conormal edge gradients."""
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    if face_fields.shape[0] != len(operators["faces"]):
        raise ValueError("dual-gradient fields must have one row per face")
    gradient = np.asarray(operators["dual_gradient_matrix"] @ face_fields)
    return gradient[:, 0] if scalar_input else gradient


def project_edge_flux(raw_edge_flux, operators):
    """Project an edge flux to the closed-surface constraint ``d1 u=0``."""
    raw_edge_flux = np.asarray(raw_edge_flux).ravel()
    if raw_edge_flux.shape != (len(operators["edges"]),):
        raise ValueError("raw_edge_flux must have one value per surface edge")
    divergence = np.asarray(operators["d1"] @ raw_edge_flux).ravel()
    pressure = np.zeros(len(operators["faces"]))
    free_faces = operators["free_faces"]
    pressure[free_faces] = operators["poisson_factorization"].solve(
        divergence[free_faces]
    )
    pressure -= np.dot(operators["face_areas"], pressure) / np.sum(
        operators["face_areas"]
    )
    correction = np.asarray(operators["pressure_gradient"] @ pressure).ravel()
    return pressure, raw_edge_flux - correction


def reconstruct_and_project(chart_fields, operators):
    """Reconstruct and project ``sum_i mu_i grad_S(lambda_i)``."""
    chart_fields = np.asarray(chart_fields)
    expected_shape = (len(operators["faces"]), CLEBSCH_FIELD_COUNT)
    if chart_fields.shape != expected_shape:
        raise ValueError(
            f"chart_fields must have face-based shape {expected_shape}, "
            f"got {chart_fields.shape}"
        )
    edge_mu = np.asarray(operators["face_edge_average"] @ chart_fields[:, MU])
    lambda_flux = dual_gradient(chart_fields[:, LAMBDA], operators)
    raw_flux = np.sum(edge_mu * lambda_flux, axis=1)
    return project_edge_flux(raw_flux, operators)


def relative_divergence(edge_flux, operators):
    residual = np.asarray(operators["d1"] @ edge_flux).ravel()
    scale = np.asarray(abs(operators["d1"]) @ np.abs(edge_flux)).ravel()
    return np.linalg.norm(residual, np.inf) / max(
        np.linalg.norm(scale, np.inf), 1e-15
    )


def face_field_gradients(fields, operators):
    """Recover intrinsic gradients of face-centered fields in ambient R^3."""
    fields = np.asarray(fields)
    scalar_input = fields.ndim == 1
    if scalar_input:
        fields = fields[:, None]
    first = np.asarray(operators["face_gradient_1"] @ fields)
    second = np.asarray(operators["face_gradient_2"] @ fields)
    gradients = (
        first[:, :, None] * operators["tangent_1"][:, None, :]
        + second[:, :, None] * operators["tangent_2"][:, None, :]
    )
    return gradients[:, 0] if scalar_input else gradients


def chart_metrics(chart_fields, operators):
    lambda_gradients = face_field_gradients(chart_fields[:, LAMBDA], operators)
    first_column = np.einsum(
        "fai,fi->fa", lambda_gradients, operators["tangent_1"]
    )
    second_column = np.einsum(
        "fai,fi->fa", lambda_gradients, operators["tangent_2"]
    )
    jacobian = np.stack((first_column, second_column), axis=2)
    normalized_jacobian = np.einsum(
        "fai,fij->faj", jacobian, operators["chart_metric_normalizer"]
    )
    singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, 1]
    log_sigma_max = np.log(np.maximum(sigma_max, 1e-15))
    log_condition = np.log(
        np.maximum(sigma_max / np.maximum(sigma_min, 1e-15), 1.0)
    )
    area_scale = sigma_max * sigma_min
    return log_sigma_max, log_condition, area_scale, sigma_min


def _closest_surface_points(query_points, operators, candidate_count=24):
    query_points = np.asarray(query_points, dtype=np.float64)
    triangles = operators["reference_triangles"]
    candidate_count = min(candidate_count, len(triangles))
    _, candidates = operators["reference_face_tree"].query(
        query_points, k=candidate_count
    )
    if candidate_count == 1:
        candidates = candidates[:, None]
    flat_candidates = candidates.reshape(-1)
    repeated_query = np.repeat(query_points, candidate_count, axis=0)
    closest = trimesh.triangles.closest_point(
        triangles[flat_candidates], repeated_query
    )
    squared_distance = np.sum((closest - repeated_query) ** 2, axis=1).reshape(
        len(query_points), candidate_count
    )
    best = np.argmin(squared_distance, axis=1)
    selected_faces = candidates[np.arange(len(query_points)), best]
    selected_points = closest.reshape(len(query_points), candidate_count, 3)[
        np.arange(len(query_points)), best
    ]
    return selected_faces, selected_points


def interpolate_surface_face_fields(query_points, face_fields, operators):
    """Closest-point sample a linearly reconstructed face field."""
    face_fields = np.asarray(face_fields)
    scalar_input = face_fields.ndim == 1
    if scalar_input:
        face_fields = face_fields[:, None]
    selected_faces, selected_points = _closest_surface_points(
        query_points, operators
    )
    gradients = face_field_gradients(face_fields, operators)
    offsets = selected_points - operators["face_centers"][selected_faces]
    samples = face_fields[selected_faces] + np.einsum(
        "nsi,ni->ns", gradients[selected_faces], offsets
    )
    return samples[:, 0] if scalar_input else samples


def compose_surface_backmap(local_lambda, reference_to_initial, operators):
    return interpolate_surface_face_fields(
        local_lambda, reference_to_initial, operators
    )


def _fit_identity_mu(target_flux, baseline_mu, operators, damping=0.001):
    """Fit three face mu fields for lambda equal to the embedding coordinates."""
    recovery = operators["identity_chart_flux_matrix"]
    baseline = np.concatenate(tuple(baseline_mu[:, component] for component in range(3)))
    target_flux = np.asarray(target_flux).ravel()
    residual = target_flux - recovery @ baseline
    row_scale = np.sqrt(
        np.mean(np.asarray(recovery.multiply(recovery).sum(axis=1)).ravel())
    )
    iteration_limit = min(4000, max(600, 2 * min(recovery.shape)))
    best = None
    for damping_factor in (damping, 0.1 * damping, 0.0):
        solution = lsqr(
            recovery,
            residual,
            damp=damping_factor * max(row_scale, 1e-15),
            atol=1e-11,
            btol=1e-11,
            iter_lim=iteration_limit,
        )
        coefficients = baseline + solution[0]
        fitted_flux = np.asarray(recovery @ coefficients).ravel()
        relative_residual = np.linalg.norm(fitted_flux - target_flux) / max(
            np.linalg.norm(target_flux), 1e-15
        )
        if np.all(np.isfinite(coefficients)) and np.isfinite(relative_residual):
            if best is None or relative_residual < best[0]:
                best = (float(relative_residual), coefficients, int(solution[1]))
            if relative_residual <= 5e-4:
                break
    if best is None or best[0] > 5e-3:
        residual_text = "non-finite" if best is None else f"{best[0]:.3e}"
        raise FloatingPointError(
            f"identity-chart Clebsch fit failed; relative residual={residual_text}"
        )
    relative_residual, coefficients, _ = best
    face_count = len(operators["faces"])
    mu = np.column_stack(
        tuple(
            coefficients[component * face_count : (component + 1) * face_count]
            for component in range(3)
        )
    )
    return mu, relative_residual


def initialize_fields(operators, settings: Settings):
    """Initialize three face Clebsch pairs and a passive vertex dye."""
    vertices = operators["vertices"]
    face_centers = operators["face_centers"]
    vertical_axis = np.array((0.0, 1.0, 0.0))

    # q=d0 psi is exactly divergence-free because d1*d0=0.  On a sphere,
    # psi=y produces azimuthal rotation; on the bunny it gives its intrinsic
    # curved-surface analogue.
    stream_function = vertices @ vertical_axis
    target_flux = np.asarray(operators["d0"] @ stream_function).ravel()
    target_vectors = edge_flux_to_face_vectors(target_flux, operators)
    target_scale = settings.target_speed / max(
        float(np.percentile(np.linalg.norm(target_vectors, axis=1), 95)), 1e-15
    )
    target_flux *= target_scale
    target_vectors *= target_scale
    mu, fit_residual = _fit_identity_mu(target_flux, target_vectors, operators)
    clebsch = np.column_stack((mu, face_centers))
    _, initial_flux = reconstruct_and_project(clebsch, operators)
    realized_speed = np.linalg.norm(
        edge_flux_to_face_vectors(initial_flux, operators), axis=1
    )
    realized_scale = settings.target_speed / max(
        float(np.percentile(realized_speed, 95)), 1e-15
    )
    clebsch[:, MU] *= realized_scale
    target_flux *= realized_scale
    _, initial_flux = reconstruct_and_project(clebsch, operators)

    dye_direction = np.array((1.0, 0.25, 0.15))
    dye_direction /= np.linalg.norm(dye_direction)
    dye_center_index = int(np.argmax(vertices @ dye_direction))
    dye_center = vertices[dye_center_index]
    width = settings.dye_width_fraction
    dye = np.exp(
        -np.sum((vertices - dye_center) ** 2, axis=1)
        / max(2.0 * width * width, 1e-15)
    )
    initialization = {
        "stream_axis": vertical_axis,
        "target_flux": target_flux,
        "fit_relative_residual": fit_residual,
        "dye_center_vertex": dye_center_index,
        "dye_center": dye_center,
        "dye_width": width,
    }
    return clebsch, dye, initialization


def centered_advection_matrix(edge_flux, operators):
    """Return the conservative face/dual-vertex centered generator."""
    weak_operator = (
        -operators["d1"]
        @ sp.diags(np.asarray(edge_flux))
        @ operators["face_edge_average"]
    ).tocsr()
    return (
        sp.diags(1.0 / operators["dual_star0"]) @ weak_operator
    ).tocsr()


def advection_rhs(transported_fields, edge_flux, operators, settings: Settings):
    """Advect face-based Clebsch dual 0-cochains by shared edge fluxes."""
    transported_fields = np.asarray(transported_fields)
    scalar_input = transported_fields.ndim == 1
    if scalar_input:
        transported_fields = transported_fields[:, None]
    if transported_fields.shape[0] != len(operators["faces"]):
        raise ValueError("face-transported fields must have one row per face")

    edge_values = operators["face_edge_average"] @ transported_fields
    weak_rhs = np.asarray(
        -operators["d1"] @ (np.asarray(edge_flux)[:, None] * edge_values)
    )
    if settings.stabilization == "graph" and settings.graph_viscosity > 0.0:
        edge_faces = operators["edge_faces"]
        strength = settings.graph_viscosity * np.abs(edge_flux)
        difference = (
            transported_fields[edge_faces[:, 1]]
            - transported_fields[edge_faces[:, 0]]
        )
        contribution = strength[:, None] * difference
        np.add.at(weak_rhs, edge_faces[:, 0], contribution)
        np.add.at(weak_rhs, edge_faces[:, 1], -contribution)
    result = weak_rhs / operators["dual_star0"][:, None]
    return result[:, 0] if scalar_input else result


def vertex_advection_rhs(transported_fields, edge_flux, operators, settings: Settings):
    """Advect passive vertex fields with the recovered tangent face velocity."""
    transported_fields = np.asarray(transported_fields)
    scalar_input = transported_fields.ndim == 1
    if scalar_input:
        transported_fields = transported_fields[:, None]
    if transported_fields.shape[0] != len(operators["vertices"]):
        raise ValueError("vertex-transported fields must have one row per vertex")

    velocity = edge_flux_to_face_vectors(edge_flux, operators)
    velocity_dot_basis_gradient = np.einsum(
        "fij,fj->fi", operators["p1_gradients"], velocity
    )
    face_mean = np.mean(transported_fields[operators["faces"]], axis=1)
    face_weak_rhs = (
        operators["face_areas"][:, None, None]
        * velocity_dot_basis_gradient[:, :, None]
        * face_mean[:, None, :]
    )
    weak_rhs = np.zeros_like(transported_fields)
    np.add.at(
        weak_rhs,
        operators["faces"].ravel(),
        face_weak_rhs.reshape(-1, transported_fields.shape[1]),
    )

    if settings.stabilization == "graph" and settings.graph_viscosity > 0.0:
        strength = (
            settings.graph_viscosity
            * operators["face_areas"]
            * np.max(np.abs(velocity_dot_basis_gradient), axis=1)
            / 3.0
        )
        face_values = transported_fields[operators["faces"]]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            difference = face_values[:, second] - face_values[:, first]
            contribution = strength[:, None] * difference
            np.add.at(weak_rhs, operators["faces"][:, first], contribution)
            np.add.at(weak_rhs, operators["faces"][:, second], -contribution)
    result = weak_rhs / operators["star0"][:, None]
    return result[:, 0] if scalar_input else result


def stage_rhs(face_state, dye, operators, settings: Settings):
    chart = face_state[:, :CLEBSCH_FIELD_COUNT]
    _, edge_flux = reconstruct_and_project(chart, operators)
    return (
        advection_rhs(face_state, edge_flux, operators, settings),
        vertex_advection_rhs(dye, edge_flux, operators, settings),
    )


def ssprk3_step(
    chart_fields, material_fields, dye, time_step, operators, settings: Settings
):
    """Advance local/material face states and vertex dye with stage feedback."""
    face_state = np.hstack((chart_fields, material_fields))
    face_rate, dye_rate = stage_rhs(face_state, dye, operators, settings)
    face_stage1 = face_state + time_step * face_rate
    dye_stage1 = dye + time_step * dye_rate

    face_rate, dye_rate = stage_rhs(face_stage1, dye_stage1, operators, settings)
    face_stage2 = 0.75 * face_state + 0.25 * (
        face_stage1 + time_step * face_rate
    )
    dye_stage2 = 0.75 * dye + 0.25 * (
        dye_stage1 + time_step * dye_rate
    )

    face_rate, dye_rate = stage_rhs(face_stage2, dye_stage2, operators, settings)
    next_face_state = (1.0 / 3.0) * face_state + (2.0 / 3.0) * (
        face_stage2 + time_step * face_rate
    )
    next_chart, next_material = np.hsplit(next_face_state, 2)
    next_dye = (1.0 / 3.0) * dye + (2.0 / 3.0) * (
        dye_stage2 + time_step * dye_rate
    )
    pressure, edge_flux = reconstruct_and_project(next_chart, operators)
    return next_chart, next_material, next_dye, pressure, edge_flux


def stable_time_step(edge_flux, operators, settings: Settings):
    velocity = edge_flux_to_face_vectors(edge_flux, operators)
    speed_rate = np.max(
        np.linalg.norm(velocity, axis=1)
        / np.maximum(operators["minimum_altitudes"], 1e-15)
    )
    face_flux = np.asarray(abs(operators["d1"]) @ np.abs(edge_flux)).ravel()
    flux_rate = np.max(face_flux / np.maximum(operators["face_areas"], 1e-15))
    diffusion_rate = (
        2.0 * settings.graph_viscosity * flux_rate
        if settings.stabilization == "graph"
        else 0.0
    )
    maximum_rate = max(
        float(speed_rate), float(flux_rate), float(diffusion_rate), 1e-15
    )
    return min(settings.maximum_time_step, settings.cfl / maximum_rate), maximum_rate


def kinetic_energy(edge_flux, operators):
    velocity = edge_flux_to_face_vectors(edge_flux, operators)
    return 0.5 * np.sum(
        operators["face_areas"] * np.sum(velocity * velocity, axis=1)
    )


def clebsch_vorticity(chart_fields, operators):
    """Return sum_i n dot (grad(mu_i) cross grad(lambda_i)) per face."""
    gradients = face_field_gradients(chart_fields, operators)
    vorticity = np.zeros(len(operators["faces"]))
    for mu_index, lambda_index in zip(range(MU1, MU3 + 1), range(LAMBDA1, LAMBDA3 + 1)):
        cross_gradient = np.cross(
            gradients[:, mu_index], gradients[:, lambda_index]
        )
        vorticity += np.sum(cross_gradient * operators["face_normals"], axis=1)
    return vorticity


def recovered_velocity_vorticity(edge_flux, operators):
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    vertex_velocity = face_vectors_to_vertex_vectors(face_velocity, operators)
    velocity_gradients = np.einsum(
        "fvi,fvc->fci",
        operators["p1_gradients"],
        vertex_velocity[operators["faces"]],
    )
    curl = np.column_stack(
        (
            velocity_gradients[:, 2, 1] - velocity_gradients[:, 1, 2],
            velocity_gradients[:, 0, 2] - velocity_gradients[:, 2, 0],
            velocity_gradients[:, 1, 0] - velocity_gradients[:, 0, 1],
        )
    )
    return np.sum(curl * operators["face_normals"], axis=1)


def discrete_enstrophy(vorticity, face_areas):
    """Return ``0.5 * integral_M omega^2 dA`` for face vorticity values."""
    vorticity = np.asarray(vorticity)
    face_areas = np.asarray(face_areas)
    return 0.5 * float(np.sum(face_areas * vorticity * vorticity))


def relative_enstrophy_change(enstrophy, initial_enstrophy, epsilon=1e-15):
    """Return the signed percentage change in enstrophy from its initial value."""
    difference = float(enstrophy) - float(initial_enstrophy)
    if abs(initial_enstrophy) <= epsilon:
        return 0.0 if abs(difference) <= epsilon else np.nan
    return 100.0 * difference / abs(float(initial_enstrophy))


def face_scalar_invariants(fields, operators):
    fields = np.asarray(fields)
    mass = operators["face_areas"] @ fields
    l2 = np.sqrt(
        np.sum(operators["face_areas"][:, None] * fields * fields, axis=0)
    )
    return mass, l2


def vertex_scalar_invariants(field, operators):
    return (
        float(operators["star0"] @ field),
        float(np.sqrt(operators["star0"] @ (field * field))),
    )


def rechart(chart_fields, edge_flux, reference_to_initial, operators):
    """Reset lambda to (x,y,z) and refit three face mu fields to the flux."""
    edge_flux_before = np.asarray(edge_flux).copy()
    global_labels_before = compose_surface_backmap(
        chart_fields[:, LAMBDA], reference_to_initial, operators
    )
    baseline_mu = edge_flux_to_face_vectors(edge_flux_before, operators)
    recovered_mu, fit_residual = _fit_identity_mu(
        edge_flux_before, baseline_mu, operators
    )
    reset_chart = chart_fields.copy()
    reset_chart[:, MU] = recovered_mu
    reset_chart[:, LAMBDA] = operators["face_centers"]
    pressure, edge_flux_after = reconstruct_and_project(reset_chart, operators)

    new_reference_to_initial = global_labels_before.copy()
    global_labels_after = compose_surface_backmap(
        reset_chart[:, LAMBDA], new_reference_to_initial, operators
    )
    compatible_before = np.sqrt(
        np.sum(edge_flux_before * edge_flux_before / operators["conductance"])
    )
    compatible_jump = np.sqrt(
        np.sum(
            (edge_flux_after - edge_flux_before) ** 2
            / operators["conductance"]
        )
    ) / max(compatible_before, 1e-15)
    if not np.isfinite(compatible_jump) or compatible_jump > 5e-3:
        raise FloatingPointError(
            f"rechart rejected: relative compatible-flux jump={compatible_jump:.3e}"
        )
    energy_before = kinetic_energy(edge_flux_before, operators)
    transition = {
        "velocity_jump": float(compatible_jump),
        "energy_jump": float(
            (kinetic_energy(edge_flux_after, operators) - energy_before)
            / max(energy_before, 1e-15)
        ),
        "fit_relative_residual": fit_residual,
        "global_label_jump": float(
            np.max(
                np.linalg.norm(global_labels_after - global_labels_before, axis=1),
                initial=0.0,
            )
        ),
    }
    return (
        reset_chart,
        pressure,
        edge_flux_after,
        new_reference_to_initial,
        transition,
    )


def simulate(operators, settings: Settings):
    material_fields, dye, initialization = initialize_fields(operators, settings)
    chart_fields = material_fields.copy()
    pressure, edge_flux = reconstruct_and_project(chart_fields, operators)
    reference_to_initial = operators["face_centers"].copy()
    output_times = np.linspace(0.0, settings.final_time, settings.frames)
    histories = {
        "times": [], "material_fields": [], "dye": [],
        "composition_error": [], "projection_potential": [], "edge_fluxes": [],
        "clebsch_vorticity": [], "velocity_vorticity": [],
        "clebsch_enstrophy": [], "velocity_enstrophy": [],
        "clebsch_enstrophy_change": [], "velocity_enstrophy_change": [],
        "log_sigma_max": [], "log_condition": [], "area_scale": [],
        "divergence": [], "energy": [], "mass": [], "l2": [],
        "maximum_speed": [], "step_cfl": [], "chart": [], "chart_age": [],
    }
    initial_enstrophy = {}
    reset_events = []
    current_time = 0.0
    next_output = 1
    step = 0
    chart_index = 0
    chart_start = 0.0
    last_rechart_time = -np.inf
    maximum_cfl_since_output = 0.0

    def store():
        nonlocal maximum_cfl_since_output
        log_sigma, log_condition, area_scale, _ = chart_metrics(
            chart_fields, operators
        )
        composed_labels = compose_surface_backmap(
            chart_fields[:, LAMBDA], reference_to_initial, operators
        )
        material_labels = material_fields[:, LAMBDA]
        omega_c = clebsch_vorticity(chart_fields, operators)
        omega_u = recovered_velocity_vorticity(edge_flux, operators)
        enstrophy_c = discrete_enstrophy(omega_c, operators["face_areas"])
        enstrophy_u = discrete_enstrophy(omega_u, operators["face_areas"])
        if not initial_enstrophy:
            initial_enstrophy["clebsch"] = enstrophy_c
            initial_enstrophy["velocity"] = enstrophy_u
        clebsch_mass, clebsch_l2 = face_scalar_invariants(
            material_fields, operators
        )
        dye_mass, dye_l2 = vertex_scalar_invariants(dye, operators)
        face_velocity = edge_flux_to_face_vectors(edge_flux, operators)

        histories["times"].append(current_time)
        histories["material_fields"].append(material_fields.copy())
        histories["dye"].append(dye.copy())
        histories["composition_error"].append(
            np.linalg.norm(composed_labels - material_labels, axis=1)
        )
        histories["projection_potential"].append(pressure.copy())
        histories["edge_fluxes"].append(edge_flux.copy())
        histories["clebsch_vorticity"].append(omega_c)
        histories["velocity_vorticity"].append(omega_u)
        histories["clebsch_enstrophy"].append(enstrophy_c)
        histories["velocity_enstrophy"].append(enstrophy_u)
        histories["clebsch_enstrophy_change"].append(
            relative_enstrophy_change(
                enstrophy_c, initial_enstrophy["clebsch"]
            )
        )
        histories["velocity_enstrophy_change"].append(
            relative_enstrophy_change(
                enstrophy_u, initial_enstrophy["velocity"]
            )
        )
        histories["log_sigma_max"].append(log_sigma)
        histories["log_condition"].append(log_condition)
        histories["area_scale"].append(area_scale)
        histories["divergence"].append(relative_divergence(edge_flux, operators))
        histories["energy"].append(kinetic_energy(edge_flux, operators))
        histories["mass"].append(np.concatenate((clebsch_mass, (dye_mass,))))
        histories["l2"].append(np.concatenate((clebsch_l2, (dye_l2,))))
        histories["maximum_speed"].append(
            float(np.max(np.linalg.norm(face_velocity, axis=1)))
        )
        histories["step_cfl"].append(maximum_cfl_since_output)
        histories["chart"].append(chart_index)
        histories["chart_age"].append(current_time - chart_start)
        maximum_cfl_since_output = 0.0

    store()
    tolerance = 1e-12 * max(1.0, settings.final_time)
    while current_time < settings.final_time - tolerance:
        if next_output >= len(output_times):
            raise RuntimeError("output scheduler exhausted before final time")
        target_time = float(output_times[next_output])
        proposed_step, rate = stable_time_step(edge_flux, operators, settings)
        remaining = min(target_time - current_time, settings.final_time - current_time)
        time_step = min(proposed_step, remaining)
        if not np.isfinite(time_step) or time_step <= tolerance:
            if remaining <= tolerance:
                current_time = target_time
                store()
                next_output += 1
                continue
            raise FloatingPointError(
                f"time step collapsed at t={current_time:.8g}; dt={time_step:.3e}"
            )

        chart_fields, material_fields, dye, pressure, edge_flux = ssprk3_step(
            chart_fields, material_fields, dye, time_step, operators, settings
        )
        current_time += time_step
        step += 1
        maximum_cfl_since_output = max(
            maximum_cfl_since_output, time_step * rate
        )
        if not (
            np.all(np.isfinite(chart_fields))
            and np.all(np.isfinite(material_fields))
            and np.all(np.isfinite(dye))
            and np.all(np.isfinite(edge_flux))
        ):
            raise FloatingPointError(
                f"surface solution became non-finite at t={current_time:.8g}"
            )

        log_sigma, _, _, sigma_min = chart_metrics(chart_fields, operators)
        distorted = float(np.max(log_sigma)) > settings.rechart_cutoff
        singular = float(np.min(sigma_min)) <= settings.singular_value_floor
        if distorted or singular:
            (
                chart_fields, pressure, edge_flux, reference_to_initial, transition
            ) = rechart(chart_fields, edge_flux, reference_to_initial, operators)
            chart_index += 1
            chart_start = current_time
            last_rechart_time = current_time
            reset_events.append(
                {
                    "time": current_time,
                    "chart": chart_index,
                    "reason": (
                        "sigma_min+sigma_max"
                        if singular and distorted
                        else "sigma_min" if singular else "sigma_max"
                    ),
                    "pre_log_sigma_max": float(np.max(log_sigma)),
                    "pre_sigma_min": float(np.min(sigma_min)),
                    **transition,
                }
            )

        if current_time >= target_time - tolerance:
            current_time = target_time
            scheduled = (
                settings.rechart_every_frames > 0
                and next_output % settings.rechart_every_frames == 0
                and abs(current_time - last_rechart_time) > tolerance
            )
            if scheduled:
                log_sigma, _, _, sigma_min = chart_metrics(chart_fields, operators)
                (
                    chart_fields, pressure, edge_flux,
                    reference_to_initial, transition,
                ) = rechart(chart_fields, edge_flux, reference_to_initial, operators)
                chart_index += 1
                chart_start = current_time
                last_rechart_time = current_time
                reset_events.append(
                    {
                        "time": current_time,
                        "chart": chart_index,
                        "reason": "frame interval",
                        "pre_log_sigma_max": float(np.max(log_sigma)),
                        "pre_sigma_min": float(np.min(sigma_min)),
                        **transition,
                    }
                )
            store()
            next_output += 1

    result = {key: np.asarray(values) for key, values in histories.items()}
    float32_limit = np.finfo(np.float32).max
    for key in (
        "material_fields", "dye", "composition_error", "projection_potential",
        "edge_fluxes", "clebsch_vorticity", "velocity_vorticity",
        "log_sigma_max", "log_condition", "area_scale",
    ):
        values = result[key]
        # Keep extreme or non-finite diagnostic histories in float64.  Blind
        # downcasting can overflow and obscure the chart-composition failure
        # that the diagnostic is intended to reveal.
        if np.all(np.isfinite(values)) and np.max(np.abs(values), initial=0.0) <= float32_limit:
            result[key] = values.astype(np.float32)
    result["reset_events"] = reset_events
    result["number_of_steps"] = step
    result["initialization"] = initialization
    return result


def relative_change_history(values, epsilon=1e-15):
    """Return signed percentage changes, with a defined zero-reference case."""
    values = np.asarray(values, dtype=float)
    initial_value = float(values[0])
    differences = values - initial_value
    if abs(initial_value) <= epsilon:
        return np.where(np.abs(differences) <= epsilon, 0.0, np.nan)
    return 100.0 * differences / abs(initial_value)


def percentage_change(values, frame=-1):
    return float(relative_change_history(values)[frame])


def diagnostic_time_series(result):
    """Return scalar histories for optional diagnostic plotting.

    This helper is post-processing only.  The third-panel entries are signed
    relative changes of the two area-weighted discrete enstrophies already
    recorded by :func:`simulate`.
    """
    return {
        "time": np.asarray(result["times"], dtype=float),
        "relative_divergence": np.asarray(result["divergence"], dtype=float),
        "energy_change_percent": relative_change_history(result["energy"]),
        "velocity_enstrophy_change_percent": np.asarray(
            result["velocity_enstrophy_change"], dtype=float
        ),
        "clebsch_enstrophy_change_percent": np.asarray(
            result["clebsch_enstrophy_change"], dtype=float
        ),
    }


def plot_diagnostic_histories(result, settings: Settings, output_path=None, show=False):
    """Plot divergence, energy, and enstrophy diagnostics versus physical time.

    Matplotlib is imported only when this optional output is requested.  Plot
    generation cannot change the simulation state or any stored diagnostic.
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

    # Exact zeros cannot be shown on a logarithmic axis.  This display-only
    # floor does not alter the stored divergence values.
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
        times, series["energy_change_percent"],
        color="tab:green", linewidth=1.8,
    )
    axes[1].axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axes[1].set_ylabel("Relative energy change [%]")

    axes[2].plot(
        times, series["velocity_enstrophy_change_percent"],
        color="tab:red", linewidth=1.8,
        label="from projected edge velocity",
    )
    axes[2].plot(
        times, series["clebsch_enstrophy_change_percent"],
        color="tab:purple", linewidth=1.8, linestyle="--",
        label="from local-chart Clebsch gradients",
    )
    axes[2].axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axes[2].set_ylabel("Relative enstrophy change [%]")
    axes[2].set_xlabel("Physical time")
    axes[2].legend(loc="best", fontsize=8)

    reset_events = result.get("reset_events", [])
    if len(reset_events) <= 50:
        for event_index, event in enumerate(reset_events):
            label = "rechart" if event_index == 0 else None
            for axis in axes:
                axis.axvline(
                    float(event["time"]), color="0.35", linewidth=0.7,
                    linestyle="--", alpha=0.35, label=label,
                )
    else:
        # Hundreds of full-height lines hide the data.  A top-edge event rug
        # preserves timing and density without washing out all three panels.
        event_times = np.asarray(
            [float(event["time"]) for event in reset_events], dtype=float
        )
        for axis in axes:
            axis.vlines(
                event_times, 0.965, 1.0,
                transform=axis.get_xaxis_transform(), color="0.35",
                linewidth=0.45, alpha=0.55,
            )
        axes[0].plot(
            [], [], color="0.35", linewidth=0.8,
            label=f"recharts ({len(reset_events)}; top-edge rug)",
        )
    if reset_events:
        axes[0].legend(loc="best", fontsize=8)

    for axis in axes:
        axis.grid(True, which="both", linewidth=0.45, alpha=0.3)
        axis.margins(x=0.0)
    surface_stem = Path(settings.mesh_path).stem.replace("_", " ")
    surface_name = (
        "Stanford Bunny"
        if surface_stem.casefold() == "bunny"
        else surface_stem.title()
    )
    figure.suptitle(
        f"Surface Clebsch Euler diagnostics: {surface_name}"
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
    if signed:
        return -magnitude, magnitude
    return float(np.min(values)), float(np.percentile(values, 99.5))


def print_report(result, operators, mesh_path):
    target_flux = result["initialization"]["target_flux"]
    initial_flux_error = np.sqrt(
        np.sum(
            (result["edge_fluxes"][0] - target_flux) ** 2
            / operators["conductance"]
        )
        / max(
            np.sum(target_flux * target_flux / operators["conductance"]),
            1e-15,
        )
    )
    face_mass_scale = np.sum(
        operators["face_areas"][:, None]
        * np.abs(result["material_fields"][0]),
        axis=0,
    )
    dye_mass_scale = operators["star0"] @ np.abs(result["dye"][0])
    mass_scale = np.concatenate((face_mass_scale, (dye_mass_scale,)))
    mass_change = 100.0 * (
        result["mass"][-1] - result["mass"][0]
    ) / np.maximum(mass_scale, 1e-15)
    l2_change = 100.0 * (
        result["l2"][-1] / np.maximum(result["l2"][0], 1e-15) - 1.0
    )

    print(f"Mesh:                           {mesh_path}")
    print(
        f"Vertices / edges / faces:       {len(operators['vertices']):,} / "
        f"{len(operators['edges']):,} / {len(operators['faces']):,}"
    )
    print(
        f"Euler characteristic / genus:   {operators['euler_characteristic']} / "
        f"{operators['genus']}"
    )
    print(
        f"Minimum angle / max aspect:     {np.min(operators['minimum_angles']):.3f} deg / "
        f"{np.max(operators['aspect_ratios']):.3f}"
    )
    print(f"Face-based Clebsch state:       {CLEBSCH_FIELD_COUNT} fields (three pairs)")
    print(
        f"Initial raw fit / projected error: "
        f"{result['initialization']['fit_relative_residual']:.6e} / "
        f"{initial_flux_error:.6e}"
    )
    print(f"SSPRK3 steps:                   {result['number_of_steps']}")
    print(f"Final time:                     {result['times'][-1]:.8f}")
    print(f"Maximum stored CFL:             {np.max(result['step_cfl']):.6e}")
    print(f"Maximum relative divergence:    {np.max(result['divergence']):.6e}")
    print(f"Relative kinetic-energy change: {percentage_change(result['energy']):+.6e}%")
    print(
        f"Projected-u enstrophy change:   "
        f"{result['velocity_enstrophy_change'][-1]:+.6e}%"
    )
    print(
        f"Local-chart Clebsch enstrophy change: "
        f"{result['clebsch_enstrophy_change'][-1]:+.6e}%"
    )
    for index, name in enumerate(FIELD_NAMES):
        location = "face material" if index < DYE_INDEX else "vertex dye"
        print(
            f"  {location} {name:8s} mass/L2:      "
            f"{mass_change[index]:+.3e}% / {l2_change[index]:+.3e}%"
        )
    print(f"Number of recharts:             {len(result['reset_events'])}")
    if result["reset_events"]:
        print(
            "Maximum rechart flux / energy jump: "
            f"{max(event['velocity_jump'] for event in result['reset_events']):.3e} / "
            f"{max(abs(event['energy_jump']) for event in result['reset_events']):.3e}"
        )
        print(
            "Maximum composition jump:        "
            f"{max(event['global_label_jump'] for event in result['reset_events']):.3e}"
        )


def run_diagnostic_tests(operators, settings: Settings):
    """Run algebra, transport, rechart, and short nonlinear regressions."""
    fields, dye, initialization = initialize_fields(operators, settings)
    _, edge_flux = reconstruct_and_project(fields, operators)
    rng = np.random.default_rng(20260911)
    results = []

    def check(name, error, tolerance):
        passed = bool(np.isfinite(error) and error <= tolerance)
        results.append(passed)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name:47s} "
            f"error={error:.3e}  tolerance={tolerance:.3e}"
        )

    face_count = len(operators["faces"])
    vertex_count = len(operators["vertices"])
    check(
        "three Clebsch pairs are face-based",
        float(fields.shape != (face_count, CLEBSCH_FIELD_COUNT)),
        0.0,
    )
    check(
        "passive dye remains vertex-based",
        float(dye.shape != (vertex_count,)),
        0.0,
    )
    check("incidence exactness ||d1 d0||", operators["exactness_error"], 1e-14)
    check(
        "closed orientation gives zero d1 column sums",
        np.linalg.norm(np.asarray(operators["d1"].sum(axis=0)).ravel(), np.inf),
        0.0,
    )
    check(
        "dual gradient annihilates constants",
        np.linalg.norm(dual_gradient(np.ones(face_count), operators), np.inf),
        1e-13,
    )
    check(
        "surface conductances are positive",
        float(np.any(operators["conductance"] <= 0.0)),
        0.0,
    )
    log_sigma, log_condition, area_scale, sigma_min = chart_metrics(
        fields, operators
    )
    identity_metric_error = max(
        np.max(np.abs(log_sigma)), np.max(np.abs(log_condition)),
        np.max(np.abs(area_scale - 1.0)), np.max(np.abs(sigma_min - 1.0)),
    )
    check("identity embedding has unit chart metric", identity_metric_error, 5e-12)

    target_flux = initialization["target_flux"]
    fit_error = np.sqrt(
        np.sum((edge_flux - target_flux) ** 2 / operators["conductance"])
        / max(np.sum(target_flux * target_flux / operators["conductance"]), 1e-15)
    )
    check("initial Clebsch fit reproduces target flux", fit_error, 5e-4)
    check("initial projection is divergence-free", relative_divergence(edge_flux, operators), 3e-12)
    face_velocity = edge_flux_to_face_vectors(edge_flux, operators)
    tangency_error = np.max(
        np.abs(np.sum(face_velocity * operators["face_normals"], axis=1))
    )
    check("recovered face velocity is tangent", tangency_error, 2e-13)
    predicted_outward_flux = np.einsum(
        "fej,fj->fe", operators["integrated_conormals"], face_velocity
    )
    exact_outward_flux = (
        operators["face_edge_signs"]
        * edge_flux[operators["face_edges"]]
    )
    roundtrip_error = np.linalg.norm(
        predicted_outward_flux - exact_outward_flux
    ) / max(np.linalg.norm(exact_outward_flux), 1e-15)
    check("face-vector flux round trip", roundtrip_error, 2e-12)

    random_flux = rng.standard_normal(len(operators["edges"]))
    _, projected = project_edge_flux(random_flux, operators)
    _, projected_twice = project_edge_flux(projected, operators)
    check("projection is divergence-free", relative_divergence(projected, operators), 3e-12)
    check(
        "projection is idempotent",
        np.linalg.norm(projected_twice - projected) / max(np.linalg.norm(projected), 1e-15),
        3e-12,
    )
    gradient_flux = np.asarray(
        operators["pressure_gradient"] @ rng.standard_normal(face_count)
    ).ravel()
    _, removed_gradient = project_edge_flux(gradient_flux, operators)
    check(
        "projection removes compatible gradients",
        np.linalg.norm(removed_gradient) / max(np.linalg.norm(gradient_flux), 1e-15),
        3e-11,
    )

    constant_fields = np.tile(
        (0.25, -0.5, 0.75, 1.0, -1.25, 1.5), (face_count, 1)
    )
    vertex_constants = face_fields_to_vertex_fields(constant_fields, operators)
    check(
        "display projection preserves constant fields",
        np.max(np.abs(vertex_constants - constant_fields[0])),
        3e-14,
    )
    random_fields = rng.standard_normal((face_count, 3))
    random_rate = advection_rhs(random_fields, projected, operators, settings)
    derivative = centered_advection_matrix(projected, operators)
    operator_scale = max(
        float(np.linalg.norm(derivative.data, np.inf)), 1.0
    )
    check(
        "face transport preserves constants",
        np.linalg.norm(derivative @ np.ones(face_count), np.inf) / operator_scale,
        3e-13,
    )
    check(
        "face transport preserves scalar mass",
        np.linalg.norm(operators["face_areas"] @ random_rate, np.inf),
        3e-11,
    )
    weak_derivative = sp.diags(operators["face_areas"]) @ derivative
    skew_defect = weak_derivative + weak_derivative.T
    skew_error = np.sqrt(float(skew_defect.multiply(skew_defect).sum())) / max(
        np.sqrt(float(weak_derivative.multiply(weak_derivative).sum())), 1e-15
    )
    check("centered face transport is weighted-skew", skew_error, 3e-12)

    step_size, maximum_rate = stable_time_step(edge_flux, operators, settings)
    check(
        "adaptive step obeys requested CFL",
        max(0.0, step_size * maximum_rate - settings.cfl),
        3e-14,
    )
    graph_settings = replace(
        settings, stabilization="graph", graph_viscosity=25.0
    )
    graph_step, graph_rate = stable_time_step(
        edge_flux, operators, graph_settings
    )
    face_flux_rate = np.max(
        np.asarray(abs(operators["d1"]) @ np.abs(edge_flux)).ravel()
        / operators["face_areas"]
    )
    check(
        "graph-viscosity stiffness enters CFL rate",
        max(0.0, 2.0 * graph_settings.graph_viscosity * face_flux_rate - graph_rate),
        3e-12,
    )
    check(
        "graph-stabilized step obeys requested CFL",
        max(0.0, graph_step * graph_rate - graph_settings.cfl),
        3e-14,
    )
    next_chart, next_material, next_dye, _, next_flux = ssprk3_step(
        fields, fields.copy(), dye, 0.25 * step_size, operators, settings
    )
    initial_mass, _ = face_scalar_invariants(fields, operators)
    next_mass, _ = face_scalar_invariants(next_material, operators)
    dye_mass, _ = vertex_scalar_invariants(dye, operators)
    next_dye_mass, _ = vertex_scalar_invariants(next_dye, operators)
    check(
        "coupled SSPRK3 preserves transported mass",
        max(np.max(np.abs(next_mass - initial_mass)), abs(next_dye_mass - dye_mass)),
        3e-11,
    )
    check(
        "one SSPRK3 stage remains finite",
        float(
            not (
                np.all(np.isfinite(next_chart))
                and np.all(np.isfinite(next_flux))
            )
        ),
        0.0,
    )

    reset_chart, _, reset_flux, _, transition = rechart(
        next_chart, next_flux, operators["face_centers"].copy(), operators
    )
    check(
        "rechart restores identity surface labels",
        np.max(np.abs(reset_chart[:, LAMBDA] - operators["face_centers"])),
        1e-14,
    )
    check("rechart keeps flux transfer small", transition["velocity_jump"], 5e-4)
    check("evolved rechart is composition-continuous", transition["global_label_jump"], 3e-12)
    check("recharted flux remains divergence-free", relative_divergence(reset_flux, operators), 3e-12)

    short_settings = replace(
        settings,
        final_time=0.006,
        frames=4,
        maximum_time_step=0.001,
        rechart_cutoff=99.0,
        rechart_every_frames=1,
        singular_value_floor=0.0,
    )
    short_result = simulate(operators, short_settings)
    check(
        "simulation stores every requested output",
        float(len(short_result["times"]) != short_settings.frames),
        0.0,
    )
    check(
        "simulation reaches final time by positive steps",
        max(
            abs(short_result["times"][-1] - short_settings.final_time),
            float(short_result["number_of_steps"] <= 0),
        ),
        3e-14,
    )
    check(
        "stored nonlinear fluxes satisfy divergence",
        float(np.max(short_result["divergence"])),
        3e-12,
    )
    check(
        "enstrophy-change histories start at zero",
        max(
            abs(short_result["velocity_enstrophy_change"][0]),
            abs(short_result["clebsch_enstrophy_change"][0]),
        ),
        0.0,
    )
    expected_velocity_enstrophy = discrete_enstrophy(
        short_result["velocity_vorticity"][-1], operators["face_areas"]
    )
    expected_clebsch_enstrophy = discrete_enstrophy(
        short_result["clebsch_vorticity"][-1], operators["face_areas"]
    )
    check(
        "stored enstrophies use face-area weighting",
        max(
            abs(
                short_result["velocity_enstrophy"][-1]
                - expected_velocity_enstrophy
            ) / max(abs(short_result["velocity_enstrophy"][-1]), 1e-15),
            abs(
                short_result["clebsch_enstrophy"][-1]
                - expected_clebsch_enstrophy
            ) / max(abs(short_result["clebsch_enstrophy"][-1]), 1e-15),
        ),
        2e-5,
    )
    expected_velocity_change = relative_enstrophy_change(
        short_result["velocity_enstrophy"][-1],
        short_result["velocity_enstrophy"][0],
    )
    expected_clebsch_change = relative_enstrophy_change(
        short_result["clebsch_enstrophy"][-1],
        short_result["clebsch_enstrophy"][0],
    )
    check(
        "enstrophy changes use signed relative change",
        max(
            abs(
                short_result["velocity_enstrophy_change"][-1]
                - expected_velocity_change
            ),
            abs(
                short_result["clebsch_enstrophy_change"][-1]
                - expected_clebsch_change
            ),
        ),
        2e-12,
    )
    test_vorticity = np.array((1.0, -2.0, 3.0))
    test_areas = np.array((0.2, 0.3, 0.5))
    test_enstrophy = discrete_enstrophy(test_vorticity, test_areas)
    check(
        "enstrophy scales quadratically with vorticity",
        abs(
            relative_enstrophy_change(
                1.25**2 * test_enstrophy, test_enstrophy
            ) - 56.25
        ),
        2e-12,
    )
    check(
        "scheduled evolved recharts occur once per interval",
        float(len(short_result["reset_events"]) != short_settings.frames - 1),
        0.0,
    )
    check(
        "scheduled rechart transfers remain small",
        max(event["velocity_jump"] for event in short_result["reset_events"]),
        5e-4,
    )
    energy_ratio = short_result["energy"] / short_result["energy"][0]
    check(
        "short nonlinear kinetic energy stays bounded",
        max(0.0, float(np.max(np.abs(energy_ratio - 1.0))) - 0.02),
        0.0,
    )
    passed = sum(results)
    print("-" * 88)
    print(f"Mandatory tests: {passed}/{len(results)} passed")
    return passed == len(results)


def visualize(result, operators, settings: Settings):
    import polyscope as ps
    import polyscope.imgui as psim

    face_velocities = np.asarray(
        [
            edge_flux_to_face_vectors(edge_flux, operators)
            for edge_flux in result["edge_fluxes"]
        ]
    )
    vertex_velocities = np.asarray(
        [
            face_vectors_to_vertex_vectors(face_velocity, operators)
            for face_velocity in face_velocities
        ]
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

    names = (
        "Speed |u|", "Local-chart Clebsch vorticity",
        "Recovered velocity vorticity", "Velocity change",
        "log(sigma_max)", "log(condition number)",
        "Relative chart area scale - 1",
        "mu_1 (material, face-based)", "mu_2 (material, face-based)",
        "mu_3 (material, face-based)", "lambda_1 = Psi_0 X",
        "lambda_2 = Psi_0 Y", "lambda_3 = Psi_0 Z",
        "Persistent dye (vertex-based)", "Projection potential Phi",
        "Chart-composition label error",
    )
    speed_view = 0
    velocity_change_view = 3
    material_views = {
        7: MU1, 8: MU2, 9: MU3,
        10: LAMBDA1, 11: LAMBDA2, 12: LAMBDA3,
    }
    dye_view = 13
    scalar_data = (
        face_speed,
        result["clebsch_vorticity"],
        result["velocity_vorticity"],
        np.linalg.norm(face_velocity_change, axis=2),
        result["log_sigma_max"],
        result["log_condition"],
        result["area_scale"] - 1.0,
        *(result["material_fields"][:, :, component] for component in range(6)),
        result["dye"],
        result["projection_potential"],
        result["composition_error"],
    )
    vertex_views = {dye_view}
    nonnegative_views = {
        speed_view, velocity_change_view, 5, dye_view, 15
    }
    defined_on = tuple(
        "vertices" if index in vertex_views else "faces"
        for index in range(len(names))
    )
    color_maps = tuple(
        "viridis" if index in nonnegative_views else "coolwarm"
        for index in range(len(names))
    )
    limits = [
        visualization_limits(values, index not in nonnegative_views)
        for index, values in enumerate(scalar_data)
    ]
    limits[speed_view] = (
        0.0,
        max(
            float(np.percentile(face_speed, 99.5)),
            float(np.percentile(vertex_speed, 99.5)),
            1e-12,
        ),
    )
    face_change_norm = np.linalg.norm(face_velocity_change, axis=2)
    vertex_change_norm = np.linalg.norm(vertex_velocity_change, axis=2)
    limits[velocity_change_view] = (
        0.0,
        max(
            float(np.percentile(face_change_norm, 99.5)),
            float(np.percentile(vertex_change_norm, 99.5)),
            1e-12,
        ),
    )
    for view, component in material_views.items():
        magnitude = max(
            float(np.percentile(np.abs(scalar_data[view]), 99.5)),
            float(
                np.percentile(
                    np.abs(vertex_material_fields[:, :, component]), 99.5
                )
            ),
            1e-12,
        )
        limits[view] = (-magnitude, magnitude)

    arrow_faces = np.arange(
        0, len(operators["faces"]), max(settings.arrow_stride, 1)
    )
    mesh_scale = float(np.linalg.norm(np.ptp(operators["vertices"], axis=0)))
    arrow_points = (
        operators["face_centers"][arrow_faces]
        + 0.002 * mesh_scale * operators["face_normals"][arrow_faces]
    )
    target_length = 0.04 * mesh_scale
    velocity_scale = target_length / max(
        float(np.percentile(face_speed[:, arrow_faces], 95)), 1e-15
    )
    change_scale = target_length / max(
        float(np.percentile(face_change_norm[:, arrow_faces], 95)), 1e-15
    )

    ps.set_use_prefs_file(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.init()
    ps.set_ground_plane_mode("none")
    surface = ps.register_surface_mesh(
        "Face-based three-pair Clebsch bunny",
        operators["vertices"],
        operators["faces"],
        smooth_shade=True,
        edge_width=0.0,
    )
    arrows = ps.register_point_cloud(
        "Velocity samples", arrow_points, radius=0.0007 * mesh_scale
    )
    frame = 0
    selected = 0
    current_time = 0.0
    playing = False
    loop_playback = True
    show_arrows = True
    show_edges = False
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
            displayed_scalars = vertex_change_norm[frame]
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
            names[selected],
            displayed_scalars,
            defined_on=scalar_location,
            enabled=True,
            cmap=color_maps[selected],
            vminmax=limits[selected],
        )

        if selected == velocity_change_view:
            vector_field = (
                vertex_velocity_change if smooth_display else face_velocity_change
            )
            scale, color = change_scale, (0.95, 0.10, 0.05)
        else:
            vector_field = vertex_velocities if smooth_display else face_velocities
            scale, color = velocity_scale, (0.05, 0.25, 1.0)
        if smooth_display:
            vectors = vector_field[frame][
                operators["faces"][arrow_faces]
            ].mean(axis=1)
            normals = operators["face_normals"][arrow_faces]
            vectors -= np.sum(vectors * normals, axis=1)[:, None] * normals
        else:
            vectors = vector_field[frame, arrow_faces]
        arrows.add_vector_quantity(
            "Vectors",
            scale * vectors,
            enabled=show_arrows,
            vectortype="ambient",
            color=color,
            radius=0.0025,
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
        psim.SetNextWindowPos(
            (width - 12.0 * scale, 12.0 * scale), pivot=(1.0, 0.0)
        )
        psim.SetNextWindowBgAlpha(0.94)
        flags = (
            psim.ImGuiWindowFlags_AlwaysAutoResize
            | psim.ImGuiWindowFlags_NoCollapse
            | psim.ImGuiWindowFlags_NoMove
            | psim.ImGuiWindowFlags_NoSavedSettings
        )
        begin = psim.Begin("Three-pair surface Clebsch Euler", True, flags=flags)
        visible = begin[0] if isinstance(begin, tuple) else begin
        if visible:
            psim.PushItemWidth(255.0 * scale)
            changed_view, new_selected = psim.Combo("Field", selected, names)
            changed_time, new_time = psim.SliderFloat(
                "Time", current_time, 0.0, settings.final_time
            )
            if psim.Button("Pause" if playing else "Play"):
                playing = not playing
                accumulator = 0.0
            psim.SameLine()
            previous = psim.Button("Previous")
            psim.SameLine()
            following = psim.Button("Next")
            changed_rate, new_rate = psim.SliderFloat(
                "Playback frames/s", playback_rate, 1.0, 60.0
            )
            changed_loop, new_loop = psim.Checkbox("Loop", loop_playback)
            changed_arrows, new_arrows = psim.Checkbox(
                "Velocity arrows", show_arrows
            )
            changed_smoothing, new_smoothing = psim.Checkbox(
                "Smooth velocity/Clebsch display", smooth_display
            )
            changed_edges, new_edges = psim.Checkbox(
                "Show triangle edges", show_edges
            )
            new_frame = int(np.argmin(np.abs(result["times"] - new_time)))
            if previous:
                playing, new_frame, changed_time = False, max(frame - 1, 0), True
            elif following:
                playing, new_frame, changed_time = (
                    False, min(frame + 1, len(result["times"]) - 1), True
                )
            if changed_rate:
                playback_rate = new_rate
            if changed_loop:
                loop_playback = new_loop
            if changed_arrows:
                show_arrows = new_arrows
                arrows.set_enabled(show_arrows)
            if changed_edges:
                show_edges = new_edges
                surface.set_edge_width(0.5 if show_edges else 0.0)
            if changed_view or changed_smoothing or (
                changed_time and new_frame != frame
            ):
                remove_quantities()
                selected, frame = new_selected, new_frame
                smooth_display = new_smoothing
                current_time = float(result["times"][frame])
                add_quantities()

            l2_loss = 100.0 * (
                1.0
                - result["l2"][frame]
                / np.maximum(result["l2"][0], 1e-15)
            )
            psim.Separator()
            psim.TextUnformatted("Closed surface; three face-based Clebsch pairs")
            psim.TextUnformatted("lambda^0 = bunny embedding coordinates (x,y,z)")
            psim.TextUnformatted("SSPRK3; stabilization: " + settings.stabilization)
            psim.TextUnformatted(
                f"Frame: {frame + 1}/{len(result['times'])}; "
                f"t={result['times'][frame]:.5f}"
            )
            psim.TextUnformatted(
                f"Chart: {int(result['chart'][frame])}; "
                f"age={result['chart_age'][frame]:.4f}"
            )
            psim.TextUnformatted(
                f"CFL since previous frame: {result['step_cfl'][frame]:.3e}"
            )
            psim.TextUnformatted(
                f"Maximum speed: {result['maximum_speed'][frame]:.3e}"
            )
            psim.TextUnformatted(
                f"Maximum log(sigma_max): "
                f"{np.max(result['log_sigma_max'][frame]):.3e}"
            )
            psim.TextUnformatted(
                f"Minimum relative chart area: "
                f"{np.min(result['area_scale'][frame]):.3e}"
            )
            psim.TextUnformatted(
                f"Relative divergence: {result['divergence'][frame]:.3e}"
            )
            psim.TextUnformatted(
                "Relative energy change: "
                f"{percentage_change(result['energy'], frame):+.3e}%"
            )
            psim.TextUnformatted(
                "Projected-u enstrophy change: "
                f"{result['velocity_enstrophy_change'][frame]:+.3e}%"
            )
            psim.TextUnformatted(
                "Local-chart Clebsch enstrophy change: "
                f"{result['clebsch_enstrophy_change'][frame]:+.3e}%"
            )
            psim.TextUnformatted(
                "Material mu L2 loss 1/2/3: "
                f"{l2_loss[MU1]:+.2e}% / {l2_loss[MU2]:+.2e}% / "
                f"{l2_loss[MU3]:+.2e}%"
            )
            psim.TextUnformatted(
                "Material lambda L2 loss 1/2/3: "
                f"{l2_loss[LAMBDA1]:+.2e}% / {l2_loss[LAMBDA2]:+.2e}% / "
                f"{l2_loss[LAMBDA3]:+.2e}%"
            )
            psim.TextUnformatted(
                f"Persistent dye L2 loss: {l2_loss[DYE_INDEX]:+.3e}%"
            )
            psim.TextUnformatted(
                f"Maximum composition error: "
                f"{np.max(result['composition_error'][frame]):.3e}"
            )
            psim.TextUnformatted(
                f"Recharts so far: {int(result['chart'][frame])}"
            )
            psim.TextUnformatted(
                "Display smoothing only; face state and DEC flux are unchanged"
                if smooth_display
                else "Showing raw piecewise-constant face fields and velocity"
            )
            psim.PopItemWidth()
        psim.End()

    ps.set_user_callback(callback)
    ps.show()


def parse_arguments():
    defaults = Settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mesh", dest="mesh_path", type=Path, default=defaults.mesh_path,
        help=f"closed triangular surface (default: {defaults.mesh_path})",
    )
    parser.add_argument("--final-time", type=float, default=defaults.final_time)
    parser.add_argument("--frames", type=int, default=defaults.frames)
    parser.add_argument("--cfl", type=float, default=defaults.cfl)
    parser.add_argument(
        "--maximum-time-step", type=float, default=defaults.maximum_time_step
    )
    parser.add_argument("--target-speed", type=float, default=defaults.target_speed)
    parser.add_argument(
        "--rechart-cutoff", type=float, default=defaults.rechart_cutoff
    )
    parser.add_argument(
        "--rechart-every-frames", type=int,
        default=defaults.rechart_every_frames,
        help="reset the hidden local chart every N stored-frame intervals; 0 disables",
    )
    parser.add_argument(
        "--singular-value-floor", type=float,
        default=defaults.singular_value_floor,
    )
    parser.add_argument(
        "--dye-width-fraction", type=float,
        default=defaults.dye_width_fraction,
    )
    parser.add_argument(
        "--stabilization", choices=("centered", "graph"),
        default=defaults.stabilization,
    )
    parser.add_argument(
        "--graph-viscosity", type=float, default=defaults.graph_viscosity
    )
    parser.add_argument("--arrow-stride", type=int, default=defaults.arrow_stride)
    parser.add_argument(
        "--playback-fps", type=float, default=defaults.playback_fps
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="run the simulation and report without opening Polyscope",
    )
    parser.add_argument(
        "--diagnostics-plot", metavar="PATH",
        help=(
            "save a PNG time-series plot of relative divergence, kinetic-energy "
            "change, and the two relative enstrophy changes"
        ),
    )
    parser.add_argument(
        "--show-diagnostic-plots", action="store_true",
        help="show the diagnostic time-series plot in a Matplotlib window",
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="run the bunny operator/rechart/nonlinear regression suite and exit",
    )
    arguments = parser.parse_args()
    if (
        arguments.final_time <= 0.0
        or arguments.frames < 2
        or not (0.0 < arguments.cfl <= 1.0)
        or arguments.maximum_time_step <= 0.0
        or arguments.target_speed <= 0.0
        or arguments.rechart_cutoff <= 0.0
        or arguments.rechart_every_frames < 0
        or arguments.singular_value_floor < 0.0
        or arguments.dye_width_fraction <= 0.0
        or arguments.graph_viscosity < 0.0
        or arguments.arrow_stride < 1
        or arguments.playback_fps <= 0.0
    ):
        parser.error(
            "require positive times/speed/width/playback, frames >= 2, "
            "0 < cfl <= 1, nonnegative rechart interval/singular floor/viscosity, "
            "and arrow-stride >= 1"
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


def main():
    (
        settings, headless, run_tests,
        diagnostics_plot, show_diagnostic_plots,
    ) = parse_arguments()
    vertices, faces, mesh_path = load_surface_mesh(settings.mesh_path)
    operators = build_operators(vertices, faces)
    if run_tests:
        if not run_diagnostic_tests(operators, settings):
            raise SystemExit(1)
        return

    result = simulate(operators, settings)
    print_report(result, operators, mesh_path)
    if diagnostics_plot is not None or show_diagnostic_plots:
        plot_diagnostic_histories(
            result, settings,
            output_path=diagnostics_plot,
            show=show_diagnostic_plots,
        )
    if not headless:
        try:
            visualize(result, operators, settings)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("polyscope"):
                raise SystemExit(
                    "Polyscope is required for interactive viewing; install polyscope "
                    "or rerun with --headless."
                ) from error
            raise


if __name__ == "__main__":
    main()
