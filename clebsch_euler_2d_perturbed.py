#!/usr/bin/env python3
"""Perturbed-Taylor--Green entry point for the face-based 2D Clebsch solver.

The initial streamfunction on ``[0, 2*pi]^2`` is

    psi = sin(x) sin(y) + epsilon sin(2x) sin(3y),

so ``u = (partial_y psi, -partial_x psi)`` gives

    u_x = sin(x) cos(y) + 3 epsilon sin(2x) cos(3y),
    u_y = -cos(x) sin(y) - 2 epsilon cos(2x) sin(3y).

The default is ``epsilon=0.25`` and can be changed with
``--perturbation-amplitude``.  At face centroids, the shared solver initializes
``lambda=(x,y)`` and ``mu=(u_x,u_y)``, then reconstructs and projects the edge
flux.  All remaining numerics and visualization come from
``clebsch_euler_2d.py`` so this file differs from the TGV experiment only in
its prescribed initial condition.  The perturbation contains a ``k_y=3`` mode;
for quantitative work, increase the mesh resolution (for example,
``--resolution 32``) and perform a convergence study.
"""

import clebsch_euler_2d as solver


def main():
    solver.main(
        fixed_initial_condition=solver.PERTURBED,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
