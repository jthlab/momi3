# docstring style: Google
"Code for projecting onto a polyhedron"

import cvxpy as cp
import jax
import numpy as np


def _make_projection_dpp(A, b, G, h, tol):
    """Project point x onto a polyhedron defined by Ax = b, Gx <= h.

    Params:
        A: The matrix in the equality constraint Ax = b.
        b: The vector in the equality constraint Ax = b.
        G: The matrix in the inequality constraint Gx <= h.
        h: The vector in the inequality constraint Gx <= h.

    Returns:
        The projection of x onto the polyhedron defined by Ax = b, Gx <= h.
    """
    # Define and solve the optimization problem.
    x = cp.Parameter(A.shape[1])
    y = cp.Variable(x.shape)
    # minimize the squared distance between x and y
    objective = cp.Minimize(cp.norm(x - y))
    constraints = []
    # cp doesn't like empty constraints
    if A.shape[0] > 0:
        constraints.append(A @ y == b)
    if G.shape[0] > 0:
        constraints.append(G @ y <= h - tol)
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    return problem


def project_polyhedron(A, b, G, h, verbose: bool = False, tol: float = 1e-6):
    """Jittable function that projects onto given polyedron"""
    prob = _make_projection_dpp(A, b, G, h, tol)

    def solve(x):
        try:
            xp = prob.parameters()[0]
            y = prob.variables()[0]
            xp.value = np.array(x)
            prob.solve(solver=cp.CLARABEL, verbose=verbose)
        except ValueError as e:
            raise ValueError(f"Projection failed when x={x}") from e
        return y.value

    def ret(x):
        result_shape_dtypes = jax.ShapeDtypeStruct(x.shape, x.dtype)
        ret = jax.pure_callback(solve, result_shape_dtypes, x)
        return ret

    return ret
