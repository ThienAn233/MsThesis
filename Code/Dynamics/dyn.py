import numpy as np

# -----------------------------------------------------------
# Utility: rotation matrix from ZYX Euler angles
# -----------------------------------------------------------
def R_zyx(phi, theta, psi):
    """Rotation matrix R = Rz(psi) * Ry(theta) * Rx(phi)."""
    c1, c2, c3 = np.cos([phi, theta, psi])
    s1, s2, s3 = np.sin([phi, theta, psi])
    Rz = np.array([
        [c3, -s3, 0],
        [s3,  c3, 0],
        [0,    0, 1]
    ])
    Ry = np.array([
        [c2, 0, s2],
        [0,  1, 0],
        [-s2, 0, c2]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, c1, -s1],
        [0, s1,  c1]
    ])
    return Rz @ Ry @ Rx


def skew(v):
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

# -----------------------------------------------------------
# MAIN DYNAMICS ROUTINES
# -----------------------------------------------------------


def inertia_matrix(q, m, Ib):
    """
    Build full 6x6 inertia matrix M(q) for:
        q_b = [x,y,z,phi,theta,psi]  (floating base)
    m  = total body mass
    Ib = 3x3 body inertia tensor in body frame
    """
    # Extract Euler angles
    phi, theta, psi = q[3], q[4], q[5]

    # Rotation from body to world
    R = R_zyx(phi, theta, psi)

    # Base inertia block 6x6
    Mbb = np.zeros((6, 6))
    Mbb[0:3, 0:3] = m * np.eye(3)                 # linear inertia
    Mbb[3:6, 3:6] = R @ Ib @ R.T                  # rotational inertia
    return Mbb


def gravity_vector(m, g=9.81):
    """
    Gravity vector g(q) for 6-DOF quadruped.
    ONLY affects base because legs are massless.
    """
    gvec = np.zeros(6)
    gvec[2] = -m * g     # only z-axis gravity force
    return gvec
print(np.shape(gravity_vector(1)))


def coriolis_term(dq, m, Ib):
    """
    Compute C(q,dq)*dq for floating base.
    Massless legs => zero contribution.
    """
    # Extract velocities
    v = dq[0:3]      # linear velocity of base
    omega = dq[3:6]  # angular velocity of base

    # Coriolis term for a rigid body
    C_term = np.zeros(6)

    # Linear part: m * (omega x v)
    C_term[0:3] = m * np.cross(omega, v)

    # Angular part: omega x (I*omega)
    C_term[3:6] = np.cross(omega, Ib @ omega)

    return C_term


def nonlinear_term(dq, m, Ib, g=9.81):
    """
    b(q,dq) = C(q,dq)*dq + g(q)
    """
    Cqdq = coriolis_term(dq, m, Ib)
    G = gravity_vector(m, g)
    return Cqdq + G


def J_contact_base_one_point(r_cb):
    """
    r_cb : 3-vector from base COM to contact point
    Returns J_cb : (3x6)
    [ I   -S(r) ]
    """
    J_trans = np.eye(3)
    J_rot   = -skew(r_cb)
    return np.hstack([J_trans, J_rot])

def J_contact_base(r_cb):
    """
    r_cb : list of N 3d-vectors from base COM to contact points (N contact points)
    Returns J_cb : (3N x 6)
    """
    J = np.zeros((3*len(r_cb), 6))
    for i, r in enumerate(r_cb):
        J[i*3:(i+1)*3, :] = J_contact_base_one_point(r)
    return J

def J_contact_joint_one_leg(p_c, joint_positions, joint_axes):
    """
    p_c : (3,) contact position in base frame
    joint_positions : list of joint origins p_k (each 3,) in base frame
    joint_axes : list of joint axes z_k (each 3,) in base frame
    Returns J_leg : (3x3)
    """
    J = np.zeros((3, len(joint_axes)))

    for k, (p_k, z_k) in enumerate(zip(joint_positions, joint_axes)):
        J[:, k] = np.cross(z_k, p_c - p_k)

    return J


def J_contact_joint(q_j, fk_leg_funcs):
    """
    q_j : (N,3) joint vector
    fk_leg_funcs : list of N FK functions (one per leg)

    Each fk_leg(q_leg) must return:
      {
        "p_c": (3,),
        "joint_positions": [p1, p2, p3],
        "joint_axes": [z1, z2, z3]
      }

    Returns:
      J_cj : (12x12)
    """
    N = len(q_j)
    J = np.zeros((3*N, 3*N))

    for leg_id in range(N):
        q_leg = q_j[leg_id]
        fk = fk_leg_funcs[leg_id](q_leg)

        J_leg = J_contact_joint_one_leg(
            fk["p_c"],
            fk["joint_positions"],
            fk["joint_axes"]
        )

        # Insert block
        J[3*leg_id:3*(leg_id+1),
          3*leg_id:3*(leg_id+1)] = J_leg

    return J


# -----------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------
if __name__ == "__main__":
    # Example body mass and inertia
    m = 10.0  # kg
    Ib = np.diag([0.2, 0.3, 0.15])  # inertia tensor

    # Example state
    q = np.zeros(6)     # all positions = 0
    dq = np.zeros(6)    # all velocities = 0

    M = inertia_matrix(q, m, Ib)
    gvec = gravity_vector(m)
    bvec = nonlinear_term(dq, m, Ib)

    print("shape of M:", M.shape)
    print("shape of gvec:", gvec.shape)
    print("shape of bvec:", bvec.shape)
