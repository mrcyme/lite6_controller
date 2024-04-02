
# %%
import swift
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve
import time


env = swift.Swift()
env.launch(realtime=True)

# Create a puma in the default zero pose
lite6 = rtb.models.URDF.Lite6()
lite6.q = lite6.qz
env.add(lite6, robot_alpha=True, collision_alpha=False)

def jacobian_i_k_i(q, v):
    # jacobian inverse kinematics with pseudo-inverse
    J = lite6.jacobe(q)
    qd = np.linalg.inv(J) @ v
    return qd


def jacobian_i_k_pi(q, v):
    # jacobian inverse kinematics with pseudo-inverse
    J = lite6.jacobe(q)
    qd = np.linalg.pinv(J) @ v
    return qd

def jacobian_i_k_v2(q, v,  dt):
    success = False
    pos_d = v.rpy()
    quat_d = R.from_euler('xyz', pos_d).as_quat()
    iters = 1000
    tol = 0.01
    damp = 1
    for _ in range(iters):
        pose = lite6.fkine(q)
        rpy = pose.rpy()
        t = pose.t
        quat = R.from_euler('xyz', rpy).as_quat()
        error = p.zeros(6)
        error[:3] = pos_d - t
        orientation_error = quat[0] * quat_d[1:] - quat_d[0] * quat_d[1:] -np.cross(quat_d[1:], quat[1:])
        error[3:] = orientation_error
        if np.linalg.norm(error) < tol:
            success = True
            break
        J = lite6.jacobe(q)
        JT = np.transpose(J)
        qd = JT @ np.linalg.inv(J @ JT + damp**2 * np.eye(6)) @ error
        return qd, success
    
def jacobian_i_k_optimisation(q, v, v_max=1.2):
    # jacobian inverse kinematics with optimisation
    J = lite6.jacobe(q)
    prog = MathematicalProgram()
    v_opt = prog.NewContinuousVariables(6, "v_opt")

    # Define the error term for the cost function
    error = J @ v_opt - v
    prog.AddCost(error.dot(error))

    # Add bounding box constraint for joint velocities
    lower_bounds = [-v_max] * 6  # Lower bounds for each joint velocity
    upper_bounds = [v_max] * 6   # Upper bounds for each joint velocity
    prog.AddBoundingBoxConstraint(lower_bounds, upper_bounds, v_opt)

    # Solve the optimization problem
    result = Solve(prog)

    return result.is_success(), result.GetSolution(v_opt)


def jacobian_i_k_optimisation_v2(q, v, v_max=1.2):
    # jacobian inverse kinematics with optimisation
    J = lite6.jacobe(q)
    prog = MathematicalProgram()
    v_opt = prog.NewContinuousVariables(6, "v_opt")
    alpha = prog.NewContinuousVariables(1, "alpha")
    lower_bounds = [-v_max] * 6  # Lower bounds for each joint velocity
    upper_bounds = [v_max] * 6   # Upper bounds for each joint velocity
    prog.AddBoundingBoxConstraint(lower_bounds, upper_bounds, v_opt)

    alpha_min = 0
    alpha_max = 1
    prog.AddBoundingBoxConstraint(alpha_min, alpha_max, alpha)
    prog.AddCost(-alpha[0])
    V = J @ v_opt
    for i in range(len(V)):
        prog.AddConstraint(V[i] == alpha[0] * v[i])
    # Define the error term for the cost function
    #error = J @ v_opt - v
    #prog.AddCost(error.dot(error))


    # Solve the optimization problem
    result = Solve(prog)

    return result.is_success(), result.GetSolution(v_opt)

    

Tep = lite6.fkine(lite6.q)*sm.SE3.Tx(0.2)*sm.SE3.Ty(0.2)*sm.SE3.Tz(0.1)

axes = sg.Axes(length=0.1, pose=Tep)
env.add(axes)
dt = 0.05
arrived = False
while not arrived:
    v, arrived = rtb.p_servo(lite6.fkine(lite6.q), Tep, gain=1, threshold=0.01)
    print(v)
    #lite6.qd = jacobian_i_k_pi(lite6.q, v, dt)
    qd = jacobian_i_k_optimisation(lite6.q, v, v_max=1)[1]
    time.sleep(dt)
    lite6.qd = qd
    env.step(dt)


"""
for _ in range(100):, randn
    env.step(dt)
"""
# Uncomment to stop the browser tab from closing
env.hold()


# %%
import roboticstoolbox as rtb
lite6 = rtb.models.URDF.Lite6()
lite6.plot(lite6.qz)

# %%
