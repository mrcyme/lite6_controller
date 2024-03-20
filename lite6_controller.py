import swift
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve

env = swift.Swift()
env.launch(realtime=True)

# Create a puma in the default zero pose
lite6 = rtb.models.URDF.Lite6()
lite6.q = lite6.qz
env.add(lite6, robot_alpha=True, collision_alpha=False)


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

Tep = lite6.fkine(lite6.q)*sm.SE3.Tx(0.2)*sm.SE3.Ty(0.2)*sm.SE3.Tz(0.1)

def go_to(pose):
    axes = sg.Axes(length=0.1, pose=Tep)
    env.add(axes)
    dt = 0.05
    arrived = False
    while not arrived:
        v, arrived = rtb.p_servo(lite6.fkine(lite6.q), pose, gain=1, threshold=0.01)
        #lite6.qd = jacobian_i_k_pi(lite6.q, v, dt)
        qd = jacobian_i_k_optimisation(lite6.q, v, v_max=1)[1]
        lite6.qd = qd
        env.step(dt)
    return arrived

Tep = lite6.fkine(lite6.q)*sm.SE3.Tx(0.2)*sm.SE3.Ty(0.2)*sm.SE3.Tz(0.1)
arrived = go_to(Tep)