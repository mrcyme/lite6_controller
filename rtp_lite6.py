# %%
import swift
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve
import time
import math

MODES = ["simulation"]
LITE6 = rtb.models.URDF.LITE6()
LITE6.q = LITE6.qz


if "real" in MODES:
    from xarm.wrapper import XArmAPI
    arm = XArmAPI("192.168.1.159")
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(1)
    arm.reset(wait=True)
if "simulation" in MODES:
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(LITE6, robot_alpha=True, collision_alpha=False)


def degrees_to_radians(angle_list):
    return [math.radians(angle) for angle in angle_list]

def jacobian_i_k_optimisation(q, v, v_max=1.2):
    # jacobian inverse kinematics with optimisation
    J = LITE6.jacobe(q)
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


def move_to(dest, dt, gain=1, treshold=0.01, modes=["simulation"]):
    dest = LITE6.fkine(dest)
    if "simulation" in modes:
        axes = sg.Axes(length=0.1, pose=dest)
        env.add(axes)
    if "real" in modes:
        # set joint velocity control mode
        arm.set_mode(4)
        arm.set_state(0)
        time.sleep(0.1)
    arrived = False
    while not arrived:
        v, arrived = rtb.p_servo(LITE6.fkine(LITE6.q), dest, gain=gain, threshold=treshold)
        qd = jacobian_i_k_optimisation(LITE6.q, v, v_max=1)[1]
        LITE6.qd = qd
        if "real" in modes:
            arm.vc_set_joint_velocity(qd, is_radian=True)
        if "simulation" in modes:
            env.step(dt)
    return arrived
            
dest = [50, 0, 16, 14, 7, 0]
dest = degrees_to_radians(dest)
move_to(dest, 0.05, gain=1, treshold=0.01, modes=MODES)
# Uncomment to stop the browser tab from closing
#env.hold()
# %%
