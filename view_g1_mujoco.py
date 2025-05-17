"""
Script to load and inspect the G1 robot model in Mujoco.
Displays joint information, actuator details, and opens the viewer.
"""

import os
import time
import mujoco
import mujoco.viewer
import numpy as np

def print_model_info(model: mujoco.MjModel, data: mujoco.MjData):
    """Print detailed information about the Mujoco model."""
    print("\n=== G1 Robot Model Information ===")
    
    # Basic model info
    print(f"\nGeneral Info:")
    print(f"Number of DOFs (nv): {model.nv}")
    print(f"Number of Joints (njnt): {model.njnt}")
    print(f"Number of Actuators (nu): {model.nu}")
    print(f"Number of Bodies (nbody): {model.nbody}")
    print(f"Simulation Timestep: {model.opt.timestep}")
    
    # Joint information
    print(f"\nJoint Information:")
    print(f"{'Index':<6} {'Type':<12} {'Name':<30} {'qpos idx':<10} {'qvel idx':<10} {'Limited':<8} {'Range':<20}")
    print("-" * 90)
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        jnt_type = model.jnt_type[i]
        jnt_type_str = ['free', 'ball', 'slide', 'hinge'][jnt_type]
        qpos_start = model.jnt_qposadr[i]
        qvel_start = model.jnt_dofadr[i]
        limited = "yes" if model.jnt_limited[i] else "no"
        if model.jnt_limited[i]:
            range_str = f"{model.jnt_range[i][0]:.2f}, {model.jnt_range[i][1]:.2f}"
        else:
            range_str = "N/A"
        print(f"{i:<6} {jnt_type_str:<12} {jnt_name:<30} {qpos_start:<10} {qvel_start:<10} {limited:<8} {range_str:<20}")
    
    # Actuator information
    print(f"\nActuator Information:")
    print(f"{'Index':<6} {'Name':<30} {'Joint Controlled':<30}")
    print("-" * 70)
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        # Get the joint name this actuator controls
        joint_id = model.actuator_trnid[i, 0]  # First element is the joint id
        if joint_id >= 0:
            joint_name = model.joint(joint_id).name
        else:
            joint_name = "N/A"
        print(f"{i:<6} {actuator_name:<30} {joint_name:<30}")
    
    # Body information
    print(f"\nBody Information:")
    print(f"{'Index':<6} {'Name':<30} {'Parent':<30} {'Mass':<10}")
    print("-" * 80)
    for i in range(model.nbody):
        body_name = model.body(i).name
        parent_id = model.body_parentid[i]
        parent_name = model.body(parent_id).name if parent_id >= 0 else "world"
        mass = model.body_mass[i]
        print(f"{i:<6} {body_name:<30} {parent_name:<30} {mass:<10.3f}")

def main():
    # Load the model
    xml_path = os.path.join(os.path.dirname(__file__), "assets/robots/g1/scene_29dof.xml")
    print(f"Loading Mujoco model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Print model information
    print_model_info(model, data)
    
    # Launch the viewer
    print("\nLaunching viewer... Press Ctrl+C to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial pose
        data.qpos[2] = 0.78  # Set initial height
        viewer.sync()
        
        # Keep the viewer open
        try:
            while viewer.is_running():
                step_start = time.time()
                
                # Step the simulation
                # mujoco.mj_step(model, data)
                
                # Sync the viewer
                viewer.sync()
                
                # Cap the simulation at real time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main() 