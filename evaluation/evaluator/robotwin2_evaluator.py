from evaluation.policies.robotwin2_vla_agent import RoboTwin2VLAAgent


def encode_obs(observation):  # Post-Process Observation
    head_camera_rgb = observation["observation"]["head_camera"]["rgb"]
    front_camera_rgb = observation["observation"]["front_camera"]["rgb"]
    left_camera_rgb = observation["observation"]["left_camera"]["rgb"]
    right_camera_rgb = observation["observation"]["right_camera"]["rgb"]
    current_state = observation["joint_action"]["vector"]

    obs = {
        "head_camera_rgb": head_camera_rgb,
        "front_camera_rgb": front_camera_rgb,
        "left_camera_rgb": left_camera_rgb,
        "right_camera_rgb": right_camera_rgb,
        "current_state": current_state,
    }
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    camera_names = usr_args["cameras"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_horizon = usr_args.get("action_horizon", 8)
    action_mode = usr_args.get("action_mode", "relative")
    api_style = usr_args.get("api_style", "legacy")
    sampling = usr_args.get("sampling", None)
    client = RoboTwin2VLAAgent(
        usr_args["base_url"],
        camera_names,
        action_horizon,
        action_mode,
        api_style=api_style,
        sampling=sampling,
    )
    return client


def eval(TASK_ENV, model: RoboTwin2VLAAgent, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    rgbs = []
    for camera_name in model.use_cameras:
        rgb = obs[camera_name]
        rgbs.append(rgb)

    actions = model.get_action(
        instruction=instruction, 
        rgbs=rgbs,
        state=obs["current_state"],
    )  # Get Action according to observation chunk
    if model.action_mode == "absolute":
        pass
    elif model.action_mode == "relative":
        actions = model.convert_relative_to_absolute_action(
            state=obs["current_state"], 
            action_chunk=actions,
        )
    else:  # delta
        actions = model.convert_delta_to_absolute_action(
            state=obs["current_state"], 
            action_chunk=actions,
        )

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        

def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset()
