from unified_vla.prompt import build_vlm_prompt, IMAGE_PLACEHOLDER


class TestBuildVLMPrompt:
    def test_returns_string(self):
        out = build_vlm_prompt("robot", "do something")
        assert isinstance(out, str)

    def test_contains_im_start_end(self):
        prompt = build_vlm_prompt("robot", "do something")
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in prompt

    def test_image_placeholders_count(self):
        for n_cam in (1, 3, 5):
            prompt = build_vlm_prompt("robot", "task", num_cameras=n_cam)
            assert prompt.count(IMAGE_PLACEHOLDER) == n_cam

    def test_no_state_line(self):
        """Proprio is no longer textualized in the prompt."""
        prompt = build_vlm_prompt("robot", "task")
        assert "State:" not in prompt
        assert "tx=" not in prompt
        assert "grip=" not in prompt

    def test_task_wrapped_in_tags(self):
        prompt = build_vlm_prompt("robot", "pick up the cup")
        assert "<task>pick up the cup</task>" in prompt

    def test_ends_with_assistant_turn(self):
        prompt = build_vlm_prompt("robot", "task")
        assert prompt.rstrip().endswith("<|im_start|>assistant")

    def test_robot_description_present(self):
        desc = "7-DOF Trossen arm with parallel gripper"
        prompt = build_vlm_prompt(desc, "task")
        assert f"Robot: {desc}" in prompt

    def test_system_prompt_present(self):
        prompt = build_vlm_prompt("robot", "task")
        assert "robot manipulation assistant" in prompt

    def test_section_ordering(self):
        """system → user (images, robot, task) → assistant."""
        prompt = build_vlm_prompt("robot", "task")
        sys_pos = prompt.index("<|im_start|>system")
        user_pos = prompt.index("<|im_start|>user")
        img_pos = prompt.index("<|vision_start|>")
        robot_pos = prompt.index("Robot:")
        task_pos = prompt.index("<task>")
        asst_pos = prompt.index("<|im_start|>assistant")

        assert sys_pos < user_pos < img_pos < robot_pos < task_pos < asst_pos
