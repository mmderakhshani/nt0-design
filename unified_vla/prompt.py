SYSTEM_PROMPT = (
    "You are a robot manipulation assistant. Given camera observations, robot state, and a "
    "task instruction, understand the scene by identifying objects, their spatial relationships, "
    "colors, shapes, sizes, and textures. Pay attention to graspable surfaces, contact points,"
    " and geometric constraints relevant to manipulation."
)

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def build_vlm_prompt(
    robot_description: str,
    task_instruction: str,
    num_cameras: int = 3,
) -> str:
    """Build the VLM prompt string following the Qwen-VL chat template.

    Embodiment is no longer spliced into the VLM segment — it lives as
    learned tokens inside the PC and Action segments via
    `SharedEmbodimentEmbedding`. The textual `Robot: {description}` line
    still appears here so the frozen VLM has a semantic embodiment cue.

    Proprio is also not textualized — it enters the Action expert as
    projected tokens via `ProprioEncoder`.

    Args:
        robot_description: e.g. "7-DOF Trossen arm with parallel gripper".
        task_instruction: e.g. "Pick up the red cup and place it on the white plate."
        num_cameras: number of camera images (default 3).
    Returns:
        Prompt string ready for Qwen-VL tokenizer.
    """
    image_tokens = "\n".join(IMAGE_PLACEHOLDER for _ in range(num_cameras))

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{image_tokens}\n"
        f"Robot: {robot_description}\n"
        f"<task>{task_instruction}</task><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
