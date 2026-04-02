import os
import re
import string
from typing import Dict, List, Optional

from loguru import logger as eval_logger


HF_DATASETS_CACHE = os.environ.get(
    "HF_DATASETS_CACHE", "~/.cache/huggingface"
)
BASE_CACHE_DIR = os.path.expanduser(HF_DATASETS_CACHE)


def doc_to_visual(
    doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None
) -> List[str]:
    """Load video path from the dataset."""
    video_rel = doc["video"]
    video_path = os.path.join(BASE_CACHE_DIR, video_rel)
    if not os.path.exists(video_path):
        eval_logger.error(f"Video path: {video_path} does not exist")
    return [video_path]


def doc_to_text(
    doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None
) -> str:
    """Format multiple-choice question with options."""
    question = doc["question"]
    candidates = doc["candidates"]
    option_letters = string.ascii_uppercase

    option_prompt = ""
    for i, option in enumerate(candidates):
        option_prompt += f"({option_letters[i]}) {option}\n"

    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return (
        f"Question:{question}\n"
        f"Option:\n{option_prompt}"
        f"{post_prompt}"
    )


def _extract_letter(text: str) -> str:
    """Extract a single option letter (A-Z) from model prediction."""
    text = text.strip()

    match = re.match(r"^([A-Z])\.\s", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.match(r"^\(?([A-Z])\)?$", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Z])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return text.strip().upper()


def process_results(doc: dict, results: list) -> Dict[str, dict]:
    """Process model prediction and compute accuracy."""
    pred = results[0].strip()
    candidates = doc["candidates"]
    answer = str(doc["answer"])
    option_letters = string.ascii_uppercase

    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        gt_letter = answer.upper()
    else:
        gt_letter = None
        for i, candidate in enumerate(candidates):
            if str(candidate) == answer:
                gt_letter = option_letters[i]
                break
        if gt_letter is None:
            eval_logger.warning(
                f"Answer '{answer}' not found in candidates: "
                f"{candidates}"
            )
            gt_letter = "?"

    pred_letter = _extract_letter(pred)
    score = 1 if pred_letter == gt_letter else 0

    return {
        "direction_testbed_ablation_accuracy": {
            "pred_answer": pred,
            "pred_letter": pred_letter,
            "gt_answer": gt_letter,
            "score": score,
        }
    }


def aggregate_results(results: list) -> float:
    """Compute accuracy over all results."""
    total = 0
    correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total += 1
            correct += result["score"]
    return 100 * correct / total if total > 0 else 0
