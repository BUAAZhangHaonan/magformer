from __future__ import annotations

import ast
from pathlib import Path


EVALUATE_TTA = Path(__file__).resolve().parents[1] / "tools" / "evaluate_tta.py"


def _source_tree() -> ast.Module:
    return ast.parse(EVALUATE_TTA.read_text(encoding="utf-8"))


def _is_args_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def _list_of_strings(node: ast.AST) -> list[str] | None:
    if not isinstance(node, ast.List):
        return None
    values = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return values


def _parser_add_argument_calls(tree: ast.Module) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "add_argument":
            calls.append(node)
    return calls


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def test_tta_cli_defaults_to_bbox_and_segm_iou_types() -> None:
    tree = _source_tree()

    for call in _parser_add_argument_calls(tree):
        if not call.args:
            continue
        first_arg = call.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != "--iou-types":
            continue

        assert isinstance(_keyword(call, "nargs"), ast.Constant)
        assert _keyword(call, "nargs").value == "+"
        assert _list_of_strings(_keyword(call, "default")) == ["bbox", "segm"]
        return

    raise AssertionError("evaluate_tta.py must expose --iou-types")


def test_tta_evaluator_uses_cli_iou_types() -> None:
    tree = _source_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "COCOEvaluator":
            assert _is_args_attr(_keyword(node, "iou_types"), "iou_types")
            return

    raise AssertionError("evaluate_tta.py must construct COCOEvaluator")


def test_tta_coco_export_uses_separate_export_score_threshold() -> None:
    tree = _source_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "predictions_to_coco_instances":
            assert isinstance(_keyword(node, "score_threshold"), ast.Name)
            assert _keyword(node, "score_threshold").id == "export_score_thresh"
            assert isinstance(_keyword(node, "mask_threshold"), ast.Name)
            assert _keyword(node, "mask_threshold").id == "export_mask_thresh"
            return

    raise AssertionError("evaluate_tta.py must export predictions through predictions_to_coco_instances")


def test_tta_cli_exposes_separate_pre_and_export_thresholds() -> None:
    tree = _source_tree()
    exposed = set()

    for call in _parser_add_argument_calls(tree):
        if not call.args:
            continue
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant):
            exposed.add(first_arg.value)

    assert "--pre-score-thresh" in exposed
    assert "--export-score-thresh" in exposed
    assert "--cluster-mask-thresh" in exposed
    assert "--export-mask-thresh" in exposed
