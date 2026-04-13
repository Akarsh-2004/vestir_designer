import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import benchmark_label_models as bm

ROOT = Path("/Users/akarshsaklani/Desktop/vestir")
SCRIPTS_DIR = ROOT / "vestir-prototype" / "scripts"
TERMINALS_DIR = Path("/Users/akarshsaklani/.cursor/projects/Users-akarshsaklani-Desktop-vestir/terminals")

FILENAME_DATASET = ROOT / "for label testing dataset "
DF2_ROOT = ROOT / "vestir-prototype" / "vision-sidecar" / "datasets" / "df2_v1"

DF2_CLASS_TO_CANON = {0: {"top"}, 1: {"jacket"}, 2: {"pants"}, 3: {"dress"}}


def parse_pred_lines(text: str) -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = {}
    rx = re.compile(r"^\[pred\]\s+(\w+)\s+\|\s+(.+?)\s+->\s+(\[.*\])\s*$")
    for line in text.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        model = m.group(1).strip()
        fname = m.group(2).strip()
        labels_raw = m.group(3).strip()
        try:
            labels = set(ast.literal_eval(labels_raw))
        except Exception:
            labels = set()
        out.setdefault(model, {})[fname] = labels
    return out


def read_df2_gt(split: str, filename: str) -> Tuple[List[int], Set[str]]:
    label_path = DF2_ROOT / "labels" / split / f"{Path(filename).stem}.txt"
    if not label_path.exists():
        return [], set()
    ids: List[int] = []
    labels: Set[str] = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except Exception:
            continue
        ids.append(cid)
        labels |= DF2_CLASS_TO_CANON.get(cid, set())
    return ids, labels


def as_csv_labels(s: Set[str]) -> str:
    return ",".join(sorted(s))


def write_test1_report(preds: Dict[str, Dict[str, Set[str]]]) -> Path:
    images = sorted([p for p in FILENAME_DATASET.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out = SCRIPTS_DIR / "test1_filename_detailed_comparison.txt"
    lines: List[str] = []
    lines.append("TEST 1 - Filename dataset detailed expected vs predicted")
    lines.append(f"images={len(images)}")
    lines.append("")
    for img in images:
        gt = bm.extract_labels(img.stem)
        lines.append(f"file: {img.name}")
        lines.append(f"  expected(filename): {as_csv_labels(gt)}")
        for model in ("yolo", "siglip", "florence", "ollama"):
            p = preds.get(model, {}).get(img.name, set())
            lines.append(f"  predicted[{model}]: {as_csv_labels(p)}")
        # hybrid from available model preds
        y = preds.get("yolo", {}).get(img.name, set())
        s = preds.get("siglip", {}).get(img.name, set())
        f = preds.get("florence", {}).get(img.name, set())
        o = preds.get("ollama", {}).get(img.name, set())
        h = bm.hybrid_vote({img.name: y}, {img.name: s}, {img.name: f}, {img.name: o})[img.name]
        lines.append(f"  predicted[hybrid]: {as_csv_labels(h)}")
        lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_test2_df2_report(preds: Dict[str, Dict[str, Set[str]]], split: str = "val", max_images: int = 40) -> Path:
    images = sorted([p for p in (DF2_ROOT / "images" / split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:max_images]
    out = SCRIPTS_DIR / "test2_df2_detailed_comparison.txt"
    lines: List[str] = []
    lines.append("TEST 2 - DF2 detailed expected annotation vs predicted (ollama-only run)")
    lines.append(f"split={split} images={len(images)}")
    lines.append("")
    for img in images:
        class_ids, gt = read_df2_gt(split, img.name)
        p = preds.get("ollama", {}).get(img.name, set())
        lines.append(f"file: {img.name}")
        lines.append(f"  expected_df2_class_ids: {','.join(str(x) for x in class_ids)}")
        lines.append(f"  expected_mapped_labels: {as_csv_labels(gt)}")
        lines.append(f"  predicted[ollama]: {as_csv_labels(p)}")
        lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_test3_report(preds: Dict[str, Dict[str, Set[str]]]) -> Path:
    images = sorted([p for p in FILENAME_DATASET.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out = SCRIPTS_DIR / "test3_filename_detailed_comparison.txt"
    lines: List[str] = []
    lines.append("TEST 3 - Filename dataset detailed expected vs predicted (latest patched run)")
    lines.append(f"images={len(images)}")
    lines.append("")
    for img in images:
        gt = bm.extract_labels(img.stem)
        lines.append(f"file: {img.name}")
        lines.append(f"  expected(filename): {as_csv_labels(gt)}")
        for model in ("yolo", "siglip", "florence", "ollama"):
            p = preds.get(model, {}).get(img.name, set())
            lines.append(f"  predicted[{model}]: {as_csv_labels(p)}")
        y = preds.get("yolo", {}).get(img.name, set())
        s = preds.get("siglip", {}).get(img.name, set())
        f = preds.get("florence", {}).get(img.name, set())
        o = preds.get("ollama", {}).get(img.name, set())
        h = bm.hybrid_vote({img.name: y}, {img.name: s}, {img.name: f}, {img.name: o})[img.name]
        lines.append(f"  predicted[hybrid]: {as_csv_labels(h)}")
        lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_master_analysis(test1: Path, test2: Path, test3: Path) -> Path:
    label_report = SCRIPTS_DIR / "benchmark_label_models_report.json"
    df2_report = SCRIPTS_DIR / "benchmark_df2_small_ops_report.json"
    label_data = json.loads(label_report.read_text()) if label_report.exists() else {}
    df2_data = json.loads(df2_report.read_text()) if df2_report.exists() else {}

    out = SCRIPTS_DIR / "tests_1_2_3_master_analysis.txt"
    lines: List[str] = []
    lines.append("MASTER REPORT: Test1 + Test2 + Test3")
    lines.append("")
    lines.append("=== Test 1 / Test 3 (filename-labeled dataset) ===")
    if label_data:
        scores = label_data.get("scores", {})
        lat = label_data.get("latency_seconds", {})
        for name in ("yolo", "siglip", "florence", "ollama_1", "ollama_2", "hybrid"):
            if name in scores:
                s = scores[name]
                lines.append(
                    f"{name}: hit_rate={s.get('hit_rate',0):.3f}, exact_match={s.get('exact_match',0):.3f}, avg_jaccard={s.get('avg_jaccard',0):.3f}"
                )
        lines.append("")
        for key in ("yolo", "siglip", "florence", "ollama_1", "ollama_2", "total"):
            if key in lat:
                lines.append(f"latency[{key}]={lat[key]:.2f}s")
    else:
        lines.append("No completed benchmark_label_models_report.json found.")

    lines.append("")
    lines.append("=== Test 2 (DF2 small, annotation GT) ===")
    if df2_data:
        s = (df2_data.get("scores") or {}).get("ollama_single", {})
        lines.append(
            f"ollama_single: hit_rate={s.get('hit_rate',0):.3f}, exact_match={s.get('exact_match',0):.3f}, avg_jaccard={s.get('avg_jaccard',0):.3f}"
        )
        lat = df2_data.get("latency_seconds", {})
        lines.append(f"latency[ollama_single]={lat.get('ollama_single',0):.2f}s")
        lines.append(f"latency[total]={lat.get('total',0):.2f}s")
        lines.append(f"df2_split={df2_data.get('split')} images={df2_data.get('num_images')} ollama_images={df2_data.get('ollama_num_images')}")
    else:
        lines.append("No completed benchmark_df2_small_ops_report.json found.")

    lines.append("")
    lines.append("=== Detailed per-file outputs ===")
    lines.append(f"test1_file={test1}")
    lines.append(f"test2_file={test2}")
    lines.append(f"test3_file={test3}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- YOLO is fastest and stable; label granularity can be noisy.")
    lines.append("- SigLIP (Marqo) is strongest semantic matcher but slower.")
    lines.append("- Florence gives balanced quality with occasional empty/noisy tags.")
    lines.append("- Ollama vision is inconsistent in this environment (empty or over-broad outputs).")
    lines.append("- Hybrid is useful when Ollama is healthy; otherwise use YOLO+SigLIP+Florence.")

    out.write_text("\n".join(lines))
    return out


def main() -> None:
    def _read_term(name: str, fallback: str = "") -> str:
        p = TERMINALS_DIR / name
        return p.read_text() if p.exists() else fallback

    t25 = _read_term("25.txt", "")
    t30 = _read_term("30.txt", "")
    # 520825 is a detailed patched run terminal id from agent-side execution
    t520 = _read_term("520825.txt", t25)
    if not t30:
        # Fallback: use latest ollama-only run log if 30.txt is unavailable.
        t30 = _read_term("95197.txt", t25)

    preds1 = parse_pred_lines(t25)
    preds2 = parse_pred_lines(t30)
    preds3 = parse_pred_lines(t520)

    f1 = write_test1_report(preds1)
    f2 = write_test2_df2_report(preds2, split="val", max_images=40)
    f3 = write_test3_report(preds3)
    master = write_master_analysis(f1, f2, f3)

    print(f"Saved: {f1}")
    print(f"Saved: {f2}")
    print(f"Saved: {f3}")
    print(f"Saved: {master}")


if __name__ == "__main__":
    main()

