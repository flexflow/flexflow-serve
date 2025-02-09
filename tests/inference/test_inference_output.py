import os
import re
import glob
import pytest

OUTPUT_DIR = os.path.join("inference", "output")

def get_line(filepath, line_index):
    """
    Returns the specified line (0-based) from the file, or '' if the file
    doesn't have that many lines.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines[line_index] if len(lines) > line_index else ""

def compare_single_line(file_a, file_b):
    """
    Compare a single line in two files:
      - If filename starts with 'spec_infer' or 'incr_dec', compare line index = 1 (2nd line).
      - If filename starts with 'huggingface_', compare line index = 0 (1st line).
    Raise AssertionError if they differ.
    """
    base_a = os.path.basename(file_a)
    base_b = os.path.basename(file_b)

    if base_a.startswith(("spec_infer", "incr_dec")):
        line_a = get_line(file_a, 1)
    else:
        line_a = get_line(file_a, 0)

    if base_b.startswith(("spec_infer", "incr_dec")):
        line_b = get_line(file_b, 1)
    else:
        line_b = get_line(file_b, 0)

    list_a = line_a[len("token IDs: "):].split(",")
    list_b = line_b[len("token IDs: "):].split(",")

    # check if the first 50 elements are equal
    for i in range(min(50, len(list_a), len(list_b))):
        if list_a[i] != list_b[i]:
            raise AssertionError(
                f"File contents differ at position {i}:\n  {file_a} -> {list_a[i]}\n  {file_b} -> {list_b[i]}"
            )


def group_model_files(prefix):
    """
    Returns a dict of the form: {model_name: [list of files]} for files matching:
       [prefix]-python-<model_name>-half_prec*
    """
    pattern = os.path.join(OUTPUT_DIR, f"{prefix}-python-*-half_prec*")
    files = glob.glob(pattern)
    grouped = {}
    for fpath in files:
        basename = os.path.basename(fpath)
        # Example: spec_infer-python-opt-6.7b-half_prec_1_tp_4_pp.json
        remainder = basename[len(prefix + "-python-"):]
        parts = remainder.split("-half_prec", 1)
        if len(parts) < 2:
            continue
        model_name = parts[0]
        grouped.setdefault(model_name, []).append(fpath)
    return grouped

def collect_file_comparisons():
    """
    Yields tuples (file_a, file_b) for all pairwise comparisons among
    spec_infer or incr_dec files that share a model name,
    plus the comparison with huggingface_<model_name>.txt if it exists.
    """
    for prefix in ["spec_infer", "incr_dec"]:
        grouped = group_model_files(prefix)
        for model_name, file_group in grouped.items():
            # Pairwise among spec_infer / incr_dec files
            for i in range(len(file_group)):
                for j in range(i+1, len(file_group)):
                    yield file_group[i], file_group[j]
            # Compare with huggingface_<model_name>.txt
            hf_file = os.path.join(OUTPUT_DIR, f"huggingface_{model_name}.txt")
            if os.path.exists(hf_file) and file_group:
                yield file_group[0], hf_file


def _extract_llm_decoding_steps(line):
    """
    Given a string like:
      [Profile] guid(26516) llm_decoding_steps(69) latency(123456)
    parse and return the integer after llm_decoding_steps(...).
    Return None if not found.
    """
    match = re.search(r'llm_decoding_steps\((\d+)\)', line)
    return int(match.group(1)) if match else None

def collect_spec_infer_incr_dec_pairs():
    """
    Yields (spec_file, incr_file) for files that have the same trailing name
    after the prefix spec_infer- / incr_dec-.
    """
    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.*"))  # .txt/.json
    spec_infer = {}
    for f in all_files:
        base = os.path.basename(f)
        if base.startswith("spec_infer-"):
            rest = base[len("spec_infer-"):]
            spec_infer[rest] = f
    for f in all_files:
        base = os.path.basename(f)
        if base.startswith("incr_dec-"):
            rest = base[len("incr_dec-"):]
            if rest in spec_infer:
                yield (spec_infer[rest], f)

@pytest.mark.parametrize("file_a,file_b", collect_file_comparisons(), 
                         ids=lambda f: os.path.basename(f))
def test_output_alignment(file_a, file_b):
    """
    Each file pair is tested and reported separately.
    """
    compare_single_line(file_a, file_b)



@pytest.mark.parametrize("spec_file,incr_file", collect_spec_infer_incr_dec_pairs(),
                         ids=lambda f: os.path.basename(f))
def test_decoding_steps(spec_file, incr_file):
    """
    For each matching pair (same suffix), compare the first line:
    "[Profile] guid(...) llm_decoding_steps(...) latency(...)"
    Ensure that spec_infer's llm_decoding_steps is <= incr_dec's steps / 1.5.
    """
    with open(spec_file, "r", encoding="utf-8") as fs:
        spec_line = fs.readline()
    with open(incr_file, "r", encoding="utf-8") as fi:
        incr_line = fi.readline()

    spec_steps = _extract_llm_decoding_steps(spec_line)
    incr_steps = _extract_llm_decoding_steps(incr_line)

    # If we don't have valid numbers in one or both lines, skip
    if spec_steps is None or incr_steps is None:
        pytest.skip(f"No valid llm_decoding_steps found in {spec_file} or {incr_file}")

    # Check ratio
    if not (spec_steps <= incr_steps / 1.5):
        raise AssertionError(
            f"[{os.path.basename(spec_file)} vs {os.path.basename(incr_file)}] "
            f"spec_infer has llm_decoding_steps={spec_steps}, which is not "
            f"<= incr_dec steps={incr_steps}/1.5 = {incr_steps/1.5:.1f}"
        )