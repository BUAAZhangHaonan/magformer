# MAGFormer Repository Review And Environment Bring-Up Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit the migrated `magformer` workspace, document repository structure and dependency risks, then create a unified `magformer` conda environment that can import and run the main project plus baseline entrypoints.

**Architecture:** First perform a parallel repository review across the core package, `docs/`, and `baselines/`, then consolidate findings into a written audit. After that, create one conda environment centered on a Detectron2-compatible PyTorch stack and install local editable packages plus runtime dependencies. Finish by running smoke verifications for core tools and baseline wrappers, fixing environment issues discovered during validation.

**Tech Stack:** Conda, Python, PyTorch, Detectron2, Mask2Former, AdelaiDet/UOAIS, vendored baseline repos, pytest, shell smoke tests.

---

### Task 1: Inventory The Repository And Dependency Surface

**Files:**
- Create: `docs/reviews/2026-03-20-repo-audit.md`
- Reference: `README.md`
- Reference: `pyproject.toml`
- Reference: `requirements.txt`
- Reference: `docs/`
- Reference: `baselines/`
- Reference: `magformer/`
- Reference: `tools/`
- Reference: `tests/`

**Step 1: Capture repository structure and installation entrypoints**

Run:

```bash
find . -maxdepth 2 -type d | sort
rg --files -g 'README*' -g 'pyproject.toml' -g 'setup.py' -g 'setup.cfg' -g 'requirements*.txt' -g 'environment*.yml' -g 'environment*.yaml'
```

Expected: top-level structure, packaging files, and dependency files are identified.

**Step 2: Parallel-review independent domains**

Run focused inspections for:

```bash
find docs -maxdepth 2 -type f | sort
find magformer tools tests configs -maxdepth 3 -type f | sort
find baselines -maxdepth 3 -type f | sort
```

Expected: enough context to classify core modules, baseline families, and documentation clusters.

**Step 3: Record dependency and compatibility hotspots**

Document:
- C++/CUDA extensions under `magformer/models/ops/`
- Detectron2-dependent wrappers under `baselines/run_*.py`
- Vendored repos that need editable installs
- Data-path assumptions and dataset contracts

Expected: a concrete compatibility checklist for environment design.

### Task 2: Write The Repository Review Document

**Files:**
- Create: `docs/reviews/2026-03-20-repo-audit.md`

**Step 1: Draft the audit**

Include sections for:
- repository overview
- core package architecture
- documentation map
- baseline taxonomy
- environment/dependency risks
- recommended verification commands

Expected: a collaborator can understand the migrated workspace without reading commit history.

**Step 2: Add actionable environment guidance**

Include:
- recommended Python version and rationale
- recommended PyTorch/CUDA line and rationale
- packages/extensions that must compile locally
- known optional or brittle components

Expected: the audit becomes the handoff document for environment bring-up.

### Task 3: Create The Unified Conda Environment

**Files:**
- Create: `environment.magformer.yml`
- Modify: `requirements.txt` if missing hard runtime dependencies are discovered
- Modify: `requirements-dev.txt` if missing tooling needed for verification

**Step 1: Choose the compatibility baseline**

Decision target:
- keep Python at the newest version that still supports Detectron2, Mask2Former, AdelaiDet/UOAIS, and existing local extensions
- keep PyTorch on the newest line that can compile and import the vendored extensions reliably on this machine

Expected: one environment matrix is selected and documented.

**Step 2: Create the environment**

Run:

```bash
conda create -n magformer python=<chosen-version> -y
conda run -n magformer python --version
```

Expected: environment exists and the interpreter version matches the chosen baseline.

**Step 3: Install main and baseline dependencies**

Install:
- PyTorch / torchvision
- core runtime requirements
- editable installs for `.` and vendored repos that require package registration
- any additional packages required by smoke tests

Expected: imports succeed for the main project and baseline wrappers.

### Task 4: Verify Main Project And Baseline Entry Points

**Files:**
- Modify: `environment.magformer.yml` if verification reveals missing packages or incompatible pins
- Modify: `docs/reviews/2026-03-20-repo-audit.md` with final verification notes

**Step 1: Verify core imports and tests**

Run:

```bash
conda run -n magformer python -c "import torch; import magformer; print(torch.__version__)"
conda run -n magformer pytest tests -q
```

Expected: core project imports cleanly and the test suite passes or produces a bounded list of real issues.

**Step 2: Verify baseline wrapper imports**

Run smoke commands such as:

```bash
conda run -n magformer python baselines/run_detectron2_0831_1k.py --help
conda run -n magformer python baselines/run_official_mask2former_0831_1k.py --help
conda run -n magformer python baselines/run_msmformer_0831_1k.py --help
conda run -n magformer python baselines/run_uoais_0831_1k.py --help
conda run -n magformer python baselines/run_ucn_0831_1k.py --help
conda run -n magformer python baselines/yolo_export_coco.py --help
```

Expected: wrapper CLIs start without import failures.

**Step 3: Verify compiled extensions when applicable**

Run:

```bash
conda run -n magformer python -c "import detectron2"
conda run -n magformer python -c "from magformer.models.ops.modules import MSDeformAttn"
```

Expected: Detectron2 and MAGFormer deformable attention import successfully.

**Step 4: Update the audit with final state**

Record:
- exact environment choices
- commands used to verify
- failures that remain and why
- recommended follow-up if some baseline remains optional

Expected: the final audit doubles as a reproducible bring-up note.
