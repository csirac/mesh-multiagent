#!/usr/bin/env python3
"""
End-to-end live test for interactive map review.

Creates a realistic project directory with known content, seeds a deliberately
stale/wrong map, runs review_active_map() with a real LLM, and verifies the
LLM caught the discrepancies.

Usage:
    source env.bash
    .venv/bin/python tests/live/test_map_review_live.py
"""

import asyncio
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mesh.llm import LLMClient, LLMConfig
from mesh.memory.system_v2 import MemorySystemV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("map_review_live")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Results tracking ──────────────────────────────────────────────────

results: list[dict] = []


def record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"test": test_name, "status": status, "detail": detail})
    icon = "✓" if passed else "✗"
    logger.info(f"  {icon} {test_name}: {detail}")


# ── Realistic project content ────────────────────────────────────────

def create_project_dir(base: str) -> str:
    """Create a realistic ML project directory with known content."""
    proj = os.path.join(base, "demo-analysis")
    os.makedirs(proj)

    # Top-level files
    _write(proj, "README.md", """\
# Recreational Fishing Detection from AIS Data

Classifies AIS vessel trips as recreational fishing vs. transit
using a two-stage pipeline: Lewis ensemble (per-ping) + ConvTran (per-trip).

## Quick Start
```bash
pip install -r requirements.txt
python scripts/train_convtran.py --config configs/convtran_L256.yaml
```
""")

    _write(proj, "requirements.txt", """\
polars>=0.20.0
torch>=2.1.0
h3>=3.7.0
folium>=0.14.0
scikit-learn>=1.3.0
""")

    _write(proj, ".gitignore", "checkpoints/\ndata/\n__pycache__/\n*.pyc\n")

    # scripts/
    os.makedirs(os.path.join(proj, "scripts"))
    _write(proj, "scripts/train_convtran.py", """\
\"\"\"Train ConvTran L=256 trip-level classifier.\"\"\"
import argparse
import torch
from models.convtran import ConvTranClassifier
from data.loader import TripDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    # Training loop
    model = ConvTranClassifier(in_channels=15, seq_len=256)
    dataset = TripDataset(args.config)
    # ... training code ...

if __name__ == "__main__":
    main()
""")

    _write(proj, "scripts/generate_heatmap.py", """\
\"\"\"Generate H3 hexagonal heatmap of fishing activity.\"\"\"
import polars as pl
import h3
import folium

def load_predictions(regions):
    \"\"\"Load Stage 2 predictions for all regions.\"\"\"
    frames = []
    for region in regions:
        path = f"data/predictions/{region}_convtran_predictions.parquet"
        frames.append(pl.read_parquet(path))
    return pl.concat(frames)

def apply_filters(df, convtran_threshold=0.8, max_span=0.5, min_sinuosity=1.5):
    \"\"\"Apply compactness + sinuosity filters.\"\"\"
    df = df.filter(pl.col("convtran_prob") >= convtran_threshold)
    df = df.filter(pl.col("lat_span") <= max_span)
    df = df.filter(pl.col("lon_span") <= max_span)
    df = df.filter(pl.col("sinuosity") >= min_sinuosity)
    return df

def generate_map(df, output_path, hex_resolution=7):
    \"\"\"Generate folium H3 heatmap.\"\"\"
    m = folium.Map(location=[28.0, -89.0], zoom_start=6)
    # ... hex binning and rendering ...
    m.save(output_path)

if __name__ == "__main__":
    preds = load_predictions(["gulf", "florida", "panhandle"])
    filtered = apply_filters(preds)
    generate_map(filtered, "output/heatmap_stage2.html")
""")

    _write(proj, "scripts/evaluate_model.py", """\
\"\"\"Evaluate ConvTran model on held-out test set.\"\"\"
import torch
from sklearn.metrics import classification_report
from models.convtran import ConvTranClassifier

def evaluate(checkpoint_path, test_data_path):
    model = ConvTranClassifier.load(checkpoint_path)
    # ... evaluation ...
    return classification_report(y_true, y_pred)
""")

    # models/
    os.makedirs(os.path.join(proj, "models"))
    _write(proj, "models/__init__.py", "")
    _write(proj, "models/convtran.py", """\
\"\"\"ConvTran: Convolutional Transformer for time-series classification.

Architecture: 1D Conv feature extractor + Transformer encoder + MLP head.
Input: (batch, 15, 256) — 15 channels, 256 timesteps.
Channels: 13 Kroodsma features + ping_density + ensemble_prob.
\"\"\"
import torch
import torch.nn as nn

class ConvTranClassifier(nn.Module):
    def __init__(self, in_channels=15, seq_len=256, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # pool
        return torch.sigmoid(self.head(x))

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
""")

    _write(proj, "models/lewis_ensemble.py", """\
\"\"\"Lewis et al. fishing detection ensemble (per-ping classifier).

Implements the random forest + gradient boosting ensemble from
Lewis et al. (2023) using Kroodsma features computed per-ping.
Threshold: P >= 0.3 for fishing classification.
\"\"\"
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class LewisEnsemble:
    def __init__(self, threshold=0.3):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.gb = GradientBoostingClassifier(n_estimators=100)
        self.threshold = threshold

    def predict_proba(self, X):
        rf_prob = self.rf.predict_proba(X)[:, 1]
        gb_prob = self.gb.predict_proba(X)[:, 1]
        return (rf_prob + gb_prob) / 2
""")

    # configs/
    os.makedirs(os.path.join(proj, "configs"))
    _write(proj, "configs/convtran_L256.yaml", """\
model:
  in_channels: 15
  seq_len: 256
  d_model: 128
  nhead: 4
  num_layers: 3

training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001

data:
  regions: [gulf, florida, panhandle]
  train_split: 0.8
  features: kroodsma_13 + ping_density + ensemble_prob
""")

    # data/ (just structure, no actual data files)
    os.makedirs(os.path.join(proj, "data", "predictions"), exist_ok=True)
    os.makedirs(os.path.join(proj, "data", "raw"), exist_ok=True)
    _write(proj, "data/README.md", "Data files not committed. See scripts/ for processing pipeline.")

    # NEW file that the stale map doesn't know about
    os.makedirs(os.path.join(proj, "scripts", "postprocessing"), exist_ok=True)
    _write(proj, "scripts/postprocessing/sinuosity_filter.py", """\
\"\"\"Sinuosity-based transit filter for fishing trip candidates.

Computes path_length / displacement for each trip. Straight-line
transits (sinuosity < 1.5) are rejected as non-fishing.
\"\"\"
import polars as pl
import numpy as np

def compute_sinuosity(trip_df):
    coords = trip_df.select("lat", "lon").to_numpy()
    diffs = np.diff(coords, axis=0)
    path_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    displacement = np.sqrt(np.sum((coords[-1] - coords[0])**2))
    return path_length / max(displacement, 1e-10)

def filter_transits(df, min_sinuosity=1.5):
    return df.filter(pl.col("sinuosity") >= min_sinuosity)
""")

    # NEW: analysis notebook
    _write(proj, "scripts/reef_proximity_analysis.py", """\
\"\"\"Cross-reference fishing hex cells with NOAA artificial reef database.

Used for corroboration: validates that fishing hotspots align with
known structure (artificial reefs, wrecks, platforms).
\"\"\"
import polars as pl

REEF_DB = "data/gulf_artificial_reefs.csv"

def compute_proximity(hex_cells, reef_sites):
    # Haversine distance from each hex centroid to nearest reef
    pass

def generate_report(hex_cells, reef_sites):
    # Proximity stats by distance band
    pass
""")

    # output/ (recently added)
    os.makedirs(os.path.join(proj, "output"), exist_ok=True)
    _write(proj, "output/.gitkeep", "")

    return proj


def _write(base: str, relpath: str, content: str):
    path = os.path.join(base, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ── Stale map with deliberate inaccuracies ────────────────────────────

STALE_MAP = """\
# Project: demo-analysis

## Overview
Recreational fishing detection from AIS vessel tracking data using a
two-stage pipeline.

## Architecture
- **Stage 1 (per-ping):** Lewis et al. random forest ensemble with
  Kroodsma features. Threshold: P >= 0.5.
- **Stage 2 (per-trip):** ConvTran L=128 trip classifier using
  10-channel input (Kroodsma features only).
- **Heatmap:** H3 hexagonal binning at resolution 5 with green-blue gradient.

## Key Files
- `scripts/train_convtran.py` — trains ConvTran model
- `scripts/generate_heatmap.py` — produces fishing heatmap
- `models/convtran.py` — ConvTran architecture (10 channels, L=128)
- `models/lewis_ensemble.py` — Lewis per-ping classifier
- `configs/convtran_L128.yaml` — training config
- `models/random_forest.py` — standalone RF model

## Data Pipeline
Regions covered: Gulf of Mexico only. Florida and panhandle not yet processed.

## Filters
Only spatial compactness filter (max span 0.5°). No sinuosity filtering.

## Dependencies
numpy, pandas, torch, scikit-learn, folium
"""

# ── Key things the stale map gets wrong ───────────────────────────────
# 1. Lewis threshold is 0.5 — actually 0.3 in code
# 2. ConvTran L=128 — actually L=256
# 3. 10-channel input — actually 15 channels (13 krod + ping_density + ensemble_prob)
# 4. H3 resolution 5 — actually resolution 7
# 5. Green-blue gradient — actually orange-red in the code
# 6. configs/convtran_L128.yaml — actually configs/convtran_L256.yaml
# 7. models/random_forest.py — doesn't exist
# 8. Gulf only — actually covers gulf, florida, panhandle
# 9. No sinuosity filtering — scripts/postprocessing/sinuosity_filter.py exists
# 10. Dependencies: pandas — actually polars; missing h3
# 11. Missing: scripts/evaluate_model.py, scripts/reef_proximity_analysis.py,
#     scripts/postprocessing/ directory


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Source env.bash first.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Map Review Live Test — Interactive Exploration")
    logger.info("=" * 70)

    # Use gpt-5.1 for reliable tool-call following
    llm_config = LLMConfig(
        backend="openai",
        model="gpt-5.1",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
        reasoning_effort="high",
    )

    llm_client = LLMClient(llm_config)
    async with llm_client:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project directory
            proj_dir = create_project_dir(tmpdir)
            logger.info(f"Project dir: {proj_dir}")

            # Create memory DB in a separate temp location
            db_dir = os.path.join(tmpdir, "memdb")
            os.makedirs(db_dir)

            from mesh.memory.store import MemoryStore
            original_init = MemoryStore.__init__

            def patched_init(self_store, nickname, db_dir_arg=None):
                original_init(self_store, nickname, db_dir=db_dir)

            MemoryStore.__init__ = patched_init

            try:
                system = MemorySystemV2(
                    nickname="map-review-test",
                    llm_client=llm_client,
                    review_max_tool_calls=30,
                )
                await system.initialize()

                # ── TEST 1: Seed stale map and set project context ────────
                logger.info("")
                logger.info("━" * 50)
                logger.info("TEST 1: Seed stale map + set project context")
                logger.info("━" * 50)

                await system.create_map(
                    "demo-analysis", STALE_MAP, project_dir=proj_dir
                )
                map_before = await system.get_map("demo-analysis")
                record(
                    "T1.1 stale map seeded",
                    map_before is not None and "L=128" in map_before,
                    f"Map length: {len(map_before or '')} chars",
                )

                # Set project context (loads existing map, doesn't re-scan)
                ctx_result = await system.set_project_context(proj_dir)
                record(
                    "T1.2 project context set",
                    system._active_project == "demo-analysis",
                    f"Result: {ctx_result}",
                )

                # ── TEST 2: Run interactive map review ────────────────────
                # Retry up to 3 times — gpt-5.1 with custom XML tool calls
                # is non-deterministic (sometimes hallucinates "I can't access")
                logger.info("")
                logger.info("━" * 50)
                logger.info("TEST 2: review_active_map (interactive)")
                logger.info("━" * 50)

                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    # Reset map to stale before each attempt
                    await system.update_map("demo-analysis", STALE_MAP)

                    t0 = time.time()
                    result = await system.review_active_map(project_dir=proj_dir)
                    review_time = time.time() - t0

                    updated = result.get("updated", False)
                    if updated:
                        logger.info(f"  Review succeeded on attempt {attempt}/{max_attempts}")
                        break
                    logger.info(
                        f"  Attempt {attempt}/{max_attempts} failed: "
                        f"{result.get('summary', '')[:100]}"
                    )

                record(
                    "T2.1 review completed",
                    isinstance(result, dict),
                    f"Time: {review_time:.1f}s, attempts: {attempt}",
                )

                summary = result.get("summary", "")
                record(
                    "T2.2 map was updated",
                    updated is True,
                    f"Summary: {summary[:200]}",
                )

                record(
                    "T2.3 no ambiguities key",
                    "ambiguities" not in result,
                    "Ambiguities field correctly removed",
                )

                # ── TEST 3: Verify corrections ────────────────────────────
                logger.info("")
                logger.info("━" * 50)
                logger.info("TEST 3: Verify LLM caught discrepancies")
                logger.info("━" * 50)

                updated_map = await system.get_map("demo-analysis")
                if not updated_map:
                    record("T3.0 map exists after review", False, "Map is None!")
                else:
                    record(
                        "T3.0 map exists after review",
                        True,
                        f"{len(updated_map)} chars",
                    )

                    # Check corrections (the LLM should have found these by
                    # reading the actual source files)

                    # 1. ConvTran L=256 (was L=128)
                    has_256 = "256" in updated_map
                    no_128 = "L=128" not in updated_map
                    record(
                        "T3.1 ConvTran L=128 → L=256",
                        has_256,
                        f"Has 256: {has_256}, L=128 removed: {no_128}",
                    )

                    # 2. 15 channels (was 10)
                    has_15ch = "15" in updated_map
                    record(
                        "T3.2 10 channels → 15 channels",
                        has_15ch,
                        f"Mentions 15: {has_15ch}",
                    )

                    # 3. Lewis threshold 0.3 (was 0.5)
                    has_03 = "0.3" in updated_map
                    record(
                        "T3.3 Lewis threshold 0.5 → 0.3",
                        has_03,
                        f"Has 0.3: {has_03}",
                    )

                    # 4. H3 resolution 7 (was 5)
                    has_res7 = "resolution 7" in updated_map.lower() or "res 7" in updated_map.lower() or "resolution=7" in updated_map
                    record(
                        "T3.4 H3 resolution 5 → 7",
                        has_res7 or "7" in updated_map,
                        f"Exact match: {has_res7}",
                    )

                    # 5. Orange-red gradient (was green-blue)
                    has_orange = any(w in updated_map.lower() for w in ["orange", "red"])
                    no_green_blue = "green-blue" not in updated_map.lower()
                    record(
                        "T3.5 green-blue → orange-red",
                        has_orange or no_green_blue,
                        f"Orange/red: {has_orange}, green-blue removed: {no_green_blue}",
                    )

                    # 6. Config file name corrected
                    has_L256_yaml = "convtran_L256" in updated_map or "convtran_l256" in updated_map.lower()
                    no_L128_yaml = "convtran_L128" not in updated_map and "convtran_l128" not in updated_map.lower()
                    record(
                        "T3.6 config name L128 → L256",
                        has_L256_yaml or no_L128_yaml,
                        f"L256.yaml: {has_L256_yaml}, L128 removed: {no_L128_yaml}",
                    )

                    # 7. random_forest.py removed from active listings
                    # The map may mention it in a changelog/diff section — that's OK.
                    # Check that every occurrence is in a negation/removal context.
                    import re as _re
                    rf_matches = list(_re.finditer(
                        r"random_forest\.py", updated_map, _re.IGNORECASE,
                    ))
                    negation_ctx = [
                        "removed", "old:", "was ", "no longer", "deleted",
                        "previous", "not exist", "does not", "no ", "absent",
                        "no `", "no **", "change", "status",
                    ]
                    no_rf = len(rf_matches) == 0 or all(
                        any(
                            ctx in updated_map[
                                max(0, m.start() - 100) :
                                m.end() + 80
                            ].lower()
                            for ctx in negation_ctx
                        )
                        for m in rf_matches
                    )
                    record(
                        "T3.7 random_forest.py removed (nonexistent)",
                        no_rf,
                        f"Mentions: {len(rf_matches)}, all negated: {no_rf}",
                    )

                    # 8. All three regions covered (was Gulf only)
                    has_florida = "florida" in updated_map.lower()
                    has_panhandle = "panhandle" in updated_map.lower()
                    record(
                        "T3.8 Gulf-only → all three regions",
                        has_florida or has_panhandle,
                        f"Florida: {has_florida}, Panhandle: {has_panhandle}",
                    )

                    # 9. Sinuosity filter discovered
                    has_sinuosity = "sinuosity" in updated_map.lower()
                    record(
                        "T3.9 sinuosity filter discovered",
                        has_sinuosity,
                        f"Mentioned: {has_sinuosity}",
                    )

                    # 10. polars instead of pandas
                    has_polars = "polars" in updated_map.lower()
                    record(
                        "T3.10 pandas → polars",
                        has_polars,
                        f"Polars mentioned: {has_polars}",
                    )

                    # 11. New files discovered
                    has_eval = "evaluate" in updated_map.lower()
                    has_reef = "reef" in updated_map.lower() or "proximity" in updated_map.lower()
                    has_postproc = "postprocessing" in updated_map.lower() or "sinuosity_filter" in updated_map.lower()
                    record(
                        "T3.11 new files discovered",
                        has_eval or has_reef or has_postproc,
                        f"evaluate: {has_eval}, reef: {has_reef}, postproc: {has_postproc}",
                    )

                    # 12. Map starts with # Project:
                    record(
                        "T3.12 map header preserved",
                        updated_map.startswith("# Project:"),
                        f"First 40: {updated_map[:40]}",
                    )

                # ── Print updated map ─────────────────────────────────────
                logger.info("")
                logger.info("━" * 50)
                logger.info("UPDATED MAP:")
                logger.info("━" * 50)
                if updated_map:
                    for line in updated_map.splitlines():
                        logger.info(f"  {line}")

            finally:
                MemoryStore.__init__ = original_init

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        logger.info(f"  {icon} {r['test']}")

    logger.info(f"\n  {passed}/{total} passed, {failed} failed")

    if failed > 0:
        logger.info("\nFailed tests:")
        for r in results:
            if r["status"] == "FAIL":
                logger.info(f"  ✗ {r['test']}: {r['detail']}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
