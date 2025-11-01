# Phase 1 Quick Start

## TL;DR

Extract all ~75k IR images for self-supervised pre-training:

```bash
./scripts/run_phase1.sh
```

That's it! Takes 15-25 minutes.

---

## What This Does

1. Extracts ALL IR images from 5 flights (~75,000 total)
2. Applies preprocessing (vignetting correction, averaging)
3. Splits into train (95%) and val (5%)
4. Saves to efficient HDF5 format
5. Verifies data integrity
6. Generates statistics and sample visualizations

---

## Output

```
data_ssl/images/
├── train.h5                  # ~71k images, ~1.2 GB
├── val.h5                    # ~3.7k images, ~65 MB
├── extraction_stats.yaml     # Statistics
└── sample_images.png         # Visual check
```

---

## Requirements

- 5 GB free disk space
- ~2 GB RAM
- 15-25 minutes runtime
- CPU only (no GPU needed)

---

## Verification

After completion, check:

- ✅ Console shows "✅ Phase 1 complete!"
- ✅ Both train.h5 and val.h5 exist
- ✅ extraction_stats.yaml shows ~75k total images
- ✅ No NaN/Inf values detected
- ✅ All 5 flights processed (flights_failed: 0)

---

## Troubleshooting

**"File not found"**
→ Check `data_directory` in `configs/ssl_extract.yaml`

**"Out of memory"**
→ Close other applications (script uses chunked loading)

**"Flight X failed"**
→ Script continues with other flights; check that flight's HDF5 file

---

## Next Step

After successful Phase 1:

```bash
python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml
```

(Phase 2 - to be implemented)

---

## Manual Execution

If automated script fails, run manually:

```bash
# Step 1: Extract
./venv/bin/python scripts/extract_all_images.py \
    --config configs/ssl_extract.yaml \
    --output-dir data_ssl/images

# Step 2: Verify
./venv/bin/python scripts/verify_extraction.py \
    --data-dir data_ssl/images \
    --plot-samples
```

---

## Full Documentation

- Detailed guide: `PHASE1_EXTRACTION_GUIDE.md` (481 lines)
- Implementation details: `PHASE1_READY.md` (273 lines)
- Project status: `PROJECT_STATUS.md`
- Master plan: `Agent Scope of Work: A Research Program.md`

---

**Status:** 🟢 Ready to execute

**Command:** `./scripts/run_phase1.sh`
