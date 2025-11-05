# SOW Sprint 3: Documentation Index

**Project:** Physics-Constrained CBH Model Validation  
**Document ID:** SOW-AGENT-CBH-WP-001  
**Status:** WP-1 & WP-2 Complete, WP-3 & WP-4 Ready for Implementation

---

## 📖 Start Here

### New to This Project?
1. **Read:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 5 min overview
2. **Read:** [README.md](README.md) - 15 min comprehensive guide
3. **Run:** `./run_sow.sh --verbose` - Execute WP-1 and WP-2

### Need Technical Details?
- **Full Guide:** [SOW_IMPLEMENTATION_GUIDE.md](SOW_IMPLEMENTATION_GUIDE.md)
- **Requirements:** [../ScopeWorkSprint3.md](../ScopeWorkSprint3.md)

### Want to See What's Done?
- **Summary:** [WORK_COMPLETED_SUMMARY.md](WORK_COMPLETED_SUMMARY.md)

---

## 📚 Documentation Hierarchy

```
INDEX.md (you are here) ────┬──► QUICK_REFERENCE.md
                             │   └─ Quick commands, status, workflow
                             │
                             ├──► README.md
                             │   └─ Comprehensive guide, getting started
                             │
                             ├──► SOW_IMPLEMENTATION_GUIDE.md
                             │   └─ Complete technical specifications
                             │
                             ├──► WORK_COMPLETED_SUMMARY.md
                             │   └─ What's been built, what's left
                             │
                             └──► ../ScopeWorkSprint3.md
                                 └─ Official requirements (SOW)
```

---

## 🎯 Quick Decision Tree

**Q: "What should I do first?"**
```
Are WP-1 and WP-2 features extracted?
├─ NO  → Run: ./run_sow.sh --verbose
└─ YES → Are they validated and look good?
         ├─ NO  → See: QUICK_REFERENCE.md "Data Inspection"
         └─ YES → Is WP-3 implemented?
                  ├─ NO  → See: SOW_IMPLEMENTATION_GUIDE.md Section 5
                  └─ YES → Did WP-3 pass (R² > 0)?
                           ├─ NO  → Analyze failure, revise approach
                           └─ YES → Implement WP-4
                                    See: SOW_IMPLEMENTATION_GUIDE.md Section 6
```

---

## 📁 Files by Purpose

### 🚀 Execution
- `run_sow.sh` - Automated execution script for all work packages
- `wp1_geometric_features.py` - WP-1: Shadow-based geometric features ✅
- `wp2_atmospheric_features.py` - WP-2: ERA5 atmospheric features ✅
- `wp3_physical_baseline.py` - WP-3: Physical baseline validation ⏳
- `wp4_hybrid_models.py` - WP-4: Hybrid model integration ⏳

### 📖 Documentation
- `INDEX.md` - This file (master index)
- `QUICK_REFERENCE.md` - Quick commands and workflow (1 page)
- `README.md` - Comprehensive getting started guide
- `SOW_IMPLEMENTATION_GUIDE.md` - Complete technical specifications
- `WORK_COMPLETED_SUMMARY.md` - Implementation status and achievements

### 📊 Data (Generated)
- `wp1_geometric/WP1_Features.hdf5` - Geometric features (933 samples)
- `wp2_atmospheric/WP2_Features.hdf5` - Atmospheric features (933 samples)
- `wp3_baseline/WP3_Report.json` - Physical baseline results
- `wp4_hybrid/final_features.hdf5` - All features combined
- `wp4_hybrid/WP4_Report.json` - Hybrid model results
- `models/final_gbdt_models/` - Trained models

---

## 🎨 Document Purposes

| Document | Read Time | Purpose | When to Use |
|----------|-----------|---------|-------------|
| **QUICK_REFERENCE.md** | 5 min | Commands, status, quick lookup | Need to run something quickly |
| **README.md** | 15 min | Getting started, overview | First time using the system |
| **SOW_IMPLEMENTATION_GUIDE.md** | 30-60 min | Technical specs, algorithms | Implementing WP-3 or WP-4 |
| **WORK_COMPLETED_SUMMARY.md** | 10 min | What's done, what's left | Status check, planning |
| **INDEX.md** | 2 min | Navigation, decision tree | Finding the right document |

---

## 🔄 Workflow Stages

### Stage 1: Feature Extraction ✅
**Status:** COMPLETE  
**Documents:**
- Quick start: `QUICK_REFERENCE.md` → "Quick Start" section
- Details: `README.md` → "WP-1" and "WP-2" sections
- Run: `./run_sow.sh --verbose`

### Stage 2: Physical Baseline Validation ⏳
**Status:** READY TO IMPLEMENT  
**Documents:**
- Overview: `README.md` → "WP-3" section  
- Implementation: `SOW_IMPLEMENTATION_GUIDE.md` → Section 5
- Reference: `../ScopeWorkSprint3.md` → Section 5

**Key Requirement:** This is the GO/NO-GO gate (must achieve R² > 0)

### Stage 3: Hybrid Model Integration ⏳
**Status:** WAITING FOR WP-3 TO PASS  
**Documents:**
- Overview: `README.md` → "WP-4" section
- Implementation: `SOW_IMPLEMENTATION_GUIDE.md` → Section 6
- Reference: `../ScopeWorkSprint3.md` → Section 6

**Key Requirement:** Must use spatial MAE features (NOT CLS token)

### Stage 4: Final Validation & Reporting ⏳
**Status:** NOT STARTED  
**Documents:**
- Requirements: `SOW_IMPLEMENTATION_GUIDE.md` → Section 7
- Template: `../ScopeWorkSprint3.md` → Section 7 (Table 7.3a)

---

## 🎯 Success Criteria Checklist

- [ ] **WP-1:** Geometric features extracted for 933 samples
- [ ] **WP-2:** Atmospheric features extracted for 933 samples
- [ ] **WP-3:** Physical baseline achieves LOO CV R² > 0 (GO/NO-GO)
- [ ] **WP-4:** Hybrid model achieves LOO CV R² > 0.3 (TARGET)
- [ ] **Final:** All deliverables generated per Section 7 of SOW

---

## 📊 Key Metrics Tracking

| Metric | Baseline (Failed) | WP-3 Target | WP-4 Target | Actual |
|--------|-------------------|-------------|-------------|--------|
| Angles-Only R² | -4.46 | - | - | -4.46 |
| MAE CLS R² | < 0 | - | - | < 0 |
| Physical Baseline R² | - | **> 0** | - | ⏳ TBD |
| Hybrid Full R² | - | - | **> 0.3** | ⏳ TBD |

---

## 🛠️ Quick Commands

```bash
# Get started
cd cloudMLPublic
./sow_outputs/run_sow.sh --verbose

# Check what's implemented
ls -lh sow_outputs/wp*_*.py

# Check what's been generated
ls -lh sow_outputs/*/

# View help
./sow_outputs/run_sow.sh --help

# Run specific work package
python sow_outputs/wp1_geometric_features.py --help
python sow_outputs/wp2_atmospheric_features.py --help
```

---

## 📞 Getting Help

### General Questions
- Start: `README.md`
- Quick lookup: `QUICK_REFERENCE.md`

### Implementation Questions
- WP-1/WP-2: See code comments in `.py` files
- WP-3/WP-4: See `SOW_IMPLEMENTATION_GUIDE.md` detailed specs

### Troubleshooting
- Common issues: `QUICK_REFERENCE.md` → "Common Issues"
- Testing: `README.md` → "Testing Checklist"
- Debugging: `SOW_IMPLEMENTATION_GUIDE.md` → "Known Limitations and Risks"

### Project Context
- Requirements: `../ScopeWorkSprint3.md`
- Background: `../docs/project_status_report.pdf`
- Summary: `../docs/ONE_PAGE_SUMMARY.md`

---

## 🔗 Related Files Outside This Directory

```
../
├── ScopeWorkSprint3.md           ← Official SOW requirements
├── configs/bestComboConfig.yaml  ← Configuration file
├── docs/
│   ├── project_status_report.pdf ← Project background
│   └── ONE_PAGE_SUMMARY.md       ← Quick project summary
├── scripts/
│   └── validate_hybrid_loo.py    ← Reference for LOO CV implementation
└── src/
    ├── hdf5_dataset.py           ← Dataset loading
    └── evaluate_model.py         ← Metrics computation
```

---

## ✅ Immediate Next Steps

1. **If you haven't run anything yet:**
   - Read: `QUICK_REFERENCE.md` (5 min)
   - Run: `./run_sow.sh --verbose` (2-4 hours)

2. **If features are extracted:**
   - Check outputs (see `QUICK_REFERENCE.md` → "Data Inspection")
   - Read: `SOW_IMPLEMENTATION_GUIDE.md` Section 5 (WP-3)
   - Implement: `wp3_physical_baseline.py`

3. **If WP-3 is done and passed:**
   - Read: `SOW_IMPLEMENTATION_GUIDE.md` Section 6 (WP-4)
   - Implement: `wp4_hybrid_models.py`
   - Generate final deliverables

---

## 📈 Project Progress

```
[████████████████████░░░░░░░░] 50% Complete

✅ WP-1: Geometric Features (100%)
✅ WP-2: Atmospheric Features (100%)
⏳ WP-3: Physical Baseline (0%)
⏳ WP-4: Hybrid Models (0%)
⏳ Final Deliverables (0%)
```

**Estimated Time to Completion:** 20-35 hours

---

## 🎓 Learning Path

**For someone new to the project:**

1. **Context (30 min):**
   - `../docs/ONE_PAGE_SUMMARY.md` - What's this project about?
   - `../ScopeWorkSprint3.md` Section 1 - Why this approach?

2. **Quick Start (5 min):**
   - `QUICK_REFERENCE.md` - How to run it?

3. **Understanding (15 min):**
   - `README.md` - What does each work package do?

4. **Implementation (as needed):**
   - `SOW_IMPLEMENTATION_GUIDE.md` - How to build WP-3/WP-4?

5. **Execution:**
   - `./run_sow.sh` - Run the pipeline

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025 | Initial implementation (WP-1, WP-2 complete) |

---

## 🎯 Success Statement

> **Goal:** Validate that physics-constrained features (shadow geometry + atmospheric thermodynamics) enable cross-flight generalization in CBH retrieval, where pure ML approaches have catastrophically failed.

> **Success Metric:** Physical baseline R² > 0, Hybrid model R² > 0.3

> **Current Status:** Foundation complete, ready for validation phase

---

**Ready to begin? Start here:**
```bash
cd cloudMLPublic && ./sow_outputs/run_sow.sh --verbose
```

**Questions? Check:** `README.md` or `QUICK_REFERENCE.md`

**Need details? See:** `SOW_IMPLEMENTATION_GUIDE.md`

---

**END OF INDEX**