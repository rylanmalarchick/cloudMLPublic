# CloudML Project - Future Work & Strategic Roadmap
**Last Updated:** November 24, 2025  
**Status:** Paper 1 submitted to journal (Jan 2026), Paper 2 planning phase  
**Primary Goal:** Atmospheric ML + domain adaptation + publication pipeline

---

## Executive Summary

CloudML has two orthogonal research directions that should **not compete** but rather **inform each other**:

1. **Paper 1 (Published):** Atmospheric Features vs Images for Cloud Base Height (GBDT vs CNN on 933 samples)  
   - Status: Preprint released Nov 13, 2025
   - Next: Submit to journal (Atmospheric Measurement Techniques or JGR Atmospheres) by Jan 31, 2026
   - Focus: Negative result (ensemble doesn't help), strong domain shift finding

2. **Paper 2 (Next):** Self-Supervised Learning for Cloud Base Height Retrieval (Domain Adaptation Track)  
   - Status: Planning phase now
   - Timeline: Spring 2026 → Submit by June 2026
   - Focus: Fix domain shift problem using contrastive learning (SimCLR/MoCo)

3. **Paper 3+ (Optional):** Advanced methods downstream  
   - Physics-informed neural networks
   - Domain-adversarial neural networks (DANN)
   - Multi-source fusion (satellite + ground + airborne)

---

## Timeline & Milestones

### **Phase 1: Journal Submission (Dec 2025 - Feb 2026)**

**Objective:** Get Paper 1 accepted to peer-reviewed journal

| Task | Deadline | Owner | Notes |
|------|----------|-------|-------|
| Fix 3-4 critical issues from feedback doc | Dec 28 | You | Remove "state-of-the-art" oversell, clarify baselines |
| Format for target journal | Jan 10 | You | Atmospheric Measurement Techniques preferred |
| Submit Paper 1 | Jan 31 | You | Expect 3-month review cycle |
| Begin Paper 2 literature review | Jan 15 | You | Parallel with submission |

**Success Criteria:**
- Paper 1 submitted to first-choice journal
- Clean GitHub repo with reproducible results
- README clearly states limitations (8-bit data, 933 samples, domain shift)

---

### **Phase 2: Paper 2 Development (Jan 2026 - May 2026)**

**Objective:** Solve domain shift problem, produce publication-ready results

**Research Question:** Can unsupervised domain adaptation (contrastive learning) improve cross-flight generalization?

#### **2.1 Literature Sprint (Jan 15 - Jan 31)**
- [ ] Review contrastive learning papers (SimCLR, MoCo, BYOL)
- [ ] Study atmospheric ML domain adaptation (search for papers on satellite→airborne, etc.)
- [ ] Identify 3-5 baseline methods to compare against
- [ ] Create comparison table: method, data efficiency, cross-flight performance

**Key Papers to Read:**
- Chen et al. 2020: SimCLR - Simple Framework for Contrastive Learning
- He et al. 2020: MoCo - Momentum Contrast Unsupervised ViR
- Tuia et al. 2016: Domain Adaptation for Remote Sensing (survey)

#### **2.2 Experiment Design (Feb 1 - Feb 14)**
Define experimental protocol before coding:

```
Baseline Experiment:
  - Train on flights 1-4 (882 samples)
  - Test on flight 18Feb25 (44 samples)
  - Measure: R², MAE, RMSE
  - Expected baseline: R² ≈ -0.98 (catastrophic failure from Paper 1)

SimCLR Experiment:
  - Step 1: Unsupervised pretraining on unlabeled cloud imagery
    * Use all ER-2 camera images (labeled + unlabeled)
    * Learn representations without cloud base labels
  - Step 2: Linear probe on downstream task
    * Freeze encoder, train linear layer on labeled data
    * Cross-flight evaluation
  - Expected improvement: R² → 0.2-0.4 range?

MoCo Experiment:
  - Similar to SimCLR but with momentum encoder
  - Potentially better for smaller datasets

Supervised Baseline (transfer learning):
  - ImageNet pretrained ResNet-50
  - Fine-tune on cloud imagery
  - Cross-flight evaluation
```

**Milestones:**
- [ ] Experimental protocol document (1-2 pages)
- [ ] Hypothesis: "Contrastive pretraining on atmospheric imagery will improve cross-flight generalization by learning invariant features to flight-specific visual artifacts"

#### **2.3 Implementation (Feb 15 - Apr 15)**

**Code Structure:**
```
cloudml/
├── data/
│   ├── ER2_flights_1_to_5.hdf5  (existing)
│   ├── ER2_flights_unlabeled.hdf5  (collect if possible)
│   └── preprocess_for_contrastive.py
├── models/
│   ├── simclr_encoder.py
│   ├── moco_encoder.py
│   └── downstream_probes.py
├── experiments/
│   ├── baseline_lofo.py
│   ├── simclr_pretrain.py
│   ├── moco_pretrain.py
│   ├── transfer_learning.py
│   └── evaluation.py
├── results/
│   └── paper2_results.csv
└── README_Paper2.md
```

**Sprints:**
- **Week 1-2 (Feb 15-28):** Implement SimCLR baseline
  - Data pipeline for contrastive learning
  - Augmentation strategy for cloud imagery
  - Training loop with 100 epochs
  
- **Week 3-4 (Mar 1-14):** Run baselines + analysis
  - Paper 1 baseline (supervised, within-distribution)
  - Catastrophic failure case (cross-flight LOFO)
  - SimCLR pretrain + linear probe
  
- **Week 5-6 (Mar 15-28):** MoCo + transfer learning
  - Implement MoCo variant
  - ImageNet transfer learning baseline
  - Compare all methods
  
- **Week 7-8 (Apr 1-15):** Analysis & writing
  - SHAP analysis for best model
  - Error analysis for remaining failures
  - Begin paper draft

**Milestones:**
- [ ] SimCLR pretrain + probe working end-to-end (Mar 7)
- [ ] Baseline comparison results (Mar 21)
- [ ] Paper 2 first draft (Apr 30)

#### **2.4 Results & Analysis (Apr 15 - May 15)**

**Expected Outcomes:**
1. Quantify improvement: "SimCLR pretraining improves cross-flight R² from -0.98 to X"
2. Identify best method (SimCLR vs MoCo vs transfer learning)
3. Per-flight analysis: which flights improve, which remain hard
4. Failure mode analysis: when does contrastive learning help/hurt

**Figures for Paper 2:**
- Figure 1: Experimental design (diagram)
- Figure 2: Cross-flight performance comparison (bar chart)
- Figure 3: LOFO validation for best method vs Paper 1 baseline
- Figure 4: UMAP projection of learned representations (supervised vs unsupervised)
- Figure 5: Error analysis - residuals by method

**Writing (Draft by Apr 30, final by May 31):**
- Motivation: Atmospheric ML domain shift is unsolved
- Methods: SimCLR/MoCo pretraining + linear evaluation protocol
- Results: R² improvement, generalization gains
- Discussion: Why contrastive learning works for atmospheric data
- Limitations: Still limited by 933 samples, needs more unlabeled data

**Target Journal:** AI for Earth and Space Science (lower bar than paper 1) or NeurIPS 2026 Climate Change AI track

---

### **Phase 3: Optional Extensions (Summer 2026+)**

Only pursue if Paper 2 results are strong AND you have time after PhD apps:

#### **3.1 Paper 3: Physics-Informed Domain Adaptation**
- Add physics constraints: predictions must respect LCL, atmospheric stability
- Differentiable physics module: force model to obey cloud formation equations
- Timeline: Summer 2026 (if internship allows)

#### **3.2 Data Collection Sprint**
- Coordinate with NASA to acquire additional ER-2 flight data
- Expand dataset from 933 → 2000+ samples
- Diverse geographic regions / seasons
- **Only if Paper 2 shows promise**

#### **3.3 Satellite Data Integration**
- Compare ER-2 patterns to MODIS/GOES satellite data
- Test if learned representations transfer to satellite domain
- Real-world impact paper

---

## Decision Framework

**When to pivot away from CloudML:**
- ❌ Paper 1 rejected from first two journals
- ❌ Paper 2 shows <5% cross-flight improvement by May 1
- ❌ Domain shift remains unsolved after 2 months of effort
- ✅ Paper 1 accepted OR under review by Jan 31
- ✅ Paper 2 shows >20% improvement in cross-flight R² by Apr 1
- ✅ Results are novel enough for atmospheric ML community

**Realistic outcome:** Paper 1 gets published (low bar, novel dataset paper), Paper 2 gets rejected 1-2 times, then accepted to lower-tier venue. Goal is NOT Nature/Science, goal is **publishable + adds to your PhD narrative**.

---

## Integration with QubitPulseOpt & PhD Applications

**How CloudML fits your broader story:**

| Project | Technical Focus | Career Message |
|---------|-----------------|-----------------|
| **CloudML** | Data-limited ML, domain shift, atmospheric science | I solve real-world ML problems in Earth science |
| **QubitPulseOpt** | Quantum optimal control, hardware integration, FPGA future | I design control systems for quantum hardware |
| **AirHound** | Real-time embedded ML, robotics, perception | I integrate AI with physical systems |

**PhD Application Narrative:**
> "My research spans data-limited machine learning (CloudML), quantum optimal control (QubitPulseOpt), and embedded AI systems (AirHound). Common thread: I design and validate intelligent systems that bridge simulation and hardware. For my PhD, I want to combine these into FPGA-accelerated quantum control."

**Timeline Alignment:**
- Paper 1 (CloudML) submitted by Jan 31 → mention in PhD applications Dec 2026
- Paper 2 (CloudML) submitted by June 30 → could update PhD apps if accepted
- QubitPulseOpt Paper 1 submitted by Dec 2026 → main PhD application material
- FPGA integration by Dec 2026 → differentiator for quantum programs

---

## Success Metrics

### Paper 1 (Dec 2025 - Feb 2026)
- [ ] Submitted to journal
- [ ] Clean, reproducible code on GitHub
- [ ] 93.5% test coverage maintained

### Paper 2 (Jan 2026 - May 2026)
- [ ] Experimental protocol document created
- [ ] SimCLR implementation complete + working
- [ ] Cross-flight R² improvement >10% over Paper 1 baseline
- [ ] First draft complete by May 1
- [ ] Submitted to journal by June 30

### Overall CloudML Project
- [ ] 2 papers (1 published or accepted, 1 submitted)
- [ ] Open-source framework with 93%+ test coverage
- [ ] Clear narrative: "Solved domain shift in atmospheric ML"

---

## Notes & Constraints

**Practical Realities:**
1. You finish semester Dec 10 → limited time Dec 10-31
2. PhD apps due Dec 15 2026 → CloudML Paper 2 must be submitted by Oct 31 at latest
3. Internship likely Summer 2026 → may interrupt work Apr-June
4. QubitPulseOpt is higher priority for PhD apps

**Resource Allocation:**
- 60% QubitPulseOpt (higher ROI for PhD)
- 30% CloudML Paper 2 (if Paper 1 accepted)
- 10% Other (AirHound, maintenance)

**If time is scarce:**
- Prioritize Paper 1 journal submission (Dec 2025 - Jan 2026)
- Skip Paper 2 if Paper 1 is rejected from first 2 journals
- Move FPGA work to top priority if internship is Fermilab/IBM

---

## References & Useful Links

**Contrastive Learning Papers:**
- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722

**Target Journals:**
- Atmospheric Measurement Techniques (AMT)
- Journal of Geophysical Research (JGR) Atmospheres
- AI for Earth and Space Science (if Paper 2)

**Your Repos:**
- Paper 1: https://github.com/rylanmalarchick/cloudMLPublic
- Preprint: Nov 13, 2025 on GitHub

---

**Questions?** Refer back to this doc when deciding what to work on next month. If you find Paper 1 gets rejected or you run out of time, you've made an explicit decision *not* to pursue Paper 2 heavily—that's OK. Your QubitPulseOpt + FPGA work is the higher-leverage play for PhD admission anyway.