# CloudML Preprint Documents

This directory contains LaTeX preprint documents for the Cloud Base Height (CBH) Retrieval project.

## Available Documents

### 1. Academic Preprint (RECOMMENDED for arXiv)

**File:** `cloudml_academic_preprint.tex` / `cloudml_academic_preprint.pdf`

**Title:** *Atmospheric Features Outperform Images for Cloud Base Height Retrieval: A Systematic Comparison Using NASA Airborne Observations*

**Author:** Rylan Malarchick (Embry-Riddle Aeronautical University)

**Status:** Ready for arXiv submission

**Pages:** 22

**Style:** Academic research paper

**Key Features:**
-  Academic tone and structure (Introduction, Related Work, Methods, Results, Discussion)
-  60+ citations to relevant literature
-  Solo authorship with NASA acknowledgment
-  Framed as independent research continuing from NASA internship
-  Emphasizes scientific contributions and negative results
-  No "Sprint" or "Production" terminology
-  arXiv-ready formatting (PDFLaTeX compatible)

**Target Venues:**
- Primary: arXiv (cs.LG, cs.CV, physics.ao-ph)
- Future: NeurIPS 2026 Datasets & Benchmarks, KDD 2026, IGARSS 2026

---

### 2. Sprint 6 Technical Report (Original)

**File:** `sprint6_cbh_preprint.tex` / `sprint6_cbh_preprint.pdf`

**Title:** *Production-Ready Cloud Base Height Retrieval: Sprint 6 Validation and Ensemble Methods*

**Author:** Research Team, NASA High Altitude Research Program

**Status:** Internal technical documentation

**Pages:** 12

**Style:** Technical report / deliverable documentation

**Key Features:**
- Sprint-oriented structure and language
- Production deployment focus
- Internal SOW references
- Detailed implementation metrics
- Code quality and compliance reporting

**Use Case:** Internal NASA documentation, project archive

---

## Quick Start: Academic Preprint

### Compilation

```bash
cd preprint
pdflatex cloudml_academic_preprint.tex
pdflatex cloudml_academic_preprint.tex  # Second pass for references
```

**Output:** `cloudml_academic_preprint.pdf` (22 pages, ~1.7 MB)

### Requirements

Ubuntu/Debian:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

macOS:
```bash
brew install --cask mactex
```

---

## Key Results (Both Documents)

### Model Performance (Stratified 5-Fold CV, n=933)

| Model | R² | MAE (m) | RMSE (m) |
|-------|----|---------| ---------|
| **GBDT (Atmospheric)** | **0.744 ± 0.037** | **117.4 ± 7.4** | **187.3 ± 15.3** |
| CNN (Image) | 0.351 ± 0.075 | 236.8 ± 16.7 | 299.1 ± 18.2 |
| Weighted Ensemble | 0.739 ± 0.096 | 122.5 ± 19.8 | 195.0 ± 23.4 |

### Key Scientific Findings

1. **Atmospheric features dominate:** GBDT using ERA5 reanalysis achieves 2.1× lower error than CNN on airborne imagery
2. **Limited multi-modal benefit:** Ensemble methods provide < 1% R² improvement, indicating minimal complementarity
3. **Severe domain shift:** Leave-one-flight-out validation reveals R² = -0.98 for Flight F4 (catastrophic failure)
4. **Under-calibrated uncertainty:** 90% prediction intervals achieve only 77% coverage

---

## Document Comparison

| Aspect | Academic Preprint | Sprint 6 Report |
|--------|------------------|-----------------|
| **Audience** | Research community | NASA team / internal |
| **Tone** | Formal academic | Technical / operational |
| **Citations** | 60+ references | Minimal |
| **Sections** | Intro, Related Work, Discussion | Executive Summary, Deployment |
| **Authorship** | Solo (Rylan Malarchick) | Team (NASA HAR Program) |
| **Focus** | Scientific contributions | Production readiness |
| **Length** | 22 pages | 12 pages |
| **Figures** | 6 core scientific figures | All 24 validation figures |
| **arXiv Ready** |  Yes |  No (internal language) |

---

## Recommended Workflow

### For arXiv Submission

1. Use **`cloudml_academic_preprint.tex`**
2. Compile PDF and verify figures render correctly
3. Select arXiv categories: `cs.LG` (primary), `cs.CV`, `physics.ao-ph` (secondary)
4. License: CC-BY 4.0
5. Upload source `.tex` + figures + compiled PDF

### For Internal Documentation

1. Use **`sprint6_cbh_preprint.tex`**
2. Keep as project archive and SOW deliverable
3. Reference in technical handoff documents

---

## Academic Preprint Structure

### Section Breakdown

1. **Abstract** (250 words) - Condensed findings, no "Sprint" language
2. **Introduction** (2 pages)
   - Motivation: Cloud base height importance
   - Feature representation question
   - Research questions and contributions
3. **Related Work** (1.5 pages)
   - Cloud base height retrieval methods
   - GBDT for atmospheric science
   - Computer vision for remote sensing
   - Ensemble methods and domain adaptation
4. **Dataset and Methods** (3 pages)
   - NASA ER-2 platform (933 samples)
   - ERA5 reanalysis features (28 features)
   - Model architectures (GBDT, CNN, ensembles)
   - Experimental protocol (stratified 5-fold CV)
5. **Results** (3.5 pages)
   - Model comparison, ensemble analysis
   - Feature importance, error analysis
   - Uncertainty quantification, domain adaptation
6. **Discussion** (2.5 pages)
   - Why atmospheric features outperform images
   - Limited ensemble complementarity
   - Domain shift and generalization challenges
7. **Limitations and Future Work** (2 pages)
8. **Conclusion** (1 page)
9. **Acknowledgments** - NASA context clearly stated
10. **Code/Data Availability** - GitHub link, data sources
11. **References** - 60+ citations

---

## Citation

### Academic Preprint

```bibtex
@article{malarchick2025cloudml,
  title={Atmospheric Features Outperform Images for Cloud Base Height Retrieval: 
         A Systematic Comparison Using NASA Airborne Observations},
  author={Malarchick, Rylan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  note={Code available at https://github.com/rylanmalarchick/CloudMLPublic}
}
```

### Sprint 6 Report

```bibtex
@techreport{nasa_cbh_sprint6_2025,
  title={Production-Ready Cloud Base Height Retrieval: 
         Sprint 6 Validation and Ensemble Methods},
  author={{NASA CBH Retrieval Team}},
  institution={NASA High Altitude Research Program},
  year={2025},
  type={Technical Report}
}
```

---

## Additional Resources

### Code and Documentation

- **Source Code:** `../src/cbh_retrieval/` (22 production modules)
- **Tests:** `../tests/cbh/` (93.5% coverage)
- **Documentation:** `../docs/cbh/`
  - `MODEL_CARD.md` - Model specifications
  - `DEPLOYMENT_GUIDE.md` - Production deployment
  - `REPRODUCIBILITY_GUIDE.md` - Full reproduction steps
  - `SPRINT6_FINAL_DELIVERY.md` - Complete deliverables

### Results and Figures

- **Reports:** `../results/cbh/reports/` (13 JSON validation reports)
- **Figures:** `../results/cbh/figures/` (24 publication-ready figures)
  - `paper/` - Core manuscript figures
  - `ensemble/` - Ensemble analysis plots
  - `uncertainty/` - UQ calibration curves
  - `domain_adaptation/` - Few-shot learning results

### Data

- **NASA ER-2 Imagery:** https://har.gsfc.nasa.gov/
- **CPL Lidar Data:** Available upon request from NASA GSFC
- **ERA5 Reanalysis:** https://cds.climate.copernicus.eu/
- **Preprocessed Features:** `../outputs/preprocessed_data/Integrated_Features.hdf5`

---

## Authorship and Acknowledgment

The **academic preprint** (`cloudml_academic_preprint.tex`) is authored solely by **Rylan Malarchick** and frames the work as independent research conducted following a NASA OSTEM internship. The acknowledgments section clearly states:

> *"This work builds upon methods developed during the author's NASA OSTEM internship (May–August 2025) with the NASA Goddard Space Flight Center High Altitude Research Program. The author thanks Dr. Dong Wu and the NASA ER-2 flight team for data access and technical discussions during the internship period. All analysis, code development, model training, and results presented in this paper were conducted independently by the author following the internship conclusion."*

This framing:
-  Respects NASA collaboration context
-  Clarifies independent contribution scope
-  Allows solo authorship for fellowship applications
-  Maintains scientific integrity

---

## Next Steps

### Immediate (Week 1)

1.  Academic preprint drafted (`cloudml_academic_preprint.tex`)
2.  PDF compiled successfully (22 pages)
3.  Review figures and ensure all paths resolve correctly
4.  Email Dr. Wu for final authorship confirmation (optional but recommended)
5.  Proofread for typos and clarity

### arXiv Submission (Week 2)

1. Final PDF compilation
2. Prepare ancillary files (figures as separate uploads if needed)
3. Submit to arXiv (cs.LG primary, cs.CV + physics.ao-ph cross-list)
4. Select CC-BY 4.0 license
5. Update CV and fellowship applications with arXiv link

### Conference Submission (Future)

- **NeurIPS 2026 Datasets & Benchmarks** (May 2026 deadline) - 6 months
- **KDD 2026 Applied Data Science** (July 2026) - 8 months
- **IGARSS 2026** (Jan 2026) - 2 months (tight, may need co-authors)

---

## Contact

- **GitHub:** https://github.com/rylanmalarchick/CloudMLPublic
- **Email:** rylan.malarchick@erau.edu
- **Issues:** https://github.com/rylanmalarchick/CloudMLPublic/issues

---

**Last Updated:** November 2025  
**Academic Preprint Version:** 1.0 (arXiv-ready)  
**Sprint 6 Report Version:** 1.0 (archival)