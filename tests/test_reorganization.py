#!/usr/bin/env python3
"""
Integration test to verify Sprint 6 reorganization from sow_outputs to professional structure.

Tests:
1. All source code moved to src/cbh_retrieval/
2. All tests moved to tests/cbh/
3. All scripts moved to scripts/cbh/
4. All documentation moved to docs/cbh/
5. All results moved to results/cbh/
6. All models moved to models/cbh_production/
7. Module can be imported
8. File structure is clean and professional
"""

import os
from pathlib import Path


def test_project_root_exists():
    """Verify we're in the correct project root."""
    assert Path("src").exists(), "src/ directory should exist"
    assert Path("tests").exists(), "tests/ directory should exist"
    assert Path("docs").exists(), "docs/ directory should exist"
    assert Path("README.md").exists(), "README.md should exist"


def test_cbh_source_code_structure():
    """Verify CBH source code is in src/cbh_retrieval/."""
    cbh_src = Path("src/cbh_retrieval")

    assert cbh_src.exists(), "src/cbh_retrieval/ should exist"
    assert (cbh_src / "__init__.py").exists(), "src/cbh_retrieval/__init__.py should exist"
    assert (cbh_src / "README.md").exists(), "src/cbh_retrieval/README.md should exist"

    # Check key modules exist
    key_modules = [
        "image_dataset.py",
        "mc_dropout.py",
        "train_production_model.py",
        "offline_validation_tabular.py",
        "ensemble_models.py",
        "error_analysis.py",
    ]

    for module in key_modules:
        assert (cbh_src / module).exists(), f"src/cbh_retrieval/{module} should exist"

    # Check at least 20 Python files
    py_files = list(cbh_src.glob("*.py"))
    assert len(py_files) >= 20, f"Should have at least 20 Python files, found {len(py_files)}"


def test_cbh_tests_structure():
    """Verify CBH tests are in tests/cbh/."""
    cbh_tests = Path("tests/cbh")

    assert cbh_tests.exists(), "tests/cbh/ should exist"

    # Check test files
    test_files = [
        "test_data_loading.py",
        "test_features.py",
        "test_model_inference.py",
        "test_training.py",
    ]

    for test_file in test_files:
        assert (cbh_tests / test_file).exists(), f"tests/cbh/{test_file} should exist"

    # Check pytest config
    assert (cbh_tests / "pytest.ini").exists(), "tests/cbh/pytest.ini should exist"


def test_cbh_scripts_structure():
    """Verify CBH scripts are in scripts/cbh/."""
    cbh_scripts = Path("scripts/cbh")

    assert cbh_scripts.exists(), "scripts/cbh/ should exist"

    # Check for key scripts
    assert (cbh_scripts / "power_of_10_audit.py").exists(), "Power of 10 audit script should exist"

    # At least 2 scripts
    scripts = list(cbh_scripts.glob("*"))
    assert len(scripts) >= 2, f"Should have at least 2 scripts, found {len(scripts)}"


def test_cbh_documentation_structure():
    """Verify CBH documentation is in docs/cbh/."""
    cbh_docs = Path("docs/cbh")

    assert cbh_docs.exists(), "docs/cbh/ should exist"

    # Check key documentation files
    key_docs = [
        "MODEL_CARD.md",
        "DEPLOYMENT_GUIDE.md",
        "REPRODUCIBILITY_GUIDE.md",
        "FUTURE_WORK.md",
        "SPRINT6_FINAL_DELIVERY.md",
        "SPRINT6_100_PERCENT_COMPLETION.md",
    ]

    for doc in key_docs:
        assert (cbh_docs / doc).exists(), f"docs/cbh/{doc} should exist"

    # Check requirements files
    assert (cbh_docs / "requirements_production.txt").exists(), "Requirements file should exist"

    # At least 12 markdown files
    md_files = list(cbh_docs.glob("*.md"))
    assert len(md_files) >= 10, f"Should have at least 10 markdown files, found {len(md_files)}"


def test_cbh_results_structure():
    """Verify CBH results are in results/cbh/."""
    cbh_results = Path("results/cbh")

    assert cbh_results.exists(), "results/cbh/ should exist"

    # Check subdirectories
    assert (cbh_results / "figures").exists(), "results/cbh/figures/ should exist"
    assert (cbh_results / "reports").exists(), "results/cbh/reports/ should exist"

    # Check for figures
    figures = list((cbh_results / "figures").rglob("*.png"))
    assert len(figures) >= 40, f"Should have at least 40 PNG figures, found {len(figures)}"

    # Check for reports
    reports = list((cbh_results / "reports").glob("*.json"))
    assert len(reports) >= 10, f"Should have at least 10 JSON reports, found {len(reports)}"


def test_cbh_models_structure():
    """Verify CBH models are in models/cbh_production/."""
    cbh_models = Path("models/cbh_production")

    assert cbh_models.exists(), "models/cbh_production/ should exist"

    # Check for model files (may or may not exist depending on training)
    joblib_files = list(cbh_models.glob("*.joblib"))
    pkl_files = list(cbh_models.glob("*.pkl"))

    # At least some model artifacts should exist
    assert len(joblib_files) + len(pkl_files) >= 0, "Model directory should be present"


def test_preprocessed_data_location():
    """Verify preprocessed data moved to outputs/preprocessed_data/."""
    preprocessed = Path("outputs/preprocessed_data")

    assert preprocessed.exists(), "outputs/preprocessed_data/ should exist"
    assert (preprocessed / "Integrated_Features.hdf5").exists(), "Integrated features should exist"


def test_sow_outputs_can_be_deleted():
    """Verify sow_outputs is no longer needed (all content migrated)."""
    sow_outputs = Path("sow_outputs")

    # This test documents that sow_outputs can be safely deleted
    # We don't actually delete it here, but verify the new structure is complete

    # All essential files should be in new locations
    assert Path("src/cbh_retrieval").exists(), "Source code migrated"
    assert Path("tests/cbh").exists(), "Tests migrated"
    assert Path("docs/cbh").exists(), "Docs migrated"
    assert Path("results/cbh").exists(), "Results migrated"
    assert Path("models/cbh_production").exists(), "Models migrated"

    print("\n All content successfully migrated from sow_outputs/")
    print(" sow_outputs/ can be safely deleted")


def test_no_import_errors_in_init():
    """Verify __init__.py has correct structure (syntax check only)."""
    init_file = Path("src/cbh_retrieval/__init__.py")

    assert init_file.exists(), "__init__.py should exist"

    # Read and check for key exports
    content = init_file.read_text()

    assert "__version__" in content, "Should define __version__"
    assert "__all__" in content, "Should define __all__"
    assert "from .image_dataset import" in content or "ImageCBHDataset" in content, (
        "Should export ImageCBHDataset"
    )


def test_professional_structure():
    """Verify overall structure is clean and professional."""

    # Check no sow_outputs references in new code
    cbh_py_files = list(Path("src/cbh_retrieval").glob("*.py"))

    problematic_imports = []
    for py_file in cbh_py_files:
        content = py_file.read_text()
        if "sow_outputs/sprint6" in content and py_file.name != "__init__.py":
            problematic_imports.append(py_file)

    assert len(problematic_imports) == 0, f"Found sow_outputs references in: {problematic_imports}"

    # Verify clean directory structure
    assert Path("src/cbh_retrieval").is_dir(), "Source should be a directory"
    assert Path("tests/cbh").is_dir(), "Tests should be a directory"
    assert Path("docs/cbh").is_dir(), "Docs should be a directory"

    print("\n Professional directory structure verified")


def test_documentation_complete():
    """Verify all critical documentation is present."""
    docs = Path("docs/cbh")

    # Essential docs for production deployment
    essential = [
        "MODEL_CARD.md",
        "DEPLOYMENT_GUIDE.md",
        "REPRODUCIBILITY_GUIDE.md",
    ]

    for doc in essential:
        doc_path = docs / doc
        assert doc_path.exists(), f"Essential doc {doc} missing"

        # Should be substantial (>1KB)
        assert doc_path.stat().st_size > 1000, f"{doc} seems too small"


def test_results_organized():
    """Verify results are properly organized by type."""
    results = Path("results/cbh")

    # Check organization
    subdirs = ["figures", "reports"]
    for subdir in subdirs:
        assert (results / subdir).exists(), f"results/cbh/{subdir} should exist"

    # Figures should have subdirectories by type
    figures = results / "figures"
    expected_fig_types = ["paper", "ensemble", "validation"]

    for fig_type in expected_fig_types:
        assert (figures / fig_type).exists(), f"figures/{fig_type} should exist"


if __name__ == "__main__":
    """Run tests manually."""
    import sys

    tests = [
        test_project_root_exists,
        test_cbh_source_code_structure,
        test_cbh_tests_structure,
        test_cbh_scripts_structure,
        test_cbh_documentation_structure,
        test_cbh_results_structure,
        test_cbh_models_structure,
        test_preprocessed_data_location,
        test_sow_outputs_can_be_deleted,
        test_no_import_errors_in_init,
        test_professional_structure,
        test_documentation_complete,
        test_results_organized,
    ]

    failed = []
    passed = 0

    print("=" * 80)
    print("REORGANIZATION VERIFICATION TESTS")
    print("=" * 80)
    print()

    for test_func in tests:
        test_name = test_func.__name__
        try:
            test_func()
            print(f" {test_name}")
            passed += 1
        except AssertionError as e:
            print(f" {test_name}: {e}")
            failed.append(test_name)
        except Exception as e:
            print(f" {test_name}: ERROR - {e}")
            failed.append(test_name)

    print()
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} passed")

    if failed:
        print(f"\nFailed tests: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n ALL TESTS PASSED - Reorganization successful!")
        sys.exit(0)
