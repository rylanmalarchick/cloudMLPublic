# torch.compile State Dict Loading Fix

**Date:** January 18, 2025  
**Issue:** RuntimeError when loading pretrained model with torch.compile  
**Status:** ✅ FIXED

---

## The Problem

After successful pretraining with `torch.compile` enabled, the final training phase crashed when trying to load the pretrained checkpoint:

```
RuntimeError: Error(s) in loading state_dict for MultimodalRegressionModel:
	Missing key(s) in state_dict: "cnn_layers.0.weight", ...
	Unexpected key(s) in state_dict: "_orig_mod.cnn_layers.0.weight", ...
```

### Why This Happens

When you use `torch.compile()` on a model, PyTorch wraps the model and adds a prefix `_orig_mod.` to all layer names in the compiled model. When you save this compiled model:

```python
# During pretraining with torch.compile
model = torch.compile(model)
torch.save({"model_state_dict": model.state_dict()}, "checkpoint.pth")
# Saved keys: "_orig_mod.cnn_layers.0.weight", "_orig_mod.cnn_layers.0.bias", ...
```

But when you try to load this checkpoint into a fresh, uncompiled model:

```python
# During final training
model = MultimodalRegressionModel(config)  # Fresh uncompiled model
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])  # ❌ FAILS!
# Expected keys: "cnn_layers.0.weight", "cnn_layers.0.bias", ...
# Got keys: "_orig_mod.cnn_layers.0.weight", "_orig_mod.cnn_layers.0.bias", ...
```

The key names don't match → loading fails.

---

## The Solution

Detect and remove the `_orig_mod.` prefix from checkpoint keys before loading:

```python
if pre_ckpt:
    print(f"Initialized model from {pre_ckpt}")
    checkpoint = torch.load(pre_ckpt, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Remove '_orig_mod.' prefix added by torch.compile if present
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        print("  → Cleaning torch.compile prefixes from checkpoint...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
```

### What This Does

1. **Load checkpoint** - Get the saved state dict
2. **Check for prefix** - Look for any keys starting with `_orig_mod.`
3. **Clean keys** - Remove the prefix from all keys
4. **Load cleaned state** - Now the keys match the uncompiled model

---

## When Does This Matter?

This fix is needed when **all** of these conditions are true:

1. ✅ You use `torch.compile()` during training
2. ✅ You save model checkpoints
3. ✅ You load those checkpoints into a fresh model instance
4. ✅ The fresh model is **not** compiled when you load the checkpoint

### Common Scenarios

#### Scenario 1: Pretraining → Final Training (THIS ONE)
```python
# Pretraining phase
model = torch.compile(model)  # Compiled
train_and_save(model, "pretrain.pth")  # Saves with _orig_mod. prefix

# Final training phase
model = MultimodalRegressionModel(config)  # Fresh, uncompiled
model.load_state_dict(torch.load("pretrain.pth")["model_state_dict"])  # ❌ Needs fix
model = torch.compile(model)  # Compile after loading
```

#### Scenario 2: Training → Inference
```python
# Training
model = torch.compile(model)
torch.save(model.state_dict(), "model.pth")

# Inference
model = MultimodalRegressionModel(config)  # Fresh, uncompiled
model.load_state_dict(torch.load("model.pth"))  # ❌ Needs fix
```

#### Scenario 3: Does NOT need fix
```python
# Training
model = torch.compile(model)
torch.save(model.state_dict(), "model.pth")

# Loading into compiled model
model = MultimodalRegressionModel(config)
model = torch.compile(model)  # Compile FIRST
model.load_state_dict(torch.load("model.pth"))  # ✅ Works (both have _orig_mod.)
```

---

## Code Changes

**File:** `src/pipeline.py`

**Function:** `run_final_training_and_evaluation()`

**Before:**
```python
if pre_ckpt:
    print(f"Initialized model from {pre_ckpt}")
    model.load_state_dict(
        torch.load(pre_ckpt, weights_only=False)["model_state_dict"]
    )
```

**After:**
```python
if pre_ckpt:
    print(f"Initialized model from {pre_ckpt}")
    checkpoint = torch.load(pre_ckpt, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Remove '_orig_mod.' prefix added by torch.compile if present
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        print("  → Cleaning torch.compile prefixes from checkpoint...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
```

---

## Expected Output

When the fix is applied and you restart training:

```
--- Final full-flight training ---
Initialized model from models/trained/pretrain_30Oct24_baseline_full_20251018_232936.pth
  → Cleaning torch.compile prefixes from checkpoint...  ✅ Fix applied
Applying torch.compile() with mode='default'...
  → Model compiled successfully
Starting training for 50 epochs on CUDA
Epoch 1/50 | Train Loss: 3.1234 | Val Loss: 1.0123 | Time: 95s
...
```

The key indicator is the line:
```
→ Cleaning torch.compile prefixes from checkpoint...
```

This means the fix detected and cleaned the `_orig_mod.` prefixes.

---

## Why Not Save Without Prefixes?

You might wonder: "Why not just save the model without the `_orig_mod.` prefix in the first place?"

**Answer:** You could, but it requires extra complexity:

```python
# Option 1: Save compiled model (simple but has _orig_mod. prefix)
torch.save({"model_state_dict": model.state_dict()}, "checkpoint.pth")

# Option 2: Save underlying model (complex)
if hasattr(model, "_orig_mod"):
    state_dict = model._orig_mod.state_dict()  # Access underlying model
else:
    state_dict = model.state_dict()
torch.save({"model_state_dict": state_dict}, "checkpoint.pth")
```

**Our approach is cleaner:** Save simply, clean when loading. This way:
- ✅ Saving code stays simple
- ✅ Loading handles both compiled and uncompiled checkpoints automatically
- ✅ One fix location (loading code) instead of multiple save locations
- ✅ Backward compatible with old checkpoints

---

## Alternative Solutions

### Alternative 1: Compile before loading
```python
model = MultimodalRegressionModel(config)
model = torch.compile(model)  # Compile FIRST
model.load_state_dict(checkpoint["model_state_dict"])  # Now keys match
```

**Pros:** No state dict cleaning needed  
**Cons:** Forces you to always compile in the same order; less flexible

### Alternative 2: Use model._orig_mod when saving
```python
if hasattr(model, "_orig_mod"):
    torch.save({"model_state_dict": model._orig_mod.state_dict()}, "checkpoint.pth")
else:
    torch.save({"model_state_dict": model.state_dict()}, "checkpoint.pth")
```

**Pros:** Checkpoints don't have prefix  
**Cons:** Every save location needs this logic; accessing private attributes is fragile

### Alternative 3: Disable torch.compile for pretraining
```python
# Don't compile during pretraining
model = MultimodalRegressionModel(config)  # No compile
train(model)  # Slower but saves clean checkpoints
```

**Pros:** No prefix issues  
**Cons:** 15-25% slower pretraining; lose compile benefits

### ✅ Our Solution (Best)
Clean prefixes when loading - simple, flexible, one fix location.

---

## Testing

To verify the fix works:

1. **Pretrain with compile:**
   ```python
   model = torch.compile(model)
   train(model)
   save(model, "pretrain.pth")
   ```

2. **Load into fresh model:**
   ```python
   model = MultimodalRegressionModel(config)
   checkpoint = torch.load("pretrain.pth")
   model.load_state_dict(checkpoint["model_state_dict"])  # Should work now ✅
   ```

3. **Check for cleaning message:**
   ```
   Initialized model from pretrain.pth
   → Cleaning torch.compile prefixes from checkpoint...  ✅
   ```

---

## Related Issues

This fix is related to but different from:
- **CUDA graph error** (fixed by changing compile mode to "default")
- **Gradient checkpointing compatibility** (fixed in same commit)

All three issues stem from using `torch.compile()` with advanced features, but have different root causes and fixes.

---

## References

- PyTorch torch.compile docs: https://pytorch.org/docs/stable/generated/torch.compile.html
- Issue tracking: https://github.com/pytorch/pytorch/issues/99031
- State dict compatibility: https://pytorch.org/tutorials/beginner/saving_loading_models.html

---

**Status:** ✅ Fixed in commit 1235a77  
**Files Modified:** `src/pipeline.py`  
**Lines Added:** 9  
**Impact:** Enables loading pretrained torch.compile checkpoints into fresh models
