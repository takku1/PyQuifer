# Bugs Identified During Code Review

This document lists minor bugs and issues identified during the initial review of the PyQuifer codebase. These are primarily related to example usage and clarity, rather than core logic errors.

## 1. Missing `matplotlib.pyplot` Import in `src/pyquifer/core.py` Example

**Location:** `src/pyquifer/core.py` (within the `if __name__ == '__main__':` block)

**Description:**
The example usage block at the end of `core.py` utilizes `matplotlib.pyplot` for plotting the archetype evolution and loss history during training (`fig = plt.figure(...)`, `ax1 = fig.add_subplot(...)`, `plt.tight_layout()`, `plt.show()`). However, `matplotlib.pyplot` is not imported anywhere in the file.

**Impact:**
Running `core.py` directly will result in a `NameError` because `plt` is not defined.

**Suggested Fix:**
Add the following import statement at the beginning of `src/pyquifer/core.py` (or within the `if __name__ == '__main__':` block if preferred, though top-level is common for widely used libraries):

```python
import matplotlib.pyplot as plt
```