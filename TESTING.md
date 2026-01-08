# Testing the Cliff Data Pipeline

To ensure the data preparation script is working correctly before you process your large 35GB files, you can run the provided test suite.

### 1. Requirements
Ensure you have the following installed in your environment:
- `pytest`
- `laspy`
- `numpy`
- `torch`

### 2. Running the Tests
From the root of the repository, run:
```bash
# Ensure the current directory is in your python path
export PYTHONPATH=$PYTHONPATH:.

# Run the tests
pytest tests/test_prep_data.py
```

### 3. What is being tested?
- **Voxelization Test:** Ensures that points falling into the same spatial grid (e.g., 0.02m) are correctly merged.
- **Tiling Pipeline Test:** Creates a small dummy `.las` file, runs the tiling script, and verifies that the output `.pth` files contain the expected dictionary keys (`coord`, `feat`, `segment`, `grid_size`) required by the PTv3 `Point` class.
