"""
Record the PyTorch reference mask for the cases ``test_torch_parity.py`` checks.

Run with torch installed: ``python -m mass_screen_rs.tests.make_torch_reference``.
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "torch_reference.npz"
# name: (n, m, b, density, threshold, fill_value, seed). The first case spans several column blocks
# and pixel tiles of the kernel; the second is small and dense.
CASES = {
    "blocks_and_tiles": (24, 2 * 8192 + 5, 3 * 64 + 9, 0.03, 2.24, 0.25, 1),
    "small_dense": (40, 300, 50, 0.3, 0.5, 1.0, 2),
}


def torch_mask(response, spectra, threshold, fill_value):
    """
    Return the mask of ``muse.inversions.inversions._mass_screen_eager``, computed the same way.
    """
    import torch  # noqa: PLC0415 - only the recording and the live test need torch

    rg = torch.from_numpy(response)
    ys = torch.where(torch.from_numpy(spectra) <= 0, fill_value, torch.from_numpy(spectra))
    max_bnd = torch.amin(ys[:, :, None] / rg[:, None, :], dim=0, keepdims=True)
    keep = torch.any(rg[:, None, :] * max_bnd >= threshold, dim=0)
    return torch.where(keep, 1, float("nan")).numpy()


def make_case(n, m, b, density, seed):
    """
    Build a random case.

    The response is sparse with blocks of zero and of tiny components; about a quarter of the
    spectra are ``<= 0``.
    """
    rng = np.random.default_rng(seed)
    response = np.where(rng.random((n, m)) < density, rng.random((n, m)), 0).astype(np.float32)
    response[:, : m // 10] = 0
    response[:, m // 10 : m // 5] *= np.float32(1e-10)
    spectra = (rng.random((n, b)) * 4 - 1).astype(np.float32)
    return response, spectra


def main():
    arrays = {}
    for name, (n, m, b, density, threshold, fill_value, seed) in CASES.items():
        response, spectra = make_case(n, m, b, density, seed)
        kept = torch_mask(response, spectra, threshold, fill_value) == 1
        arrays[f"{name}/response"] = response
        arrays[f"{name}/spectra"] = spectra
        arrays[f"{name}/kept"] = kept
        arrays[f"{name}/params"] = np.array([threshold, fill_value], dtype=np.float32)
        print(f"{name}: {response.shape}, {spectra.shape}, kept {kept.mean() * 100:.1f} %")  # noqa: T201
    np.savez_compressed(DATA, **arrays)
    print(f"wrote {DATA} ({DATA.stat().st_size / 1024:.0f} KiB)")  # noqa: T201


if __name__ == "__main__":
    main()
