"""Thermocouple calibration: temperature against voltage, by least squares.

Writes ``voltage-linear-model.png`` for the least-squares chapter of the
PID book. This script is shown to the reader in
``least-squares-modelling/least-squares-exercises.rst``, so it is kept
short and explicit.

Usage
-----
    uv run --with numpy --with matplotlib python voltage_linear_model.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Measured thermocouple voltage [mV] and the reference temperature [K].
x = np.array([0.01, 0.12, 0.24, 0.38, 0.51, 0.67, 0.84, 1.01, 1.15, 1.31])
y = np.array([273, 293, 313, 333, 353, 373, 393, 413, 433, 453])

n = len(x)
X = np.column_stack([np.ones(n), x])       # intercept and slope columns

# Solve the normal equations for the two coefficients.
coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
predictions = X @ coefficients

residuals = y - predictions                # e = y - Xb
RSS = np.sum(residuals**2)                 # residual sum of squares
TSS = np.sum((y - np.mean(y)) ** 2)        # total sum of squares
R2 = 1 - RSS / TSS
standard_error = np.sqrt(RSS / (n - len(coefficients)))

print(f"Temperature = {coefficients[0]:.1f} + {coefficients[1]:.1f} x voltage")
print(f"R2 = {R2:.4f}, standard error = {standard_error:.1f} K")

fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(color="#DDDDDD", linewidth=0.8)
ax.plot(x, y, "o", color="#0072B2", markersize=9, label="Original data")
ax.plot(x, predictions, color="#D55E00", linewidth=2.5, label="Fitted line")
ax.plot(x, predictions + 2 * standard_error, "--", color="#D55E00", linewidth=1.5)
ax.plot(x, predictions - 2 * standard_error, "--", color="#D55E00", linewidth=1.5)
ax.text(0.75, 320, f"Standard error = {standard_error:.1f} K")
ax.set_xlabel("Voltage [mV]")
ax.set_ylabel("Temperature [K]")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("voltage-linear-model.png", dpi=300)
