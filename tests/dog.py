import numpy as np
import matplotlib.pyplot as plt


def generate_symmetric_unitary_householder(n):
    """
    Generate symmetric unitary matrix using Householder reflections.
    For real matrices, symmetric + unitary = symmetric + orthogonal.
    """
    # Start with identity
    U = np.eye(n)

    # Apply random Householder reflections
    for _ in range(n // 2):  # Apply several reflections
        # Generate random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        # Householder matrix: H = I - 2*v*v^T
        H = np.eye(n) - 2 * np.outer(v, v)

        # Update U (Householder matrices are symmetric and orthogonal)
        U = H @ U @ H

    return U


def generate_symmetric_unitary_exponential(n):
    """
    Generate symmetric unitary matrix using matrix exponential of
    skew-symmetric matrix: U = exp(iA) where A is skew-symmetric.
    For real case: U = exp(A) where A is antisymmetric, but this gives
    orthogonal not necessarily symmetric.

    Instead, use U = cos(A) + i*sin(A) for symmetric A.
    For real symmetric unitary, we need eigenvalue constraints.
    """
    # Generate random symmetric matrix
    A = np.random.randn(n, n)
    A = (A + A.T) / 2

    # Scale to ensure eigenvalues give unitary result
    # For real symmetric matrix to be unitary, eigenvalues must be ±1
    eigenvals, eigenvecs = np.linalg.eigh(A)

    # Force eigenvalues to be ±1 randomly
    new_eigenvals = np.random.choice([-1, 1], size=n)

    # Reconstruct matrix
    U = eigenvecs @ np.diag(new_eigenvals) @ eigenvecs.T

    return U


def generate_symmetric_unitary_givens(n):
    """
    Generate symmetric unitary matrix using composition of Givens rotations
    applied symmetrically.
    """
    U = np.eye(n)

    # Apply symmetric Givens rotations
    for i in range(n):
        for j in range(i + 1, n):
            # Random angle
            theta = np.random.uniform(0, 2 * np.pi)

            # Create Givens rotation matrix
            G = np.eye(n)
            G[i, i] = np.cos(theta)
            G[j, j] = np.cos(theta)
            G[i, j] = -np.sin(theta)
            G[j, i] = np.sin(theta)

            # Apply symmetrically: U = G * U * G^T
            U = G @ U @ G.T

    return U


def generate_random_permutation_matrix(n):
    """
    Generate a random permutation matrix (special case of symmetric unitary).
    """
    perm = np.random.permutation(n)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1
    return P


def verify_properties(U, name="Matrix"):
    """
    Verify that matrix is symmetric and unitary.
    """
    n = U.shape[0]

    # Check if symmetric
    is_symmetric = np.allclose(U, U.T, atol=1e-10)

    # Check if unitary
    UUT = U @ U.T
    is_unitary = np.allclose(UUT, np.eye(n), atol=1e-10)

    # Check determinant (should be ±1 for orthogonal matrices)
    det = np.linalg.det(U)

    print(f"\n{name} ({n}x{n}):")
    print(f"  Symmetric: {is_symmetric}")
    print(f"  Unitary: {is_unitary}")
    print(f"  Determinant: {det:.6f}")
    print(f"  Max eigenvalue magnitude: {np.max(np.abs(np.linalg.eigvals(U))):.6f}")

    if is_symmetric and is_unitary:
        print("  ✓ Valid symmetric unitary matrix!")
    else:
        print("  ✗ Not a valid symmetric unitary matrix")

    return is_symmetric and is_unitary


def visualize_matrix(U, title="Matrix"):
    """
    Visualize the matrix structure.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(U, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(label="Matrix Element Value")
    plt.title(f"{title}\n(Blue: negative, Red: positive)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


# Main demonstration
if __name__ == "__main__":
    print("Symmetric Unitary Matrix Generator")
    print("=" * 50)

    n = 4  # Matrix size

    # Method 1: Householder reflections
    print("\nMethod 1: Householder Reflections")
    U1 = generate_symmetric_unitary_householder(n)
    verify_properties(U1, "Householder Method")
    print(f"Matrix:\n{U1}")

    # Method 2: Eigenvalue forcing
    print("\nMethod 2: Eigenvalue Forcing")
    U2 = generate_symmetric_unitary_exponential(n)
    verify_properties(U2, "Eigenvalue Method")
    print(f"Matrix:\n{U2}")

    # Method 3: Givens rotations
    print("\nMethod 3: Symmetric Givens Rotations")
    U3 = generate_symmetric_unitary_givens(n)
    verify_properties(U3, "Givens Method")
    print(f"Matrix:\n{U3}")

    # Method 4: Simple examples
    print("\nMethod 4: Simple Examples")

    # 2x2 case
    theta = np.random.uniform(0, 2 * np.pi)
    U_2x2 = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
    verify_properties(U_2x2, "2x2 Example")
    print(f"2x2 Matrix:\n{U_2x2}")

    # Identity and negative identity
    I = np.eye(n)
    neg_I = -np.eye(n)
    verify_properties(I, "Identity")
    verify_properties(neg_I, "Negative Identity")

    # Permutation matrix example
    P = generate_random_permutation_matrix(n)
    verify_properties(P, "Permutation Matrix")
    print(f"Permutation Matrix:\n{P}")

    print("\n" + "=" * 50)
    print("Note: For real matrices, symmetric + unitary = symmetric + orthogonal")
    print("All eigenvalues must have magnitude 1 (i.e., ±1 for real matrices)")
    print("These matrices represent reflections and rotations that preserve")
    print("distances and have symmetric structure.")
