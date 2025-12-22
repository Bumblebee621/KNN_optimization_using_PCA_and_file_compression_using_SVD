import pickle
import numpy as np
import os
import time
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_training_data(path):
    all_data = []
    all_labels = []
    # Load all 5 batches
    for i in range(1, 6):
        file_path = os.path.join(path, f'data_batch_{i}')
        batch_dict = unpickle(file_path)
        all_data.append(batch_dict[b'data'])
        all_labels.extend(batch_dict[b'labels'])
    return np.vstack(all_data), np.array(all_labels)

def load_test_data(path):
    testpath = os.path.join(path, 'test_batch')
    batch_dict = unpickle(testpath)
    return np.array(batch_dict[b'data']), np.array(batch_dict[b'labels'])

def to_grayscale(data):
    data = data.reshape(-1, 3, 32, 32)
    gray_data = 0.299 * data[:, 0, :, :] + 0.587 * data[:, 1, :, :] + 0.114 * data[:, 2, :, :]
    return gray_data.reshape(-1, 1024)

def fit_transform_pca(X_train, X_test, s):
    """
    PCA using SVD on the Data Matrix (Reverted as requested).
    """
    #Transpose the data so each column is a sample
    X_train_T = X_train.T
    X_test_T = X_test.T

    # 1. Center the data

    mean = np.mean(X_train_T, axis=1, keepdims=True)
    X_train_centered = X_train_T - mean
    X_test_centered = X_test_T - mean

    # 2. SVD on the Data Matrix
    u, _, _ = np.linalg.svd(X_train_centered, full_matrices=False)

    # 3. Select top s components from U
    Us = u[:, :s]

    # 4. Project data
    train_reduced_T = np.dot(Us.T, X_train_centered)
    test_reduced_T = np.dot(Us.T, X_test_centered)

    # Transpose back to (Samples, Features) for the KNN step
    return train_reduced_T.T, test_reduced_T.T

def compute_distances_vectorized(X_train, X_test):
    """
    Computes Euclidean distance matrix.
    """
    # 1. Compute squared norms
    train_sq = np.sum(np.square(X_train), axis=1)
    test_sq = np.sum(np.square(X_test), axis=1)

    # 2. Compute dot product
    # This creates the massive (10000 x 50000) matrix
    dot_prod = np.dot(X_test, X_train.T)

    # 3. Apply formula (a-b)^2 = -2ab + a^2 + b^2
    dists = -2 * dot_prod + train_sq + test_sq[:, None]

    # Numerical stability
    np.maximum(dists, 0, out=dists)

    return np.sqrt(dists)

def predict_knn(X_train, y_train, X_test, k_values):
    """
    KNN.
    Computes the entire distance matrix at once.
    """
    num_test = X_test.shape[0]

    # 1. Compute ALL distances at once
    dists = compute_distances_vectorized(X_train, X_test)

    predictions = {k: np.zeros(num_test, dtype=int) for k in k_values}
    max_k = max(k_values)

    # 2. Find nearest neighbors for the whole matrix
    # We only need the top max(k) neighbors so we use argpartition instead of full sort.
    knn_indices = np.argpartition(dists, max_k, axis=1)[:, :max_k]

    # Retrieve distances for the top k to sort them correctly
    row_indices = np.arange(num_test)[:, None]
    knn_dists = dists[row_indices, knn_indices]

    # Sort within the top k
    sorted_order = np.argsort(knn_dists, axis=1)
    sorted_knn_indices = knn_indices[row_indices, sorted_order]

    # 3. Retrieve labels
    nearest_labels = y_train[sorted_knn_indices]

    # 4. Vote for each k
    for k in k_values:
        k_nearest_labels = nearest_labels[:, :k]

        # Loop over rows to do the bincount (voting)
        for i in range(num_test):
            predictions[k][i] = np.bincount(k_nearest_labels[i]).argmax()

    return predictions

def main():
    path = 'cifar-10-batches-py'

    #Loading data
    x_train_raw, y_train = load_training_data(path)
    x_test_raw, y_test = load_test_data(path)

    #Converting to grayscale
    x_train_gray = to_grayscale(x_train_raw)
    x_test_gray = to_grayscale(x_test_raw)

    Ss = [3, 5, 8, 10, 30, 50, 1024]
    ks = [3, 5, 7, 9]

    results_table = {s: {} for s in Ss}

    total_start_time = time.time()

    for s in Ss:
        print(f"\n--- Running for s={s} ---")
        step_start = time.time()

        # 1. Apply PCA
        if s < 1024:
            x_train_pca, x_test_pca = fit_transform_pca(x_train_gray, x_test_gray, s)
        else:
            x_train_pca, x_test_pca = x_train_gray, x_test_gray

        # 2. Run KNN
        preds_dict = predict_knn(x_train_pca, y_train, x_test_pca, ks)

        # 3. Compute Errors
        for k in ks:
            error = np.mean(y_test != preds_dict[k])
            results_table[s][k] = error
            print(f"k={k}, s={s}, error={error:.4f}")

        print(f"Time for s={s}: {time.time() - step_start:.2f} seconds")

    print(f"\nTotal Execution Time: {time.time() - total_start_time:.2f} seconds")

    # --- Summary Table Printing ---
    print("\n" + "=" * 65)
    print("Summary Table: KNN Classification Error")
    print("=" * 65)

    # Define widths
    first_col_w = 10
    col_w = 12

    # Header
    header = f" {'s \\ k':<{first_col_w}} |"
    for k in ks:
        header += f" {f'k={k}':^{col_w}} |"
    print(header)
    print("-" * len(header))

    # Rows
    for s in Ss:
        row_str = f" {f's={s}':<{first_col_w}} |"
        for k in ks:
            val = results_table[s][k]
            row_str += f" {val:^{col_w}.4f} |"
        print(row_str)
    print("=" * 65)


if __name__ == '__main__':
    main()