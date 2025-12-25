from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        image = Image.open("DanGarber.jpeg")
    except FileNotFoundError:
        print("Image not found.")
        return

    pix = np.array(image)

    rim = pix[:, :, 0]
    gim = pix[:, :, 1]
    bim = pix[:, :, 2]

    rU, rS, rVT = np.linalg.svd(rim, full_matrices=False, compute_uv=True)
    gU, gS, gVT = np.linalg.svd(gim, full_matrices=False, compute_uv=True)
    bU, bS, bVT = np.linalg.svd(bim, full_matrices=False, compute_uv=True)

    k_values = [5, 10, 20, 40, 60, 80]

    plt.figure(figsize=(15, 10))

    for i, k in enumerate(k_values):
        rk = compressIm(rU, rS, rVT, k)
        gk = compressIm(gU, gS, gVT, k)
        bk = compressIm(bU, bS, bVT, k)

        r_err = (computeError(rk, rim) / computeFnorm(rim))**2
        g_err = (computeError(gk, gim) / computeFnorm(gim))**2
        b_err = (computeError(bk, bim) / computeFnorm(bim))**2

        print(f"k={k} | R_Error: {r_err:.4f}, G_Error: {g_err:.4f}, B_Error: {b_err:.4f}")

        compressed_im = np.stack((rk, gk, bk), axis=2)
        compressed_im = np.clip(compressed_im, 0, 255).astype('uint8')

        plt.subplot(2, 3, i + 1)
        plt.imshow(compressed_im)
        plt.title(f"k = {k}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def compressIm(u, s, vt, k):
    ak = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
    return ak


def computeError(compressed, original):
    return np.linalg.norm(original - compressed)


def computeFnorm(matrix):
    return np.linalg.norm(matrix)


if __name__ == "__main__":
    main()