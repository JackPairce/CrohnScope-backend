from typing import List
import cv2
import numpy as np


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    """Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    """

    HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))
        ),
    )
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))
        ),
    )
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    return Inorm, H, E


def DataAugmentation(patch: np.ndarray) -> List[np.ndarray]:
    """Apply data augmentation techniques to the image patch.

    Args:
        patch: Image patch to augment.

    Returns:
        List of augmented image patches.
    """
    augmented_patches = []
    
    # apply Flipping Augmentation
    # Horizontal Flip Augmentation
    flipped_patch = np.flipud(patch)
    augmented_patches.append(flipped_patch)
    # Vertical Flip Augmentation
    flipped_patch = np.fliplr(patch)
    augmented_patches.append(flipped_patch)
    # Horizontal and Vertical Flip Augmentation
    flipped_patch = np.flipud(np.fliplr(patch))
    augmented_patches.append(flipped_patch)    

    # apply Orientation Augmentation
    for angle in [0, 90, 180, 270]:
        rotated_patch = np.rot90(patch, k=angle // 90)
        augmented_patches.append(rotated_patch)
        
    # apply brightness Augmentation (.8x, 1.2x)
    for brightness in [0.8, 1.2]:
        jittered_patch = np.clip(patch * brightness, 0, 255).astype(np.uint8)
        augmented_patches.append(jittered_patch)
        
    # apply Contrast Augmentation (.8x, 1.2x)
    for contrast in [0.8, 1.2]:
        jittered_patch = np.clip(patch * contrast, 0, 255).astype(np.uint8)
        augmented_patches.append(jittered_patch)
        
    # apply noise Augmentation (Gaussian Noise)
    noise = np.random.normal(0, 25, patch.shape).astype(np.uint8)
    noisy_patch = cv2.add(patch, noise)
    augmented_patches.append(noisy_patch)
    
    # apply gamma correction Augmentation (0.8, 1.2)
    for gamma in [0.8, 1.2]:
        gamma_corrected_patch = np.clip(patch ** gamma, 0, 255).astype(np.uint8)
        augmented_patches.append(gamma_corrected_patch)
        
    # apply Zoom/Scale Augmentation
    zoomed_patch = cv2.resize(patch, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    zoomed_patch = cv2.resize(zoomed_patch, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
    augmented_patches.append(zoomed_patch)
    
    # TODO: apply Elastic Transform Augmentation

    return augmented_patches



def QualityControl():
    # TODO: Implement quality control checks for the image preprocessing
    ...