import numpy as np
import numpy.linalg as LA

def LRA_grayscale(img, rank):
    
    # Define image size
    length, width = img.shape
    
    # Compute the SVD
    U, Sig, VT = LA.svd(img)
        
    # Compute the LRA for U, Sigma, VT at the rank provied
    U_LRA = U[:, :rank]
    Sigma_LRA = np.diag(Sig[:rank])
    VT_LRA = VT[:rank, :]
    
    A_LRA = U_LRA @ Sigma_LRA @ VT_LRA
    A_LRA = A_LRA.astype(int)
        
    return A_LRA

def LRA(img, rank):

    # Check if the input image is RGB (3 channels) or grayscale (2D)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # determine the shape of the image
        length, width, channels = img.shape
        
        # Split the image into its R, G, B channels
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Perform SVD on each channel separately
        U_R, Sig_R, VT_R = LA.svd(R)
        U_G, Sig_G, VT_G = LA.svd(G)
        U_B, Sig_B, VT_B = LA.svd(B)

        # Compute the LRA for each channel

        U_R_LRA, Sig_R_LRA, VT_R_LRA = U_R[:, :rank], np.diag(Sig_R[:rank]), VT_R[:rank, :]
        U_G_LRA, Sig_G_LRA, VT_G_LRA = U_G[:, :rank], np.diag(Sig_G[:rank]), VT_G[:rank, :]
        U_B_LRA, Sig_B_LRA, VT_B_LRA = U_B[:, :rank], np.diag(Sig_B[:rank]), VT_B[:rank, :]

        # Reconstruct the image using the low-rank approximations of each channel
        R_LRA = (U_R_LRA @ Sig_R_LRA @ VT_R_LRA)
        G_LRA = (U_G_LRA @ Sig_G_LRA @ VT_G_LRA)
        B_LRA = (U_B_LRA @ Sig_B_LRA @ VT_B_LRA)

        # Combine the R, G, B channels into an RGB image
        A_LRA = np.stack((R_LRA, G_LRA, B_LRA), axis=-1)
        A_LRA = A_LRA.astype(int)

        #New Image 
        return A_LRA
    
    else:
        # For grayscale images, use the original function
        return LRA_grayscale(img, rank)

