import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to load the signature image
def load_signature_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    return image

# Function to verify the signature using structural similarity
def verify_signature(original_signature, test_signature, threshold=0.75):
    # Resize the images to the same dimensions for comparison
    height, width = original_signature.shape
    test_signature = cv2.resize(test_signature, (width, height))

    # Calculate the structural similarity index
    similarity_index = ssim(original_signature, test_signature)

    if similarity_index > threshold:
        return True, similarity_index
    else:
        return False, similarity_index

# Driver code (only executed when this script is run directly)
if __name__ == "__main__":
    try:
        # Load the original signature image
        original_signature = load_signature_image("original_signature.png")

        # Load the test signature image (to be verified)
        test_signature = load_signature_image("test_signature.png")

        # Verify the signature
        result, similarity_index = verify_signature(original_signature, test_signature)

        # Output the result
        if result:
            print("Signature verified! Similarity index:", similarity_index)
        else:
            print("Signature not verified. Similarity index:", similarity_index)
    except FileNotFoundError as e:
        print(e)
