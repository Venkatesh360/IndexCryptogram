import cv2
import os
import numpy as np
import hashlib

def normalize_coordinates(coordinates):
    coords_array = np.array(coordinates)
    centroid = np.mean(coords_array, axis=0)
    centered_coords = coords_array - centroid
    max_abs_value = np.max(np.abs(centered_coords))
    normalized_coords = centered_coords / max_abs_value
    return normalized_coords.tolist()

def process_fingerprints():
    # Create an empty list to store keypoints and descriptors
    KeypointsAndDescriptors = []

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Iterate over each file in the "samples" directory
    for file in os.listdir("samples"):
        # Read the image
        image = cv2.imread(os.path.join("samples/", file))
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        # Append keypoints and descriptors to the list
        KeypointsAndDescriptors.append((keypoints, descriptors))

    # Initialize a BFMatcher
    bf = cv2.BFMatcher()

    # Initialize match_detail list
    match_detail = []

    # Iterate over each pair of keypoints and descriptors
    for i, (k1, d1) in enumerate(KeypointsAndDescriptors):
        stats = [i + 1]
        for j, (k2, d2) in enumerate(KeypointsAndDescriptors):
            if j == i:
                continue
            
            matches = bf.knnMatch(d1, d2, k=2)
            
            good_matches = [m for m, n in matches if m.distance < 0.322 * n.distance]
            
            if len(good_matches) >= 30:
                stats.append((j + 1, len(good_matches)))
        
        match_detail.append(stats)

    # Find the worst case
    worst_case = min(match_detail, key=len)

    # Extract keypoints for the worst case
    keypoints = {i + 1: keypoints for i, (keypoints, _) in enumerate(KeypointsAndDescriptors)}

    final_sample = keypoints.get(worst_case[0], [])

    # Sort keypoints by response value
    response_coordinates = sorted([(kp.response, kp.pt) for kp in final_sample], key=lambda x: x[0], reverse=True)[:30]

    coordinates = [(x,y) for _, (x, y) in response_coordinates]

    normal_coordinates = normalize_coordinates(coordinates)

    normal_coordinate_str = ''.join([str(coord) for coord in normal_coordinates])

    # Hash the concatenated string to generate an integer value
    hashed_value = int(hashlib.sha256(normal_coordinate_str.encode()).hexdigest(), 16) % (2**128)  # Adjust modulus as needed

    num_bytes = (hashed_value.bit_length() + 7) // 8
    
    return hashed_value.to_bytes(num_bytes, byteorder='big')

