
# IndexCryptogram

A Python application that generates a 128-bit key using a fingerprint 


## Fingerprint
A fingerprint contains various identifiers that enable the unique identification and distinction of each print. When an algorithm seeks these identifiers, it commonly examines minutiae points, the precise locations and orientations of ridge endings and bifurcations (divisions) along a ridge's trajectory. For example when a fingerprint is processed using using the [SIFT (Scale-Invariant Feature Transform)](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) method from the OpenCV library,
The `sift.detectandcompute()` function returns a tuple and a numpy data array(`numpy.ndarray`). The tuple consists of 48 `cv2 keypoint` objects and the numpy data array  contains descriptors for each keypoint that was detected. Each `Keypoint` object contains data about a certain keypoint which includes:


+ `pt`: This is a tuple (x, y) representing the coordinates of the keypoint in the image.
+ `size`: The diameter of the meaningful keypoint neighborhood.
+ `angle`: The orientation of the keypoint in degrees (0-360).
+ `response`: The strength of the keypoint.
+ `octave`: The octave (pyramid layer) from which the keypoint has been extracted.
+ `class_id`: An integer that can be used to cluster keypoints by some criteria. 


## Approach
In each session, multiple fingerprint images of the same finger undergo processing using `sift.detectandcompute()`. Each image is then compared with the others using `cv2.BFMatcher()` to identify both the best and worst image.  
__Using Noise as Filter :__
The image with the fewest matches is presumed to have the most noise, resulting in only the prominent keypoints being detected. 
   
First, the keypoints of the image with the fewest good matches (the worst case) are identified. These keypoints are then sorted in descending order based on their response values. Finally, the coordinates of the top 30 keypoints with the highest response values are extracted.
  
This coordinates are then normalised using Centoid Normalisation Technique. This Normalised coordinates are then used as input to SHA256 hashing algorithm that generates a 128 bit intaeger
## Future use

By enhancing keypoint recognition techniques and calibration, coupled with better quality fingerprint sensors, every fingerprint can consistently generate a unique integer. This eliminates the necessity to store cryptographic keys, consequently enhancing security measures.
## Fingerprint Processing
```python
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
            
            good_matches = [m for m, n in matches if m.distance <           0.322 * n.distance]
            
            if len(good_matches) >= 30:
                stats.append((j + 1, len(good_matches)))
        
        match_detail.append(stats)

    # Find the worst case
    worst_case = min(match_detail, key=len)

    # Extract keypoints for the worst case
    keypoints = {i + 1: keypoints for i, (keypoints, _) in e    numerate(KeypointsAndDescriptors)}

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


```