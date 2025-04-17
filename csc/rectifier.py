import cv2 as cv
import numpy as np
import os
from csc.standard_stereo_calibrator import log_message


# Skew-symmetric matrix for a vector
def skew_symmetric(t):
    # Flatten or reshape `t` to ensure it's a 1D array
    t = t.flatten()  # Convert 2D array to 1D
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


# Extract intrinsic camera matrices from projection matrices
def extract_intrinsics(proj_matrix):
    # Decompose projection matrix using RQ decomposition
    K, R = cv.decomposeProjectionMatrix(proj_matrix)[:2]
    K = K / K[-1, -1]  # Normalize intrinsic matrix
    return K


class Rectifier:
    def __init__(self, calibration_file="stereoMap.xml", visualization_dims=(960, 540)):
        """
        Initialize the Rectifier class and load precomputed calibration data.

        Args:
            calibration_file (str): Path to the XML file containing stereo calibration data.
            visualization_dims (tuple): Target resolution (width, height) for visualization purposes.
        """
        self.calibration_file = calibration_file
        self.visualization_dims = visualization_dims  # For visualization only
        self.load_calibration_results()

    def load_calibration_results(self):
        """
        Load stereo calibration results (e.g., stereo maps) from an XML file.
        """
        cv_file = cv.FileStorage(self.calibration_file, cv.FILE_STORAGE_READ)

        if not cv_file.isOpened():
            raise FileNotFoundError(
                f"Cannot open calibration file: {self.calibration_file}"
            )

        self.projMatrixL = cv_file.getNode("projMatrixL").mat()
        self.projMatrixR = cv_file.getNode("projMatrixR").mat()
        self.stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
        self.stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
        self.stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
        self.stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()
        self.cameraMatrixL = cv_file.getNode("cameraMatrixL").mat()
        self.distL = cv_file.getNode("distL").mat()
        self.cameraMatrixR = cv_file.getNode("cameraMatrixR").mat()
        self.distR = cv_file.getNode("distR").mat()
        self.R = cv_file.getNode("R").mat()
        self.T = cv_file.getNode("T").mat()
        self.Q = cv_file.getNode("Q").mat()

        cv_file.release()
        log_message("Calibration data loaded successfully!", level="SUCCESS")

    # Calculate the Fundamental matrix F
    def compute_fundamental_matrix(self):
        # Step 1: Extract intrinsic matrices
        K_L = extract_intrinsics(self.projMatrixL)
        K_R = extract_intrinsics(self.projMatrixR)

        # Step 2: Compute the Essential matrix E
        t_skew = skew_symmetric(self.T)
        E = t_skew @ self.R  # E = [t]_x R

        # Step 3: Compute the Fundamental matrix F from E
        F = np.linalg.inv(K_R).T @ E @ np.linalg.inv(K_L)

        return F

    def rectify_image(self, left_image_path, right_image_path):
        """
        Rectify stereo pair images using precomputed rectification maps.

        Args:
            left_image_path (str): Path to the left image.
            right_image_path (str): Path to the right image.

        Returns:
            rectified_left (numpy.ndarray): Rectified left image.
            rectified_right (numpy.ndarray): Rectified right image.
        """
        # Load the images
        img_left = cv.imread(left_image_path)
        img_right = cv.imread(right_image_path)

        if img_left is None or img_right is None:
            raise ValueError(
                "Could not load input images. Please check the file paths."
            )

        # Apply the rectification maps without resizing
        rectified_left = cv.remap(
            img_left, self.stereoMapL_x, self.stereoMapL_y, cv.INTER_LANCZOS4
        )
        rectified_right = cv.remap(
            img_right, self.stereoMapR_x, self.stereoMapR_y, cv.INTER_LANCZOS4
        )

        return rectified_left, rectified_right

    def draw_epilines_on_image(
        self, img, epilines, points, color=(0, 255, 0), thickness=2
    ):
        """
        Helper function to draw epilines on an image.

        Args:
            img (numpy.ndarray): The input image on which to draw epilines.
            epilines (numpy.ndarray): The epilines computed for the points.
            points (list of numpy.ndarray): Points corresponding to the epilines.
            color (tuple): The color of the epilines.
            thickness (int): The thickness of the epilines.

        Returns:
            numpy.ndarray: The image with epilines drawn.
        """
        img_with_lines = img.copy()

        for line, point in zip(epilines, points):
            a, b, c = line  # Epiline coefficients: ax + by + c = 0
            x0, y0 = 0, int(-c / b) if b != 0 else 0  # Line at the left image boundary
            x1, y1 = img.shape[1], (
                int(-(c + a * img.shape[1]) / b) if b != 0 else 0
            )  # Line at the right boundary

            # Draw the epiline on the image
            cv.line(img_with_lines, (x0, y0), (x1, y1), color, thickness)

            # Draw the corresponding point in the image
            pt = tuple(int(x) for x in point.ravel())
            cv.circle(
                img_with_lines, pt, radius=5, color=(255, 0, 0), thickness=-1
            )  # Mark the point

        return img_with_lines

    def visualize_epipolar(self, left_image_path, right_image_path, save=False):
        """
        Visualize epipolar geometry by drawing epilines on both images.

        Args:
            left_image_path (str): Path to the left image.
            right_image_path (str): Path to the right image.
            save (bool, optional): Whether to save the visualization images to disk. Defaults to False.
        """
        # Rectify the images
        rectified_left, rectified_right = self.rectify_image(
            left_image_path, right_image_path
        )

        # Perform ORB (or any feature detector) to find keypoints in the rectified images
        orb = cv.ORB_create()
        keypointsL, descriptorsL = orb.detectAndCompute(rectified_left, None)
        keypointsR, descriptorsR = orb.detectAndCompute(rectified_right, None)

        # Use BFMatcher to match descriptors
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptorsL, descriptorsR)

        # Select a few random matches for visualization
        matches = sorted(matches, key=lambda x: x.distance)[:20]

        # Extract points from matches
        pointsL = np.float32([keypointsL[m.queryIdx].pt for m in matches])
        pointsR = np.float32([keypointsR[m.trainIdx].pt for m in matches])

        # Compute epilines for points in both images
        pointsL = pointsL.reshape(-1, 1, 2)  # Shape required for epiline computation
        pointsR = pointsR.reshape(-1, 1, 2)

        F = self.compute_fundamental_matrix()

        epilinesR = cv.computeCorrespondEpilines(pointsL, 1, F).reshape(-1, 3)
        epilinesL = cv.computeCorrespondEpilines(pointsR, 2, F).reshape(-1, 3)

        # Draw epilines on the rectified images
        imgL_with_epilines = self.draw_epilines_on_image(
            rectified_left, epilinesR, pointsL
        )
        imgR_with_epilines = self.draw_epilines_on_image(
            rectified_right, epilinesL, pointsR
        )

        # Resize for visualization
        imgL_viz = cv.resize(
            imgL_with_epilines, self.visualization_dims, interpolation=cv.INTER_AREA
        )
        imgR_viz = cv.resize(
            imgR_with_epilines, self.visualization_dims, interpolation=cv.INTER_AREA
        )

        # Display the rectified images with epilines side-by-side
        combined_output = np.hstack((imgL_viz, imgR_viz))
        cv.imshow("Epipolar Geometry - Rectified Images with Epilines", combined_output)

        if save:
            save_path = "output/rectifier/epipolar_visualizations"
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(save_path, "epilines_visualization.jpg")
            cv.imwrite(output_path, combined_output)
            log_message(
                f"Epipolar visualization saved to {output_path}", level="SUCCESS"
            )

        cv.waitKey(8000)  # Display for 8 seconds or until a key is pressed
        cv.destroyAllWindows()


# Example usage of Rectifier class
if __name__ == "__main__":
    # Configuration variables
    calibration_id = "C250416SN2_v0.1"
    base_img_path = "demo/samples"
    base_out_path = f"output/rectifier/{calibration_id}"

    # Initialize the rectifier with visualization dimensions
    rectifier = Rectifier(
        calibration_file=f"{calibration_id}_stereoMap.xml",
        visualization_dims=(960, 540),  # Resize for visualization only
    )

    # Example stereo image paths
    image_timestamp = "1744817521"  # Assuming this is dynamic in real usage
    left_image_path = f"{base_img_path}/{calibration_id}_{image_timestamp}_left.jpg"
    right_image_path = f"{base_img_path}/{calibration_id}_{image_timestamp}_right.jpg"

    # Rectify stereo pairs
    rectified_left, rectified_right = rectifier.rectify_image(
        left_image_path, right_image_path
    )

    # Create output directory if it doesn't exist
    import os

    os.makedirs(base_out_path, exist_ok=True)

    # Save rectified images
    output_prefix = f"{base_out_path}/{image_timestamp}"
    cv.imwrite(f"{output_prefix}_left_rectified.jpg", rectified_left)
    cv.imwrite(f"{output_prefix}_right_rectified.jpg", rectified_right)
    log_message(
        f"Rectified images saved to {base_out_path} successfully!", level="SUCCESS"
    )

    # Visualize epipolar geometry with epilines
    rectifier.visualize_epipolar(left_image_path, right_image_path, save=True)
