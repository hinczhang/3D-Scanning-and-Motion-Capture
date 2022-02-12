import numpy as np
# from pypfm import PFMLoader
import re
import os
import cv2
from stereomideval.exceptions import PathNotFound
from stereomideval.dataset import Dataset
# credit to: https://www.programcreek.com/python/?CodeExample=save+depth



class CalibrationData:
    """
    Calibration data

    Used to make returning and accessing calibration data simple.
    """
    def __init__(self, width, height, c_x, c_y, focal_length, doffs, baseline, ndisp, vmin, vmax):
        """
        Initalisaiton of CalibrationData structure

        Parameters:
            c_x (float): Principle point in X
            c_y (float): Principle point in Y
            focal_length (float): focal length
            doffs (float): x-difference of principal points, doffs = cx1 - cx0
            baseline (float): baseline
        """
        self.width = width
        self.height = height
        self.c_x = c_x
        self.c_y = c_y
        self.focal_length = focal_length
        self.doffs = doffs
        self.baseline = baseline
        self.ndisp = ndisp
        self.vmin = vmin
        self.vmax = vmax


def save_depth_to_pgm(depth, pgm_file_path):
    """
    Save the depth map to PGM file
    :param depth: depth map with 2D ND-array
    :param pgm_file_path: output file path
    """
    max_depth = np.max(depth)
    depth_copy = np.copy(depth)
    depth_copy = 65535.0 * (depth_copy / max_depth)
    depth_copy = depth_copy.astype(np.uint16)

    with open(pgm_file_path, 'wb') as f:
        f.write(bytes("P5\n", encoding="ascii"))
        f.write(bytes("# %f\n" % max_depth, encoding="ascii"))
        f.write(bytes("%d %d\n" % (depth.shape[1], depth.shape[0]), encoding="ascii"))
        f.write(bytes("65535\n", encoding="ascii"))
        f.write(depth_copy.tobytes()) 
        
#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
def disp_to_depth(disp, focal_length, doffs, baseline):
        """
        Convert from disparity to depth using calibration file
        Parameters:
            disp (numpy): 2D disparity image (result of stereo matching)
            focal_length (float): Focal length of left camera (in pixels)
            doffs (float): x-difference of principal points, doffs = cx1 - cx0
            baseline (float): Baseline distance between left and right camera (in mm)
        Returns:
            depth (numpy): 2D depth image (units meters)
        """
        # Calculate depth from disparitiy
        # Z = baseline * f / (disp + doeff)
        z_mm = baseline * focal_length / (disp + doffs)
        # Z is in mm, convert to meters
        depth = z_mm / 1000
        return depth


def load_scene_disparity(scene_name, dataset_folder, display_images=False,
                             display_time=500, save_path=None, load_perfect=False):
        """
        Load disparity image from scene folder
        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.
        Returns:
            disp_image (numpy): 2D disparity image
                loaded from scene data (result of stereo matching)
        """
        left_disp_filename = "disp0.pfm"
        left_disp_filepath = os.path.join(dataset_folder,
                                           scene_name,
                                           left_disp_filename)
        # Check disparity file exists
        if not os.path.exists(left_disp_filepath):
            print("Disparity pfm file does not exist")
            print(left_disp_filepath)
            raise PathNotFound(left_disp_filepath, "Disparity pfm file does not exist")
        # Load disparity file to numpy image
        disp_image, _ = Dataset.load_pfm(left_disp_filepath)
        # disp_image = cv2.imread(disp_filename, cv2.IMREAD_UNCHANGED)

        if display_images:
            # Display disparity image in opencv window
            norm_disp_image = Dataset.normalise_pfm_data(disp_image)
            norm_disp_image_resize = cv2.resize(norm_disp_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', cv2.applyColorMap(norm_disp_image_resize, cv2.COLORMAP_JET))
            cv2.waitKey(display_time)
        if save_path is not None:
            norm_disp_image = Dataset.normalise_pfm_data(disp_image)
            norm_disp_image_resize = cv2.resize(norm_disp_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imwrite(save_path, cv2.applyColorMap(norm_disp_image_resize, cv2.COLORMAP_JET))
        return disp_image

# Fcuntion to read pfm file
# def read_pfm_file(path):
#     """
#     Read from .pfm file
#     :param flow_file: name of the flow file
#     :return: optical flow data in matrix
#     """
#     loader = PFMLoader(color=False, compress=False)
#     data = loader.load_pfm(path)
#     return data 

def read_pfm_file(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale 


def load_scene_stereo_pair(scene_name, dataset_folder, display_images=False, display_time=500):
        """
        Load stereo pair images from scene folder

        Parameters:
            scene_name (string): Scene name to load data (use 'get_scene_list()'
                to see possible scenes) Must have already been downloaded to the
                dataset foldering using 'download_scene_data()'
            dataset_folder (string): Path to folder where dataset has been downloaded
            display_image (bool): Optional. Should the scene data be displayed when it is loaded.
            display_time (int): Optional. Length of time to display each image once loaded.
            image_suffix (string): Optional. Addition suffix to load alternate left views
                                                'E' means exposure changed between views
                                                'L' means lighting changed between views

        Returns:
            left_image (numpy): 2D image from left camera, loaded from scene data
            right_image (numpy): 2D image from right camera, loaded from scene data
        """
        left_image_filename = "im0.png"
        right_image_filename = "im1.png"
        
        # Define left and right image files in scene folder
        left_image_filepath = os.path.join(dataset_folder,
                                           scene_name,
                                           left_image_filename)
        right_image_filepath = os.path.join(dataset_folder,
                                            scene_name,
                                            right_image_filename)
        # Check left and right image files exist
        if not os.path.exists(left_image_filepath) or not os.path.exists(right_image_filepath):
            print("Left or right image file does not exist")
            print(left_image_filepath)
            print(right_image_filepath)
            raise PathNotFound(
                left_image_filepath+","+right_image_filepath,
                "Left or right image file does not exist")
        # Read left and right image files to numpy image
        left_image = cv2.imread(left_image_filepath, cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(right_image_filepath, cv2.IMREAD_UNCHANGED)
        if display_images:
            # Display left and right image files to OpenCV window
            left_image_resize = cv2.resize(left_image, dsize=(0, 0), fx=0.2, fy=0.2)
            right_image_resize = cv2.resize(right_image, dsize=(0, 0), fx=0.2, fy=0.2)
            cv2.imshow('image', left_image_resize)
            cv2.waitKey(display_time)
            cv2.imshow('image', right_image_resize)
            cv2.waitKey(display_time)
        return left_image, right_image

def load_cal(scene_name, dataset_folder):
        """
        Load camera parameters from calibration file

        Parameters:
            cal_filepath (string): filepath to calibration file (usually calib.txt)
                Expected format:
                    cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
                    cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
                    doffs=131.111
                    baseline=193.001
                    width=2964
                    height=1988
                    ndisp=280
                    isint=0
                    vmin=31
                    vmax=257
                    dyavg=0.918
                    dymax=1.516

        Returns:
            depth (numpy): 2D depth image (units meters)
        """

        # Check calibration file exists
        calib_filname = "calib.txt"
        cal_filepath = os.path.join(dataset_folder,
                                    scene_name,
                                    calib_filname)
        if not os.path.exists(cal_filepath):
            print("Calibration file not found")
            print(cal_filepath)
            raise PathNotFound(cal_filepath, "Calibration file not found")

        # Open calibration file
        file = open(cal_filepath, 'rb')
        # Read first line
        # expected format: "cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]"
        cam0_line = file.readline().decode('utf-8').rstrip()
        # Read second line but ignore the data as cam0 and cam1 have the same parameters
        _ = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "doffs=131.111")
        doffs_line = file.readline().decode('utf-8').rstrip()
        # Read third line (expected format: "baseline=193.001")
        baseline_line = file.readline().decode('utf-8').rstrip()
        # Read 4th line (expected format: "width=2964")
        width_line = file.readline().decode('utf-8').rstrip()
        # Read 5th line (expected format: "height=19881")
        height_line = file.readline().decode('utf-8').rstrip()
        # Read 6th line (expected format: "ndisp=280")
        ndisp_line = file.readline().decode('utf-8').rstrip()
        vmin_line = file.readline().decode('utf-8').rstrip()
        vmax_line = file.readline().decode('utf-8').rstrip()

        # Read all numbers from cam0 line using regex
        nums = re.findall("\\d+\\.\\d+", cam0_line)
        # Get camera parmeters from file data
        cam0_f = float(nums[0])
        cam0_cx = float(nums[1])
        cam0_cy = float(nums[3])

        # Get doffs and baseline from file data
        doffs = float(re.findall("\\d+", doffs_line)[0])
        baseline = float(re.findall("\\d+", baseline_line)[0])
        width = float(re.findall("\\d+", width_line)[0])
        height = float(re.findall("\\d+", height_line)[0])
        ndisp = float(re.findall("\\d+", ndisp_line)[0])
        vmin = float(re.findall("\\d+", vmin_line)[0])
        vmax = float(re.findall("\\d+", vmax_line)[0])

        cal_data = CalibrationData(width, height, cam0_cx, cam0_cy, cam0_f, doffs, baseline, ndisp, vmin, vmax)
        return cal_data