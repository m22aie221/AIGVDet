import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pdb
from tqdm import tqdm

sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from natsort import natsorted
from utils1.utils import get_network, str2bool, to_cuda
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score,roc_auc_score




DEVICE = 'cuda'
# DEVICE = 'cpu'  # Changed to 'cpu'


def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()

    images = glob.glob(os.path.join(output_folder, '*.png')) + \
             glob.glob(os.path.join(output_folder, '*.jpg'))
    images = sorted(images)
    
    return images


# Function to generate optical flow images for a single video
def OF_video_generate(video_path, output_rgb_folder, output_flow_folder, device='cuda'):
    # Initialize the RAFT model for optical flow

    model = torch.nn.DataParallel(RAFT())  # Adjust RAFT initialization as needed
    model.load_state_dict(torch.load('raft_model/raft-things.pth', map_location=torch.device(device)))
    
    # Remove DataParallel wrapper and move the model to the specified device
    model = model.module
    model.to(device)
    model.eval()

    # Check and create output directories if they don't exist
    if not os.path.exists(output_rgb_folder):
        os.makedirs(output_rgb_folder)
        print(f'Created folder for RGB frames: {output_rgb_folder}')

    if not os.path.exists(output_flow_folder):
        os.makedirs(output_flow_folder)
        print(f'Created folder for optical flow: {output_flow_folder}')

    # Extract and save RGB frames from the input video
    images = video_to_frames(video_path, output_rgb_folder)
    images = natsorted(images)  # Sort frames in natural order

    # Generate optical flow for consecutive frames
    with torch.no_grad():
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # Load images
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # Pad images for compatibility with RAFT
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Compute optical flow
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            # Visualize and save optical flow
            viz(image1, flow_up, output_flow_folder, imfile1)

    print(f"Optical flow generation complete for {video_path}.")

# Function to process all videos in a folder
def process_videos_in_folder(original_subfolder_path, optical_subfolder_path, device='cuda'):
    # Iterate over all mp4 files in the original subfolder
    for video_file in os.listdir(original_subfolder_path):
        if video_file.lower().endswith('.mp4'):
            video_path = os.path.join(original_subfolder_path, video_file)

            # Extract the base name of the video file without the extension
            video_name = os.path.splitext(video_file)[0]

            # Create separate folders for RGB frames and optical flow images for each video
            output_rgb_folder = os.path.join(optical_subfolder_path, video_name, 'RGB_Frames')
            output_flow_folder = os.path.join(optical_subfolder_path, video_name, 'Optical_Flow')

            # Call the OF_gen function to process the video
            OF_video_generate(video_path, output_rgb_folder, output_flow_folder, device)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# Function to visualize optical flow and save the result as an image
def viz(img, flo, folder_optical_flow_path, imfile1):
    # Convert the tensor 'img' from shape [1, C, H, W] to [H, W, C] and move it to CPU as a NumPy array
    img = img[0].permute(1, 2, 0).cpu().numpy()
    
    # Convert the tensor 'flo' (optical flow) from shape [1, C, H, W] to [H, W, C] and move it to CPU as a NumPy array
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    
    # Map the optical flow 'flo' to a color-coded RGB image using a flow visualization function
    flo = flow_viz.flow_to_image(flo)
    
    # Concatenate the original image and the color-coded optical flow image vertically (along axis 0)
    img_flo = np.concatenate([img, flo], axis=0)

    # Print the base folder path where the optical flow image will be saved
    print(folder_optical_flow_path)
    
    # Extract only the filename from 'imfile1' (handling both Windows and Unix-style paths)
    filename = os.path.basename(imfile1)
    
    # Construct the full path for saving the optical flow image using the base folder path and filename
    output_path = os.path.join(folder_optical_flow_path, filename)
    
    # Print the final output path
    print(output_path)
    
    # Save the color-coded optical flow image to the specified output path using OpenCV
    cv2.imwrite(output_path, flo)





# Function to generate optical flow images
def OF_gen(args):
    # Initialize the RAFT model for optical flow using DataParallel for multi-GPU support
    model = torch.nn.DataParallel(RAFT(args))
    
    # Load the pre-trained model weights from the specified path
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    # Remove the DataParallel wrapper and use the original model
    model = model.module
    
    # Move the model to the specified device (e.g., GPU or CPU)
    model.to(DEVICE)
    
    # Set the model to evaluation mode (disables training-specific behaviors like dropout)
    model.eval()

    # Check if the directory for saving optical flow images exists; if not, create it
    if not os.path.exists(args.folder_optical_flow_path):
        os.makedirs(args.folder_optical_flow_path)
        print(f'{args.folder_optical_flow_path}')  # Print the path of the created directory

    # Disable gradient calculation for inference to save memory and computation
    with torch.no_grad():

        # Convert the input video to individual frames and save them to a specified folder
        images = video_to_frames(args.path, args.folder_original_path)
        
        # Sort the frame file names in natural order (e.g., image1.jpg, image2.jpg, ..., image10.jpg)
        images = natsorted(images)

        # Iterate through consecutive pairs of images to calculate the optical flow
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # Load the first image in the pair
            image1 = load_image(imfile1)
            # Load the second image in the pair
            image2 = load_image(imfile2)

            # Initialize an InputPadder to pad the images so their dimensions are compatible with the model
            padder = InputPadder(image1.shape)
            # Pad both images
            image1, image2 = padder.pad(image1, image2)

            # Compute the optical flow between the two images using the model
            # `iters=20` specifies the number of iterations for the RAFT model
            # `test_mode=True` indicates that the model is in test mode
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            # Visualize the optical flow and save it to the specified folder
            viz(image1, flow_up, args.folder_optical_flow_path, imfile1)



if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    
    # Argument to specify the model checkpoint path
    parser.add_argument('--model', help="Path to the model checkpoint", default="raft_model/raft-things.pth")
    
    # Argument to specify the path of the input video
    parser.add_argument('--path', help="Path to the dataset for evaluation", default="video/000000.mp4")
    
    # Argument to specify the folder where video frames will be saved
    parser.add_argument('--folder_original_path', help="Folder for original video frames", default="frame/000000")
    
    # Boolean flag to indicate whether to use a small version of the model
    parser.add_argument('--small', action='store_true', help='Use a smaller model version')
    
    # Boolean flag for using mixed precision during evaluation
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision computation')
    
    # Boolean flag for using an efficient correlation implementation
    parser.add_argument('--alternate_corr', action='store_true', help='Use an efficient correlation implementation')
    
    # Argument to specify the folder where optical flow results will be saved
    parser.add_argument('--folder_optical_flow_path', help="Folder to save optical flow results", default="optical_result/000000")
    
    # Argument for the path to the model checkpoint used for optical flow detection
    parser.add_argument("-mop", "--model_optical_flow_path", type=str, default="checkpoints/optical.pth")
    
    # Argument for the path to the model checkpoint used for original frame detection
    parser.add_argument("-mor", "--model_original_path", type=str, default="checkpoints/original.pth")
    
    # Argument to set the detection threshold
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    
    # Boolean flag to use CPU instead of GPU
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU for computation instead of GPU")
    
    # Argument to specify the architecture of the network
    parser.add_argument("--arch", type=str, default="resnet50")
    
    # Argument to enable or disable data augmentation normalization
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Generate optical flow images using the specified arguments
    OF_gen(args)

    # Load the first model for optical flow detection
    model_op = get_network(args.arch)
    state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")
    
    # If the state dictionary contains a 'model' key, extract the model weights
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Load the model weights and set the model to evaluation mode
    model_op.load_state_dict(state_dict)
    model_op.eval()
    
    # Move the model to GPU if not using CPU
    if not args.use_cpu:
        model_op.cuda()

    # Load the second model for original frame detection
    model_or = get_network(args.arch)
    state_dict = torch.load(args.model_original_path, map_location="cpu")
    
    # If the state dictionary contains a 'model' key, extract the model weights
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Load the model weights and set the model to evaluation mode
    model_or.load_state_dict(state_dict)
    model_or.eval()
    
    # Move the model to GPU if not using CPU
    if not args.use_cpu:
        model_or.cuda()

    # Define the image transformation: center crop to 448x448 and convert to tensor
    trans = transforms.Compose(
        (
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
        )
    )

    # Print a separator for better output formatting
    print("*" * 30)

    # Path for the original and optical flow frames
    original_subsubfolder_path = args.folder_original_path
    optical_subsubfolder_path = args.folder_optical_flow_path

    # List of all original image files in the specified folder
    original_file_list = sorted(
        glob.glob(os.path.join(original_subsubfolder_path, "*.jpg")) +
        glob.glob(os.path.join(original_subsubfolder_path, "*.png")) +
        glob.glob(os.path.join(original_subsubfolder_path, "*.JPEG"))
    )
    
    # Initialize a variable to accumulate the probabilities for original frames
    original_prob_sum = 0

    # Loop over each image in the original file list
    for img_path in tqdm(original_file_list, dynamic_ncols=True, disable=len(original_file_list) <= 1):
        # Load the image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Apply the transformations to the image
        img = trans(img)
        
        # Apply normalization if specified
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Add a batch dimension to the image tensor
        in_tens = img.unsqueeze(0)
        
        # Move the tensor to GPU if not using CPU
        if not args.use_cpu:
            in_tens = in_tens.cuda()
        
        # Perform inference with the model without computing gradients
        with torch.no_grad():
            prob = model_or(in_tens).sigmoid().item()
            original_prob_sum += prob

    # Compute the average probability for the original frames
    original_predict = original_prob_sum / len(original_file_list)
    print("Original probability:", original_predict)

    # List of all optical flow image files in the specified folder
    optical_file_list = sorted(
        glob.glob(os.path.join(optical_subsubfolder_path, "*.jpg")) +
        glob.glob(os.path.join(optical_subsubfolder_path, "*.png")) +
        glob.glob(os.path.join(optical_subsubfolder_path, "*.JPEG"))
    )
    
    # Initialize a variable to accumulate the probabilities for optical flow frames
    optical_prob_sum = 0

    # Loop over each image in the optical flow file list
    for img_path in tqdm(optical_file_list, dynamic_ncols=True, disable=len(original_file_list) <= 1):
        # Load the image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Apply the transformations to the image
        img = trans(img)
        
        # Apply normalization if specified
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Add a batch dimension to the image tensor
        in_tens = img.unsqueeze(0)
        
        # Move the tensor to GPU if not using CPU
        if not args.use_cpu:
            in_tens = in_tens.cuda()
        
        # Perform inference with the model without computing gradients
        with torch.no_grad():
            prob = model_op(in_tens).sigmoid().item()
            optical_prob_sum += prob

    # Compute the average probability for the optical flow frames
    optical_predict = optical_prob_sum / len(optical_file_list)
    print("Optical probability:", optical_predict)

    # Compute the final prediction by averaging the probabilities from both models
    predict = original_predict * 0.5 + optical_predict * 0.5
    print(f"Final prediction: {predict}")

    # Compare the prediction with the threshold to determine if the video is real or fake
    if predict < args.threshold:
        print("Real video")
    else:
        print("Fake video")

