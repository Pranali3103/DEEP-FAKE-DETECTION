from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(input_image:Image.Image):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers=[model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, face_with_mask
def extract_frames(video_path, output_folder):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    cap = cv2.VideoCapture(video_path)
    frame_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1


    cap.release()

    print(f"{frame_count} frames extracted and saved in {output_folder}")


def process_frame(frame):
 
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Predict using the provided function
    confidences, _ = predict(pil_image)
    return confidences



def detect_fake_or_real_from_frames(frames, mtcnn):
    total_confidences = {'real': 0, 'fake': 0}


    for frame in frames:
 
        try:
            confidences = process_frame(frame)
        except Exception as e:
            print("Error processing frame:", e)
            continue 


        total_confidences['real'] += confidences['real']
        total_confidences['fake'] += confidences['fake']


    total_frames = len(frames)
    avg_confidences = {label: confidence / total_frames for label, confidence in total_confidences.items()}
    return avg_confidences

def process_video_frames(frames_folder, mtcnn):
    frames = []

    # Read frames from the folder
    for filename in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path)
        frames.append(frame)

    # Detect fake or real from frames
    result = detect_fake_or_real_from_frames(frames, mtcnn)
    return result

app = Flask(__name__)

# Initialize MTCNN model
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
).eval()

# API endpoint to process video and detect fake or real frames
@app.route('/predict_video', methods=['POST'])
def detect_deepfake():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded video file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # Extract frames from the video
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'frames')
        extract_frames(video_path, output_folder)

        # Process frames and detect deepfake
        result = process_video_frames(output_folder, mtcnn)

        return jsonify(result)

if __name__ == '__main__':
    # Define the upload folder
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run the Flask app
    app.run(host='0.0.0.0', port=9000, debug=True)
