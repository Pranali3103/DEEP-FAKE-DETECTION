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

app = Flask(__name__)

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

class CAMCleanup:
    def __init__(self, cam_instance):
        self.cam_instance = cam_instance

    def cleanup(self):
        try:
            if hasattr(self.cam_instance, 'activations_and_grads'):
                self.cam_instance.activations_and_grads.release()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def predict(input_image):
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(0)]

    try:
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)
    except AttributeError as e:
        print(f"Ignoring AttributeError: {e}")
        face_with_mask = prev_face

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

@app.route('/predict_photo', methods=['POST'])
def api_predict():
    try:
        if 'image' in request.files:
            input_image = Image.open(request.files['image']).convert('RGB')
            result, face_with_mask = predict(input_image)
            return jsonify(result)
        else:
            return jsonify({'error': 'No image provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Flask teardown handler
@app.teardown_appcontext
def teardown_cam_cleanup(error):
    if hasattr(app, 'cam_cleanup'):
        app.cam_cleanup.cleanup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


