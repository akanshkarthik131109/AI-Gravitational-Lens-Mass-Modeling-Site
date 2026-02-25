import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
from torchvision import transforms 
from torchvision import transforms as T
import math
import random
from scipy.optimize import least_squares
import cv2
import matplotlib.pyplot as plt
import lenstronomy
from lenstronomy.LensModel.lens_model import LensModel
import io

from matplotlib.figure import Figure
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from astropy.cosmology import Planck18
from astropy import constants as const
import astropy.units as u

app = FastAPI()


# 2. Add this block immediately after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allows your React app
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers
)


def load_custom_model(num_classes):
    # 1. Initialize the architecture
    # Must match the architecture you used during training
    model_path = 'lenses_model_full_data_e20_V3_Update_Loss+Normalization.pt'
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    
    
    # 2. Adjust the head if you trained on a custom number of classes
    # (e.g., if you trained on 2 classes: background + your_object)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # 3. Load the state dict
    # 'map_location' ensures it works even if you trained on GPU but are now on CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode for inference
    
    return model

model = load_custom_model(num_classes=3)

transform = transforms.Compose([
    transforms.PILToTensor(),  # Converts PIL Image to tensor and scales [0,255] → [0,1]
    transforms.ConvertImageDtype(torch.float32),  # Ensure float32 dtype
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
        std=[0.229, 0.224, 0.225]    # ImageNet RGB standard deviations
    )
])

def pdist(point1, point2):
    ax, ay = point1
    bx, by = point2
    return math.sqrt((ax-bx)**2 + (ay-by)**2)

def rotate_point(center, x, y, phi):

    #Polar
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)

    return r*math.cos(theta - phi), r*math.sin(theta - phi)

def find_algebraic_residual(params, center, x, y):
    a, b, phi = params
    h, k = center
    
    # Transform point to ellipse coordinate system
    x_centered = x - h
    y_centered = y - k
    
    # Rotate to ellipse's coordinate system
    x_rot, y_rot = rotate_point(center, x_centered, y_centered, phi)
    
    
    # Algebraic ellipse equation: (x_rot/a)^2 + (y_rot/b)^2 = 1
    # Residual: (x_rot/a)^2 + (y_rot/b)^2 - 1
    residual = (x_rot**2 / a**2) + (y_rot**2 / b**2) - 1
    
    return residual

def find_all_algebraic_residuals(params, center, x_data, y_data):
    return [find_algebraic_residual(params, center, x_data[i], y_data[i]) 
            for i in range(len(x_data))]

def get_axial_estimates(x_data, y_data, center):
    x, y = center
    distances = {}
    for i in range(len(x_data)):
        distances[(x_data[i], y_data[i])] = pdist(center, (x_data[i], y_data[i]))
    #Farthest and Closest Point
    farthest = max(distances, key=distances.get)
    semi_major = distances[farthest]
    closest = min(distances, key=distances.get)
    semi_minor = distances[closest]
    pred_phi = math.atan2((farthest[1] - y), (farthest[0] - x))

    return semi_major, semi_minor, pred_phi

def custom_fit_ellipse_fixed_center(points, center):
  #  print(f'Shape Points: {points[0]}')

    x_data = [point[0][0] for point in points]
    y_data = [point[0][1] for point in points]
    cx, cy = center
    def diffy_res(params, center, x_data, y_data):
        return sum(find_all_algebraic_residuals(params, center, x_data, y_data))
    
    bounds = ([0.001, 0.001, -np.pi], [np.inf, np.inf, np.pi])
    p_maj, p_min, p_phi = get_axial_estimates(x_data, y_data, center)
    guesses = []
    for i in range(10):
        guesses.append([np.clip(p_maj + random.randint(-15, 15), 0.001, np.inf), np.clip(p_min + random.randint(-15, 15), 0.001, np.inf), np.clip(p_phi + random.randint(-20,20) * math.pi/180, -np.pi, np.pi)])
    
    results = {}
    for guess in guesses:
        result = least_squares(find_all_algebraic_residuals, guess, args=(center, x_data, y_data), bounds=bounds)
        results[result.cost] = result.x

    a, b, phi = results[min(results.keys())]

    return center, (a, b), phi

def get_theta_e(a1, a2, pixelScale):
    # I changed the Scale here from 0.262 to the current Value 0.13099999998
    return math.sqrt( (a1) * (a2) ) * pixelScale
def phi_q2_ellipticity(phi, q):
    """Transforms orientation angle and axis ratio into complex ellipticity moduli e1,
    e2.

    :param phi: angle of orientation (in radian)
    :param q: axis ratio minor axis / major axis
    :return: eccentricities e1 and e2 in complex ellipticity moduli
    """
    e1 = (1.0 - q) / (1.0 + q) * np.cos(2 * phi)
    e2 = (1.0 - q) / (1.0 + q) * np.sin(2 * phi)
    return e1, e2

def SIE_Info(mask, image, pixelScale):
    points = cv2.findNonZero(mask)
    print(f'Points: {points}')
    center = image.size
    x, y = center
    x, y = x/2, y/2
    ellipse = custom_fit_ellipse_fixed_center(points, (x, y))
    _, axes, phi = ellipse
    major = max(axes)
    minor = min(axes)
    theta_e = get_theta_e(major, minor, pixelScale)
    print(theta_e)
    q = minor / major
    deg = math.degrees(phi)
    e1, e2 = phi_q2_ellipticity(phi, q)

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = mask.reshape(256, 256)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    cv2.ellipse(mask, (int(x), int(y)), (int(major), int(minor)), deg, 0, 360, (0, 255, 0), 2)

    # --- FIX 2: Use the numpy array for the overlay ---
    overlay = np.asarray(image.copy())
    
    # Ensure mask is the right size for the image
    mask_resized = np.asarray(mask)
    mask_indices = mask_resized > 127
    
    # This line will now work because 'overlay' is a NumPy array
  #  overlay[mask_indices] = (overlay[mask_indices] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)
    
    # --- FIX 3: Draw the ellipse using OpenCV ---
   # deg = math.degrees(phi)
   # cv2.ellipse(overlay, (int(x), int(y)), (int(major), int(minor)), deg, 0, 360, (0, 255, 0), 2)

    # --- FIX 4: Convert to Base64 so React can read it ---
   # _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    #overlay_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Return the data (Ensure you return the base64 string, not the array)
    return theta_e, e1, e2, x, y, ((int(major), int(minor)), phi)#, overlay_base64

def get_arcseconds_center(shape, center, grid_size):
    w, h = shape
    x, y = center

 #   print(f'Shape: {w}, {h}')
    
    new_x = grid_size*(x - w/2) / w
    new_y = grid_size*(h/2 - y) / h

 #   print(f"Center Arcseconds: {new_x}, {new_y}")

    return new_x, new_y



def get_convergence_map(theta_e, e1, e2, x, y, display, impath):

    
    lens_model_list = ["SIE"]
    lens_model = LensModel(lens_model_list=lens_model_list)

    h, w = display.shape
    grid_size = 12.0
    cx, cy = get_arcseconds_center((w, h), (x, y), grid_size)

    params_sie = {"theta_E": theta_e, "e1": e1, "e2": e2, "center_x":cx, "center_y":cy}


    
    pix = 256

    x = np.linspace(-grid_size/2, grid_size/2, pix)
    y = np.linspace(-grid_size/2, grid_size/2, pix)
    x_grid, y_grid = np.meshgrid(x, y)

    kappa = lens_model.kappa(x_grid, y_grid, [params_sie])
    kappa_img = np.log10(np.clip(kappa, 1e-10, None))
    
    # Create figure WITHOUT plt (important for web apps)
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()

    im = ax.imshow(
        kappa_img,
        origin="lower",
        extent=[-grid_size/2, grid_size/2, -grid_size/2, grid_size/2]
    )
    fig.colorbar(im, ax=ax, label="log10(κ)")
    ax.set_title("SIE Convergence Map")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    return buf.getvalue()

def show_overlay_mask(image, mask):
    img_array = np.array(image)
    
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)
    
    # Handle different image modes
    if len(img_array.shape) == 2:  # Grayscale
        img_array[mask_bool] = 255
    elif len(img_array.shape) == 3:  # RGB or RGBA
        img_array[mask_bool] = 255  # This sets all channels to 255
    
    # Convert back to PIL Image
    result_image = Image.fromarray(img_array)

    plt.imshow(result_image)
    plt.title("Label")
    plt.axis('off')
   # plt.show()
    
    return result_image

def get_full_mass(theta_e, e1, e2, zl, zs):
    cosmo = Planck18
    D_l = cosmo.angular_diameter_distance(zl)
    D_s = cosmo.angular_diameter_distance(zs)
    D_ls = cosmo.angular_diameter_distance_z1z2(zl, zs)

    Sigma_crit = (const.c**2 / (4*np.pi*const.G)) * (D_s / (D_l * D_ls))

    theta_e_rad = (theta_e * u.arcsec).to(u.rad).value
    R_E = theta_e_rad * D_l
    M_E = np.pi * R_E**2 * Sigma_crit

    return M_E.to(u.Msun).value


@app.post("/predict")
async def predict(file: UploadFile = File(...), pixel_scale: float = Form(0.131), lensZ: float = Form(0.1371), sourceZ: float = Form(0.7126)):
   # print(f"Received pixel_scale: {pixel_scale}")
   # print(f"Type: {type(pixel_scale)}")
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Pre-process
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Ensure model is on that device
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Run through your "Extra Pipeline Algorithms"
    # Example: if you have a custom pre-filter
    # input_tensor = my_custom_pre_filter(input_tensor)

    with torch.no_grad():
        prediction = model(input_tensor)

    # 3. Post-process (The "Extra Algorithms" part)
    # Mask R-CNN outputs are Tensors; convert to lists for JSON compatibility
    pred_masks = prediction[0]['masks'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy().tolist()
    pred_scores = prediction[0]['scores'].cpu().numpy().tolist()
    
    # Return only confident detections
    threshold = 0.0
    results = []
    for i, score in enumerate(pred_scores):
        if score > threshold:
            results.append({
                "mask": pred_masks[i],
                "label": pred_labels[i],
                "score": score
            })

    if len(results) > 0: results = results[0]
    plt.imshow(results["mask"].squeeze())
    plt.title("Label")
    plt.axis('off')
   # plt.show()
    mask_np = (results["mask"].squeeze() > 0.5).astype(np.uint8) * 255

    theta_e, e1, e2, cx, cy, ellipse = SIE_Info(mask_np, image, pixel_scale)
    conv_map_bytes = get_convergence_map(theta_e, e1, e2, cx, cy, np.array(image).mean(axis=2), "path")
    conv_map_base64 = base64.b64encode(conv_map_bytes).decode('utf-8')

    axes, phi = ellipse
    major, minor = axes
    # Inside your predict function
    overlay_image = show_overlay_mask(image, mask_np)
    ellipse_image = np.array(overlay_image)

    cv2.ellipse(ellipse_image, (int(cx), int(cy)), axes, phi, 0, 360, (0, 255, 0), 2)
    ellipse_image = Image.fromarray(ellipse_image)
    # Create a buffer to hold the image bytes
    buf = io.BytesIO()
    overlay_image.save(buf, format="PNG")  # Save PIL Image to the buffer
    overlay_bytes = buf.getvalue()        # Get the actual bytes

    # Now encode to base64
    overlay_encoded = base64.b64encode(overlay_bytes).decode('utf-8')

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    image_encoded = base64.b64encode(image_bytes).decode('utf-8')

    buf = io.BytesIO()
    ellipse_image.save(buf, format="PNG")
    ellipse_image_bytes = buf.getvalue()
    ellipse_image_encoded = base64.b64encode(ellipse_image_bytes).decode('utf-8')

    Mass = get_full_mass(theta_e, e1, e2, lensZ, sourceZ)

    
    return {
        "Image": image_encoded,
        "SIE_Info": [theta_e, e1, e2, cx, cy],
        "overlay": overlay_encoded,
        "ellipse": ellipse_image_encoded, 
        "convergence_map": conv_map_base64,
        "Mass": Mass
    }
