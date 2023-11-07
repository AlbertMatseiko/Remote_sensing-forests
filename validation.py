import os
import cv2
import numpy as np
import h5py as h5
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'), tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

path_to_val = './DATA/image_to_validate/'
h5_name = 'LC08_L2SP_02_T1_256.h5'
path_to_h5 = './DATA/h5_files/' + h5_name


def get_rgb_image(img_norm, path_to_h5=path_to_h5):
    with h5.File(path_to_h5, 'r') as f:
        MEAN = f['all/norm_params/mean_values'][:]
        SIGMA = f['all/norm_params/sigma_values'][:]
    img = (img_norm * SIGMA + MEAN)
    return img[:, :, 5:2:-1]


def draw_layers(image, predicted_classes, opacity=0.3):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Original Image", "Classes", "Overlay", ""))
    rgb = get_rgb_image(image)

    fig.add_trace(px.imshow(rgb).data[0],
                  row=1,
                  col=1)
    fig.add_trace(px.imshow(rgb).data[0],
                  row=1,
                  col=2)

    for i in np.unique(predicted_classes):
        class_mask = np.where(predicted_classes == i, i, np.nan)
        fig.add_trace(
            go.Heatmap(
                z=class_mask,
                colorscale=[[0, f"hsv({i * 360 / 20},100%,100%)"], [1, f"hsv({i * 360 / 20},100%,100%)"]],
                hoverongaps=False,
                showscale=False,
                showlegend=True,
                name=f"Class {i}",
                opacity=1
            ), row=1, col=2)

    return fig

## Converting val tifs to npy, saving
#collecting tifs into one numpy array
name_tifs = [s for s in sorted(os.listdir(path_to_val)) if s.startswith('B')]
scene = []
for nt in name_tifs:
        path_to_tif = path_to_val + nt
        #z print(path_to_tif)
        band = cv2.imread(path_to_tif)[:,:,0]
        scene.append(band)
image_npy = np.array(scene).transpose(1,2,0)


with h5.File(path_to_h5, 'r+') as f:
    means = f['all/norm_params/mean_values'][:]
    stds = f['all/norm_params/sigma_values'][:]
image_to_predict = (image_npy-means)/stds
np.save(path_to_val+'image_to_predict.npy', image_to_predict)
magnitsky_preds = cv2.imread(path_to_val+'klass10-3o-magansk.tif')[:,:,0]
image_to_predict.shape, magnitsky_preds.shape
fig_mag = draw_layers(image_to_predict, magnitsky_preds)
fig_mag.write_html(path_to_val + "/fig_mag.html")

## Loading model, making preds and html picture
#path_to_val = './DATA/image_to_validate/'
image_to_predict = np.load(path_to_val+'image_to_predict.npy')
image_to_predict = np.expand_dims(image_to_predict, 0)
# just choose the model name
model_name = '07.11.2023_ResNet_f.128.64.64_k.2.2.2_c0.1_b0.1_CLASSES.14_BS.4_loss.nmi_without_shifts'
path_to_model = './models/'+model_name+'/classificator_last/'
model = tf.keras.models.load_model(path_to_model, compile=False)
preds = model.predict(image_to_predict)
image_to_predict = image_to_predict[0]
predicted_classes = preds.argmax(axis=-1)[0]
np.save('./models/'+model_name+'/preds_val.npy', predicted_classes)
fig_nn = draw_layers(image_to_predict, predicted_classes)
fig_nn.write_html('./models/'+model_name + "/fig_nn.html")