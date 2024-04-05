from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf 
import io

app = Flask(__name__)
model = load_model("brain_tumor_classification_model.h5")

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')

# Function to preprocess the uploaded image
def preprocess_image(img):
    # Resize the image to match the input shape of the model
    img = tf.image.resize(img, (150, 150))
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict the class of the uploaded image
def predict_image(img):
    img = preprocess_image(img)
    img_array = np.expand_dims(img, axis=0)
    result = model.predict(img_array)
    return result

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        img_file = request.files['image']
        if img_file:
            # Read the image file
            img_data = img_file.read()
            img = image.load_img(io.BytesIO(img_data), target_size=(150, 150))
            # Predict the class of the image
            prediction = predict_image(img)
            # Get the predicted class label
            predicted_class = np.argmax(prediction)
            # Map class index to class label
            labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
            predicted_label = labels[predicted_class]
            return render_template('index.html', prediction=predicted_label)
    return render_template('index.html', prediction="No image uploaded")

if __name__ == '__main__':
    app.run(debug=True)
