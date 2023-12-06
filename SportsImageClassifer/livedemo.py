import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img = img_to_array(img) / 255.0
    return img

# Load your Keras model
model = load_model('c:/Users/RyanM/OneDrive/Desktop/KaggleProject/finalmodel.keras')

# Capture a screenshot (you can use the method mentioned in the previous response)
# After saving the screenshot to a file, apply preprocessing
screenshot_path = "screenshot.png"
preprocessed_image = preprocess_image(screenshot_path)

# You may need to expand the dimensions of the input image to match the model's input shape
input_image = np.expand_dims(preprocessed_image, axis=0)

# Make predictions using your model
predictions = model.predict(input_image)

# Load your class labels or create a mapping between class indices and labels
class_labels = ["Baseball", "Basketball"]  # Replace with your actual class labels

# Display the image along with the prediction percentages for each class
image = cv2.imread(screenshot_path)

# Define a color for the text (in BGR format)
text_color = (0, 255, 0)  # Green

# Initialize a Y-coordinate for text
y_coord = 30

# Loop through class labels and prediction percentages
for class_label, prediction in zip(class_labels, predictions[0]):
    prediction_percent = prediction * 100
    prediction_text = f"{class_label}: {prediction_percent:.2f}%"
    cv2.putText(image, prediction_text, (10, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    y_coord += 30

# Display the image with prediction percentages in a separate window
cv2.imshow("Image with Prediction Percentages", image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: If you want to save the annotated image
cv2.imwrite("screenshot_with_prediction.png", image)

