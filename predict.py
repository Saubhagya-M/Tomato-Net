import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model(
    "model_training/tomato_model.h5"
)
print("Model loaded from:", "model_training/tomato_model.h5")
print("Output shape:", model.output_shape)
classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_healthy"
]

def predict_disease(img_path):

    img = image.load_img(
        img_path,
        target_size=(256, 256)
    )

    img = image.img_to_array(img)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])

    predicted_class = classes[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    return predicted_class, confidence


# if __name__ == "__main__":

#     image_path = input("Enter image path: ")

#     disease, confidence = predict_disease(image_path)

#     print("Disease:", disease)
#     print("Confidence:", confidence)