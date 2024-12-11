
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Function for MobileNetV2 model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2 Model")
    
    uploaded_file = st.file_uploader("Upload an image...!!", type=["jpg", "webp", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...!!")
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score * 100:.2f}%")

# Function for FoodClassifier.h5 Model
def food_classifier():
    st.title("Image Classification with FoodClassifier.h5 Model")
    
    uploaded_file = st.file_uploader("Upload an image...!!", type=["jpg", "webp", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...!!")
        
        # Load FoodClassifier model
        model = tf.keras.models.load_model('models/FoodClassifier.h5')
        
        class_names = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 
                'fried_rice', 'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']

        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        accuracy = np.max(predictions)
        
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Function for Cifar_10.h5 Model
def cifar_10_classification():
    st.title("Image Classification with Cifar_10.h5 Model")
    
    uploaded_file = st.file_uploader("Upload an image...!!", type=["jpg", "webp", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...!!")
        
        # Load Cifar_10 model
        model = tf.keras.models.load_model('models\Cifar_10.h5')
        
        # Cifar_10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        accuracy = np.max(predictions)
        
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")


# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model:", ("MobileNetV2","FoodClassifier","Cifar-10"))
    
    if choice == "MobileNetV2":
        mobilenetv2_imagenet()
    elif choice == "FoodClassifier":
        food_classifier()
    elif choice == "Cifar-10":
        cifar_10_classification()    

if __name__ == "__main__":
    main()