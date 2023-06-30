import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError



json_file = open('Animal_10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("Animal_10.h5")
# print("Loaded model from disk")

st.title('Identify Animals By Their Images')
class_name = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

traslate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
            "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel"}

ImagePath = st.file_uploader("Choose a file")


try:
    if st.button('Predict'):
        test_image = image.load_img(ImagePath, target_size=(256, 256))
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        result = loaded_model.predict(test_image, verbose=0)

        type_ = traslate[class_name[np.argmax(result)]]

        st.header('Prediction is: ' + type_)
        st.header('Confidence is: ' + str(round(np.max(result), 4) * 100) + ' %')

except TypeError:
    st.header('Please Upload Your File !!!')

except UnidentifiedImageError:
    st.header('Input Valid File !!!')