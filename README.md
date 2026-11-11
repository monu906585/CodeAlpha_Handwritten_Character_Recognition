# CodeAlpha Handwritten Character Recognition

A beginner-friendly Machine Learning internship project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project features
- MNIST dataset (28 × 28 grayscale digit images)
- CNN model with convolution, pooling, dropout, and softmax layers
- Model training and test evaluation
- Classification report and training accuracy plot
- Streamlit web app to upload an image and predict its digit
- Confidence score and probability chart

## Folder structure
```text
CodeAlpha_Handwritten_Character_Recognition/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── models/
└── outputs/
```

## Setup (Windows)

**Recommended:** Python 3.11 or 3.12. TensorFlow may not be available for every newer Python version.

1. Open Command Prompt in this project folder.
2. Create and activate a virtual environment:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Train the model (downloads MNIST data on the first run):

```cmd
python train_model.py
```

The trained model will be saved to `models/handwritten_digit_model.keras`.

4. Start the app:

```cmd
streamlit run app.py
```

The app will open in your browser. Upload a clear image of one handwritten digit.

## How it works
1. MNIST images are loaded and pixel values are normalized to the range 0–1.
2. A CNN learns patterns from training images.
3. The model is evaluated on the held-out test dataset.
4. The web app prepares an uploaded image as a 28 × 28 grayscale input.
5. Softmax scores are used to choose the predicted digit.

## Important note
The model file is not included in this ZIP because it is generated after training. Run `python train_model.py` before launching the app.

## Internship repository name
`CodeAlpha_Handwritten_Character_Recognition`

## Future improvements
- Add a drawing canvas
- Support handwritten letters using EMNIST
- Improve image preprocessing for different backgrounds
- Add confusion matrix visualization
