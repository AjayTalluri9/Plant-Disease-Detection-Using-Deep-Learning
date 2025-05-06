# Plant-Disease-Detection-Using-Deep-Learning

## Overview

This project aims to develop a plant disease detection system using **Deep Learning**. It leverages a **ResNet9** model to identify plant diseases from images of leaves. The model was trained on a plant disease dataset and deployed using a **Flask** web application. The web interface allows users to upload leaf images, and the system will predict whether the leaf is healthy or infected, as well as identify the specific disease (if applicable).

## Dataset

The model utilizes a **custom dataset** composed of various images of plant leaves. Each image is labeled to indicate the plant species and the corresponding disease. These images are used to train and evaluate the deep learning model. The dataset includes a variety of plant species, such as corn, tomato, and apple, along with disease labels like rust, blight, and leaf mold. 

### Key Features:
- **38 plant disease classes**: Including both healthy and infected leaves across various species.
- **Data Augmentation**: Techniques like rotation, flipping, and zooming were applied during training to prevent overfitting and improve generalization

<img width="533" alt="Screenshot 2025-05-05 at 2 17 27 PM" src="https://github.com/user-attachments/assets/20530565-3acc-4c68-8209-2c0d491702cb" />

This image shows an example of the plant disease dataset used for training the model. The dataset contains images of leaves from various plants with diseases like rust, blight, and others, each labeled with the corresponding plant species and disease.




## Model Architecture: ResNet9

The model used in this project is a **ResNet9** architecture, a smaller and more efficient variant of the popular **ResNet** family. It contains several convolutional blocks with skip connections, enabling deeper networks without overfitting.

### Key Architecture Details:
- **Convolutional Layers**: Several layers that extract features from images, including edge detection, texture analysis, and lesion identification.
- **Residual Blocks**: These layers help the model retain important feature information, aiding in the detection of subtle plant disease patterns.
- **Fully Connected Layers**: After feature extraction, the model flattens the outputs and applies a fully connected layer to classify the image into one of 38 classes (healthy or specific diseases).

<img width="386" alt="Screenshot 2025-05-05 at 2 20 14 PM" src="https://github.com/user-attachments/assets/02f97240-5ea3-4ed2-b2b5-7e6a754d4733" />

This image illustrates how the code classifies plant leaf images into different disease patterns. It shows how the convolutional layers detect specific features such as color changes and lesions on the leaf, which are essential for distinguishing between healthy and diseased plants.

### Code Details:
```python
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),  # Adjusted pooling size
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, num_diseases)  # Adjust for 512 channels * 3 * 3 size
        )
```
## Flask Web Application
A Flask web application was developed to serve the model and allow user interaction. The user uploads an image through the web interface, and the model predicts the disease (if any).

### Important HTML Files:
* index.html: This is the main user interface for the application. It provides a form to upload plant leaf images and displays a preview of the selected image. The user selects an image, submits it, and the Flask backend processes the image for prediction.
* results.html: After the model predicts the disease or health status of the uploaded leaf image, the results are displayed on this page. It shows the image along with the predicted class (disease or healthy leaf).

These HTML templates are rendered using Flask’s render_template() function. Flask loads these templates and injects dynamic data (like the uploaded image and model prediction) into them.

## Features:
Image Upload: Users can upload images of plant leaves in .jpg or .png format.

## Prediction Result: 
Once the image is uploaded, the system returns the disease prediction and displays the result on a new page, showing the uploaded image along with the predicted class.

```python
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = Image.open(filepath)
            img_tensor = transform(img)
            prediction = predict_image(img_tensor, model)
            return render_template('results.html', image_file=filepath, prediction=prediction)
    return render_template('index.html')
```
# How to Run the Application
To run the Plant Disease Detection web application locally, follow these steps:

## 1. Clone the Repository
Clone the project repository to your local machine.
```bash
git clone <repo-url>
```
## 2. Install Dependencies
Create a virtual environment and install the required Python packages.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Dependencies:
1. Flask for web development.
2. PyTorch for the deep learning model.
3. Pillow for image processing.

## 3. Download the Trained Model
Download the trained ResNet9 model from the specified source or from the project repository. Ensure that the model file is placed in the same directory as app.py with the filename plant-disease-model-complete.pth.

## 4. Run the Flask Application
Start the Flask web server by running the following command:
```bash
python app.py
```
This will start the server on http://127.0.0.1:5000/.

## 5. Access the Web Interface
Open your browser and go to http://127.0.0.1:5000/. From there, you can upload a leaf image and see the disease prediction result.

# Results
The proposed deep learning-based approach for plant disease detection has shown considerable promise in improving the identification of agricultural diseases. Utilizing convolutional neural networks (CNNs) and transfer learning techniques, the model achieved a high classification accuracy of 85%, complemented by robust precision (90.9%), recall (83.3%), and an F1-score of 87%. Its efficiency is underscored by an inference time of 47.2 ms, making it suitable for real-time use in mobile and cloud applications.

<img width="428" alt="Screenshot 2025-05-05 at 2 24 24 PM" src="https://github.com/user-attachments/assets/6630b9c7-0263-4205-8f4a-603af8390deb" />

The confusion matrix illustrates the model's performance. It shows the True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN), which are important for evaluating the model's ability to correctly classify diseased and healthy plant leaves.

<img width="498" alt="Screenshot 2025-05-05 at 2 25 02 PM" src="https://github.com/user-attachments/assets/b63d7d1c-3965-4a76-85c5-4f34c2bb192b" />

This graph shows the model's performance over different epochs, displaying the accuracy and time taken for each iteration. It helps visualize the model’s learning progress and overall efficiency during training.


# Limitations and Future Work
* Generalization: The model performs well on images from the PlantVillage dataset but may struggle with real-world images that differ in lighting, background, or leaf orientation. Future improvements could include training with field images.
* Mobile Optimization: The current app is designed for desktop use. Future work could focus on optimizing the model and web app for mobile devices, allowing farmers to use the system on the go.
* Dataset Diversity: The model could benefit from more diverse plant species and disease examples to improve its classification capabilities for rare or less represented diseases.

# Conclusion
This project demonstrates a deep learning-based approach to plant disease detection using a ResNet9 model. It serves as a proof-of-concept for how machine learning can aid in agriculture by providing real-time, automated disease diagnosis from leaf images. The system is easily extendable and could be further enhanced with real-world data, additional disease classes, and mobile deployment to aid farmers in the field.
