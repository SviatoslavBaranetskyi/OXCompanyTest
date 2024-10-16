# OXCompanyTest
Anime Character Clustering Model
## Idea
This project is a machine learning system for clustering anime character images. It uses neural networks to analyze images and classify them into predefined clusters. The model is trained on a dataset of anime images and can predict clusters for new images, allowing users to quickly find and identify characters.
# Technological stack
- Python
- Flask
- REST API
- PostgreSQL
- JWT
- AES
- PyTorch
- torchvision
- Pandas
- NumPy
- CNN (ResNet18, ResNet50)
# Description:
### Model training and accuracy testing
In the clustering folder is the train_model.py file that was used to train the model. This script performs data preparation, training the model on the training set and evaluating its performance on the test dataset.

The validation results of the test dataset showed high accuracy of the model:

Test Accuracy: 96.56%,
Average Test Loss: 0.1106

These figures indicate that the model classifies the data efficiently and has low loss on the test set.
### AES for data encryption:
- it provides resistance to attacks, in particular brute-force attacks. The use of keys of different lengths (128, 192, or 256 bits) increases the level of protection.
- works quickly, which allows for efficient real-time data processing.
- is supported by many standards and protocols, making it a universal encryption solution.
- encrypting user data (e.g. email addresses) increases the level of data confidentiality.
# Starting a project
- Clone the repository to your computer:
```
git clone https://github.com/SviatoslavBaranetskyi/OXCompanyTest.git
cd OXCompanyTest
```
- Install the necessary dependencies:
```
pip install -r requirements.txt
```
- Start the application:
```
python app.py
```
## Requests that do not require a token:
- Register a user<br>
POST /register<br>
Content-Type: application/json<br>
{<br>
&nbsp;&nbsp;&nbsp;"username": "username",<br>
&nbsp;&nbsp;&nbsp;"email": "username@gmail.com",<br>
&nbsp;&nbsp;&nbsp;"password": "password",<br>
}
- User authorization<br>
POST /login<br>
Content-Type: application/json<br>
{<br>
&nbsp;&nbsp;&nbsp;"username": "username",<br>
&nbsp;&nbsp;&nbsp;"password": "password"<br>
}
## Requests that require a token (Authorization: Bearer your-access-token):
- Upload photo for analysis<br>
POST /predict<br>
Content-Type: multipart/form-data<br>
{<br>
&nbsp;&nbsp;&nbsp;"file": <image_file><br>
}
- Retrieve a list of user analyses<br>
GET /analyses
# Future plans:
- Experiment with different architectures, such as EfficientNet and Inception, to enhance classification performance.
- Advanced Analytics
- Prepare the application for deployment on cloud platforms (e.g. AWS, Heroku) to make it available to a wider audience.
## Developer
Sviatoslav Baranetskyi

Email: svyatoslav.baranetskiy738@gmail.com
