# TreeClassifier
### Goal
I completed this project as part of the Out In Tech Mentorship program. This project was an opportunity to learn more about how machine learning models are built at the most basic level. Through out this program, I followed Deep Learning with PyTorch, using it as a guide to understand some of the algorithms that make up these models

### Design 
For the Tree Classifier, I built a Convolutional Neural Network that would tell the difference between 5 different types of trees native to Brooklyn. In the CNN, it passes through three rounds of convolutions. Then I use a fully connected layer to map my convolutions to the test image at hand to identify it. For each of the nn library main functions used in CNNs: BatchNorm2D, Conv2D, MaxPooling2D, and ReLu, I rebuilt my own versions of the functions.
