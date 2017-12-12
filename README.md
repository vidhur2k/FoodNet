# Convolutional Neural Network built for food item recognition in the TADA application

Technology Assisted Dietary Assessment (TADA) has been one of Purdue EPICS' most valuable insights for mounting nutrition intervention programs. With the growing concern about obesity, the need to accurately measure food intake has become imperative. For example, dietary assessment among adolescents is problematic as this group has irregular eating patterns and less enthusiasm for recording food intake. Preliminary studies among adolescents suggest that the innovative use of technology may improve the accuracy of dietary information from young people. Recognition of emerging advancements in technology, e.g., higher resolution pictures, improved memory capacity, faster processors, allow these devices to process information not previously possible.

Our goal is to develop, implement, and evaluate a mobile device food record (mdFR) that will translate to an accurate account of daily food and nutrient intake among adolescents and adults. Our first steps include further development of our pilot mobile computing device to include digital images, a nutrient database, and image processing for identification and quantification of food consumption. Mobile computing devices provide a unique vehicle for collecting dietary information that reduces burden on record keepers. Images of food can be marked with a variety of input methods that link the item for image processing and analysis to estimate the amount of food. Images before and after foods are eaten can estimate the amount of food consumed.

The Image Processing team for Fall 2017 has decided to work on three specific modules:
1. A Convolutional Neural Network for food image recognition.
2. A barcode scanner that provides nutritional information.
3. Graph Based Image Segmentation for accurate food item estimation.

This notebook will summarize the work done in developing the Convolutional Neural Network (CNN) for food image recognition. Primarily, the framework that we intended to use was Tensorflow. But, in order to quickly prototype our network architecture, we decided to use Keras, a Deep Learning framework that is built on top of Tensorflow and provides a high level API for users to work with.

