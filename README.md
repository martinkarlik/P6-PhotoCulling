# Introduction

This projects proposes a solution to automate the process of culling photos. The design of the system is informed by etnographic studies with professional photographers: the system first clusters similar images, then evalutes their aesthetics within the created clusters and suggest the top choices for each moment. Because of the highly subjective nature of assessing aesthetics of an image, the system is designed to learn from the photographer with minimal interaction inteference: instead of assessing the aesthetics of each image idividually, the system compares two images at a time and predicts the aesthetically superiour image. The training data is infered from the photographers as they cull, by pairing the accepted set of images against the rejected set of images within a cluster. This way, even if the photographer agrees with the proposal and doesn't correct the selection, the system can improve. This method also exploits the observation that photographers look for subtle diferences between two similar images, when deciding which image to choose.

# Implementation

We divide the methods for Image Aesthetics Assessment (IAA) into Generic vs Personalized (whether or not the system uses personal data to learn) and Individual vs Comparative (one image -> aesthetic score or two images -> index of the better one). Using AVA dataset, we trained a Generic Individual IAA model to predicts a user distribution of scores. This we reused as the image encoder of a Generic Comparative IAA model, which we built as a Siamese neural network with custom comparison layer. The system takes two images at a time, encodes them into their feature vectors (in our case simply the distribution of user scores) and then manually compares the means of these histograms to finally predicts the index of a more aesthetic image. Finally, the model is personalized on ~25k image pairs of horses (pairing accepted and rejected horse images from the same clusters). The performance of the different models is documented in their respective evaluation scripts.

# Deployment

The Personalized Comparative IAA model is deployed on a website, the interaction of which was designed together with professional photographers.

Main view
<img src=mainView.PNG width="800" alt="Main view" float="left">

Fullscreen view
<img src=fullscreenView.PNG width="800" alt="Fullscreen view" float="right">
