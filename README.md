# dogdetection
To detect the presence and two type of dog species, namely Pomeranian and Chihuahua in the video


### Task : Detect Two Dog Species : Pomeranian And Chihuahua
### Objective: 
To detect the presence and two type of dog species, namely Pomeranian and Chihuahua in the video. 
### Specific Domain Difficulty: 
Pomeranian and Chihuahua are similar sized dogs. Common colours are brown, black & white. There are two types of Chihuahua – long haired and short haired. Long haired versions of Chihuahua are very similar to typical Pomeranian. Only the face/head features are useful distinction between them. Within Chihuahua species, there are some with folded ears, some with pointed ears. Both Pomeranian and long hair Chihuahua have curly tails. Long haired Chihuahua with folded ears are especially difficult to distinguish from Pomeranian. 

![alt text](https://github.com/kenaimachine/dogdetection/blob/main/DetectPomeranianAndChihuahuaFromVideo.png?raw=true)

### Features Of Video Used For Detection Of Pomeranian And Chihuahua.
The video itself comprises of four separate scenes. 
1)	The 1st scene contains only two Pomeranian, one black and one brown. The black Pomeranian is largely stationary while the brown Pomeranian moves around it. This scene is to evaluate the sensitivity of the model in detecting Pomeranians while stationary, moving and in close proximity to each other. 
2)	 The 2nd scene has two Chihuahua moving around a largely stationary Pomeranian. This is to evaluate the detection of Chihuahua vs Pomeranian. 
3)	The 3rd scene involves two Pomeranian (one black and another brown) with two Chihuahuas. This is to evaluate model’s ability to detect different dogs in close interaction and in motion.
4)	The 4th scene, is to evaluate model’s detection of dogs in running pose and also in close contact interaction . As the dogs run about and interact, the ‘squashing’ of furs and pose of the dogs constantly changes. This is a real challenge to maintain detection with correct identification.    


### Collection Of Images
In order to train the model that is able to detect Pomeranian and Chihuahua, the following steps are used:
1)	The images are collected with the above four scenes/scenarios in mind. Firstly, images of each dog type are collected separately. Then images with two dog types together in close interaction are collected. Next, add images of dogs in running pose, sitting pose, looking to side, images of dogs from top view and images with slight variation of dog features such as ears, tails, size and colour. To ensure more variability, images of dogs with different background environment and pictures at different angles are also collected.
2)	There are also a few images of other dog species beside Chihuahua and Pomeranian in the same scene as Pomeranian or Chihuahua. This is to train the model to ignore other dog types. 
3)	80% of the images will be used for training. 20% for testing. Testing set are randomly selected from the pool of images.
4)	Approx. 200 images per dog type are initially collected. More will be added after processing in the next step : Image Augmentation.

### Image Augmentations
Image augmentation is required because the dataset of images are taken in limited set of conditions and may not even be similar to the scenes in the video. As such, there is a need to account for variation in the situations by training the network with augmented data. 

To ensure further variation of each image, a two step process is used. 

The first step is to add more images of Pomeranian and Chihuahua in close proximity / close contact in a single image. Having two dog types in single image is important to train model to differentiate the two dog types. This will increase the number of such images from the initial collection of 400 images (200 per dog species) with 80 more images added to this pool. Half (40images) of the images are augmented using steps defined in next step:

The second step involves augmenting 40 of these images.  
The ‘imgaug’ library is used for image augmentation. Essentially, this step is an offline augmentation. The 40 images as mentioned above are augmented using the python code found in ‘dataaugmentation.ipynb’.These 40 augmented images and its un-augmented version are added to the pool of 400 images to increase the size of dataset. The sequence of image augmentation is randomly applied.  The augmentation sequence applies affine transformations to images, flips some images horizontally, adds some blurring and noise and also changes the contrast and brightness. 

Some reasons for adding these augmentations:
1)	Introducing noise to images helps to reduce overfitting because the model will tend to learn high frequency features from the dataset. Noises distort these high frequency features and can enhance learning capability. 
2)	Translations are added because the objects (i.e. Dogs) can be located anywhere in the image. This forces the network to look at wider area of the image.
3)	Affine transformations are applied with symmetric modes. When images are augmented, it does not have any information outside its boundary. Most of the time its assumed to be constant 0 (black), but since its not always the case in real world context, ‘symmetric’ mode is selected which will make a copy of the edge pixels to fill these gaps instead of just constant black.

A second augmentation process is executed within Tensorflow Object Detection API as defined in the pipeline.config file. This augmentation step is applied to all 480 images. All these images are further augmented with resizing, random pixel value scaling, horizontal flipping, vertical flipping, rotation, random self-concatenating (either vertically or horizontally) and random cropping with padding. 

### Data Annotation
Image labelling is done using ‘LabelImg” library. Each image is labelled with box drawn around area of interest. To ensure variation in categorization of the images:
1)	Some images have bounding box drawn only on the heads of the dogs. This is to ensure that model’s ability to recognize the head features. Since head features are the most distinctive feature to differentiate the species. 
2)	Some have bounding boxes drawn on the full body features, so that model can recognize the dogs side, top or rear profiles. This is important to detect dogs in varying pose such as running or sitting. 
3)	Bounding boxes are drawn as close to the area of interest as possible. This is to reduce the background being detected as part of a dog’s features.  


## 2.	Training/ Validation Process And Discussion Of Results

Two pretrained models are used. There are:
1)	SSD MobileNet V1 FPN 640x640
I.	Video : PomeranianandChihuahua3_detected_ssd_mobilenet_v2.mp4
II.	Results: Please refer to Appendix A.
III.	mAP: 0.551 , Steps: 21k, Time To Train: 4h 21min.
2)	EfficientDet D1 640x640
I.	Video: PomeranianandChihuahua3_detected_EfficientDet1
II.	mAP: 0.5278, Steps: 37k, Time To Train: 8h 54min
III.	mAP@0.5IOU : 0.8
IV.	Results: Please refer to Appendix B.

Due to page limitations, only EfficientDet will be discussed in the following sections.
SSD_MobileNet will not be discussed but its results will be included for comparison. 


## Discussion of Process and Results:
Why EfficientDet Is Selected For This Project?
EfficientDet is a recent family of object detectors which has achieve better efficiency across a wide spectrum of resource constraints.  Key components of EfficientDet include a weighted bi-directional feature pyramid network (BiFPN) which allows fast multiscale feature fusion. Another key component is a compound scaling method which uniformly scales the resolution, depth and width for all backbone, feature network and box/class prediction networks at the same time. 

Scaling matters in the context of CNNs. Generally, deeper network can capture richer complex features but vanishing gradient is a problem. Wider network capture more fine-grained features, but with shallow models, accuracy saturates quickly. And high resolution images encapsulate more fine-grained features, but accuracy gain diminishes quickly. 

Scaling up any dimension of network (width, depth or resolution) improves accuracy. But accuracy gain diminishes for larger models. Therefore it is critical to balance all dimensions of a network during CNN scaling to get better accuracy and efficiency. EfficientDet uses compound scaling to scale network width, depth and resolution in a defined way to achieved the balance required.


On Learning Rate & Overfitting and Batch Size:
Tensorflow Object Detection API allows the setting of decay of learning in the pipeline.config file. 

Several experiments were conducted in this project such as increasing the warmup learning rates and warmup steps. Warmup learning rates is a way to handle ‘early over-fitting’ of strongly-featured observations. When warmup steps is too low, such as 2000, mAP generally fluctuates around lower averages. A lower learning base rate and more total steps has been found to improve the overall mAP. A high learning base rate resulted in more oscillations in mAP over training steps and also lower maximum mAP. Hence, given the ‘highly irregular” feature space, it is decided to proceed with lower learning rate at the expense of longer training time. In addition, using a smoothed learning rate decay with the default Cosine Learning Rate Decay lead to more stable training, and helps prevent overfitting. 

With regard to Batch Size, after some experimentation, it was observed that smaller batch size such as 1 or 2 does not produce good results. Generally, larger batch sizes produce better mAP. But due to GPU memory limitation (about 15GB), the maximum batch size possible is 8. This is relatively small. In addition, larger batch sizes have also been observed to increase training time per step. 

Loss Function Changes:
When classification and localization weights are equal, it was observed that the model can generally localized the objects but less able to categorize the objects correctly in the initial training steps. 

Instead of adjusting these weights, for this project, the project adjust the classification_loss’s weighted_sigmoid_focal loss -gamma and alpha parameters.  The key idea is to downplay easy examples (i.e. background objects) and increase weights for hard or easily misclassified objects (i.e. object against noisy background or object that is partially obstructed). This adjustment approach lead to better results. So after some experimentation, the parameter gamma is increased to 2.5 from 2 and alpha is reduced to 0.15 from 0.25.

On Data Augmentation in ‘pipeline.config’:
Increasing more data augmentation operations in Tensorflow Object Detection API through pipeline.config settings, increases training time per step. 

However, when there are too many data augmentation operations, the mAP results dropped significantly. The key takeaway here is that data augmentation must make sense for the dataset, and not increase irrelevant or badly modified data which the model will never encounter in the real scenario. 

Post-Processing And Overfitting Prevention.
TFOD API already have a method called non-max suppression that’s used within EfficientDet. What matters here is the adjustment of score_threshold which defines a min confidence score for classification (i.e. the box that will be filtered out). The default setup is close to zero, meaning almost all box proposals are accepted. To make the training more stable, score_threshold was set to 0.2 in order to improve convergence and lower chances of overfitting. 

On Regularization Loss:
It appears that as regularization loss decline over 30k steps and above, the total loss for evaluation increases slightly. This could be an indication of overfitting. So instead of continuing the training beyond 40k steps, the project stopped training at 40k step to prevent overfitting.  


Others Observations:
From tensorboard chart, model starts to exhibit overfitting for classification beyond 30k steps. The peak mAP is reached at 37k steps at 0.5278. 

Number parameters for EfficientDet-D1 is 6.6 million. Taken in context with the much smaller number of training dataset, we have a complex model with small training set. 



## 3.	References
1)	EfficientDet: Towards Scalable And Efficient Object Detection
https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html

2)	EfficientDet: Scalable And Efficient Object Detection
https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf

3)	Learning Data Augmentation Strategies For Object Detection
https://arxiv.org/pdf/1906.11172.pdf
