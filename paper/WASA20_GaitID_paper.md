# GaitID: Robust Wi-Fi Based Gait Recognition

Yi Zhang<sup>1</sup>, Yue Zheng<sup>1</sup>, Guidong Zhang<sup>1</sup>, Kun Qian<sup>2</sup>, Chen Qian<sup>1</sup>, and Zheng Yang<sup>1(B)</sup> 

<sup>1</sup> Tsinghua University, Beijing, China 

zhangyithss@gmail.com, cczhengy@gmail.com, zhanggd18@gmail.com, 

chen.cronus.qian@gmail.com, hmilyyz@gmail.com 

<sup>2</sup> University of California San Diego, San Diego, CA, USA qiank10@gmail.com 

Abstract. Gait, the walking manner of a person, has been perceived as a physical and behavioral trait for human identification. Compared with cameras and wearable sensors, Wi-Fi based gait recognition is more attractive because Wi-Fi infrastructure is almost available everywhere and is able to sense passively without the requirement of on-body devices. However, existing Wi-Fi sensing approaches impose strong assumptions of fixed user walking trajectory and suficient training data. In this paper, we present <sup>GaitID</sup>, a Wi-Fi based human identification system, to overcome above unrealistic assumptions. To deal with various walking trajectories and speeds, <sup>GaitID</sup> first extracts target specific features that best characterize gait patterns and applies novel normalization algorithms to eliminate gait irrelevant perturbation in signals. On this basis, <sup>GaitID</sup> reduces the training eforts in new deployment scenarios by transfer learning. Extensive experiments have been conducted on the implementation and the outcomes are satisfying. To the best of our knowledge, <sup>GaitID</sup> is the first gait-based identification approach without any restriction on walking trajectory and speed. 

Keywords: Gait recognition Channel state information Wi-Fi 

## 1 Introduction

Various sensor modalities, e.g., cameras [3], inertial sensors on wearables [19] and Wi-Fi signals [14] emitted by wireless devices [12], have been certificated to be able to extract gait of a user for person identification. Among these sensors, video has the risk of privacy leakage and inertial sensors require users to actively carry mobile devices. In contrast, Wi-Fi signal becomes a more attractive carrier for gait-based person identification since Wi-Fi infrastructure is ubiquitously available [13] and able to work without user’s perception. 

Current state of Wi-Fi based gait identification approaches [12,17,18], however, rely on extensive training eforts for every target person in each monitoring area. Such cumbersomeness stems from three limitations of the existing approaches. First, Wi-Fi signals reflected by a target person not only possess the gait signature of the person, but also are distorted by the surrounding multipath environment. Thus, the recognition model directly trained with raw Wi-Fi features or their statistics, as WiWho [17], may overfit the environment where the data is collected and cannot be generalized to new environments without retraining. Second, besides the efect of environment factors, features related to the gait of the person in Wi-Fi signals still depend on how the person moves relative to the Wi-Fi devices. WiFiU [12] derives parameters of gaits from Doppler Frequency Spectrum (DFS). However, it requires that the target person walks right towards or away from the Wi-Fi devices on fixed trajectories to ensure the consistency of the DFS, which limits the practicality of the approach. Third, as the learning model becomes more and more sophisticated, e.g., in terms of the number of parameters that need to be trained, a suficiently large amount of training data is required when each new person is added. 

![](images/a0d31fd0941045c3cb6f3852d077a590e8c06cb208f806eec8b4e9462a248942.jpg)


![](images/215299e0480487949b1ab4f0070321292936b731039d21d642f11affa98d694b.jpg)



(a) Cross-track perfor- (b) Dataset scale demance. manded by DNN.


![](images/96e7d8eba94a81ef168321e7d1f8db2cb48610b0995c6a5922f937dc538161c1.jpg)



Fig. 1. System overview.



Fig. 2. Challenges of Wi-Fi based gait identifi cation.


In this paper, we present GaitID, an ubiquitous Wi-Fi based person identification framework, which is robust to walking manners and environment variance, and reduces training eforts significantly as Fig. 1 shows. GaitID has two key characteristics that enhance the robustness of this system with limited training samples. On one hand, GaitID is immune to environmental variations and motion status (e.g., location and velocity) of target persons, and retains prominent generalizability between environments and trajectories. On the other hand, GaitID is able to transfer the learning model of existing persons to newcomers with only a small amount of training data collected from the person. To support these features, we overcome two critical challenges. 

The first challenge is to overcome the negative impact of environmental variations and motion status of target person during walking. GaitID borrows the idea of body-coordinate velocity profile (BVP) in [20], which represents the velocities of body parts during walking, and proposes a dedicated feature GBVP for gait characterization. The GBVP is resilient to scenario factors, including environmental changes, and the location and orientation of walking trajectories. To adapt GBVP to realtime gait identification, we propose a bunch of agile extraction and normalization algorithms to boost its robustness to gait-irrelevant factors. The designed feature GBVP is theoretically both environment and trajectory independent, which mitigates gait-irrelevant components. The second challenge is to efectively train the model with a small quantity of data collected from each new user. GaitID adopts deep neural networks as the gait identification model, which is proved to be efective but has so sophisticated structure that requires a large amount of data samples to get fully trained. To overcome the challenge, GaitID exploits transfer learning to avoid retraining of the partial network which extracts high-level gait features from the input velocity profile and has the same network parameters shared by all persons. 

In summary, we make the following contributions. First, we propose an agile algorithm to extract gait-specific feature GBVP that is resilient to environment and trajectory change, and thus relieves restrictions on walking manners. Second, the proposed gait identification approach requires little training eforts for various scenarios and persons, and thus can be easily deployed and extended. Third, we implement GaitID on COTS Wi-Fi devices and extensive experiments have demonstrated the efectiveness and robustness of the proposed system. 

## 2 Related Work and Motivation

GaitID attempts to tackle two main challenges in Wi-Fi based gait identification. 

Immune to Trajectory and Speed Variance. Both WiWho [17] and WiFiU [12] try to preserve human-specific information in their extracted features, e.g., Doppler frequency shifts, from Wi-Fi signals. Such features, however, are highly correlated with users’ relative movements to Wi-Fi devices, thus impose stringent restrictions on their monitoring tracks. Widar3.0 [20] proposes a domainindependent feature BVP, which is mainly designed for in-place activities and sensitive to the moving speed of the target. As a brief example in Fig. 2(a), gait samples are collected from two users with diferent walking manners and a classifier based on CNN and RNN is trained and tested on datasets from diferent walking tracks. While DFS and BVP perform better for the same track, they fail to hold performance for testing on diferent tracks. Whereas, GBVP is robust to track and speed variance. 

Reducing Training Data for Newcomers. To fully exploit the spatial and temporal property of motion features, existing works [2] leverage sophisticated deep neural networks to achieve high accuracy. However, a more complex structure usually means more parameters need to be trained, which leads to the requirement of a massive amount of training data. This problem becomes increasingly conspicuous when new users are added and the network should be retrained. Figure 2(b) illustrates the exponential growing trend of required training samples needed to reach specific accuracy for a typical DNN network. 

Lessons Learned. The deficiency of existing gait identification works demand to be relieved before practical usage is achieved. GaitID is designed to address these issues. 

![](images/ae34abbdf4d3b8039d6119d0f595ada33fbdcc3719bf413bb91757b45cd78ea7.jpg)



Fig. 3. Work flow of GaitID.


## 3 System Design

Figure 3 shows the work flow of our system. 

## 3.1 GBVP Extraction

Widar3.0 [20] proposed an environment-independent motion feature BVP to portray human activities, which is resilient to location, orientation and environment changes. However, BVP can not be directly used for gait recognition and the reasons are two-fold. 1) BVP are modeled with in-place activities where movements of the reflection objects can be ignored, while gait activity involves meters of torso movements. 2) The complexity of BVP extraction algorithm is especially high and can hardly be applied into realtime systems. Therefor, GaitID designed a rigorous and agile feature extraction algorithm to acquire environment-independent and gait-specific feature GBVP (gait-BVP). In the following formulations, we establish the coordinates whose origin is the location of the person and positive x-axis aligns with his/her face orientation. 

GBVP Formulation. To formulate GBVP, we first define an operator : 

$$
\mathcal {A} \otimes \mathcal {B} \triangleq \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {N} \mathcal {A} _ {(i, j, *)} \cdot \mathcal {B} _ {(i, j)},\tag{1}
$$

Where $\mathcal { A } \in R ^ { M \times N \times P }$ and $B \in R ^ { M \times N }$ . Hence, the operation result $\mathcal { A } \otimes B \in$ $R ^ { P }$ is equivalent to multiply each element of with corresponding vector in the third dimension of and then sum them up. Using the defined operator, we formulate GBVP as follows: 

$$
[ G B V P ] = \min _ {G} \sum_ {i = 1} ^ {L} | \mathrm{EMD} (D ^ {(i)} (G), [ D F S ] ^ {(i)}) | + \eta \| G \| _ {0},\tag{2}
$$

Where $G \in R ^ { N \times N }$ is GBVP. $D ^ { ( i ) } ( G ) \in R ^ { N \times N \times F }$ is the reconstructed DFS from GBVP for $i ^ { t h }$ Wi-Fi link. $[ D F S ] ^ { ( i ) }$ is the observed DFS on $i ^ { t h }$ link. L is the total number of links. N is the number of possible values of velocity components along $\mathrm { { x } / \mathrm { { y } } }$ axis. F is the number of frequency bins in DFS. $E M D ( \cdot , \cdot )$ is the Earth Mover’s Distance [8] and $| | \cdot | | _ { 0 }$ is the number of non-zero elements in GBVP. η is the sparsity coeficient. $D ^ { ( i ) } ( G )$ is given by: 

$$
D ^ {(i)} (G) = S U B (A ^ {(i)}) \otimes G,\tag{3}
$$

![](images/7e3355e82d55c335db636d4075b78b26d4080a4d19661f3c88b77521c26c1d17.jpg)



(a) Gait DFS profile.


![](images/2c0dade6e0aacd2401b6001e215834b3cad311b460922082d149dbbe8ec04301.jpg)



(b) GBVP search zone.


![](images/43e7b166449e47acc11ae123f693c377ee4168c3a91f653ad8253be36273c595.jpg)



(c) Normalized GBVP



Fig. 4. GBVP extraction and normalization


Where $A ^ { ( i ) } \in R ^ { ( N \times N \times F ) }$ is the coeficient matrix to map GBVP into DFS on $i ^ { t h }$ link. The $S U B ( \cdot )$ is the operator to cherry-pick the most relevant elements in the coeficient matrix to reduce the search space of GBVP and eventually reduce algorithm complexity. We will shortly introduce the algorithm on how to do that reduction in the Accelerating GBVP extraction part. Each element in the coeficient matrix can be determined by: 

$$
A _ {(j, k, m)} ^ {(i)} = \left\{ \begin{array}{l} 1 f _ {m} = f ^ {(i)} (<   v _ {j}, v _ {k} >) \\ 0 \text {else} \end{array} \right.,\tag{4}
$$

Where $< v _ { j } , v _ { k } >$ is the corresponding velocity of the $< i , j > ^ { t h }$ element <sup>j</sup>in GBVP matrix. $f _ { m }$ is the $m ^ { t h }$ frequency sampling point in the DFS profile. $f ^ { ( i ) } ( \cdot )$ <sup>m</sup>is a mapping function to convert target velocity into DFS observation for $i ^ { t h }$ link with $f ^ { ( i ) } ( < v _ { x } , v _ { y } > ) = a _ { x } v _ { x } + a _ { y } v _ { y }$ . Specifically: 

$$
a _ {x} = \frac {1}{\lambda} (\frac {x _ {t}}{\| (x _ {t} , y _ {t}) \| _ {2}} + \frac {x _ {r}}{\| (x _ {r} , y _ {r}) \| _ {2}}), a _ {y} = \frac {1}{\lambda} (\frac {y _ {t}}{\| (x _ {t} , y _ {t}) \| _ {2}} + \frac {y _ {r}}{\| (x _ {r} , y _ {r}) \| _ {2}})\tag{5}
$$

Where $( x _ { t } , y _ { t } )$ and $( x _ { r } , y _ { r } )$ are the locations of transmitter and receiver and should be updated whenever the target position changes during walking. We sanitize CSI with existing works [7] and acquire target user’s torso position and orientation by existing passive tracking system, e.g., IndoTrack [4] and Widar2.0 [6]. 

Accelerating GBVP Extraction. For activity recognition tasks with Wi-Fi, only a few major reflection paths are considered and Widar3.0 [20] leveraged this sparsity by adding a regular term in target function to recover BVP. However, this sparsity is not fully exploit and the algorithm is still too cumbersome to be applicable. Our key insight is that, the spatial correlation embodied in human motion could potentially be used to reduce the complexity of GBVP extraction algorithm. 

We observe that, during gait activity, the limbs would swing to opposite sides of the torso with limited speeds and the velocity of all the reflection paths would cluster around the velocity of torso. This observation can be confirmed by Fig. 4(a). This profile is constructed with CSI for 4 continuous cycles from one user. The red curve demonstrates the major energy corresponding to torso motion and the white curves demonstrate the residual energy corresponding to limbs motion. It is clearly shown that both sparsity and spatial clustering phenomenon exist in the reflected signal. Hence, if we can pinpoint the velocity of torso on GBVP matrix, then the whole body GBVP components can be searched within a small area centered on torso component. 

Based on the above observations, GaitID first identifies the maximum frequency bins in DFS from each Wi-Fi link and formulates the relationship between frequency bins and velocity with the methods provided in [6]. Solving the equations from multiple links, GaitID pinpoints the torso velocity on GBVP matrix. On top of that, GaitID crops the adjacent elements of torso component in the coeficient matrix described in Eq. 3. The subtracted coeficient matrix is then used for GBVP recover with Eq. 2. After the above process, the GBVP search space is hereby reduced. In our experiments, the crop window’s size is empirically selected and a smaller window would results in shorter running time but deteriorative accuracy, vice versa. The torso and limbs components as well as the search zone in GBVP is visualized in Fig. 4(b). 

![](images/3b7c4854675fb5d42eacc3991a2d5cd2516c0436513e0611fb4db9fa9ee8f75e.jpg)



Fig. 5. Gait recognition model.


GBVP Normalization. While GBVP is theoretically only related to the gait of the target, it requires extra normalization to increase the stability as gait indicator. The reasons are three-fold. First, literature [16] has proven that the torso movement contains little information of gait patterns and need to be removed. Second, the reflected signal power is correlated with torso position relative to transceivers. Third, diferent walking speeds correspond to diferent limbs swing speeds, which results in variation of the number of GBVP frames and value floating within each GBVP matrix. 

Thereafter, GaitID first compensates the torso speed by applying translation and rotation on GBVP. The translation displacement is $\lVert \pmb { v } _ { t o r s o } \rVert _ { 2 }$ and the rotation angle is $\angle { \pmb v } _ { t o r s o } - \angle { \pmb v } _ { r e f }$ where ${ \pmb v } _ { r e f }$ <sup>torso</sup>is the manually selected refer-<sup>torso ref ref</sup>ence orientation. This transformation procedure is similar to moving the target human to a treadmill, on which the target performs fixed-speed walking. 

GaitID then normalizes the sum of all elements in each GBVP to 1. It is based on the observation that the absolute reflection power contains environment information while the relative power distribution over physical velocities doesn’t. Lastly, GaitID scales each single GBVP with a scaling ratio $\frac { v _ { o b } } { v _ { t g } }$ , where $v _ { o b }$ is the observed walking speed and $v _ { t g }$ <sup>v</sup>is the target walking speed. GaitID then <sup>tg</sup>resamples GBVP series over time with a resampling ratio of $\frac { v _ { t g } } { v _ { o b } }$ , which is based <sup>v</sup>on the hypothesis that the total displacement of the limbs relative to the torso is analogous across diferent walking speeds. 

After normalization, only human identity information is retained while gaitirrelevant factors are removed. The normalized GBVP is visualized in Fig. 4(c), where torso speed is compensated and walking direction is normalized to a fixed direction. 

## 3.2 Recognition Mechanism

Fundamental Model. Generally speaking, each single GBVP captures limbs’ velocity distribution relative to the torso, and GBVP series exhibit how the distribution varies over time. As shown in the upper half of Fig. 5, we adopt a deep neural network (DNN) to best depict the characteristics of GBVP. 

![](images/f4b2529d8b1da6cb76f7b331c1600d09e87692e29e5ca5c629e38a763e612280.jpg)



Fig. 6. Network weights trained from two datasets (lower layers share commonalities).


The input of the fundamental DNN model is of size $2 0 \times 2 0 \times 3 0$ , as velocity is quantized into 20 bins along the axis of the body-coordinate system, and GBVP series is adjusted to 30 snapshots after normalization. GaitID first applies 3D CNN onto the GBVP series for spatial feature compression and time redundancy reduction. Convolution operations along the time domain also alleviate single GBVP estimation error. 16 convolutional filters of size $5 \times 5 \times 5$ output 16 3D matrices of size $1 6 \times 1 6 \times 2 6$ . Then the max-pooling layer is applied to downsample feature maps to the size of $8 \times 8 \times 2 6$ . By flattening the feature maps except for the time dimension, we obtain a vector series of size $1 0 2 4 \times 2 6$ . And a fully connected (FC) layer is appended. 

Recurrent layers are also incorporated into the model to model the temporal dynamics of the vector series. Considering the long-term characteristics of GBVP as a gait cycle always lasts for a duration of more than one second [5], regular RNN sufers from the vanishing gradient problem [9], which hinders them from being used for long-term information extraction. Thus, instead of regular RNN, we adopt a better variation of RNNs: Long Short Term Networks (LSTM) [10]. The output of LSTM is then encoded with the softmax layer to do multiclass classification. 

Transfer Learning for Reducing Training Eforts. Despite the fact that the structure of the fundamental model is not that sophisticated, the DNN mode still demands enormous training data to converge. And when a new user is added into the human identification system, he/she must perform massive gait activities. Evaluation results in Sect. 4 shows that there will be a rapid reduction in recognition accuracy even if the amount of training data decreases slightly. 

Our solution was inspired by the observation that neural networks trained on similar datasets often share commonalities, i.e., the model trained on similar datasets undergo analogous convergence procedure to some extent [11,15]. This characteristic is exploited in a well-known research realm called Transfer Learning. 

To testify the validity of Transfer Learning in gait recognition, we tune the fundamental model from scratch on two independent datasets separately, each of which is composed of GBVP series from two diferent users. We visualize and compare the network weights from the two converged models. As can be seen from Fig. 6, the lower layers of the neural network have an analogous distribution of weights while the upper layers vary a lot. This phenomenon paves the way to transfer the information learned from diferent datasets and alleviate data collection efort. 

To leverage this generalizability between datasets, GaitID first trains a model on the pre-collected large-scale dataset, which consists of GBVP from n class 1 persons. Then GaitID replaces the softmax layer in the fundamental model with a diferent shape of n class 2 and initiates it randomly. The remaining weights of the model are initialized with the weights copied from the pre-trained model. The lower half of Fig. 5 shows how transfer learning is applied in GaitID. We will demonstrate that starting from the transferred structure and weights, our model can converge on the new dataset with significantly fewer data instances in Sect. 4. 

## 4 Evaluation

## 4.1 Experimental Methodology

GaitID is implemented on one Wi-Fi sender and six Wi-Fi receivers, each of which is equipped with Intel 5300 wireless NIC and Linux CSI Tool [1]. We conduct experiments under two diferent indoor environments illustrated in Fig. 7(a). We designed four linear tracks, including two perpendicular lines to both axes and two diagonal lines shown in Fig. 7(b). Each of these tracks has enough length for five steps and users can walk on both ways. 

We recruited 11 volunteers(4 females and 7 males) to participate in our experiments, covering the height 155 cm to 186 cm and weight 44 kg to 75 kg and age from 20 to 28 years old. These volunteers were asked to walk normally with diferent speeds on those tracks and each data sample contains five steps. Specifically, 10 of the volunteers were asked to walk on each direction of four tracks 50 times in Hall, contributing 10 users 4 tracks 2 directions 50 instances for datasets. 3 of the volunteers in the Discussion Room contributed 3 users 4 tracks 2 directions 25 instances in the dataset. All experiments were approved by our IRB. 

![](images/e7923e4cd0f365209e8445c58c754961272ac16f4f319f7f066671434d4f0d6a.jpg)



(a) Floorplan.


![](images/fb998f17c37a213328243155a902f3097b2d0c6d70de6405f2e0c793480aefa5.jpg)



(b) Devices.


![](images/9cc566914140273a37d30025f223f1fc503f4d76dc52cfa7ac1c3067f2711041.jpg)



(a) CM.


![](images/b6af38ea1b3b493043ee635c9de475fb533dda51741e516d9fb440026ae88d9a.jpg)



(b) User diversity.



Fig. 7. Experiment setup.



Fig. 8. Overall accuracy.


## 4.2 Overall Performance

Figure 8(a) shows the confusion matrix (CM) for all of the 11 users. The overall identification accuracy is 76.2%. From the confusion matrix, most users experience an accuracy over 75% except for user E, H and J, which may be attributed to their walking manner of putting hands in pockets, leading to infrequent motion on arms and induce less features in GBVP. User J and F are likely to be confused, which may be caused by their similarities in body shape. 

Figure 8(b) further shows the identification accuracy for diferent user numbers. Basically, the identification accuracy declines with more users involved, which is intuitive because more categories would lead to more crowded feature clusters in feature space. The accuracy for two users is above 99%, meaning that the extracted gait features are distinct for identification. It’s notable that GaitID holds its accuracy above 93.2% for about 5 users, demonstrating its potential for smart home applications where there are only a few users in indoor scenarios. 

## 4.3 Generalizability Evaluation

Accelerating Performance. The process of GBVP extraction is accelerated by the novel acceleration algorithm. To validate the eficiency and efectiveness of GBVP over BVP, we collect 400 samples of CSI from three volunteers and each corresponds to 4 steps. We then extract BVP as well as GBVP with diferent accelerating windows sizes (as described in Sect. 3.1). The system is running on a server with 32 cores of Intel Xeon CPU E5-2620 v4 @ 2.10 GHz and Matlab2016b installed. The system running delay and recognition accuracy are demonstrated in Fig. 9(a). As can be seen, even with 2 s of CSI as input, the BVP extraction would last for unbearably 78 s. However, with our proposed accelerating algorithm, the feature extraction speed can be accelerated 156 times while maintaining the recognition accuracy to some extent. The GBVP extraction delay is 0.5 s with a window size of $5 \times 5$ and the accuracy holds above 83%, which enables the system responds in real-time. We believe the proposed accelerating algorithm would push the BVP to a broader application prospect on other motion recognition scenarios. 

Normalization Performance. For walking tracks independence, we randomly select two users’ data from all the datasets, using one track as test and the remaining three tracks to train our model, ignoring their walking speed. As is shown in Fig. 9(b), the overall accuracy with normalized GBVP is significantly above that without normalization. The second track benefits least from normalization because we selected its direction as the reference direction and the GBVP from other tracks are rotated to match this orientation. For walking speed independence, we classify all the collected data into 6 categories, each with quantized speed. We then select one category as test and the others to train the model. Figure 9(c) demonstrates the remarkable improvements with normalization. The fifth category benefits the least from normalization because we selected its speed as the reference speed and GBVP with other speeds are normalized to this case. 

![](images/0bdc703282ec0b591e3897b9e470e600c80626fed083c3efa6a010aabc128540.jpg)



(a) System delay.


![](images/0a063a2eddb97a1081c68b7fc2ce7d2eb2ea6f8cc7f05d023e83b892deb4ad3c.jpg)



(b) Cross tracks.


![](images/030a7ada9ad1d21815762341a0c39e299290acfc153ad24b882e971c10128490.jpg)



(c) Cross speeds.


![](images/e7eac349347f6d8cf6f9a0240009158885604e41013458927eb9b954247f6b00.jpg)



(d) Samples.


![](images/eeceb6d5821ae62f21efeb3051db6f826aad5277f9c83ef3009d2ba7a2a44015.jpg)



(e) Cross user pairs.


![](images/25bb0d1ddb5c7dfa4eff4e529b8f10793925ce794464461ffe2ee5ceed14a04d.jpg)



(f) Steps number.


![](images/16ca8d446b9f086659b737daa924870b2a6e147616722dc32c1614e3fafa36f6.jpg)



(g) Wi-Fi links.


![](images/06001a9397bf2775a18f71788e3de93ab3a0c0f850c1f7c08bfdcf6ea8da79a3.jpg)



(h) Orient. error.



Fig. 9. Experimental results of GaitID.


Transfer Learning Performance. Transfer learning needs fewer data to retrain the model. To verify the number of samples needed for convergence, we randomly selected four([A,B,C,D]) users’ data with 4 400 samples containing all the tracks and speeds. These samples are feed into a pre-trained model that has already been tuned on users [A,E]. Results can be found in Fig. 9(d). With transferred model structure and weights, only 12.5% of the data samples are needed to keep accuracy above 80%. But when training from scratch, accuracy can hardly exceed above 60%. We also evaluated the transformation between different user pairs. Figure 9(e) exhibits a consistency in generalizability between diferent user set pairs. The slight degradation of accuracy for user pair (A,B)- (C,D,E,F,G) may attributes to user G’s great similarities in gait patterns with the other users. 

## 4.4 Parameter Study

Impact of Gait Instances. To evaluate the impact of gait instances on identification accuracy, we randomly select four users and let them to perform 5 steps of walk. We then manually split steps into diferent numbers by DFS peaks and valleys. Results can be found in Fig. 9(f). The performance falls slightly from 5 steps to 2 steps but tumbles to below 78% with a single step. The reason is that a full gait cycle contains 2 consecutive steps, each of which is insuficient for representations of identity. Meanwhile, gait is a periodic motion and the repetition of gait cycles introduces trivial extra information for gait characteristics. Result from temporal memorability of LSTM in our model, GaitID is capable of retaining distinctions from single gait cycles. Hence, we claim that 2 consecutive steps are suficient for human identification. In practice, we suggest to use four steps for a more robust performance. 

Impact of Link Numbers. In the formulation of extracting GBVP for gait, we adopted 6 Wi-Fi links, which potentially contains redundancy. In this section, we evaluate the impact of link numbers to the performance. We randomly selected four users for classification and randomly prune partial links to reduce link numbers. Results can be found in Fig. 9(g).The accuracy gradually slides when involved Wi-Fi links reduced. This is because less links captures less reflection paths caused by human body, and theoretically at least 3 links are necessary to recover valid GBVP. With only 2 links, accuracy drops to below 80% and are hardly beyond research usage. 

Impact of Orientation Error. In the GBVP normalization process, we rotate GBVP to identify with the reference orientation, which demands a precise estimation of walking direction. However, orientation extracted from state-of-the-art motion tracking techniques contains prominent errors. To evaluate the impact of orientation error on human identification accuracy, we generated training and testing set by manually providing trace orientation, and added controllable orientation disturbance to testing set. Results can be found in Fig. 9(h). As the illustration shown, an orientation error within 50◦ doesn’t noticeably deteriorate accuracy, while an orientation error above 50◦ witnesses an unacceptable dilution in identification accuracy. Hence, the four tracks with eight orientations designed in our evaluation implementation are suficient to represent more complicated walking traces. 

## 5 Conclusion

In this paper, we present GaitID, a Wi-Fi based person identification framework which is robust to walking trajectory with few training eforts. GaitID first proposes an enhanced gait-specific feature, which is theoretically environment, trajectory and speed independent, and then reduces the training eforts for new users by transfer learning technique. 

Acknowledgments. This work is supported in part by the National Key Research Plan under grant No. 2016YFC0700100. 

## References



1. Halperin, D., Hu, W., Sheth, A., Wetherall, D.: Tool release: gathering 802.11n traces with channel state information. In: SIGCOMM CCR, p. 53, January 2011 





2. Jiang, W., et al.: Towards environment independent device free human activity recognition. In: Proceedings of ACM MobiCom, pp. 289–304 (2018) 





3. Han, J., Bhanu, B.: Statistical feature fusion for gait-based human recognition. In: Proceedings of IEEE CVPR, p. II (2004) 





4. Li, X., et al.: Indotrack: device-free indoor human tracking with commodity WI-FI. In: Proceedings of ACM IMWUT, pp. 72:1–72:22, September 2017 





5. Murray, M.P., Drought, A.B., Kory, R.C.: Walking patterns of normal men, March 1964 





6. Qian, K., Wu, C., Zhang, Y., Zhang, G., Yang, Z., Liu, Y.: Widar2.0: passive human tracking with a single Wi-Fi link. In: Proceedings of ACM MobiSys, pp. 350–361 (2018) 





7. Qian, K., Wu, C., Zhou, Z., Zheng, Y., Yang, Z., Liu, Y.: Inferring motion direction using commodity Wi-Fi for interactive exergames. In: Proceedings of ACM CHI (2017) 





8. Rubner, Y., Tomasi, C., Guibas, L.J.: The earth mover’s distance as a metric for image retrieval. Int. J. Comput. Vis. 40, 99–121 (2000). https://doi.org/10.1023 A:1026543900054 





9. Sak, H., Senior, A.W., Beaufays, F.: Long short-term memory based recurrent neural network architectures for large vocabulary speech recognition. CoRR abs/1402.1128 (2014) 





10. Sherstinsky, A.: Fundamentals of recurrent neural network (RNN) and long shortterm memory (LSTM) network. CoRR abs/1808.03314 (2018) 





11. Taylor, M.E., Stone, P.: Transfer learning for reinforcement learning domains: a survey. J. Mach. Learn. Res. 10, 1633–1685 (2009) 





12. Wang, W., Liu, A.X., Shahzad, M.: Gait recognition using WiFi signals. In: Proceedings of ACM IMWUT, pp. 363–373 (2016) 





13. Yang, Z., Wu, C., Liu, Y.: Locating in fingerprint space: wireless indoor localization with little human intervention. In: Proceedings of ACM MobiCom, pp. 269–280 (2012) 





14. Yang, Z., Zhou, Z., Liu, Y.: From RSSI to CSI: indoor localization via channel response. ACM Comput. Surv. 46, 25:1–25:32 (2013) 





15. Yosinski, J., Clune, J., Bengio, Y., Lipson, H.: How transferable are features in deep neural networks? CoRR abs/1411.1792 (2014) 





16. Zatsiorky, V., Werner, S., Kaimin, M.: Basic kinematics of walking: step length and step frequency: a review. J. Sports Med. Phys. Fitness 34, 109–134 (1994) 





17. Zeng, Y., Pathak, P.H., Mohapatra, P.: WiWho: WiFi-based person identification in smart spaces. In: Proceedings of ACM/IEEE IPSN, pp. 1–12 (2016) 





18. Zhang, J., Wei, B., Hu, W., Kanhere, S.S.: WiFi-id: Human identification using WiFi signal. In: Proceedings of DCOSS, pp. 75–82 (2016) 





19. Zhang, Y., Pan, G., Jia, K., Lu, M., Wang, Y., Wu, Z.: Accelerometer-based gait recognition by sparse representation of signature points with clusters. IEEE Trans. Cybern. 45, 1864–1875 (2015) 





20. Zheng, Y., et al.: Zero-efort cross-domain gesture recognition with Wi-Fi. In: Proceedings of ACM MobiSys, pp. 313–325 (2019) 

