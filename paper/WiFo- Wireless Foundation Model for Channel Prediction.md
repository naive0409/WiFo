. RESEARCH PAPER . 

# WiFo: Wireless Foundation Model for Channel Prediction

Boxun LIU1, Shijian GAO2, Xuanyu LIU1, Xiang CHENG1* & Liuqing Yang2,3 

1State Key Laboratory of Photonics and Communications, School of Electronics, Peking University, Beijing 100871, China; 2Internet of Things Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511400, China; 3Department of Electronic and Computer Engineering and Department of Civil and Environmental Engineering, The Hong Kong University of Science and Technology, Hong Kong, China 

Abstract Channel prediction permits to acquire channel state information (CSI) without signaling overhead. However, almost all existing channel prediction methods necessitate the deployment of a dedicated model to accommodate a specific configuration. Leveraging the powerful modeling and multi-task learning capabilities of foundation models, we propose the first space-timefrequency (STF) wireless foundation model (WiFo) to address time-frequency channel prediction tasks in a unified manner. Specifically, WiFo is initially pre-trained over massive and extensive diverse CSI datasets. Then, the model will be instantly used for channel prediction under various CSI configurations without any fine-tuning. We propose a masked autoencoder (MAE)- based network structure for WiFo to handle heterogeneous STF CSI data, and design several mask reconstruction tasks for selfsupervised pre-training to capture the inherent 3D variations of CSI. To fully unleash its predictive power, we build a large-scale heterogeneous simulated CSI dataset consisting of 160K CSI samples for pre-training. Simulations validate its superior unified learning performance across multiple datasets and demonstrate its state-of-the-art (SOTA) zero-shot generalization performance via comparisons with other full-shot baselines. 

Keywords channel prediction, channel state information, foundation model, self-supervised pre-training, zero-shot learning 

Citation WiFo: Wireless Foundation Model for Channel Prediction. Sci China Inf Sci, for review 

## 1 Introduction

Massive multi-input multi-output (MIMO) and orthogonal frequency division multiplexing (OFDM) have been two cornerstone technologies over the past decade [1]. The performance of MIMO-OFDM systems hinges upon accurate channel state information (CSI), typically obtained through channel estimation [2]. However, in highly-dynamic scenarios, the channel coherence time is significantly limited, resulting in a sharp increase in channel estimation overhead. Therefore, channel prediction is proposed as a promising technology to reduce the overhead of CSI acquisition, where the estimated CSI is extrapolated in time or frequency. Extensive studies have been conducted on model-based channel prediction methods, including traditional autoregressive and polynomial extrapolation methods, as well as advanced vector Prony-based prediction methods [3] and high-resolution parameter estimation schemes [4]. Nevertheless, due to the difficulty of modeling complicated practical channels, the prediction accuracy of model-based approaches is limited. 

Deep learning has emerged as a powerful tool to capture complex patterns in a data-driven manner without any prior assumption. It has been widely applied in various aspects of wireless communications [5, 6], including time-domain and frequency-domain channel prediction addressed in this paper. For instance, several neural networks, including transformer [7], ConvLSTM [8], and variational autoencoders (VAE) [9], have been explored for channel prediction to learn temporal and frequency variations. In addition, several channel prediction methods based on physics-informed deep learning have been proposed to incorporate prior knowledge of the CSI or the propagation environment [10]. Recently, pre-trained large language models (LLMs) have been applied to MIMO-OFDM channel prediction [11] for enhanced accuracy and generalizability. However, existing deep learning-based channel prediction research is still at the starting stage and awaiting several improvements. First, due to the limited model scale, existing small model-based methods are not well poised to predict complex practical spatial-temporal-frequency channels accurately. Secondly, existing deep learning-based schemes are only trained under specific CSI distributions and specific system configurations. Correspondingly, these methods require retraining when scenarios and system parameters change, which leads to significant training overhead. Moreover, separate networks are designed and deployed for time-domain and frequency-domain channel prediction, increasing the storage, computation, and management overhead at the base station (BS). 

Recently, the emergence of foundational models [12] has led to a paradigm shift in deep learning. Specifically, large-scale neural networks pre-trained on vast and diverse data in a self-supervised manner can achieve remarkable generalization capabilities across various downstream tasks, significantly outperforming task-specific models. Its powerful modeling capabilities and multi-task learning potential offer promise for addressing the limitations of existing channel prediction methods [13]. Most recently, a few studies [14, 15] apply self-supervised pre-training schemes to wireless channel representation learning for CSI-related downstream tasks, such as fingerprint localization and cross-band feature extraction. Specifically, the pre-trained model is fine-tuned for specific wireless tasks through few-shot learning. Nevertheless, these cannot be directly adopted for channel prediction. First, these schemes can only handle space-frequency two-dimensional CSI and cannot address temporal variations. Secondly, they utilize the encoder-only network instead of an autoencoder, rendering these approaches suitable for classification or regression tasks but less effective for reconstruction tasks such as channel prediction. Last but not least, they can only reduce, but not eliminate, the retraining overhead for different tasks and scenarios. 

Recognizing the limitations of existing studies, we make the pioneering attempt to apply foundation models to channel prediction tasks. Specifically, we aim to establish a large-scale wireless foundation model pre-trained on extensive CSI data, and directly apply it to various channel prediction tasks under different CSI configurations and scenarios without any fine-tuning. Evidently, building such a foundational model is quite challenging. First, it is challenging to address different types of channel prediction tasks with a single network. Secondly, it is also difficult to use a single network to simultaneously cope with diverse CSI data, where the number of antennas, sub-carriers, and time samples vary. 

In order to overcome these challenges, we propose the first wireless foundation model for channel prediction tasks, termed as WiFo. Different from all existing works, we formulate time-domain and frequency-domain channel prediction tasks as a unified channel reconstruction problem, which captures the complete CSI based on partial CSI. Inspired by the masked autoencoders (MAE) for image and video self-supervised learning, we propose an MAE-based network structure suitable for CSI reconstruction. We apply 3D patching and embedding to convert diverse CSI into varying numbers of tokens, enabling efficient processing by the transformer blocks. For both the encoder and the decoder, we propose a novel positional encoding (STF-PE) structure to learn the 3D position information associated with the CSI. To capture the inherent 3D variations, we propose three self-supervised pre-training tasks, namely random, time, and frequency-masked reconstruction. The pre-trained model can be directly applied to zero-shot inference for both channel prediction tasks. To evaluate the performance of WiFo, we construct a heterogeneous CSI dataset with 16 different configurations of time sampling, sub-carrier, and antenna, containing 160K training samples. Preliminary results validate its superior multi-dataset unified learning performance and zero-shot generalization capability. It is also worth emphasizing that its zero-shot prediction performance in unseen scenarios surpasses the full-shot performance of all baselines trained on 10K samples, completely eliminating the retraining or fine-tuning costs. 1) 

The primary contributions of our work are summarized as follows. 

• We propose the first wireless foundation model (WiFo) designed to uniformly facilitate time-domain and frequency-domain channel prediction tasks. To the best of our knowledge, it is the first versatile model capable of simultaneously tackling different types of channel prediction tasks under diverse CSI configurations. 

• We develop an MAE-based network structure to cope with heterogeneous CSI data and introduce three mask reconstruction tasks for self-supervised pre-training, aimed at capturing the inherent space time-frequency correlations of CSI. The pre-trained model can be directly utilized for inference without the need for fine-tuning. 

• We construct 16 diverse datasets with various CSI configurations using the QuaDRiGa channel generator for pre-training, consisting of 160K training samples. Simulations confirm that WiFo can effectively learn across different channel prediction tasks and heterogeneous datasets and demonstrates strong zero-shot performance in new scenarios. 

Notation: $\| \cdot \| _ { F }$ denotes the Frobenius norm. a[i] is the i-th element of a vector a and $A [ i , j ]$ denotes the element of matrix A at the i-th row and the j-th column. R and C denote the set of real numbers and complex numbers, respectively. 

## 2 System Model and Problem Formulation

In this paper, we consider a multi-input single-output (MISO)-OFDM system, where the BS and the user are equipped with a uniform planar array (UPA) and a single antenna, respectively. The UPA contains $N = N _ { h } \times N _ { v }$ elements, with $N _ { h }$ and $N _ { v }$ being the number of antennas along the horizontal and vertical directions, respectively. The proposed scheme can be directly extended to the case of multi-user channel prediction by processing multiple samples in parallel. 

## 2.1 3D Channel Model

We adopt the classical geometric channel model [16] consisting of $P$ paths. The CSI between the BS and the user sampled at frequency $f$ and time t can be expressed as 

$$
\boldsymbol {h} (t, f) = \sum_ {p = 1} ^ {P} g _ {p} \boldsymbol {a} (\phi_ {p}, \theta_ {p}) e ^ {- j 2 \pi f \tau_ {p}} e ^ {j 2 \pi \nu_ {p} t}, \tag {1}
$$

where $g _ { p } , \tau _ { p } ,$ , and $\nu _ { p }$ represent the complex amplitude, the delay, and the Doppler shift associated with the p-th path. $\pmb { a } ( \phi _ { p } , \theta _ { p } ) \in \mathbb { C } ^ { N \times 1 }$ is the steering vector of UPA [11], where $\phi _ { p }$ and $\theta _ { p }$ denote the azimuth and elevation angles, respectively. 

Assume that the considered time-frequency region [11] spans T and K resource blocks (RBs) along the time and frequency dimensions, respectively. The pilot placement pattern is identical for all antennas and each RB contains a pilot. The pilot intervals along time and frequency are $\Delta t$ and $\Delta f _ { : }$ , respectively. Without loss of generality, we consider the CSI only at the pilot positions. The considered space-timefrequency CSI is denoted as $\pmb { H } \in \mathbb { C } ^ { T \times K \times N }$ , which satisfies 

$$
\boldsymbol {H} [ i, j,: ] = \boldsymbol {h} (i \Delta t, f _ {1} + (j - 1) \Delta f), \quad i = 1, \dots , T, \quad j = 1, \dots , K, \tag {2}
$$

where $f _ { 1 }$ represents the frequency at the pilot position of the first RB in the frequency domain. 

## 2.2 Problem Description

We consider two types of channel prediction tasks: time-domain prediction and frequency-domain prediction. 

1) Time-domain channel prediction: We aim to predict CSI for the future $T - T _ { h }$ RBs based on the historical $T _ { h }$ RBs along the time dimension. Denote the mapping function of time-domain channel prediction as $\Phi _ { \mathrm { t } }$ , the prediction process is derived as 

$$
\boldsymbol {H} \left[ T _ {h} + 1: T, :,: \right] = \Phi_ {\mathrm{t}} \left(\boldsymbol {H} \left[ 1: T _ {h},:,: \right]\right). \tag {3}
$$

2) Frequency-domain channel prediction: We focus on channel prediction for adjacent frequency bands. Without loss of generality, we aim to predict the last $K - K _ { \tau }$ u RBs via the first $K _ { u }$ RBs along the frequency dimension. Such an operation can be represented as 

$$
\boldsymbol {H} [:, K _ {u} + 1: K,: ] = \Phi_ {\mathrm{f}} (\boldsymbol {H} [:, 1: K _ {u},: ]), \tag {4}
$$

where $\Phi _ { \mathrm { f } }$ is the mapping function in the frequency-domain. 

Previous studies have designed separate networks to manage individual prediction tasks and various CSI configurations. However, since a BS must simultaneously handle multiple channel prediction tasks and diverse user configurations, this approach leads to significant overhead in network deployment. To address this issue, we propose to unify the channel prediction tasks into a single channel reconstruction task using our wireless foundation model. Specifically, the formulated channel reconstruction task seeks to derive the complete CSI from partial CSI. We denote the partial CSI as $H [ \Omega ]$ , where Ω represents the subset of all elements. Our goal is to develop a reconstruction function $\Phi _ { \mathrm { r e c } }$ that facilitates a universal mapping as follows: 

$$
\boldsymbol {H} = \Phi_ {\mathrm{rec}} (\boldsymbol {H} [ \Omega ]), \tag {5}
$$

![](images/1934ee4cbfe1131f28e684c94e6b76f01c91d42e15f2da5ba2367ae30ac2fddc.jpg)



Figure 1 An illustration of the proposed WiFo: the original space-time-frequency 3D CSI is firstly divided into 3D patches and masked, with the visible patches embedded as CSI tokens. The WiFo encoder processes the input tokens, and after concatenating the learnable mask tokens, the WiFo decoder reconstructs the original 3D CSI.


where Ω represents the temporal or frequency subset for time-domain and frequency-domain channel prediction, respectively. It is worth noting that the function $\Phi _ { \mathrm { r e c } }$ needs to handle diverse CSI data with arbitrary T , K, and N, while simultaneously performing reconstruction in the time or frequency domain. However, traditional deep learning-based channel prediction schemes and LLM-empowered schemes [11] are designed for fixed-size CSI and specific prediction tasks, which limits their ability to achieve this universal mapping. For instance, most existing deep-learning-based prediction schemes design neural networks under fixed input dimensions, where the adopted fully connected layers are incapable of handling a varying input size. Consequently, we aim to establish a foundation model to address this challenge. 

## 3 Wireless Foundation Model

In this section, a novel wireless foundation model, termed WiFo, is proposed to realize the universal channel prediction. First, the network structure of WiFo is introduced based on masked autoencoders (MAE). Then, a self-supervised pre-training scheme is proposed based on multiple reconstruction tasks. Finally, the pre-trained WiFo is directly applied for channel prediction without any fine-tuning. 

## 3.1 Network Structure

MAE [17–19] has been demonstrated to be a data-efficient self-supervised pre-training framework for image and video representation learning. It adopts an asymmetric encoder-decoder architecture, where an encoder operates on unmasked tokens and a lightweight decoder recovers the original image or video. Inspired by MAE, WiFo adopts a transformer-based encoder-decoder network structure, as shown in Fig. 1. It consists of four major blocks, namely CSI embedding, masking, encoder, and decoder. Each of them will be described in detail below. 

## 3.1.1 CSI Embedding

To facilitate neural network processing, the complex H is first converted into a real-valued tensor $\tilde { H } \in$ $\mathbb { R } ^ { 2 \times T \times K \times N }$ , which consists of two channels for the real and imaginary parts. To convert H˜ into 1D sequential data suitable for transformer processing, we apply 3D patching [18] to the last three dimensions. Specifically, the size of each 3D patch is $( t , k , n )$ , where t, k, and n represent the patch size along the time, frequency, and space dimensions. Then the non-overlapping 3D patches, with a total number of $\begin{array} { r } { L = \frac { T } { t } \times \frac { \mathbf { \dot { K } } } { k } \times \frac { N } { n } } \end{array}$ , are flattened and embedded into a series of tokens with dimension $D _ { \mathrm { e n c } }$ , where $D _ { \mathrm { e n c } }$ is the hidden size of the encoder. The above operation can be implemented using 3D convolution, i.e., 

$$
\boldsymbol {H} _ {\text { conv }} = \operatorname{Conv3d} (\tilde {\boldsymbol {H}}), \tag {6}
$$

where Conv3d(·) represents the 3D convolution operator with 2 input channels and $D _ { \mathrm { e n c } }$ output channels. Then the convolution result $\begin{array} { r } { H _ { \mathrm { c o n v } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times \frac { T } { t } \times \frac { K } { k } \times \frac { N } { n } } } \end{array}$ is flattened into $H _ { \mathrm { e m b } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times L }$ as the CSI tokens. 

## 3.1.2 Masking

As shown in Fig. 1, several 3D patches are masked. It is equivalent to mask partial tokens of $H _ { \mathrm { e m b } }$ and retains only a subset of tokens to be processed by the encoder module. Denote the visible tokens and the masked tokens as $H _ { \mathrm { v i s } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times L _ { \mathrm { v i s } } }$ and $H _ { \mathrm { m a s k } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times ( L - L _ { \mathrm { v i s } } ) }$ , respectively, where $L _ { \mathrm { v i s } }$ is the number of visible tokens. Then the masking process is represented as 

![](images/bd06ec2476157b888d3cfe91ad36fe08895f7357618702c7d21c19cc178b7373.jpg)



Figure 2 An illustration of the architecture of the encoder module.


$$
\left[ \boldsymbol {H} _ {\text { vis }}, \boldsymbol {H} _ {\text { mask }} \right] = M a s k (\boldsymbol {H} _ {\text { emb }}), \tag {7}
$$

where $\mathit { M a s k } ( \cdot )$ represents the masking operation with a certain masking strategy and masking ratio. 

The role of the masking operation differs between the model pre-training and inference stages. During the pre-training stage, masking strategies are performed as self-supervised pre-training reconstruction tasks for better CSI representation learning. During the inference stage, masking strategies are performed to apply the proposed model to specific channel prediction tasks. 

## 3.1.3 Encoder

The architecture of the encoder is shown in Fig. 2. Consistent with MAE [17–19], the visible tokens are sequentially added with positional encoding and processed by a series of transformer blocks implemented by vanilla Vision Transformer (ViT) backbone [20]. Denote the operation of the transformer blocks as $f _ { \mathrm { e n c } } ,$ the encoder output $H _ { \mathrm { e n c } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times L _ { \mathrm { v i s } } }$ is derived as 

$$
\boldsymbol {H} _ {\text { enc }} = f _ {\text { enc }} (\boldsymbol {H} _ {\text { vis }} + \boldsymbol {P} _ {\text { enc }}), \tag {8}
$$

where $P _ { \mathrm { e n c } } \in \mathbb { R } ^ { D _ { \mathrm { e n c } } \times L _ { \mathrm { v i s } } }$ denotes the positional encoding. 

To enable the model to learn the three-dimensional position information of CSI, we propose an STF positional encoding (STF-PE). Specifically, separate positional encoding for time, frequency, and space, $P _ { \mathrm { e n c } } ^ { \mathrm { t } } \in \mathbb { R } ^ { L _ { \mathrm { v i s } } \times D _ { \mathrm { e n c } } ^ { \mathrm { t } } } , \ P _ { \mathrm { e n c } } ^ { \mathrm { f } } \in \mathbb { R } ^ { L _ { \mathrm { v i s } } \times D _ { \mathrm { e n c } } ^ { \mathrm { f } } }$ $P _ { \mathrm { e n c } } ^ { \mathrm { s } } \in \mathbb { R } ^ { L _ { \mathrm { v i s } } \times D _ { \mathrm { e n c } } ^ { \mathrm { s } } }$ feature dimension, satisfying $D _ { \mathrm { e n c } } = D _ { \mathrm { e n c } } ^ { \mathrm { t } } + D _ { \mathrm { e n c } } ^ { \mathrm { f } } + D _ { \mathrm { e n c } } ^ { \mathrm { s } }$ . Without loss of generality, let $D _ { \mathrm { e n c } } ^ { \mathrm { t } } = D _ { \mathrm { e n c } } ^ { \mathrm { f } } =$ $\lfloor D _ { \mathrm { e n c } } / 3 \rfloor$ and $D _ { \mathrm { e n c } } ^ { \mathrm { s } } = D _ { \mathrm { e n c } } - 2 \lfloor D _ { \mathrm { e n c } } / 3 \rfloor$ , respectively. For better generalization across different sizes of CSI, each separate positional encoding of STF-PE adopts absolute SinCos positional encoding [17] instead of learnable encoding [19]. As an example, consider the time positional encoding. For i-th visible token, denote the temporal, spatial, and frequency coordinates of its corresponding 3D patch as $( p o s _ { i } ^ { \mathrm { t } } , p o s _ { i } ^ { \mathrm { f } } , p o s _ { i } ^ { \mathrm { s } } )$ , we have: 

$$
\boldsymbol {P} _ {\mathrm{enc}} ^ {\mathrm{t}} [ i, 2 j ] = \sin \frac {\operatorname{pos} _ {i} ^ {\mathrm{t}}}{1 0 0 0 0 ^ {\frac {2 j}{D _ {\mathrm{enc}} ^ {\mathrm{t}}}}}, \text {and} \boldsymbol {P} _ {\mathrm{enc}} ^ {\mathrm{t}} [ i, 2 j + 1 ] = \cos \frac {\operatorname{pos} _ {i} ^ {\mathrm{t}}}{1 0 0 0 0 ^ {\frac {2 j}{D _ {\mathrm{enc}} ^ {\mathrm{t}}}}}. \tag {9}
$$

## 3.1.4 Decoder

The decoder is designed to reconstruct the original H from the encoder output. $H _ { \mathrm { e n c } }$ is first converted to $\bar { \cal H } _ { \mathrm { e n c } } \in \mathbb { R } ^ { D _ { \mathrm { d e c } } \times L _ { \mathrm { v i } } }$ is via a fully connected layer to align with the feature dimension $D _ { \mathrm { d e c } }$ of the decoder transformer blocks. $\bar { H } _ { \mathrm { e n c } }$ is then concatenated with learnable mask tokens $M \in \mathbb { R } ^ { D _ { \mathrm { d e c } } \times ( L - L _ { \mathrm { v i s } } ) }$ , and added with decoder positional encoding $P _ { \mathrm { d e c } } \in \mathbb { R } ^ { D _ { \mathrm { d e c } } \times L }$ before processed by lightweight ViT transformer blocks, denoted as $f _ { \mathrm { d e c } } . \ P _ { \mathrm { d e c } }$ adopts the same STF-PE as the encoder. Denote the output of the decoder transformer blocks as $H _ { \mathrm { d e c } } \in \mathbb { R } ^ { D _ { \mathrm { d e c } } \times L }$ , and we have 

$$
\boldsymbol {H} _ {\mathrm{dec}} = f _ {\mathrm{dec}} \left(\left[ \boldsymbol {H} _ {\text { vis }}, \boldsymbol {M} \right] + \boldsymbol {P} _ {\mathrm{dec}}\right). \tag {10}
$$

![](images/2deb69ec6628ea53407e7e00b8cf84d30d2946e4efdc39af604882ef2dce8978.jpg)



Random-masked reconstruction


![](images/6cb54f77b69b460b77d51c6f46c2abbad2f74e7e3e0b584e1735671161daaedd.jpg)



Time-masked reconstruction


![](images/c665def7dfffc65f0173986587f58a405b736602f3cf02eabaa41a1eee230a01.jpg)



Frequency-masked reconstruction



Figure 3 An illustration of the proposed masked reconstruction tasks.


Then, $H _ { \mathrm { d e c } }$ is transformed into $\tilde { H } _ { \mathrm { p r e d } } \in \mathbb { R } ^ { 2 \times T \times K \times N }$ by a fully connected layer and reshape operation. Finally, we obtain the reconstructed complex CSI $H _ { \mathrm { p r e d } } \in \mathbb { C } ^ { T \times K \times N }$ , i.e., 

$$
\boldsymbol {H} _ {\text { pred }} = \tilde {\boldsymbol {H}} _ {\text { pred }} [ 1,:,: ] + 1 j \times \tilde {\boldsymbol {H}} _ {\text { pred }} [ 2,:,: ]. \tag {11}
$$

## 3.2 Self-Supervised Pre-Training

Unlike existing channel and prediction schemes [7–9,11] which are trained and tested on a specific dataset, the proposed WiFo is first pre-trained across multiple heterogeneous datasets and then directly applied to unseen scenarios and new CSI configurations. Specifically, several self-supervised training tasks are designed to capture the intricate inherent space-time-frequency correlations of CSI. Masked reconstruction [17–19] has been proven to be an effective pre-training task for downstream visual tasks, which masks random patches of an image and then reconstructs the complete image. Notably, temporal-domain and frequency-domain channel prediction are special types of masked reconstruction tasks, where patches are masked along the time or frequency dimensions. Therefore, we propose three masked reconstruction tasks to capture the intricate inherent space-time-frequency correlations of CSI, as shown in Fig. 3. Their detailed operations are introduced as follows. To simplify the description, we denote the temporal, spatial, and frequency coordinates of the 3D patch corresponding to the i-th token in $H _ { \mathrm { e m b } }$ as $( \overline { { p o s } } _ { i } ^ { \mathrm { t } } , \overline { { p o s } } _ { i } ^ { \mathrm { f } } , \overline { { p o s } } _ { i } ^ { \mathrm { s } } )$ . 

1) Random-masked reconstruction: It randomly masks tokens from all tokens with a R ratio. This masking strategy is isotropic across the space-time-frequency dimensions to capture the 3D structured features. 

2) Time-masked reconstruction: It is designed to enhance the time-domain channel prediction task. In this masking strategy, future tokens are masked with a certain ratio $R _ { \mathrm { t } } , \mathrm { i . e . }$ ., all tokens with post $\begin{array} { r } { \geqslant \left( { \frac { T } { t } } - \left\lfloor R _ { \mathrm { t } } { \frac { T } { t } } \right\rfloor \right) } \end{array}$ are masked. It helps the model learn the causal relationships of CSI over time. 

3) Frequency-masked reconstruction: This task is designed to improve Wifo’s performance in frequency-domain channel prediction. Unlike the time-masked reconstruction, frequency masking strategy masks tokens along frequency dimension with a ratio $R _ { \mathrm { f } }$ , so all tokens with $\begin{array} { r } { \bar { p o s } _ { i } ^ { \mathrm { f } } \geqslant \bar { ( \frac { K } { k } - \lfloor \bar { R _ { \mathrm { f } } } \frac { K } { k } \rfloor ) } } \end{array}$ are masked. This helps the model learn the variations between adjacent frequency bands. 

During self-supervised pre-training, the three pre-training tasks mentioned above are executed sequentially for each batch. Our objective is to minimize the reconstruction error, for which we adopt the mean squared error (MSE) as the loss function. The loss is computed using only the reconstructed CSI points and is defined as 

$$
\mathcal {L} = \frac {1}{| \omega |} \| \boldsymbol {H} [ \omega ] - \boldsymbol {H} _ {\text { pred }} [ \omega ] \| _ {F} ^ {2}, \tag {12}
$$

where ω represents the temporal or frequency predicted CSI subset. 

## 3.3 Model Inference

Once the pre-training stage is complete, the pre-trained WiFo can perform zero-shot inference on both the time-domain and frequency-domain tasks, as shown in Eq. 5. Specifically, the portion to be predicted is first padded with zeros, and then the zero-padded CSI is fed into WiFo. Subsequently, the corresponding time-masked or frequency-masked strategy is applied with a certain masking ratio, allowing the network to output the reconstructed part as the predicted CSI. Below, we will elaborate on the process in detail, using time-domain channel prediction as an example. 


Table 1 An illustration of the system configurations of the constructed 3D CSI datasets.


<table><tr><td>Dataset</td><td><eq>f_C</eq>(GHz)</td><td>K</td><td>Δf(kHz)</td><td>T</td><td>Δt(ms)</td><td>UPA</td><td>Scenario</td><td>User speed(km/h)</td></tr><tr><td>D1</td><td>1.5</td><td>128</td><td>90</td><td>24</td><td>1</td><td>1 × 4</td><td>UMi+NLoS</td><td>3-50</td></tr><tr><td>D2</td><td>1.5</td><td>128</td><td>180</td><td>24</td><td>0.5</td><td>2 × 4</td><td>RMa+NLoS</td><td>120-300</td></tr><tr><td>D3</td><td>1.5</td><td>64</td><td>90</td><td>16</td><td>1</td><td>1 × 8</td><td>Indoor+LoS</td><td>0-10</td></tr><tr><td>D4</td><td>1.5</td><td>32</td><td>180</td><td>16</td><td>0.5</td><td>4 × 8</td><td>UMa+LoS</td><td>30-100</td></tr><tr><td>D5</td><td>2.5</td><td>64</td><td>180</td><td>24</td><td>0.5</td><td>2 × 2</td><td>RMa+NLoS</td><td>120-300</td></tr><tr><td>D6</td><td>2.5</td><td>128</td><td>90</td><td>24</td><td>1</td><td>2 × 4</td><td>UMi+LoS</td><td>3-50</td></tr><tr><td>D7</td><td>2.5</td><td>32</td><td>360</td><td>16</td><td>0.5</td><td>4 × 8</td><td>UMa+LoS</td><td>30-100</td></tr><tr><td>D8</td><td>2.5</td><td>64</td><td>90</td><td>16</td><td>1</td><td>4 × 4</td><td>Indoor+NLoS</td><td>0-10</td></tr><tr><td>D9</td><td>4.9</td><td>128</td><td>180</td><td>24</td><td>1</td><td>1 × 4</td><td>UMi+NLoS</td><td>3-50</td></tr><tr><td>D10</td><td>4.9</td><td>64</td><td>180</td><td>24</td><td>0.5</td><td>2 × 4</td><td>RMa+LoS</td><td>120-300</td></tr><tr><td>D11</td><td>4.9</td><td>64</td><td>90</td><td>16</td><td>0.5</td><td>4 × 4</td><td>UMa+NLoS</td><td>30-100</td></tr><tr><td>D12</td><td>4.9</td><td>32</td><td>180</td><td>16</td><td>1</td><td>4 × 8</td><td>Indoor+LoS</td><td>0-10</td></tr><tr><td>D13</td><td>5.9</td><td>64</td><td>90</td><td>24</td><td>0.5</td><td>2 × 8</td><td>RMa+LoS</td><td>120-300</td></tr><tr><td>D14</td><td>5.9</td><td>128</td><td>180</td><td>24</td><td>1</td><td>2 × 4</td><td>UMi+NLoS</td><td>3-50</td></tr><tr><td>D15</td><td>5.9</td><td>64</td><td>90</td><td>16</td><td>1</td><td>4 × 4</td><td>Indoor+LoS</td><td>0-10</td></tr><tr><td>D16</td><td>5.9</td><td>32</td><td>360</td><td>16</td><td>0.5</td><td>4 × 8</td><td>UMa+NLoS</td><td>30-100</td></tr><tr><td>D17</td><td>3.5</td><td>32</td><td>180</td><td>16</td><td>0.5</td><td>4 × 8</td><td>UMa+NLoS</td><td>30-100</td></tr><tr><td>D18</td><td>6.7</td><td>64</td><td>180</td><td>24</td><td>1</td><td>4 × 4</td><td>UMi+LoS</td><td>3-50</td></tr><tr><td>D19</td><td>28</td><td>32</td><td>360</td><td>16</td><td>0.25</td><td>4 × 8</td><td>UMa+LoS</td><td>30-100</td></tr></table>

As shown in Eq. 3, we aim to predict future CSI $H [ T _ { h } + 1 : T , : , : ]$ based on historical CSI $H [ 1 : T _ { h } , : , : ]$ . First, the historical CSI is zero-padded into $\pmb { H } _ { \mathrm { i n } } \in \mathbb { C } ^ { \tilde { T } \times K \times N }$ to be fed into WiFo, satisfying 

$$
\boldsymbol {H} _ {\text { in }} [ 1: T _ {h},:, ] = \boldsymbol {H} [ 1: T _ {h},:, ] \text {   and   } \boldsymbol {H} _ {\text { in }} [ T _ {h} + 1: T,:, ] = \boldsymbol {0} _ {(T - T _ {h}) \times K \times N}, \tag {13}
$$

where ${ \bf 0 } _ { ( T - T _ { h } ) \times K \times N }$ represents a all-zero tensor of dimensional $( T - T _ { h } ) \times K \times N$ . Then the time-domain masking is applied, i.e., all tokens with $\overline { { p o s } } _ { i } ^ { \mathrm { t } } \geqslant T _ { h } + 1$ are masked. By denoting the reconstructed CSI output of WiFo as $H _ { \mathrm { o u t } }$ , the predicted future CSI can be obtained as $H _ { \mathrm { p r e } } = H _ { \mathrm { o u t } } [ T _ { h } + 1 : T , : , : ]$ . 

## 4 Experiments

In this section, we conduct extensive experiments to demonstrate the effectiveness of the proposed WiFo. First, we built a series of simulated datasets covering various system configurations for network pretraining and inference. Then, experimental setups are introduced, including network and training parameters, baselines, and the performance metric. Finally, the channel prediction performance of the proposed WiFo is comprehensively evaluated. 

## 4.1 Datasets

To fully unleash WiFo’s prediction capabilities across various CSI configurations, we have constructed a series of diverse 3D CSI datasets, generated through channel generator QuaDRiGa [21] compliant with the 3GPP standards. Consistent with our system model in Section 2.1, we consider MISO-OFDM systems, where the BS is equipped with a UPA and the user has a single antenna. The adjacent antenna spacing is half the wavelength at the central frequency. A total of 19 datasets are simulated, indexed from D1 to D19, covering various space-time-frequency CSI configurations, cell scenarios, and user speed. Among them, the first 16 datasets are used for pre-training, and the last three datasets are used for generalization testing. The detailed simulation configurations of each dataset are shown in Table 1, where $f _ { \mathrm { C } }$ represents the center frequency. There are seven 5G New Radio (NR) frequency bands, eight 3GPP [22] scenarios, and seven user speed ranges considered. For each CSI sample, the user has a random initial position and a straight-line motion trajectory, where the speed is uniformly selected within the corresponding speed range. Each dataset contains 12000 samples, which are randomly split into 9000, 1000, and 2000 samples for training, validation, and inference, respectively. All CSI samples are pre-standardized using the mean and variance of the corresponding dataset. To simulate the imperfect factors of practical CSI acquisition, complex Gaussian noise with 20 dB is added to the CSI samples during both the training and inference process. 


Table 2 Network parameters of WiFo with different sizes.


<table><tr><td>Model</td><td>Enc. depth</td><td>Enc. width</td><td>Enc. heads</td><td>Dec. depth</td><td>Dec. width</td><td>Dec. heads</td><td>Parameters</td></tr><tr><td>WiFo-Tiny</td><td>6</td><td>64</td><td>8</td><td>4</td><td>64</td><td>8</td><td>0.3M</td></tr><tr><td>WiFo-Little</td><td>6</td><td>128</td><td>8</td><td>4</td><td>128</td><td>8</td><td>1.4M</td></tr><tr><td>WiFo-Small</td><td>6</td><td>256</td><td>8</td><td>4</td><td>256</td><td>8</td><td>5.5M</td></tr><tr><td>WiFo-Base</td><td>6</td><td>512</td><td>8</td><td>4</td><td>512</td><td>8</td><td>21.6M</td></tr><tr><td>WiFo-Large</td><td>8</td><td>768</td><td>8</td><td>4</td><td>768</td><td>8</td><td>86.1M</td></tr></table>


Table 3 Pre-training parameters of WiFo.


<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW (<eq>\beta_1 = 0.9, \beta_1 = 0.999</eq>, weight decay=0.05)</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Epochs</td><td>200</td></tr><tr><td>Learning rate schedule</td><td>Cosine decay [25] (warmup epochs = 5)</td></tr><tr><td>Base learning rate</td><td><eq>5 \times 10^{-4}</eq></td></tr></table>

## 4.2 Experimental Settings

We consider both the time-domain and the frequency-domain channel prediction tasks. For time-domain channel prediction, we predict the CSI of the future $\%$ RBs according to historical $\textstyle { \frac { T } { 2 } }$ RBs along the time dimension. For frequency-domain channel prediction, we predict the CSI of the last $\begin{array} { l } { { \frac { K } { 2 } } } \end{array}$ first $\frac { K } { 2 }$ RBs as input along the frequency dimension. 

## 4.2.1 Network and Pre-training Settings

To investigate the impact of model size on performance, we consider WiFo of 5 different sizes, with the specific parameters listed in Table 2. In the experiments, we set the size of each 3D patch as (4, 4, 4). In addition, we set the masking ratio of random masking, time-domain masking, and frequency-domain masking as $R _ { \mathrm { r } } = 8 5 \% , R _ { \mathrm { t } } = 5 0 \%$ , and $R _ { \mathrm { f } } = 5 0 \%$ , respectively. 

We conduct all experiments on the same machine with 4 NVIDIA GeForce RTX4090 GPUs, AMD EPYC 7763 64-Core CPU, and 256 GB of RAM. WiFo and other baselines are trained with TF32 (TensorFloat 32) precision. The pre-training settings are illustrated in Table 3. During the pre-training process, the 16 datasets are split into batches with a certain batch size and shuffled. For each batch, the proposed three masked reconstruction tasks are applied sequentially. The final loss value for gradient descent is the mean loss of the three reconstruction tasks. 

## 4.2.2 Baselines

For a comprehensive comparison, we provide the following five baselines, covering model-based, traditional deep learning-based, and advanced LLM-powered methods. 

1) PAD [3]: PAD is a Prony-based angular-delay domain channel prediction method. It is only applied to time-domain prediction tasks. In our experiments, the predictor order is set as $N = 4$ and 6 for the case of $T = 1 6$ and 24, respectively. 

2) Transformer [7]: The transformer-based prediction scheme is proposed in [7] for parallel channel prediction. For a fair comparison, it adopts the same CSI embedding method as WiFo. Specifically, 3D patching and embedding are first applied to the original CSI, and the embedded tokens are processed by the network. The feature dimension is set as 128, while the depth of the encoder and decoder are set as 5 and 8, respectively. 

3) LSTM [24]: Long short-term memory (LSTM) is proposed for sequential processing to overcome the vanishing gradient problem. In our experiments, we consider a two-layer LSTM. For time-domain channel prediction, antenna and frequency dimensions are flattened and input into the network. For frequency-domain channel prediction, time and antenna dimensions are flattened similarly. 


Table 4 The NMSE performance of WiFo-Base and other baselines on the time-domain channel prediction task across the D1-D16 datasets. The best results are highlighted in bold, while the second-best results are underlined.


<table><tr><td>Dataset</td><td>WiFi-Base</td><td>Transformer</td><td>LSTM</td><td>3D ResNet</td><td>PAD</td><td>LLM4CP</td><td>LLM4CP*</td></tr><tr><td>D1</td><td>0.082</td><td>0.112</td><td>0.356</td><td>0.088</td><td>0.529</td><td>0.117</td><td>0.074</td></tr><tr><td>D2</td><td>0.260</td><td>0.416</td><td>0.797</td><td>0.351</td><td>1.074</td><td>0.451</td><td>0.305</td></tr><tr><td>D3</td><td>0.016</td><td>0.016</td><td>0.027</td><td>0.014</td><td>0.038</td><td>0.015</td><td>0.013</td></tr><tr><td>D4</td><td>0.048</td><td>0.107</td><td>0.418</td><td>0.055</td><td>0.317</td><td>0.106</td><td>0.060</td></tr><tr><td>D5</td><td>0.494</td><td>0.638</td><td>0.788</td><td>0.751</td><td>5.008</td><td>0.637</td><td>0.510</td></tr><tr><td>D6</td><td>0.095</td><td>0.174</td><td>0.542</td><td>0.157</td><td>0.568</td><td>0.206</td><td>0.133</td></tr><tr><td>D7</td><td>0.081</td><td>0.219</td><td>0.576</td><td>0.103</td><td>0.617</td><td>0.198</td><td>0.112</td></tr><tr><td>D8</td><td>0.018</td><td>0.024</td><td>0.092</td><td>0.016</td><td>0.073</td><td>0.025</td><td>0.016</td></tr><tr><td>D9</td><td>0.347</td><td>0.483</td><td>0.835</td><td>0.349</td><td>1.087</td><td>0.475</td><td>0.312</td></tr><tr><td>D10</td><td>0.467</td><td>0.649</td><td>0.689</td><td>0.869</td><td>3.863</td><td>0.709</td><td>0.563</td></tr><tr><td>D11</td><td>0.227</td><td>0.440</td><td>0.834</td><td>0.274</td><td>1.017</td><td>0.405</td><td>0.273</td></tr><tr><td>D12</td><td>0.023</td><td>0.035</td><td>0.166</td><td>0.025</td><td>0.132</td><td>0.035</td><td>0.026</td></tr><tr><td>D13</td><td>0.482</td><td>0.718</td><td>0.876</td><td>0.815</td><td>5.213</td><td>0.758</td><td>0.648</td></tr><tr><td>D14</td><td>0.369</td><td>0.546</td><td>0.884</td><td>0.388</td><td>1.021</td><td>0.562</td><td>0.358</td></tr><tr><td>D15</td><td>0.029</td><td>0.039</td><td>0.156</td><td>0.032</td><td>0.151</td><td>0.038</td><td>0.030</td></tr><tr><td>D16</td><td>0.318</td><td>0.591</td><td>0.944</td><td>0.329</td><td>1.034</td><td>0.545</td><td>0.349</td></tr><tr><td>Average</td><td>0.210</td><td>0.325</td><td>0.561</td><td>0.289</td><td>1.359</td><td>0.330</td><td>0.236</td></tr></table>

4) 3D ResNet [23]: As a network specifically designed for handling 3D data, a ResNet-style model [23] proposed for video recognition tasks is considered for comparison. It consists of 50 weighted layers to capture the three-dimensional relationships within the CSI. 

5) LLM4CP [11]: LLM4CP is a LLM-empowered channel prediction scheme, where GPT-2 is finetuned for cross-modality knowledge transfer. Since the original LLM4CP method cannot directly handle 3D CSI, we consider two implementation approaches. To ensure a fair comparison, the first approach considers the same 3D patching method as WiFo, termed LLM4CP. In addition, the second implementation approach adopts antenna parallel processing [11], termed LLM4CP*. 

## 4.2.3 Performance Metric

In our experiments, normalized mean square error (NMSE) is adopted to measure the prediction accuracy directly. Let $H _ { \mathrm { P } }$ and $H _ { \mathrm { G T } }$ denote the predicted part of the CSI and its corresponding ground truth, respectively. The performance metric NMSE is derived as 

$$
\mathrm{NMSE} (\boldsymbol {H} _ {\mathrm{P}}, \boldsymbol {H} _ {\mathrm{GT}}) = \frac {\| \boldsymbol {H} _ {\mathrm{P}} - \boldsymbol {H} _ {\mathrm{GT}} \| _ {F} ^ {2}}{\| \boldsymbol {H} _ {\mathrm{GT}} \| _ {F} ^ {2}}. \tag {14}
$$

## 4.3 Performance Evaluation

## 4.3.1 Multi-Dataset Unified Learning

To validate the multi-dataset unified learning capability, we first evaluate the time-domain and frequencydomain prediction performance of WiFo across the 16 pre-training datasets. WiFo is pre-trained on all training and validation samples of Dataset D1 to D16, with a total of 160k diverse CSI samples. Then, the pre-trained WiFo is evaluated on test samples of these 16 datasets. In contrast, for deep learning-based approaches, the network is individually trained and tested on each dataset for either time-domain or frequency-domain prediction tasks. The NMSE performance on the time-domain and frequency-domain task is illustrated in Table 4 and Table 5, respectively. Due to space limitations, only the performance of the base version of WiFo on each dataset is presented here, and the impact of the model size on performance is analyzed in Section 4.3.4. For both the time-domain and frequency-domain prediction tasks, WiFo achieves the SOTA average NMSE performance and shows the best or the second-best prediction performance on most datasets. The results show that WiFo can effectively perform unified learning across multiple datasets with different CSI configurations, while simultaneously mastering both time-domain and frequency-domain prediction capabilities. 


Table 5 The NMSE performance of WiFo-Base and other baselines on the frequency-domain channel prediction task across the D1-D16 datasets. The best results are highlighted in bold, while the second-best results are underlined.


<table><tr><td>Dataset</td><td>WiFi-Base</td><td>Transformer</td><td>LSTM</td><td>3D ResNet</td><td>LLM4CP</td><td>LLM4CP*</td></tr><tr><td>D1</td><td>0.318</td><td>0.532</td><td>0.705</td><td>0.839</td><td>0.392</td><td><eq>\underline{0.375}</eq></td></tr><tr><td>D2</td><td>0.181</td><td>0.556</td><td>0.763</td><td>0.647</td><td>0.419</td><td><eq>\underline{0.223}</eq></td></tr><tr><td>D3</td><td>0.027</td><td>0.016</td><td>0.037</td><td>0.071</td><td><eq>\underline{0.023}</eq></td><td>0.025</td></tr><tr><td>D4</td><td>0.073</td><td>0.270</td><td>0.475</td><td>0.215</td><td>0.211</td><td><eq>\underline{0.151}</eq></td></tr><tr><td>D5</td><td>0.152</td><td>0.315</td><td>0.577</td><td>0.386</td><td>0.267</td><td><eq>\underline{0.165}</eq></td></tr><tr><td>D6</td><td>0.081</td><td>0.310</td><td>0.540</td><td>0.458</td><td>0.193</td><td><eq>\underline{0.140}</eq></td></tr><tr><td>D7</td><td>0.092</td><td>0.392</td><td>0.578</td><td>0.354</td><td>0.318</td><td><eq>\underline{0.189}</eq></td></tr><tr><td>D8</td><td>0.061</td><td>0.024</td><td>0.348</td><td>0.139</td><td><eq>\underline{0.068}</eq></td><td>0.069</td></tr><tr><td>D9</td><td><eq>\underline{0.436}</eq></td><td>0.481</td><td>0.895</td><td>0.918</td><td>0.574</td><td>0.418</td></tr><tr><td>D10</td><td>0.087</td><td>0.261</td><td>0.451</td><td>0.257</td><td>0.163</td><td><eq>\underline{0.096}</eq></td></tr><tr><td>D11</td><td>0.245</td><td>0.723</td><td>0.859</td><td>0.823</td><td>0.621</td><td><eq>\underline{0.349}</eq></td></tr><tr><td>D12</td><td>0.023</td><td>0.048</td><td>0.131</td><td>0.029</td><td>0.032</td><td><eq>\underline{0.026}</eq></td></tr><tr><td>D13</td><td><eq>\underline{0.068}</eq></td><td>0.238</td><td>0.531</td><td>0.177</td><td>0.165</td><td>0.067</td></tr><tr><td>D14</td><td>0.395</td><td>0.744</td><td>0.911</td><td>0.924</td><td>0.637</td><td><eq>\underline{0.414}</eq></td></tr><tr><td>D15</td><td>0.023</td><td>0.053</td><td>0.083</td><td>0.045</td><td><eq>\underline{0.024}</eq></td><td><eq>\underline{0.024}</eq></td></tr><tr><td>D16</td><td>0.270</td><td>0.855</td><td>0.929</td><td>0.723</td><td>0.712</td><td><eq>\underline{0.456}</eq></td></tr><tr><td>Average</td><td>0.158</td><td>0.364</td><td>0.551</td><td>0.438</td><td>0.301</td><td><eq>\underline{0.199}</eq></td></tr></table>

![](images/b8f872254b954b02216b89f241c9a1f17805282bb16c3527f40022659c0df65a.jpg)


(a) Time-domain channel prediction.

![](images/56153ace0630f60b3e8277ca53cbaf56f4a683115c7e80fdce41018d52df7232.jpg)


(b) Frequency-domain channel prediction.

Figure 4 The zero-shot performance of WiFo and the full shot/zero-shot performance of other baselines on the D17 dataset.

## 4.3.2 Zero-shot Generalization

To evaluate the generalization ability of WiFo, its zero-shot prediction capability is evaluated. Specifically, the pre-trained WiFo performs inference directly on unseen datasets without any fine-tuning. The D17, D18, and D19 datasets are used for zero-shot inference because their operating carrier frequencies are not included in the training set. For other deep learning-based baselines, we consider both zero-shot and full-shot learning scenarios. For zero-shot learning, given that these methods struggle to generalize across CSI with varying shapes, they are trained on the D7, D13, and D7 datasets and then perform inference on the D17, D18, and D19 datasets, respectively. For full-shot learning, these methods are trained and tested on the same dataset. The performance of WiFo-Base and other baselines on the D17, D18, and D19 datasets is illustrated in Fig. 4, Fig. 5, and Fig. 6, respectively. 

It is observed that most deep learning-based methods struggle to perform zero-shot inference. Addi tionally, the transformer and LSTM show poor full-shot performance and even fail to learn, indicating that prediction on D17, D18, and D19 datasets is highly challenging. This is because the three datasets have large sub-carrier intervals and a large number of antennas, making it difficult for these models to learn the 3D variations of the CSI. Nevertheless, the zero-shot performance of WiFo outperforms the zero-shot and even full-shot performance of all other methods, demonstrating its remarkable cross-frequency generalization ability. Its superiority in zero-shot prediction can be attributed to the pre-training paradigm, where WiFo learns a generalizable channel prediction capability applicable to various CSI distributions through training on vast heterogeneous datasets. Therefore, once WiFo is trained on large-scale datasets, it can potentially be deployed instantly at the BS, significantly reducing the costs associated with data collection and model fine-tuning. 

![](images/91552fe8be874643ceac64646609ed3826834120afd1fa85caea2497dd0aaa2c.jpg)


(a) Time-domain channel prediction.

![](images/cba6c4c4576166df425ec54f713043015f0a29a0e3917517026c002620a35da8.jpg)


(b) Frequency-domain channel prediction.

Figure 5 The zero-shot performance of WiFo and the full shot/zero-shot performance of other baselines on the D18 dataset.

![](images/aff5b19414893b0e452680c0017ce323c746af69c9c5c2223ecff4549e54e15c.jpg)


(a) Time-domain channel prediction.

![](images/43935c82ac16f2edd45ff700add910970e8e1840ea5a5f4cb52f2f8434af9c14.jpg)


(b) Frequency-domain channel prediction.

Figure 6 The zero-shot performance of WiFo and the full shot/zero-shot performance of other baselines on the D19 dataset.

## 4.3.3 Ablation Study

To verify the effectiveness of the specialized designs in WiFo, we perform ablation studies based on the WiFo-Base model. The ablation results are shown in Table 6, where the corresponding multi-dataset unified learning and zero-shot generalization performance are given. The average performance is measured by the mean NMSE of columns 2 to 7, representing overall performance. Replacing the proposed STF-PE with learnable space-time positional encoding designed for video would degrade overall performance. This can be attributed to the fact that the proposed STF-PE is an absolute positional encoding [26], which is better suited to token sequences of varying lengths compared to learnable positional encodings. Additionally, removing the random-masked reconstruction task degrades the unified learning and generalization performance for both the time-domain and frequency-domain channel prediction, highlighting its effectiveness for self-supervised pre-training. It can be attributed that the additional random masking strategy facilitates WiFo’s ability to capture the intrinsic 3D relationships of CSI. Moreover, removing time-masked or frequency-masked reconstruction enhances frequency-domain or time-domain channel prediction performance but significantly degrades the other. Therefore, both time-masked and frequency-masked reconstruction tasks are essential for the model pre-training to address both prediction tasks simultaneously. 


Table 6 Results of ablation experiments. The best results are highlighted in bold, while the second-best results are underlined


<table><tr><td rowspan="2"></td><td colspan="3">Time-domain prediction</td><td colspan="3">Frequency-domain prediction</td><td rowspan="2">Average</td></tr><tr><td>D1-D16</td><td>D17</td><td>D18</td><td>D1-D16</td><td>D17</td><td>D18</td></tr><tr><td>WiFo-Base</td><td>0.210</td><td>0.305</td><td>0.420</td><td>0.158</td><td>0.229</td><td>0.280</td><td>0.267</td></tr><tr><td>w/ learnable PE [19]</td><td>0.224</td><td>0.354</td><td>0.442</td><td>0.154</td><td>0.220</td><td>0.267</td><td>0.277</td></tr><tr><td>w/o random-masked reconstruction</td><td>0.214</td><td>0.317</td><td>0.435</td><td>0.165</td><td>0.233</td><td>0.294</td><td>0.276</td></tr><tr><td>w/o time-masked reconstruction</td><td>0.497</td><td>0.754</td><td>0.723</td><td>0.145</td><td>0.199</td><td>0.257</td><td>0.429</td></tr><tr><td>w/o frequency-masked reconstruction</td><td>0.205</td><td>0.310</td><td>0.412</td><td>0.472</td><td>0.819</td><td>0.656</td><td>0.479</td></tr></table>


Table 7 The performance of WiFo is evaluated across different model sizes and various pre-training dataset scales. The best results are highlighted in bold, while the second-best results are underlined.


<table><tr><td rowspan="2">Pre-training Dataset</td><td rowspan="2">Model</td><td colspan="3">Time-domain Prediction</td><td colspan="3">Frequency-domain Prediction</td></tr><tr><td>D1,D5,D9,D15</td><td>D17</td><td>D18</td><td>D1,D5,D9,D15</td><td>D17</td><td>D18</td></tr><tr><td rowspan="5">D1-D16</td><td>WiFi-Tiny</td><td>0.315</td><td>0.444</td><td>0.506</td><td>0.343</td><td>0.440</td><td>0.399</td></tr><tr><td>WiFi-Little</td><td>0.271</td><td>0.371</td><td>0.446</td><td>0.260</td><td>0.284</td><td>0.301</td></tr><tr><td>WiFi-Small</td><td>0.245</td><td>0.326</td><td>0.421</td><td>0.239</td><td>0.245</td><td>0.299</td></tr><tr><td>WiFi-Base</td><td>0.237</td><td>0.305</td><td>0.420</td><td>0.232</td><td>0.229</td><td>0.280</td></tr><tr><td>WiFi-Large</td><td>0.234</td><td>0.314</td><td>0.416</td><td>0.226</td><td>0.232</td><td>0.276</td></tr><tr><td>D1,D2,D5,D6,D9,D12,D15,D16</td><td>WiFi-Base</td><td>0.261</td><td>0.356</td><td>0.477</td><td>0.258</td><td>0.269</td><td>0.345</td></tr><tr><td>D1,D5,D9,D15</td><td>WiFi-Base</td><td>0.306</td><td>0.960</td><td>0.636</td><td>0.320</td><td>1.036</td><td>0.523</td></tr></table>

## 4.3.4 Scaling Analysis

Scaling analysis is essential for foundation models as it highlights the effects of model size and pre-training dataset scale on performance. In our scaling experiments, we consider five model sizes and three dataset scales to evaluate their overall performance. We assess unified learning performance using the average prediction NMSE across the D1, D5, D9, and D15 datasets, and evaluate zero-shot performance on the D17 and D18 datasets, as presented in Table 7. 

From the first five rows of Table 7, we observe that, with a fixed pre-training dataset scale, increasing the model size generally enhances both the unified learning performance and zero-shot generalization capabilities within the observed range. This improvement is attributed to larger models’ ability to capture more complex patterns, resulting in better learning. However, once the model reaches a certain size, its zero-shot generalization ability plateaus, constrained by the scale of the pre-training dataset. It is due to the fact that as the model parameters increase, WiFo gradually overfits the pre-training dataset, leading to performance degradation on out-of-distribution datasets. 

Conversely, as seen in rows 4, 6, and 7, expanding the pre-training dataset scale significantly boosts both unified learning performance and zero-shot performance simultaneously. In summary, the unified learning performance and generalization performance of WiFo are improved [27] with both the increased model size and the scale of the pre-training datasets. Therefore, WiFo is a scalable model that has the potential for further improvement as computational power and dataset scale increase. 

## 4.3.5 Network Storage and Inference Cost

The number of parameters, floating point operations (FLOPs), and inference time of models are closely tied to their storage and computational overhead, directly impacting the practical deployment of channel prediction models at the BS. These overhead metrics are shown in Table 8, where the batch size is set as 8 for inference time calculation, and inference samples are taken from dataset D1. It is worth noting that inference time is not solely determined by FLOPs but also depends on the network architecture and hardware optimization. As a model-based method, PAD has a lower storage cost and FLOPs but a longer inference time due to the lack of GPU optimization. For 3D ResNet, complex 3D convolution operations and the large number of parameters result in significant inference overhead. For LLM4CP, its antenna-parallelized version has a higher inference time because the number of CSI samples processed per batch increases proportionally with the number of antennas. It is observed that WiFo has an acceptable parameter count and comparable inference time among these channel prediction schemes. 


Table 8 The number of parameters, FLOPs, and inference time per batch (batch size is set to 8) of each model.


<table><tr><td></td><td>WiFi-Base</td><td>Transformer</td><td>LSTM</td><td>3D ResNet</td><td>PAD</td><td>LLM4CP</td><td>LLM4CP*</td></tr><tr><td>Parameters(M)</td><td>21.60</td><td>0.91</td><td>1.13</td><td>31.73</td><td>0</td><td>82.35</td><td>83.32</td></tr><tr><td>FLOPs(G)</td><td>1.456</td><td>0.211</td><td>0.061</td><td>390.560</td><td>0.003</td><td>1.720</td><td>11.520</td></tr><tr><td>Inference time(ms)</td><td>9.659</td><td>6.238</td><td>5.209</td><td>86.865</td><td>119.397</td><td>4.718</td><td>7.342</td></tr></table>



21 Jaeckel S, Raschkowski L, B¨orner K, et al. QuaDRiGa: A 3-D multi-cell channel model with time evolution for enabling virtual field trials. IEEE Trans Antennas Propagat, 2014, 62: 3242-3256. 



Moreover, it is worth noting that WiFo is a versatile model capable of replacing multiple specialized channel prediction models, which can significantly reduce the number of models required on the BS. For instance, in the experiment introduced in Section 4.3.1, a single WiFo model is sufficient to perform timedomain and frequency-domain prediction tasks across 16 datasets, whereas other deep learning-based approaches require training 32 separate models for the above tasks. The parameter count of WiFo-Base is even lower than the total parameters of 32 transformer models, thereby reducing additional deployment overhead. 



22 3GPP Radio Access Network Working Group. Study on channel model for frequencies from 0.5 to 100 GHz (Release 15). 3GPP TR 38.901, 2018. 



## 5 Conclusions



23 Feichtenhofer C, Fan H, Malik J, et al. Slowfast networks for video recognition. In: Proceedings of the IEEE/CVF international conference on computer vision, Seoul, 2019. 6202-6211. 



In this paper, we have introduced a novel wireless foundation model, WiFo, designed to simultaneously facilitate time-domain and frequency-domain channel prediction tasks as well as diverse 3D CSI configurations. We developed an MAE-based network structure and implemented several mask reconstruction tasks for self-supervised pre-training to capture the intrinsic space-time-frequency features of CSI. WiFo, pre-trained on large-scale heterogeneous datasets, can be directly deployed for inference without any need for fine-tuning. Simulations demonstrate its exceptional performance in unified multi-dataset training and its superb zero-shot generalization capabilities with acceptable inference overhead. 



24 Jiang W, Schotten H D. Deep learning for fading channel prediction. IEEE Open J Commun Soc, 2020, 1: 320-332. 



## References



25 Loshchilov I, Hutter F. Sgdr: Stochastic gradient descent with warm restarts. 2016. ArXiv: 1608.03983 





26 Yuan Y, Ding J, Feng J, et al. Unist: A prompt-empowered universal model for urban spatio-temporal prediction. In: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, New York, 2024. 4095-4106. 





1 Cheng X, Zhang H, Zhang J, et al. Intelligent multi-modal sensing-communication integration: synesthesia of machines. IEEE Commun Surveys Tuts, 2024, 26: 258-301 





27 Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models. 2020. ArXiv: 2001.08361 





2 Gao S, Cheng X, Yang L. Estimating doubly-selective channels for hybrid mmWave massive MIMO systems: A doubly-sparse approach. IEEE Trans Wirel Commun, 2020, 19: 5703-5715. 





3 Yin H, Wang H, Liu Y, et al. Addressing the curse of mobility in massive MIMO with prony-based angular-delay domain channel predictions. IEEE J Select Areas Commun, 2020, 38: 2903-2917 





4 Rottenberg F, Choi T, Luo P, et al. Performance analysis of channel extrapolation in FDD massive MIMO systems. IEEE Trans Wirel Commun, 2020, 19: 2728-2741 





5 Wang Y, Gao Z, Zheng D, et al. Transformer-empowered 6G intelligent networks: From massive MIMO processing to semantic communication. IEEE Wirel Commun, 2022, 30: 127-135. 





6 Zhu G, Lyu Z, Jiao X, et al. Pushing AI to wireless network edge: An overview on integrated sensing, communication, and computation towards 6G. Sci China Inf Sci, 2023, 66: 130301. 





7 Jiang H, Cui M, Ng D W K, et al. Accurate channel prediction based on transformer: making mobility negligible. IEEE J Select Areas Commun, 2022, 40: 2717-2732 





8 Liu G, Hu Z, Wang L, et al. Spatio-temporal neural network for channel prediction in massive MIMO-OFDM systems. IEEE Trans Commun, 2022, 70: 8003-8016 





9 Liu Z, Singh G, Xu C, et al. FIRE: enabling reciprocity for FDD MIMO systems. In: Proceedings of the 27th Annual International Conference on Mobile Computing and Networking, New York, 2021. 628-641 





10 Zhang H, Gao S, Cheng X, et al. Integrated sensing and communications towards proactive beamforming in mmWave V2I via multi-modal feature fusion (MMFF). IEEE Trans Wirel Commun, 2024, 23: 15721-15735 





11 Liu B, Liu X, Gao S, et al. LLM4CP: Adapting Large Language Models for Channel Prediction. J Commun Inf Netw, 2024, 9: 113-125 





12 Bommasani R, Hudson D A, Adeli E, et al. On the opportunities and risks of foundation models. 2021. ArXiv:2108.07258 





13 Fontaine J, Shahid A, De Poorter E. Towards a Wireless Physical-Layer Foundation Model: Challenges and Strategies. 2024. ArXiv: 2403.12065 





14 Salihu A, Rupp M, Schwarz S. Self-Supervised and Invariant Representations for Wireless Localization. IEEE Trans Wirel Commun, 2024, 23: 8281-8296 





15 Alikhani S, Charan G, Alkhateeb A. Large Wireless Model (LWM): A Foundation Model for Wireless Channels. 2024. ArXiv: 2411.08872 





16 Zhang Y, Zhang J, Pei Y, et al. Latest progress for 3GPP ISAC channel modeling standardization. Sci China Inf Sci, 2024, 67: 217301 





17 He K, Chen X, Xie S, et al. Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, New Orleans, 2022. 16000-16009 





18 Tong Z, Song Y, Wang J, et al. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pretraining. In: Advances in neural information processing systems, New Orleans,2022. 10078-10093 





19 Feichtenhofer C, Li Y, He K. Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems, 2022, 35: 35946-35958. 





20 Alexey D. An image is worth 16x16 words: Transformers for image recognition at scale. 2020. ArXiv: 2010.11929 

