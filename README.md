# 1 Introduction
&nbsp; Parkinson’s disease (PD) is a progressive neurodegenerative disorder characterized by a variety of motor and non-motor symptoms. Early and accurate diagnosis of PD remains a significant challenge in the medical field. Gait analysis powered by advanced signal pro- cessing and machine learning techniques, has emerged as a promising approach to detect PD due to its ability to capture subtle changes in motor functions that are characteristic of the disease.
# 2 Gait Data and Analysis
## 2.1 What is gait analysis?
&nbsp; Gait analysis is a method used to assess and understand the mechanics of human walk- ing. It involves the systematic study of locomotion, primarily through the observation and measurement of bodily movements and the forces involved in walking. In clinical settings, gait analysis is employed to diagnose, plan treatment, and monitor the progres- sion of various conditions that affect physical movement, including neurological disorders, musculoskeletal issues, and injuries.
## 2.2 Why gait analysis is efficient in PD diagnosis?
&nbsp; Gait analysis is particularly relevant in the context of PD because the degeneration of dopamine-producing neurons in the brain, which is a hallmark of Parkinson’s disease, attribute to the changes in gait. Dopamine is crucial for smooth, purposeful muscle movement and coordination. Therefore, people with PD have the following symptoms:
* __Reduced Arm Swing:__ due to bradykinesia (slowness of movement), there’s often a reduction in the natural arm swing during walk.
* __Shorter Stride Length:__ the reduced stride length is partly due to muscle stiffness and bradykinesia.
* __Postural Instability:__ there’s often a stooped and leaned forward posture, which can affect balance and stability.
* __Shuffling Steps:__ the feet can barely leave the ground. due to diminished movement control and a struggle to lift the feet properly.
* __Freezing of Gait:__ This is a unique symptom where individuals temporarily feel as if their feet are glued to the ground, especially while starting to walk or when turning.
* __Slower Walking Speed:__ Overall, the walking speed is generally slower due to the combination of these factors.
## 2.3 Gait Data in this research
&nbsp; In the research, I used ”Gait in Parkinson’s Disease” from PhysioNet[1][2]. The dataset includes 16-channel VGRF data, which measures the force exerted by the feet against the ground during walking. 93 patients with Parkinson’s Disease and 73 healthy controls wore 8 force sensors on both their feet and performed walking experiments. Here is the illustration of the possible system that would be able to collect 16-channel VGRF data:
<p align="center">
  <img src=images/Shoes.png alt="Alt text">
  <br>
  <em>Figure 1: Example of a system that collects 16-channel VGRF data.</em>
</p>

## 2.4 EDA of the Dataset
&nbsp; As mentioned earlier the dataset consists of 93 patients with Parkinson’s Disease and 73 healthy controls. Here is the visualization of the total force for right and left feet of PD and Control individuals:
<p align="center">
  <img src=images/Total_Force.png alt="Alt text">
  <br>
  <em>Figure 2: Total Force for right and left feet of PD and Control individuals.</em>
</p>

&nbsp; The spikes of Control may be higher due to bigger mass, but the difference in the shape of these spikes is obvious. The PD patient leaves his entire foot on the ground significantly longer comparing to Control. This may be the sign of Freezing of Gait.
# 3 Signal Processing and ML Methods
## 3.1 CNN Model on Raw Data
&nbsp; I tried to train the same CNN model, that I will introduce later in this report, on the raw sequences of the VGRF signals, and the best result I got from such approach was 83% accuracy.
## 3.2 Wavelet Transform
&nbsp; To improve the performance of the model, I decided to perform the feature engineering. I selected the ’Total Force Left’ column of the dataset, which is a total force observed on the whole left foot. I selected this type of signal, since I wanted to check if such generalized signal will give us enough information about the gait of an individual.
From the EDA, one can observe that the gait data has patterns, so I made an assump- tion that the frequency components of this data will give an additional useful insight about these patterns.
&nbsp; Therefore, I chose a Continuous Wavelet Transform as a method to present the VGRF data in both time and Frequency domains.
&nbsp; The Continuous Wavelet Transform (CWT) is a powerful mathematical tool used in signal processing to analyze localized variations of power within a time series. It decomposes a continuous-time signal into wavelets, which are small waves localized in time. Unlike the Fourier transform, the CWT is capable of providing time-frequency representation of a signal, making it highly effective for analyzing non-stationary signals where frequency components vary over time. The CWT of a continuous signal $x(t)$ is defined as:
$$W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \cdot \psi^* \left( \frac{t - b}{a} \right) dt$$
&nbsp;Where:
- $W_x(a, b)$ is the wavelet coefficient,
- $a$ is the scale parameter,
- $b$ is the translation parameter,
- $ψ(t)$ is the mother wavelet, and
- $ψ^∗(t)$ is the complex conjugate of the mother wavelet.

&nbsp;As a mother wavelet for CWT I chose the Morlet wavelet, which is particularly effective for signal processing due to its good balance between time and frequency localization. The Morlet wavelet is defined as a plane wave modulated by a Gaussian window:
<p align="center">
  <img src=images/Morlet.png alt="Alt text">
  <br>
  <em>Figure 3: Real-valued Mortlet Wavelet[6].</em>
</p>

<p align="center">
  <img src=images/Wavelet_Cmor.png alt="Alt text">
  <br>
  <em>Figure 4: Complex-valued Mortlet Wavelet[7].</em>
</p>

$$\psi_{\text{Morlet}}(t) = \pi^{-1/4} e^{i \omega_0 t} e^{-t^2 / 2}$$
&nbsp;Where:
- $π^{−1/4}$ is the normalization constant ensuring the wavelet has unit energy,
- $ω_0$ is the central frequency of the wavelet,
- $e^{iω_0t}$ represents a complex sine wave (plane wave),
- $e^{−t^2/2}$ is the Gaussian window that modulates the plane wave.
&nbsp;This formulation allows the Morlet wavelet to effectively capture both frequency and location information from the signal, making it particularly suited for analyzing signals where this dual information is crucial.
## 3.3 The Results of CWT
&nbsp;Here are the results of CWT with 64 scales of Morlet wavelet:

<p align="center">
  <img src=images/Scalogram_Control.png alt="Alt text">
  <br>
  <em>Figure 5: Scalogram of the same Control individual.</em>
</p>

<p align="center">
  <img src=images/Scalogram_PD.png alt="Alt text">
  <br>
  <em>Figure 6: Scalogram of the same PD individual.</em>
</p>

&nbsp;This scalogram helps us to see at what coefficient specific wavelet scale has at a specific time. One can observe a walking pattern on these scalograms and there are noticeable differences in them, that can help to detect PD.
## 3.4 CNN Model
&nbsp;Since ML is out of the scope of this course I will shortly describe the DL model I used for my project. So, the CNN classification model[8] has the following structure:

<p align="center">
  <img src=images/CNN.png alt="Alt text">
  <br>
  <em>Figure 7: Scalogram of the same PD individual.</em>
</p>

&nbsp;Where:
- __red layers__ are 1d Convolution Layers with different dilation rates,
- __blue layers__ are Batch Normalization,
- __purple layer__ is MaxPooling,
- __output__ is SoftMax.
## 3.5 Results
&nbsp;The final CNN model has an accuracy of 94%. Here you can see the confusion matrix:
<p align="center">
  <img src=images/Confusion_Matrix.png alt="Alt text">
  <br>
  <em>Figure 8: Confusion Matrix.</em>
</p>

# 4 Conclusions and Possible Improvements
&nbsp;The Wavelet Transform proves to be a great choice for feature engineering. The result is not the best among all the other papers, but it is a decent one, if we consider the fact that the model was only trained on the generalized data from one foot. One of the possible ways to improve the accuracy is to create a second model that will utilize the data from the right foot and then the get predictions from these two models and combine these predictions. Also, attention mechanism from transformer models can improve the result.
# References
[1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... and Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[2] https://physionet.org/content/gaitpdb/1.0.0/

[3] Imanne El Maachi, Guillaume-Alexandre Bilodeau, and Wassim Bouachir. 2020. Deep 1D-Convnet for accurate Parkinson disease detection and severity prediction from gait. Expert Syst. Appl. 143, C (Apr 2020). https://doi.org/10.1016/j.eswa.2019. 113075

[4] C. Dong, Y. Chen, Z. Huan, Z. Li, B. Zhou and Y. Liu, ”Static-Dynamic Temporal Networks for Parkinson’s Disease Detection and Severity Prediction,” in IEEE Trans- actions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 2205-2213, 2023, doi: 10.1109/TNSRE.2023.3269569.

[5] Pham TD, Yan H. Tensor Decomposition of Gait Dynamics in Parkin- son’s Disease. IEEE Trans Biomed Eng. 2018 Aug;65(8):1820-1827. doi: 10.1109/TBME.2017.2779884. Epub 2017 Dec 4. PMID: 29989951.

[6] https://commons.wikimedia.org/wiki/File:MorletWaveletMathematica.svg# /media/File:MorletWaveletMathematica.svg

[7] https://commons.wikimedia.org/wiki/File:Wavelet_Cmor.svg#/media/File: Wavelet_Cmor.svg

[8] https://medium.com/mlearning-ai/audio-classification-using-wavelet-transform-and-de
