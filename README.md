# ACMIA
This is the official repository for the paper Automatic Calibration for Membership Inference Attack on Large Language Models by Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu.
# Overview
We propose ACMIA, a calibration-based framework for improving membership inference attacks (MIAs) on large language models (LLMs). Existing MIA methods often misclassify non-training texts as training data, especially when the texts are simple or share surface-level features with pre-training corpora. ACMIA addresses this issue by automatically calibrating token probabilities using a tunable temperature parameter, which refines the likelihood distribution without relying on additional reference models. This calibration is grounded in theoretical insights from maximum likelihood estimation and enhances the separation between member and non-member samples. We introduce three variants: AC, DerivAC, and NormAC, each capturing different aspects of model behavior and text complexity. To support evaluation, we benchmark ACMIA across three standard datasets: WikiMIA, MIMIR, and PatentMIA.
# ACMIA (& baselines)
We first obtain the token log probability distribution by querying the LLM with the input text, and then compute the token log probability distributions under different temperature settings.
<pre> ```bash python src/com_det_sco.py --tar_mod <model_name> --data <data_file> --max_cha <text-length> --lang <lang> ``` </pre>
We then calibrate the token probabilities by focusing only on the first occurrence of each token in the sentence, and apply our three calibration methods (AC, DerivAC, and NormAC) to adjust the token log probabilities. A final detection score is computed based on these calibrated values to assess whether the input text was part of the pretraining data.

For the DC-PDD baseline, an additional reference dataset is required to compute token frequencies. Following their approach, we use a large-scale publicly available corpus as a proxy for estimating the token frequency distribution.

# HF paths of the evaluated model in the paper

# Acknowledgement

# Citation
If you find this work helpful, please consider citing our paper:
