# google_lens

This project implements and evaluates similarity search using various methodologies, focusing on Vision Transformers (ViTs). The main goal is to compare the performance of feature extraction and fine-tuning for image similarity tasks, leveraging a subset of the CIFAR-10 dataset.

## Methods Evaluated

1. Autoencoder-based Embedding: Used an autoencoder to extract compressed embeddings of images. These embeddings were then compared using cosine similarity.

Strengths: Simple to implement; effective for generic unsupervised feature extraction.

Weaknesses: Limited semantic understanding of images; relies heavily on reconstruction quality.

2. Pre-trained Vision Transformers (ViTs): Extracted embeddings using the ViT-base-patch16-224 model pre-trained on ImageNet. Cosine similarity was used to compare embeddings.

Strengths: Leverages high-quality, pre-trained embeddings that generalize well across datasets.

Weaknesses: Sub-optimal for datasets with significantly different distributions (e.g., CIFAR-10).

Moderate performance with precision showing improvement over the autoencoder approach. Cosine similarity scores for top matches were promising but not highly discriminative.

3. Fine-tuned Vision Transformers (ViTs): Fine-tuned the pre-trained ViT model on CIFAR-10, adding a classification head and training using a cross-entropy loss.

Strengths: Adapted the model to the dataset's specific distribution, improving embedding relevance.

Weaknesses: Requires additional computational resources; depends on effective hyperparameter tuning.

Performance: Cosine similarity scores for top matches showed better clustering of semantically similar images.

4. ResNet-based Embedding: Leveraged a pre-trained ResNet-50 model to extract embeddings from CIFAR-10 images. The extracted embeddings were compared using cosine similarity.

Strengths: ResNet's residual connections allow for effective gradient flow, leading to high-quality feature extraction.

Weaknesses: Similar to ViT, pre-trained models may not align perfectly with the CIFAR-10 dataset's distribution.

5. CNN-based Embedding: Implemented a custom Convolutional Neural Network (CNN) architecture to extract features from images. The embeddings were used for similarity search.

Strengths: Customizable architecture allows tailoring to dataset-specific requirements.

Weaknesses: Requires extensive training data and hyperparameter tuning to achieve competitive performance.

Performed better than the autoencoder approach but fell short of ResNet and fine-tuned ViT.

Pre-trained ViTs and ResNet provided better embeddings than autoencoders and CNNs, but fine-tuning further enhanced relevance and similarity clustering.

While effective, cosine similarity could be explored for datasets with unique characteristics. Alternative similarity metrics (e.g., Euclidean distance) could also be explored.


# References

https://arxiv.org/abs/2010.11929

PyTorch Documentation: https://pytorch.org/docs/

https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/

Hugging Face Transformers Library: https://huggingface.co/transformers/
