## (i)
## (a) Modify the code so that you load the input_childSpeech_trainingSet.txt dataset. Briefly describe the dataset (e.g., what does it appear to contain? What is the vocabulary size? What is the length?). Do the same for the other two datasets in the folder.

### Input: input_childSpeech_trainingSet.txt

The input_childSpeech_trainingSet.txt dataset has a vocabulary size of 40. The language appears simple, resembling how a baby might talk, either to themselves or to others. However, there isn’t much context provided to clarify this. The dataset contains 10,000 lines.

### Input: input_childSpeech_testSet.txt

The input_childSpeech_testSet.txt dataset has the same vocabulary size as input_childSpeech_trainingSet.txt, which is 40. Its structure is similar to the training set, but the main difference lies in the length—this dataset has only 1,000 lines compared to the 10,000 lines in the training set.

### Input: input_shakespeare.txt

The input_shakespeare.txt dataset has a vocabulary size of 56 and contains 40,000 lines. This dataset has a more structured format compared to the other two. It provides clear context, such as identifying which characters are speaking and to whom (e.g., COMINIUS speaking to MARCIUS). The structure and length make it significantly different from the other two datasets.

## (b) Load the input_childSpeech_trainingSet.txt dataset. Downsize the model configuration so that it has less than 1 million parameters by manipulating the hyperparameters on lines 5-16 of gpt.py. Describe the rationale for your downsizing (i.e., why did you downsize a particular parameter rather than another one).

### Final answer
I changed certain important hyperparameters to balance capacity and efficiency in order to reduce the model to more than close to a million parameters (1.2 M) while still achieving respectable performance. Since it has a quadratic effect on the number of parameters, I decreased the embedding size (n_embd) from 384 to 144, which greatly reduced the model size while maintaining sufficient capacity for token representation. To accommodate the smaller embedding size and ensure effective computation without taxing the attention mechanism, I also reduced the number of attention heads (n_head) from 6 to 4. In order to balance computational efficiency with the depth of hierarchical learning, I also decreased the number of transformer layers (n_layer) from six to five. In order to increase memory efficiency, I finally changed the block size (block_size) from 256 to 128. However, I was aware that this would somewhat restrict the model's capacity to manage long-range relationships. Because of the narrower vocabulary and context window, these modifications led to a model size of roughly 1.28 million parameters with lower training (0.34) and validation losses (0.35). However, I observed that the reduced model's limited ability to produce sophisticated and contextually rich outputs is reflected in the shorter, simpler words it produces with some repetition. This scaled-down model is better suited for lesser tasks because it enables faster convergence and drastically lowers resource requirements, but at the expense of decreased coherence and limited long-range dependency handling.


## (c) Attempt another two different ways of downsizing and report the loss functions for 

## Final output
### Parameters 1 Analysis

I decreased the embedding size (n_embd) to 128 in order to reduce the model to less than 1 million parameters while still achieving acceptable performance on the input_childSpeech_trainingSet.txt dataset. This was done because it has a significant impact on the number of parameters while maintaining adequate representational capacity for the vocabulary in the dataset. To ensure compliance with the embedding size, I maintained the number of attention heads (n_head) at 4, which enables the model to focus on numerous segments of the sequence without overtaxing the attention mechanism. I also decreased the number of transformer layers (n_layer) to four in order to balance computational efficiency with hierarchical learning. Along with increasing memory efficiency, the block size (block_size) was adjusted to 128; nevertheless, this limits the model's capacity to capture long-range dependencies. With 0.818M parameters, this setup produced a model with a train loss of 0.3535 and a validation loss of 0.3557. The resulting text was a workable shrinking solution for kid speech data since it displayed straightforward and repetitive structures while maintaining coherence.

### Parameters 2 Analysis
In order to further reduce the model to a more basic configuration under severe resource restrictions, I set the embedding size (n_embd) to 96. This greatly decreased the number of parameters, but it also limited the model's ability to depict intricate patterns. Along with reducing the number of transformer layers (n_layer) to three in order to minimize depth and hierarchical learning, I also decreased the number of attention heads (n_head) to three in order to match the smaller embedding size. Block_size (block_size) was kept at 128 to ensure that the context length was constant. A model with just 0.354M parameters was produced using this setup, yielding a train loss of 0.3577 and a validation loss of 0.3597. But with sentences like "Go park" and "Look moon," the generated text was extremely basic and repetitive, demonstrating the model's diminished ability to produce imaginative and contextually rich outputs. Although this setup is effective, output quality is compromised, hence it is only appropriate for settings with severe computing constraints.

## (d) Explore and describe in the report how the inclusion of the bias terms in the self- attention layers impacts the transformer model.

| **Bias Configuration** | **Parameters (M)** | **Train Loss** | **Validation Loss** | **Text Quality**                                              |
|-------------------------|--------------------|----------------|----------------------|--------------------------------------------------------------|
| **No Bias**            | 0.818             | 0.3535         | 0.3557              | Repetitive and less diverse; moderate coherence.             |
| **Key Only**           | 0.818984          | 0.3527         | 0.3546              | Slight improvement in coherence; moderate repetition.        |
| **Key + Value**        | 0.819496          | 0.3519         | 0.3542              | Best performance; more diverse and coherent text.            |
| **Query Only**         | 0.818984          | 0.3540         | 0.3559              | Comparable to key-only; slightly less coherent and diverse.  |
| **All Biases**         | 0.820008          | 0.3523         | 0.3549              | Marginal improvement in diversity but diminishing returns.   |


Enabling key and value biases offers the optimum efficiency and performance balance, with the lowest train (0.3519) and validation losses (0.3542) and more broadened coherent text outputs, according to the examination of bias settings. The key-only and query-only configurations exhibit comparable performance, with somewhat larger losses than the key-value configuration with reasonable coherence but rather repetitive outputs. Although it produces less creative and repetitive writing, the no-bias model is the most parameter-efficient. The parameters (0.820M) are somewhat increased without a discernible improvement when all biases are enabled, demonstrating decreasing returns. Therefore, it is best to enable key and value biases judiciously for this dataset.


## (e) Explore and describe in the report how the inclusion of skip connections throughout the transformer architecture impacts the resulting model.

| **Configuration**                | **Train Loss** | **Validation Loss** | **Text Quality**                                              |
|----------------------------------|----------------|----------------------|--------------------------------------------------------------|
| **Both Layers Present**          | ~0.35          | ~0.36               | Coherent, repetitive childlike sentences with context.       |
| **No Feedforward Layer**         | ~0.376         | ~0.38               | Basic structure, repetitive text, less refined context.      |
| **No Multi-Head Attention**      | ~2.02          | ~2.02               | Completely nonsensical and incoherent.                      |
| **No Attention and Feedforward** | ~2.015         | ~2.017              | Random, nonsensical outputs; no recognizable patterns.       |

### conclusion 
The experiment demonstrates the vital roles of the feedforward and multi-head attention layers in transformer architecture. The model learns well when both layers are present, producing connected text with modest losses (~0.35). Repetitive, less sophisticated language and larger losses (~0.38) result from removing the feedforward layer. The model produces illogical results when the attention layer is absent because it is unable to recognize token associations. The worst-case situation is the removal of both layers, which results in random, meaningless text and high losses (~2.02). This illustrates how the feedforward layer improves representations while attention facilitates contextual understanding. Their fundamental significance in transformer models is highlighted by the fact that both elements are essential for learning and producing meaningful sequences.

## 2
## (a)
After determining the test loss on input_childSpeech_testSet.txt using the best model from part (i), I found that it was 2.0127, which was somewhat less than the training loss of 2.0161. This slight variation demonstrates how well the model generalises to new data without overfitting. It was evident to me that the improved model does a better job of identifying patterns in the data when I contrasted it with the baseline model, which had a greater test loss of 2.0175. However, the resulting output still contains childish and repetitive sentences and lacks meaningful context and diversity. The dataset's simplicity and the model's limited size are the causes of this constraint. Notwithstanding these limitations, the improved model performed better than the baseline, which primarily imitated the training data without successfully identifying more complex linguistic patterns. Although the revised model's performance is not outstanding, it is good for the task at hand and clearly outperforms the baseline, as evidenced by its lower test loss, which I feel indicates that it has generalised better to unknown data.
## (b)
Input_shakespeare.txt had a loss of 4.0146, which was significantly greater than the training loss of 2.0152, according to my calculation. The model's inability to generalise to the Shakespearean dataset, which is much more complicated and linguistically diverse than the training data it was refined on, is demonstrated by this notable gap. The model overfits the training data and is unable to capture Shakespeare's text's complex syntax, richer vocabulary, and subtle patterns, as evidenced by the substantial validation loss. The modified model appears to perform worse on complex datasets when compared to the baseline model, as the baseline's losses were more consistent but less efficient overall.

It's evident that the model performs significantly better on datasets that are simpler and have simple, repeating patterns when I contrast this result with question (ii)(a), where the validation loss on input_childSpeech_testSet.txt was 2.0127. This demonstrates the model's shortcomings when it comes to managing intricate datasets. A larger model with more capacity, pretraining on a broad corpus, or fine-tuning with more data indicative of Shakespearean language are some of the improvements that are necessary given the high loss on Shakespearean text. In actual use, I believe this pipeline works best for producing simpler, domain-specific writing, such as children's instructional materials. I would suggest expanding the model, utilising transfer learning, and utilising a more varied and rich training dataset in order to produce intricate and nuanced language similar to Shakespeare's.

