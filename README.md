# Model Experiments
Some basic experiments and playing around with models on HuggingFace.

The repository contains sample python code and test data / images used in them. 

The associated python dependencies and local model downloads need to be managed separately, and is not part of this repo. 


<h3>Steps that worked well for me on this journey:</h3>

1. Start with the pipelines API in Huggingface [transformers](https://huggingface.co/docs/transformers/v4.47.1/en/quicktour), simplest way to use models while abstracting all the complexities.  Post that move to learning more about using the overall transformers API. 

2. Play around with the sample code supplied with various models on Huggingface, and keep using similar constructs to write your own custom code.

	- Try at least one each of text and vision (image) models, if possible one audio.
	- For both image and text, try out playing with their raw vector embeddings & content generation from prompts.
	- A few models I tried were a [cpu version of stable diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) and Microsoft's [Florence 2](https://huggingface.co/microsoft/Florence-2-base) for images, and [miniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) & [Phi 3.5](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) for text.

3. Once played around for a bit, you'll have several questions on the theory. A good resource to get answers for these questions is the [Huggingface NLP course](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt).

4. Write some more code leveraging the theoretical knowledge gained in the previous step!


<h3>Some key learnings & gotchas:</h3>

- Most models do not work well on CPU, they expect a nvidia GPU by default. This includes most of the Standard Diffusion models.

- If you are on Windows ARM64, as of December 2024 there is very limited support for most pythin modules. Was unable to install numpy & torch, efectively making it unusable for any model experiments.

- You can brush up your python skills while experimenting itself, a dedicated python deep dive is not mandatory.


---
<h4>**Suman Ghosh, December 2024**</h4>
