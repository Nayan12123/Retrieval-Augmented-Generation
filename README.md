# RAG 
It is concept of using embeddings similarity along with the capability of LLM to answer queries which are out of the LLM's domain by providing some context from the documents.
Given Below is the brief description of the RAG process. 
<br />
<img width="910" alt="image" src="https://github.com/Nayan12123/Retrieval-Augmented-Generation/assets/91980608/2238c438-531a-47bb-82ed-c95316ecf2e2">
<br />
It mainly consists of 5 steps.
**step 1.** Loading the Document. <br />
**step 2.** Chunking the splitted document. <br />
**step 3.** Convert chunks to vector embeddings and Dumping the same to a VectorDB<br />
**step 4.** Embedding the Query and Retrieval using similarity. <br />
**step 5.** Prompting LLM to generate the response for the query by providng top-n similar retrieved chunks fromm the vector DB. <br />
<br />
The current code is compatible with nomic embeddings and any open source hugging face LLMs

<br /><br />
## LLama 3-8b Instruct
It is an instruction fined tuned LLM developed by meta AI. It consists of 8 billion parameters and is an open source LLM available on hugging face.
You can check other open source LLMs [here.](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

## Nomic embedding Model
It is an open source long context embedding model. The model architecture is similar to BERT only critical changes that were done here are:
<br />
a). Using RoPE embeddings.<br />
b). Training specifically for retrieval tasks. <br />
c). Utilizing flash attention mechanisms for faster inference. <br />
You can experiment with other open source embedding models [here.](https://huggingface.co/spaces/mteb/leaderboard)


