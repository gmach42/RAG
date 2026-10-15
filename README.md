*This project has been created as part of the 42 curriculum by gmach.*

# Description
# Instructions
# System architecture

![RAG schema](RAG_schema.png)

# Chunking strategy
# Retrieval method
# Performance analysis
# Design decisions
# Challenges faced

Problèmes de compréhension du sujet au niveau des data Models qui nous sont imposés. En effet beaucoup de modèles sont assez flous sur leur concrète utilisation/utilité dans le projet. Un exemple concret serait AnswerQuestion, qui représente une question CORRIGÉE au préalable. Elle est simplement décrite dans le sujet comme "AnsweredQuestion model represent an answered question". OUI MAIS ENCORE? Il aurait été utile de préciser lors de la présentation du modèle qu'il servait à comparer nos résultats stockés dans MinimalAnswer avec les réponses correctes (en tout cas celles attendues) de AnsweredQuestion.

# Example usage
# Resources
	Way of chunking for RAG: https://blog.stephane-robert.info/docs/developper/programmation/python/rag-chunking/
	Doc of Qwen: https://huggingface.co/Qwen/Qwen3-0.6B
	Doc for transformers.pipelines: https://huggingface.co/docs/transformers/main_classes/pipelines
	Converse with a model with pipeline: https://huggingface.co/docs/transformers/conversations
	Chat templating: https://huggingface.co/docs/transformers/chat_templating
	llm tutorial for transformers: https://huggingface.co/docs/transformers/llm_tutorial
## AI usage
	doc for transformers, chunk manipulation, bm25s, basic explanation of tools

![Saving/Loading index](save_load_index.png)

List of useful functions used throughout the project:
Pathlib functions:
```python
path.write_text("some text")
path.read_text()
path.mkdir(parents=True, exist_ok=True)
path.name
path.suffix
src_dir = Path(...)
src_dir.exists()
src_dir.rglob("*")
src_dir.rglob("*.py")
```

BM25s functions:
```python
tokens = bm25s.tokenize(["txt1", "txt2"], stopwords="en")
retriever = bm25s.BM25() # default values
retriever = bm25s.BM25.load("data/processed/bm25")
retriever.index(tokens)
retriever.save("data/processed/bm25")
ids, scores = retriever.retrieve(tokens, k=5)
ids[0], scores[0] # first retrieved document and its score
```

Pydantic functions:
```python
from pydantic import TypeAdapter

Model.model_dump()
Model.model_dump_json()
Model.model_validate()
TypeAdapter(Model).dump_json(instance)
```

