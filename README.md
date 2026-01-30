# Replication Package: LAMPS

This package includes a multi-agent pipeline using CrewAI with:
- Meta-LLaMA 3 (Instruct) for reasoning agents
- Fine-tuned CodeBERT classifier for Python file classification (malicious/benign)

## Structure

- `hybrid_pypi_classifier.py`: Main executable pipeline (CrewAI-MAS)
- `models/codebert-malware-detector`: fine-tuned CodeBERT model
- `downloads/`: PyPI packages
- `Dataset/`: dataset used for fine-tuning classifier
- `extracted/`: Python files extracted
- `prompts.md`: Prompts structure used in the pipeline


- `README.md`: This file

## Usage

Install dependencies:
```bash
pip install crewai transformers accelerate einops requests tqdm
```

Run the classifier:
```bash
python hybrid_pypi_classifier.py --package <pypi-package-name>
```

Example:
```bash
python hybrid_pypi_classifier.py --package requests
```
## How to cite

If you find our work useful for your research, please cite the papers using the following BibTex entries:
```bash
@article{UMARZESHAN2026112792,
title = {Many hands make light work: An LLM-based multi-agent system for detecting malicious PyPI packages},
journal = {Journal of Systems and Software},
volume = {236},
pages = {112792},
year = {2026},
issn = {0164-1212},
doi = {https://doi.org/10.1016/j.jss.2026.112792},
url = {https://www.sciencedirect.com/science/article/pii/S0164121226000269},
author = {Muhammad {Umar Zeshan} and Motunrayo Ibiyo and Claudio {Di Sipio} and Phuong T. Nguyen and Davide {Di Ruscio}},
keywords = {Malicious PyPI package, LLMs, Multi-agent systems},
abstract = {Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA 3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6000 setup.py files, and D2, a realistic multi-file dataset with 1296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter and TD-IDF stacking models–two state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar’s test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.}
}
```

