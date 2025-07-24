# Belgian Document Deidentification System

A waterproof deidentification system specifically designed for Belgian healthcare documents containing sensitive patient data in Dutch. This system addresses the poor performance of existing English-focused deidentification tools on Dutch clinical text.

## 🎯 Overview

This system provides state-of-the-art deidentification capabilities for Dutch clinical documents, incorporating:

- **Advanced Dutch NLP**: Built on clinlp framework and RobBERT-2023 language models
- **Multi-layered Detection**: Combines rule-based and ML approaches for superior accuracy
- **Belgian Compliance**: Full GDPR and Belgian Health Data Agency compliance
- **Production Ready**: Scalable, secure, and enterprise-grade implementation

## 🚀 Key Features

### Superior Dutch Language Support
- Specialized handling of Dutch naming conventions (tussenvoegsels)
- Belgian address format recognition
- Dutch medical terminology processing
- Context-aware entity recognition

### Advanced Architecture
- **Document Ingestion**: Secure multi-format document processing
- **Dutch Clinical NLP**: clinlp + RobBERT integration
- **Entity Recognition**: Multi-layered PHI detection
- **Deidentification Engine**: Flexible anonymization/pseudonymization
- **Quality Assurance**: Comprehensive validation and monitoring

### Regulatory Compliance
- GDPR Article 9 compliance for health data
- Belgian Data Protection Act adherence
- Belgian Health Data Agency standards
- Comprehensive audit trails

## 📊 Performance Targets

| PHI Category | Target F1-Score | Improvement vs Existing |
|--------------|----------------|------------------------|
| Person Names | ≥ 0.95 | Significant |
| Addresses | ≥ 0.93 | Substantial |
| Dates | ≥ 0.96 | Notable |
| Medical IDs | ≥ 0.94 | Major |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Document        │    │ Preprocessing    │    │ Dutch Clinical  │
│ Ingestion       │───▶│ Engine           │───▶│ NLP Engine      │
│ Service         │    │                  │    │ (clinlp+RobBERT)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Output          │    │ Deidentification │    │ Entity          │
│ Management      │◀───│ Engine           │◀───│ Recognition     │
│ Service         │    │                  │    │ System          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │ Quality          │
                       │ Assurance        │
                       │ System           │
                       └──────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker (optional)
- 8GB+ RAM recommended
- GPU support recommended for optimal performance

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/belgian-deidentification.git
cd belgian-deidentification

# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py

# Run basic setup
python setup.py install

# Start the service
python -m belgian_deidentification.main
```

### Docker Deployment

```bash
# Build the image
docker build -t belgian-deidentification .

# Run with docker-compose
docker-compose up -d
```

## 📖 Usage

### Basic Usage

```python
from belgian_deidentification import DeidentificationPipeline

# Initialize the pipeline
pipeline = DeidentificationPipeline(
    language="nl",
    mode="anonymization",  # or "pseudonymization"
    config_path="config/default.yaml"
)

# Process a document
result = pipeline.process_document(
    document_path="path/to/clinical_note.txt",
    output_path="path/to/deidentified_note.txt"
)

print(f"Processed document with {result.entities_found} PHI entities")
```

### API Usage

```bash
# Start the API server
python -m belgian_deidentification.api

# Process document via REST API
curl -X POST "http://localhost:8000/deidentify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient Jan van der Berg was born on 01/01/1980..."}'
```

### Batch Processing

```python
from belgian_deidentification import BatchProcessor

processor = BatchProcessor(
    input_dir="data/input/",
    output_dir="data/output/",
    config="config/batch.yaml"
)

results = processor.process_all()
```

## ⚙️ Configuration

### Basic Configuration

```yaml
# config/default.yaml
deidentification:
  mode: "anonymization"  # or "pseudonymization"
  entities:
    - "PERSON"
    - "ADDRESS"
    - "DATE"
    - "MEDICAL_ID"
  
nlp:
  model: "robbert-2023-dutch-large"
  use_gpu: true
  batch_size: 32

quality_assurance:
  enable_validation: true
  confidence_threshold: 0.85
  expert_review_sampling: 0.1
```

### Advanced Configuration

See [Configuration Guide](docs/configuration.md) for detailed options.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/

# Run all tests with coverage
pytest --cov=belgian_deidentification tests/
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Performance Tuning](docs/performance.md)
- [Compliance Guide](docs/compliance.md)

## 🔒 Security & Compliance

### Security Features
- End-to-end encryption
- Role-based access control
- Comprehensive audit logging
- Secure key management

### Compliance
- GDPR Article 9 compliant
- Belgian Data Protection Act adherent
- Belgian Health Data Agency standards
- ISO 27001 aligned security controls

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-org/belgian-deidentification.git
cd belgian-deidentification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **clinlp team** at University Medical Center Utrecht for the Dutch clinical NLP framework
- **RobBERT developers** at KU Leuven, UGent, and TU Berlin for the Dutch language models
- **Belgian Health Data Agency** for guidance on anonymization standards
- **Dutch clinical NLP community** for research and resources

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/belgian-deidentification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/belgian-deidentification/discussions)
- **Email**: support@belgian-deidentification.org

## 🗺️ Roadmap

- [x] Core deidentification engine
- [x] Dutch NLP integration
- [x] Quality assurance system
- [ ] Web interface
- [ ] Advanced analytics
- [ ] Multi-language support
- [ ] Cloud deployment templates

---

**Built with ❤️ for Belgian healthcare data protection**

