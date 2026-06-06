# 🧬 BlastFlow – Professional Bioinformatics Suite

BlastFlow is a modern, interactive bioinformatics platform built with Streamlit, Biopython, Plotly, and AI-powered analysis tools. It provides an all-in-one environment for sequence analysis, NCBI BLAST searches, primer design, phylogenetic visualization, protein structure exploration, and biological data interpretation.

## Features

### 🌐 NCBI BLAST Integration

* Run live BLAST searches against NCBI databases
* Support for:

  * BLASTN
  * BLASTP
  * BLASTX
  * TBLASTN
  * TBLASTX
  * MegaBLAST
  * Discontiguous MegaBLAST
* Advanced filtering and parameter customization
* Export results to CSV, Excel, FASTA, and TXT reports

### 🔬 Sequence Analyzer

* DNA, RNA, and Protein sequence detection
* Sequence statistics
* Molecular weight calculation
* GC content analysis
* Base/amino acid composition visualization
* Sequence quality assessment

### 🔀 Central Dogma Tools

* DNA → RNA transcription
* RNA → Protein translation
* Genetic code table support
* Multiple translation table options

### 📊 GC Content Dashboard

* GC percentage calculations
* Sliding window GC analysis
* Interactive visualizations
* Composition profiling

### ⚗️ Primer Design

* Primer generation support
* Melting temperature calculations
* GC content validation
* Primer quality metrics

### 🌿 Phylogenetic Analysis

* Tree visualization
* Sequence relationship exploration
* Interactive phylogenetic displays

### 🧊 3D Protein Viewer

* Interactive protein structure visualization
* PDB integration
* Multiple rendering styles:

  * Cartoon
  * Stick
  * Sphere
  * Line
* Surface rendering support

### 🤖 AI Bioinformatics Assistant

* Powered by Groq + Llama 3
* BLAST result interpretation
* Biological insights and recommendations
* Interactive scientific Q&A

### 📋 Analysis History

* Session tracking
* Search history
* Result management

---

## Technology Stack

### Frontend

* Streamlit
* Plotly
* HTML/CSS
* JavaScript

### Bioinformatics

* Biopython
* NCBI BLAST API
* Sequence Utilities

### AI

* Groq API
* Llama 3

### Data Processing

* Pandas
* NumPy
* OpenPyXL

### Visualization

* Plotly
* Matplotlib
* 3Dmol.js

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/blastflow.git
cd blastflow
```

### Install Dependencies

```bash
pip install streamlit biopython pandas numpy plotly matplotlib groq openpyxl
```

### Configure AI Features (Optional)

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_api_key_here"
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser at:

```text
http://localhost:8501
```

---

## Project Structure

```text
blastflow/
│
├── app.py
├── .streamlit/
│   └── secrets.toml
├── requirements.txt
├── README.md
└── assets/
```

---

## Export Options

BlastFlow supports exporting results in:

* CSV
* Excel (.xlsx)
* FASTA
* Text Reports (.txt)

---

## Use Cases

* Academic research
* Comparative genomics
* Sequence annotation
* Molecular biology education
* Primer design workflows
* Protein structure exploration
* Bioinformatics training

---

## Requirements

* Python 3.9+
* Internet connection (for NCBI BLAST services)
* Optional Groq API key for AI features

---

## License

This project is distributed under the MIT License.

---

## Acknowledgements

* NCBI BLAST
* Biopython
* Streamlit
* Plotly
* Groq
* RCSB Protein Data Bank
* 3Dmol.js
