# 📰 Analisador de Notícias Econômicas com IA

Este projeto tem como objetivo coletar notícias do portal [Globo.com](https://www.globo.com/), filtrar aquelas relacionadas à economia e realizar uma análise automatizada utilizando modelos de Inteligência Artificial (Groq e Hugging Face). Como fallback, também é possível usar uma análise local simples baseada em palavras-chave.

---

## 📌 Funcionalidades

- 🔎 **Raspagem de notícias** diretamente da página principal do Globo.com
- 🧠 **Análise automática** das notícias via:
  - [Groq API](https://console.groq.com/) com modelo LLaMA 3 (gratuito)
  - [Hugging Face API](https://huggingface.co/) com modelo Roberta-base
  - Análise local baseada em palavras-chave (fallback)
- 🗂️ **Filtragem por temas econômicos**
- 📊 Classificação das notícias em:
  - **Tipo**: Nacional ou Internacional
  - **Sentimento**: Boa ou Péssima notícia para o país
  - **Explicação**: Breve justificativa do impacto econômico

---

## 🚀 Como executar

### 1. Clonar o repositório

```bash
git https://github.com/leandrosouzaf30/economic_news_analyzer.git
cd economic_news_analyzer
