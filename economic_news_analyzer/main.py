import os
import json
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class NoticiaAnalise:
    titulo: str
    link: str
    tipo: str
    sentimento: str
    explicacao: str


class AnalisadorNoticiasIA:
    def __init__(self):
        self.base_url = "https://www.globo.com/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')

        self.prompt_analise = """
        Você é um analista financeiro especializado. Analise a seguinte notícia e responda APENAS no formato JSON especificado:

        NOTÍCIA: "{titulo}"

        Responda no formato JSON exato:
        {{
            "tipo": "Nacional ou Internacional",
            "sentimento": "Boa notícia para o país ou Péssima notícia para o país",
            "explicacao": "Breve explicação de 1-2 frases sobre o impacto econômico"
        }}

        Critérios:
        - Nacional: Afeta diretamente o Brasil
        - Internacional: Afeta outros países ou economia global
        - Boa notícia: Impacto econômico positivo (crescimento, emprego, investimento)
        - Péssima notícia: Impacto econômico negativo (crise, desemprego, recessão)
        """

    # ======= Extração de dados do site =======

    def extrair_conteudo_globo(self) -> str:
        try:
            print("🔍 Acessando Globo.com...")
            response = requests.get(self.base_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            print("✅ Página carregada com sucesso!")
            return response.text
        except requests.RequestException as e:
            print(f"❌ Erro ao acessar o site: {e}")
            return ""

    def extrair_noticias(self, html_content: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html_content, 'html.parser')
        noticias = []
        seletores = [
            'h2 a', 'h3 a', '.feed-post-link', '.post-title a',
            'a[href*="/economia/"]', 'a[href*="/mundo/"]', 'a[href*="/politica/"]',
            '.highlight-title a', '.chamada-principal a'
        ]
        links_processados = set()

        for seletor in seletores:
            for elemento in soup.select(seletor):
                if len(noticias) >= 30:
                    break
                titulo = self._extrair_titulo(elemento)
                link = self._extrair_link(elemento)
                if titulo and len(titulo) > 20 and link and link not in links_processados:
                    links_processados.add(link)
                    noticias.append({'titulo': titulo.strip(), 'link': link})

        print(f"📰 {len(noticias)} notícias encontradas")
        return noticias

    def _extrair_titulo(self, elemento) -> str:
        return elemento.get_text(strip=True) or elemento.get('title', '') or elemento.get('alt', '')

    def _extrair_link(self, elemento) -> str:
        href = elemento.get('href', '')
        if href.startswith('http'):
            return href
        if href.startswith('/'):
            return f"https://www.globo.com{href}"
        return href

    # ======= Filtro de notícias relevantes =======

    def filtrar_noticias_economia(self, noticias: List[Dict]) -> List[Dict]:
        palavras_chave = [
            'economia', 'econômico', 'pib', 'inflação', 'selic', 'dólar', 'real',
            'mercado', 'bolsa', 'investimento', 'banco', 'juros', 'fiscal',
            'orçamento', 'imposto', 'exportação', 'desemprego', 'emprego',
            'renda', 'consumo', 'varejo', 'indústria', 'agronegócio', 'petróleo',
            'financeiro', 'crise', 'recessão', 'crescimento', 'lucro', 'prejuízo'
        ]
        noticias_economia = [
            n for n in noticias
            if any(p in n['titulo'].lower() for p in palavras_chave)
        ]
        print(f"💰 {len(noticias_economia)} notícias de economia identificadas")
        return noticias_economia[:8]

    # ======= Métodos de análise com IA externa =======

    def analisar_com_groq(self, titulo: str) -> Optional[Dict]:
        if not self.groq_api_key:
            return None

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": self.prompt_analise.format(titulo=titulo)}],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=10
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    return json.loads(content[json_start:json_end])

        except Exception as e:
            print(f"⚠️ Erro no Groq: {e}")
        return None

    def analisar_com_huggingface(self, titulo: str) -> Optional[Dict]:
        if not self.hf_api_key:
            return None

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
                headers={"Authorization": f"Bearer {self.hf_api_key}"},
                json={"inputs": titulo},
                timeout=10
            )

            if response.status_code == 200 and isinstance(response.json(), list):
                sentimentos = response.json()[0]
                score_positivo = next((s['score'] for s in sentimentos if s['label'] == 'LABEL_2'), 0)
                score_negativo = next((s['score'] for s in sentimentos if s['label'] == 'LABEL_0'), 0)
                tipo = "Nacional" if any(p in titulo.lower() for p in ['brasil', 'brasileiro', 'nacional', 'país']) else "Internacional"
                sentimento = "Boa notícia para o país" if score_positivo > score_negativo else "Péssima notícia para o país"

                return {
                    "tipo": tipo,
                    "sentimento": sentimento,
                    "explicacao": f"Análise baseada em IA: {score_positivo:.2f} positivo vs {score_negativo:.2f} negativo"
                }

        except Exception as e:
            print(f"⚠️ Erro no Hugging Face: {e}")
        return None

    # ======= Análise local (fallback) =======

    def analisar_local_simples(self, titulo: str) -> Dict:
        titulo_lower = titulo.lower()
        tipo = "Nacional" if any(p in titulo_lower for p in ['brasil', 'brasileiro', 'nacional', 'país']) else "Internacional"
        score_pos = sum(1 for p in ['crescimento', 'alta', 'sobe', 'melhora', 'lucro', 'investimento'] if p in titulo_lower)
        score_neg = sum(1 for p in ['crise', 'queda', 'baixa', 'desemprego', 'prejuízo', 'recessão'] if p in titulo_lower)
        sentimento = "Boa notícia para o país" if score_pos > score_neg else "Péssima notícia para o país"

        return {
            "tipo": tipo,
            "sentimento": sentimento,
            "explicacao": "Análise baseada em palavras-chave locais"
        }

    # ======= Pipeline completo de análise =======

    def analisar_noticia(self, titulo: str) -> Dict:
        print(f"🤖 Analisando: {titulo[:50]}...")
        for metodo in [self.analisar_com_groq, self.analisar_com_huggingface, self.analisar_local_simples]:
            resultado = metodo(titulo)
            if resultado:
                origem = metodo.__name__.replace("analisar_com_", "")
                print(f"✅ Análise via {origem}")
                return resultado
        return {"tipo": "Desconhecido", "sentimento": "Indefinido", "explicacao": "Não foi possível analisar a notícia"}
