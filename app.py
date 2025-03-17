from flask import Flask, request, jsonify
import os
import logging
import sys
import re
import json
import tiktoken
from datetime import datetime, timedelta
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional
from langchain.callbacks import get_openai_callback
from collections import defaultdict
from flask_session import Session

# Configuración del logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
        self.chunk_relations = defaultdict(list)
        self.article_index = defaultdict(list)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def extract_metadata(self, chunk: str) -> Dict:
        metadata = {
            'pagina': None,
            'capitulo': None,
            'articulo': [],
            'tema': None,
            'es_continuacion': False,
            'chunk_id': None,
            'relacionados': []
        }

        lines = chunk.split('\n')
        for line in lines:
            if 'METADATA:' in line:
                if 'Página:' in line:
                    match = re.search(r'Página: (\d+)', line)
                    if match:
                        metadata['pagina'] = match.group(1)
                if 'Capítulo:' in line:
                    match = re.search(r'Capítulo: ([^,]+)', line)
                    if match:
                        metadata['capitulo'] = match.group(1).strip()
                if 'Artículo:' in line:
                    matches = re.findall(r'[\w\.-]+', line[line.find('Artículo:'):])
                    metadata['articulo'].extend([m.strip() for m in matches if m.strip()])
                if 'Tema:' in line:
                    match = re.search(r'Tema: (.+)', line)
                    if match:
                        metadata['tema'] = match.group(1).strip()

        content_start = chunk.find('CONTENIDO:')
        if content_start != -1:
            content = chunk[content_start:]
            article_refs = re.finditer(r'[Aa]rtículo[s]?\s+(\d+[A-Za-z]?(?:-[A-Z])?)', content)
            metadata['relacionados'].extend(match.group(1) for match in article_refs)

        metadata['es_continuacion'] = any('continuacion' in line.lower() for line in lines)
        return metadata

    def build_relations(self, chunks: List[str]):
        """
        Construye relaciones entre chunks basados en artículos y otros metadatos.
        """
        for i, chunk in enumerate(chunks):
            metadata = self.extract_metadata(chunk)
            chunk_id = f"chunk_{i}"
            metadata['chunk_id'] = chunk_id
            
            # Indexar artículos en el chunk
            for articulo in metadata['articulo']:
                self.article_index[articulo].append(chunk_id)
            
            # Construir relaciones basadas en artículos relacionados
            for relacionado in metadata['relacionados']:
                self.chunk_relations[chunk_id].append(relacionado)

class ConversationManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="coloca el nombre de tu modelo  - General o personalizado con finetunning de Open AI",
            openai_api_key=api_key
        )
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_state = {
            "last_articles": [],
            "last_topics": [],
            "context": []
        }

    def process_history(self, history: List[Dict]) -> Dict:
        """Procesa el historial para extraer información relevante"""
        processed_history = {
            "articles_mentioned": set(),
            "topics_discussed": set(),
            "last_contexts": [],
            "summary": ""
        }
        
        if not history:
            return processed_history
            
        for interaction in history:
            if "articulos_mencionados" in interaction:
                processed_history["articles_mentioned"].update(interaction["articulos_mencionados"])
            if "tema" in interaction:
                processed_history["topics_discussed"].add(interaction["tema"])
            if "contexto" in interaction:
                processed_history["last_contexts"].append(interaction["contexto"])
        
        try:
            # Generar resumen del historial
            history_context = "\n".join([f"Pregunta previa: {h.get('pregunta', '')}\nRespuesta: {h.get('respuesta', '')}\n" for h in history[-3:]])
            
            summary_prompt = f"""
            Resume brevemente el contexto de la siguiente conversación, 
            enfocándote en los temas principales y artículos discutidos:
            {history_context}
            """
            
            processed_history["summary"] = self.llm.predict(summary_prompt)
        except Exception as e:
            logger.error(f"Error generando resumen del historial: {str(e)}")
            processed_history["summary"] = "Error procesando el resumen del historial"
        
        return processed_history

    def detect_intent(self, text: str, history: List[Dict] = None) -> Dict:
        history_context = ""
        if history:
            try:
                processed_history = self.process_history(history)
                history_context = f"\nContexto del historial: {processed_history['summary']}"
            except Exception as e:
                logger.error(f"Error procesando historial en detect_intent: {str(e)}")
                history_context = ""

        intent_prompt = f"""
        Analiza el siguiente mensaje y su contexto histórico para determinar:
        1. La intención principal del usuario (saludo,agradecimiento, despedida, pregunta legal, consulta general, etc)
        2. Si hace referencia a conversación previa
        3. El tema principal
        4. Artículos mencionados
        5. Si requiere información de múltiples artículos

        Mensaje: {text}
        {history_context}

        Responde en formato JSON con:
        - intent_type: tipo de intención
        - references_history: true/false
        - main_topic: tema principal o null
        - mentioned_articles: []
        - requires_multiple_sources: true/false
        - is_followup: true/false
        """

        try:
            response = self.llm.predict(intent_prompt)
            intent_data = json.loads(response)
            return intent_data
        except Exception as e:
            logger.error(f"Error al procesar la intención: {str(e)}")
            return {
                "intent_type": "unknown",
                "references_history": False,
                "main_topic": None,
                "mentioned_articles": [],
                "requires_multiple_sources": False,
                "is_followup": False
            }
    
    def get_conversational_response(self, text: str, intent: Dict) -> Optional[str]:
        responses = {
            "saludo": [
                "¡Hola! Soy un asistente especializado en la LEY DEL IMPUESTO AL VALOR AGREGADO. ¿En qué puedo ayudarte?",
                "¡Buen día! Estoy aquí para resolver tus dudas sobre la LEY DEL IMPUESTO AL VALOR AGREGADO",
            ],
            "despedida": [
                "¡Hasta luego! No dudes en volver si tienes más preguntas sobre la LEY DEL IMPUESTO AL VALOR AGREGADO.",
                "¡Que tengas un excelente día! Estoy aquí para cuando necesites más información.",
            ],
            "agradecimiento": [
                "¡De nada! ¿Hay algo más en lo que pueda ayudarte?",
                "Es un placer ayudarte. ¿Tienes alguna otra consulta sobre la LEY DEL IMPUESTO AL VALOR AGREGADO?",
            ]
        }

        if intent["intent_type"] in responses:
            from random import choice
            return choice(responses[intent["intent_type"]])
        return None

    def add_to_history(self, question: str, answer: str):
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )

    def load_history(self, historial: List[Dict]):
        if historial:
            for interaction in historial:
                question = interaction.get('pregunta', '')
                answer = interaction.get('respuesta', '')
                if question and answer:
                    self.add_to_history(question, answer)

class QueryAnalyzer:
    def __init__(self):
        self.patterns = {
            'articulo': [
                r'art[íi]culo\s*[\w\.-]+',
                r'\bart\.\s*[\w\.-]+',
                r'\b\d+[A-Za-z]?[-\s]?[A-Za-z]?\b'
            ],
            'tema': [
                r'(sobre|acerca de|respecto a)\s+.+',
                r'(qué|como|cuando|donde)\s+.+\s+(en|para|sobre)\s+.+'  
            ],
            'definicion': [
                r'(qué|que)\s+(es|son|significa)\s+.+',
                r'(define|definición de|concepto de)\s+.+'  
            ],
            'listado': [
                r'(cuáles|cuales|que|qué)\s+(son|están)\s+.+(exentos|obligados|sujetos)',
                r'(enumera|lista|menciona)\s+.+'  
            ]
        }

    def analyze_query(self, question: str, history: List[Dict] = None) -> Dict:
        result = {
            'tipo': 'general',
            'articulos_mencionados': [],
            'temas_detectados': [],
            'tipo_consulta': None,
            'requiere_multiple_fuentes': False,
            'referencias_historicas': []
        }

        # Detectar artículos mencionados
        for pattern in self.patterns['articulo']:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                result['articulos_mencionados'].append(match.group())

        # Detectar el tipo de consulta
        for pattern in self.patterns['definicion']:
            if re.search(pattern, question, re.IGNORECASE):
                result['tipo_consulta'] = 'DEFINICION'
                result['requiere_multiple_fuentes'] = True

        for pattern in self.patterns['listado']:
            if re.search(pattern, question, re.IGNORECASE):
                result['tipo_consulta'] = 'LISTADO'
                result['requiere_multiple_fuentes'] = True

        if result['articulos_mencionados']:
            result['tipo_consulta'] = 'ARTICULO_ESPECIFICO'
            result['tipo'] = 'articulo_especifico'

        return result

class QASystem:
    def __init__(self, chunks_file_path: str):
        self.processor = DocumentProcessor()
        self.analyzer = QueryAnalyzer()
        self.knowledge_base = None
        self.chunks = []
        self.chunk_metadata = {}
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)  # Tiempo de vida del caché
        self.cache_timestamps = {}  # Para controlar la expiración del caché
        self.initialize_system(chunks_file_path)

    def initialize_system(self, chunks_file_path: str):
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            chunks = content.split("==================================================")
            self.chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            self.processor.build_relations(self.chunks)
            
            for i, chunk in enumerate(self.chunks):
                metadata = self.processor.extract_metadata(chunk)
                self.chunk_metadata[f"chunk_{i}"] = metadata

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            texts = self.chunks
            metadatas = [self.chunk_metadata[f"chunk_{i}"] for i in range(len(self.chunks))]
            self.knowledge_base = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            
            logger.info(f"Sistema inicializado con {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error inicializando sistema: {str(e)}")
            raise

    def get_related_chunks(self, initial_docs: List, analysis: Dict) -> List:
        """Obtiene chunks relacionados basados en el análisis y los documentos iniciales"""
        from langchain.schema import Document
        all_docs = list(initial_docs)
        seen_chunks = set()

        for doc in initial_docs:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                
                # Obtener chunks relacionados
                related_articles = self.processor.chunk_relations[chunk_id]
                for article in related_articles:
                    related_chunks = self.processor.article_index.get(article, [])
                    for related_chunk_id in related_chunks:
                        if related_chunk_id not in seen_chunks:
                            seen_chunks.add(related_chunk_id)
                            chunk_index = int(related_chunk_id.split('_')[1])
                            if chunk_index < len(self.chunks):  # Verificar índice válido
                                chunk_content = self.chunks[chunk_index]
                                metadata = self.chunk_metadata[related_chunk_id]
                                all_docs.append(Document(page_content=chunk_content, metadata=metadata))

        return all_docs[:10]  # Limitar a 10 documentos para evitar sobrecarga

    def check_cache(self, cache_key: str) -> Optional[Dict]:
        """Verifica si hay una respuesta en caché válida"""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key)
            if timestamp and datetime.now() - timestamp < self.cache_ttl:
                return self.cache[cache_key]
            else:
                # Limpiar caché expirado
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
        return None

    def update_cache(self, cache_key: str, result: Dict):
        """Actualiza el caché con una nueva respuesta"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()

    def process_question(self, question: str, api_key: str, history: List[Dict] = None) -> Dict:
        try:
            conversation_manager = ConversationManager(api_key)
            intent = conversation_manager.detect_intent(question, history)
            
            cache_key = f"{question}_{intent['intent_type']}"
            cached_response = self.check_cache(cache_key)
            if cached_response:
                return cached_response

            if intent["intent_type"] in ["saludo", "despedida", "agradecimiento"]:
                response = conversation_manager.get_conversational_response(question, intent)
                result = {
                    "respuesta": response,
                    "tipo": "conversacional",
                    "intent": intent,
                    "metricas": {
                        "tokens_totales": 0,
                        "costo_estimado": 0.0,
                        "chunks_relevantes": 0
                    },
                    "contexto_usado": []
                }
                self.update_cache(cache_key, result)
                return result

            analysis = self.analyzer.analyze_query(question, history)
            k_docs = 8 if analysis['requiere_multiple_fuentes'] else 5
            initial_docs = self.knowledge_base.similarity_search(question, k=k_docs)
            all_docs = self.get_related_chunks(initial_docs, analysis)

            prompt_template = f"""
            Eres un experto en la Ley del Impuesto al Valor Agregado. 
            Analiza la siguiente pregunta y el contexto proporcionado.
            
            Pregunta: {{question}}
            
            Contexto: {{context}}
            
            Estructura tu respuesta de la siguiente manera:
            
            1. CITA TEXTUAL:
            - Proporciona una respuesta clara y concisa
            - Cita los artículos específicos cuando sea relevante
            
            2. ANALISIS
            - Explica el contexto y las implicaciones
            - Menciona cualquier excepción o caso especial
            
            3. CONTEXTO LEGAL:
            - Lista los artículos y secciones citadas
            - Menciona reformas relevantes si las hay
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            llm = ChatOpenAI(
                temperature=0.3,
                model_name="coloca el nombre de tu modelo  - General o personalizado con finetunning de Open AI",
                openai_api_key=api_key
            )

            chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=all_docs, question=question)
                # Calcular el costo manualmente
                precio_prompt = 0.012 / 1000  # Precio por token de entrada
                precio_completions = 0.016 / 1000  # Precio por token de salidas
                costo_estimado = (cb.prompt_tokens * precio_prompt) + (cb.completion_tokens * precio_completions)

                context_used = [{"content": doc.page_content, "metadata": doc.metadata} for doc in all_docs]

                result = {
                    "respuesta": response,
                    "tipo": "legal",
                    "intent": intent,
                    "metricas": {
                        "tokens_totales": cb.total_tokens,
                        #"costo_estimado": cb.total_cost,
                        "costo_estimado": costo_estimado,
                        "chunks_relevantes": len(all_docs)
                    },
                    "contexto_usado": context_used,
                    "historial_procesado": history if history else []
                }

                self.update_cache(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            return {"error": str(e), "tipo": "error"}

# Configuración de Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
Session(app)

# Inicialización del sistema
qa_system = None

def init_qa_system():
    global qa_system
    if qa_system is None:
        try:
            qa_system = QASystem('data/chunks_d.txt')
            logger.info("Sistema QA inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el sistema QA: {str(e)}")
            raise

@app.before_request
def ensure_qa_system():
    if not qa_system:
        init_qa_system()

def transform_historial_to_json(plain_text_historial):
    try:
        # Manejar casos de historial vacío
        if not plain_text_historial.strip():
            return []

        # Dividir interacciones por el separador "====="
        interactions = plain_text_historial.split("=====")

        # Crear una lista para almacenar preguntas y respuestas
        conversation_list = []
        for interaction in interactions:
            parts = interaction.split("|||")  # Dividir pregunta y respuesta
            if len(parts) == 2:  # Validar que haya exactamente una pregunta y una respuesta
                conversation_list.append({
                    "pregunta": parts[0].strip(),  # Limpiar espacios
                    "respuesta": parts[1].strip()  # Limpiar espacios
                })

        if not conversation_list:  # Verificar que no esté vacío después de procesar
            raise ValueError("El historial está vacío o tiene un formato incorrecto.")

        return conversation_list
    except Exception as e:
        raise ValueError(f"Error al transformar historial: {str(e)}")

@app.route('/consulta', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        print("Historial recibido:", data.get("historial"))  # Asegúrate de que esto sea una lista o un JSON bien estructurado

        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        question = data.get("pregunta")
        api_key = data.get("api_key")
        historial = data.get("historial", [])  # Ahora esperamos una lista de diccionarios

        if not question or not api_key:
            return jsonify({"error": "Faltan datos requeridos (pregunta o api_key)"}), 400

        # Validar y transformar historial si no está en formato JSON
        if isinstance(historial, str) and historial.strip():  # Si llega como texto plano no vacío
            try:
                historial = transform_historial_to_json(historial)  # Transformar a JSON
            except Exception as e:
                return jsonify({"error": f"Historial inválido: {str(e)}"}), 400
        elif not isinstance(historial, list):  # Si no es lista, asumir vacío
            historial = []  # Usar historial vacío por defecto

        response = qa_system.process_question(question, api_key, historial)
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error en /consulta: {str(e)}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "sistema_inicializado": qa_system is not None
    }), 200

if __name__ == "__main__":
    init_qa_system()
    app.run(host="0.0.0.0", port=8080)