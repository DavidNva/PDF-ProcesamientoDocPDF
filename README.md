# API para Procesamiento de Documentos con Flask, AI y ML

## Descripción
Esta API permite el procesamiento y análisis de documentos PDF utilizando inteligencia artificial y aprendizaje automático. Implementa un sistema de preguntas y respuestas (QA) que puede extraer información relevante de documentos almacenados, utilizando modelos de NLP y bases de datos vectoriales para una búsqueda eficiente.

## Características Principales
- Procesamiento y análisis de documentos PDF.
- Sistema de preguntas y respuestas basado en IA.
- Integración con modelos de lenguaje de OpenAI (GPT).
- Base de datos vectorial con FAISS para búsquedas eficientes.
- Sistema de caché para optimizar respuestas.
- APIs REST implementadas con Flask.

---

## Requisitos Previos

### Tecnologías Utilizadas
- **Lenguaje**: Python
- **Framework Web**: Flask
- **Base de Datos Vectorial**: FAISS
- **NLP y ML**: OpenAI GPT, Hugging Face
- **Almacenamiento de Documentos**: File System
- **Autenticación y Gestión de Sesiones**: Flask-Session
- **Infraestructura en la Nube**: Google Cloud Run (opcional para despliegue)

### Dependencias
Se recomienda crear un entorno virtual antes de instalar las dependencias para evitar conflictos con otros paquetes del sistema:
```sh
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```
Luego, instala las dependencias necesarias:
```sh
pip install flask flask-session openai faiss-cpu langchain tiktoken huggingface_hub
```

---

## Arquitectura del Proyecto
![ChatPDF_API_ProcesamientoDocumentos](https://github.com/user-attachments/assets/c7323cd7-f635-4388-8990-e3c387216480)
---

## Configuración
### Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto y configura las siguientes variables:
```env
OPENAI_API_KEY=tu_clave_de_openai
HUGGINGFACE_API_KEY=tu_clave_de_huggingface
```

---

## Uso de la API
### 1. Ejecutar la Aplicación Localmente
```sh
python main.py
```
Esto iniciará la API en `tu ruta del proyecto`

### 2. Endpoints Disponibles
#### 2.1. Verificar el estado de la API
```http
GET /status
```
**Respuesta esperada:**
```json
{
  "status": "running"
}
```

#### 2.2. Subir y Procesar un Documento PDF
```
Deberas dejar un documento txt  (partiras de la conversion de un PDF a texto) convertido a chunks para poder procesarlos. 
```

#### 2.3. Realizar una Consulta sobre un Documento
```http
POST /query
Content-Type: application/json
```
**Body:**
```json
{
  "document_id": "123456",
  "question": "¿Cuál es el resumen del documento?"
}
```
**Respuesta esperada:**
```json
{
  "answer": "El documento trata sobre..."
}
```

---

## Despliegue en Google Cloud Run
Para desplegar la API en la nube, se recomienda utilizar Docker.

### 1. Autenticarse en Google Cloud
```sh
gcloud auth login
gcloud config set project [TU_PROYECTO_ID]
```

### 2. Construir la Imagen con Docker
```sh
docker build -t gcr.io/[TU_PROYECTO_ID]/procesamiento-documentos-api .
```

### 3. Subir la Imagen a Google Container Registry
```sh
docker push gcr.io/[TU_PROYECTO_ID]/procesamiento-documentos-api
```

### 4. Desplegar en Cloud Run
```sh
gcloud run deploy procesamiento-documentos-api \
  --image gcr.io/[TU_PROYECTO_ID]/procesamiento-documentos-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 5. Obtener la URL del Servicio
```sh
gcloud run services describe procesamiento-documentos-api --format 'value(status.url)'
```

---

## Contribuciones
Si deseas contribuir, por favor sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una rama con tu nueva funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Agregada nueva funcionalidad'`).
4. Sube los cambios (`git push origin nueva-funcionalidad`).
5. Abre un Pull Request en GitHub.

---

## Licencia
Este proyecto está bajo la licencia MIT.

