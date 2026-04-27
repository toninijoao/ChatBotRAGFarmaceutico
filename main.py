import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 0.0,
    api_key=api_key
)

caminhos_bulas = [
    "arquivos/bula_dipirona.pdf",
    "arquivos/bula_paracetamol.pdf",
    "arquivos/bula_dorflex.pdf",
    "arquivos/bula_ibuprofeno.pdf",
    "arquivos/bula_meleato_enalapril.pdf",
    "arquivos/bula_neosoro.pdf",
    "arquivos/bula_nimesulida.pdf",
    "arquivos/bula_omeprazol.pdf",
    "arquivos/bula_tadalafila.pdf",
    "arquivos/bula_cimegripe.pdf"
]

documentos = []

for caminho in caminhos_bulas: #percorre cada bula
    loader = PyPDFLoader(caminho)
    docs = loader.load()

    for documento in docs:
        documento.metadata["medicamento"] = caminho.split("/")[-1].replace(".pdf", "") #o nome do medicamento se torna um metadado

    documentos.extend(docs)

len(documentos)
        
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,           #tamanho maximo da chunk
    chunk_overlap = 300         #sobreposição  
)

chunks = text_splitter.split_documents(documentos)      #faz a separação em chunks
len(chunks)         #chunks geradas

for chunk in chunks:            #categorização
    texto = chunk.page_content.lower()

    if "identificação do medicamento" in texto or "composição" in texto:
        chunk.metadata["Categoria"] = "identificacao"

    elif "indicação" in texto or "para que o medicamento é indicado" in texto:
        chunk.metadata["Categoria"] = "indicacao"

    elif "como este medicamento funciona" in texto or "ação" in texto:
        chunk.metadata["Categoria"] = "funcionamento"

    elif "contraindicacao" in texto or "quando nao devo usar" in texto:
        chunk.metadata["Categoria"] = "contraindicacao"

    elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
        chunk.metadata["Categoria"] = "advertencias_precaucoes"

    elif "interação" in texto or "interações medicamentosas" in texto: 
        chunk.metadata["Categoria"] = "interacoes"

    elif (
        "posologia" in texto
        or "como devo usar" in texto
        or "modo de usar" in texto
        or "como usar" in texto
    ):
        chunk.metadata["Categoria"] = "posologia_modo_uso"

    elif any(p in texto for p in [
        "reações adversas",
        "reação adversa",
        "efeitos colaterais",
        "eventos adversos",
        "efeitos indesejáveis"
    ]):
        chunk.metadata["Categoria"] = "reacoes_adversas"

    elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
        chunk.metadata["Categoria"] = "armazenamento"

    elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto or "abuso" in texto:
        chunk.metadata["Categoria"] = "superdosagem"

    else: 
        chunk.metadata["Categoria"] = "geral"

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./chroma_bulas"
)

prompt = ChatPromptTemplate.from_template("""
    Você é um assistente especializado em leitura de bulas de medicamentos.

    Use SOMENTE o contexto fornecido.

    INSTRUÇÕES:
    - Leia TODOS os trechos antes de responder
    - Se uma informação estiver incompleta, procure continuar em outros trechos
    - UNA partes de frases que foram divididas entre trechos
    - Liste TODAS as reações adversas encontradas (não pare na primeira)
    - Ignore trechos irrelevantes

    Se encontrar a resposta:
    - responda de forma completa e detalhada

    Se NÃO encontrar:
    - diga: "Não encontrado na bula"

    Contexto:
    {context}

    Pergunta:
    {input}

    Resposta:
    """)

def reescrever_pergunta(pergunta):
    prompt_rewrite = f"""
Você é especialista em bulas de medicamentos.

Converta a pergunta do usuário para termos técnicos EXATOS usados em bulas.

Mapeamentos:
- "males", "faz mal" → "reações adversas"
- "como tomar", "quanto tomar" → "posologia"
- "para que serve" → "indicação"
- "quando usar" → "indicação"
- "efeitos" → "reações adversas"

Pergunta: {pergunta}

Retorne apenas a pergunta reescrita:
"""
    return llm.invoke(prompt_rewrite).content.strip()

document_chain = create_stuff_documents_chain(llm, prompt)

def limitar_contexto(docs, max_chars=6000):
        contexto = ""
        for doc in docs:
            trecho = doc.page_content
            if len(contexto) + len(trecho) > max_chars:
                break
            contexto += trecho + "\n\n"
        return contexto

while True:
    pergunta = input("\n" + "-" * 15 + "🤖 BOT FARMACÊUTICO 🤖" + "-" * 15 + 
    "\n\n👋 Olá, sou o Bot Farmacêutico! Sou especialista nos seguintes remédios: " \
    "\n\n💊 Dipirona" 
    "\n💊 Paracetamol" 
    "\n💊 Cimegripe" 
    "\n💊 Dorflex"
    "\n💊 Ibuprofeno" 
    "\n💊 Meleato de Enalapril" 
    "\n💊 Neosoro" 
    "\n💊 Nimesulida" 
    "\n💊 Tadalafila" 
    "\n💊 Omeprazol" 
    "\n\nQual a sua dúvida sobre algum deles?" 
    "\nDúvida: ").strip()

    pergunta_reescrita = reescrever_pergunta(pergunta)

    def detectar_medicamento(pergunta):
        p = pergunta.lower()

        medicamentos = [
            "dipirona",
            "paracetamol",
            "ibuprofeno",
            "dorflex",
            "cimegripe",
            "enalapril",
            "neosoro",
            "nimesulida",
            "tadalafila",
            "omeprazol"
        ]

        for med in medicamentos:
            if med in p:
                return f"bula_{med}"

        return None

    def detectar_categoria(pergunta):
        p = pergunta.lower()

        if "posologia" in p or "como tomar" in p or "dose" in p:
            return "posologia_modo_uso"

        elif "efeito" in p or "reação" in p or "males" in p:
            return "reacoes_adversas"

        elif "serve" in p or "indicação" in p:
            return "indicacao"

        return None

    categoria = detectar_categoria(pergunta_reescrita)
    medicamento = detectar_medicamento(pergunta_reescrita)

    filtros_lista = []

    if categoria:
        filtros_lista.append({"Categoria": categoria})

    if medicamento:
        filtros_lista.append({"medicamento": medicamento})

    if len(filtros_lista) > 1:
        filtro = {"$and": filtros_lista}
    elif len(filtros_lista) == 1:
        filtro = filtros_lista[0]
    else:
        filtro = {}

    search_kwargs = {"k": 10}

    if filtro:
        search_kwargs["filter"] = filtro

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    qa_chain = create_retrieval_chain(retriever, document_chain)
    resposta = qa_chain.invoke({"input": pergunta_reescrita})

    docs = resposta["context"]

    for doc in docs:
        print("\n========================")
        print(doc.page_content)

    if len(docs) < 3:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"medicamento": medicamento} if medicamento else {}
            }
        )

        qa_chain = create_retrieval_chain(retriever, document_chain)
        resposta = qa_chain.invoke({"input": pergunta_reescrita})
        docs = resposta["context"]

    if len(docs) < 5:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )

        qa_chain = create_retrieval_chain(retriever, document_chain)
        resposta = qa_chain.invoke({"input": pergunta_reescrita})
        docs = resposta["context"]

    print("\n" + "-"*10 + "PERGUNTA" + "-"*10 + "\n")
    print(pergunta)

    print("\n" + "-"*10 + "RESPOSTA DO BOT FARMACÊUTICO" + "-"*10 + "\n")
    print(resposta["answer"])
    docs = resposta["context"]

    print("\n" + "-"*10 + "TRECHO RESGATADOS" + "-"*10 + "\n")

    vistos = set()

    for i, doc in enumerate(docs, start=1):
        med_nome = doc.metadata.get('medicamento', 'N/A')
        categoria_doc = doc.metadata.get('Categoria', 'N/A')
        documento = doc.metadata.get('source', 'Documento Desconhecido')
        pagina = doc.metadata.get('page', 'N/A')

        chave = (med_nome, documento, pagina)

        if chave in vistos:
            continue

        vistos.add(chave)

        print(f"---TRECHO {len(vistos)}---")
        print(f"MEDICAMENTO: {med_nome}")
        print(f"CATEGORIA: {categoria_doc}")
        print(f"DOCUMENTO: {documento}")
        print(f"PAGINA: {pagina}")

    continuar = input("\nDeseja fazer outra pergunta ao Bot Farmacêutico? \n(sim/não): ").strip().lower()

    if continuar != "sim":
        print("🤖 Adeus!")
        break