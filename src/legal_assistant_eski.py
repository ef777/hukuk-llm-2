import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# PDF ve Doküman İşleme
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document

# Vektör Veritabanı ve Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM ve Zincirler - Yeni API importları
from langchain_community.llms import LlamaCpp
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Metrikler ve Değerlendirme
from sentence_transformers.util import cos_sim

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LegalAssistant")

# Document serileştirme/deserileştirme yardımcı fonksiyonları
def document_to_dict(doc):
    """Document nesnesini sözlüğe dönüştürür"""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def dict_to_document(doc_dict):
    """Sözlüğü Document nesnesine dönüştürür"""
    return Document(
        page_content=doc_dict["page_content"],
        metadata=doc_dict["metadata"]
    )

@dataclass
class DocumentMetadata:
    """Belge metadatasını tutan veri sınıfı"""
    source: str
    page: int
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = None

class QueryAnalysisResult(BaseModel):
    """Kullanıcı sorgusu analiz sonucu"""
    main_topic: str = Field(description="Sorgunun ana konusu")
    specific_question: str = Field(description="Sorgunun odaklandığı spesifik soru")
    keywords: List[str] = Field(description="Sorgudaki anahtar kelimeler")
    document_types: List[str] = Field(description="İlgili olabilecek belge türleri", default=[])

class AnswerMetadata(BaseModel):
    """Yanıt metadatasını tutan model"""
    confidence_score: float = Field(description="Yanıtın güven skoru (0-1)")
    source_documents: List[str] = Field(description="Yanıtın kaynaklandığı belgeler")
    retrieval_method: str = Field(description="Kullanılan getirme yöntemi (hybrid, semantic, keyword)")
    
class DocumentProcessor:
    """PDF belgelerini işleyen ve indeksleyen sınıf"""
    
    def __init__(self, documents_dir: str, cache_dir: str = "./cache"):
        self.documents_dir = documents_dir
        self.cache_dir = cache_dir
        self.documents = []
        self.chunks = []
        self.metadata = {}
        
        # Önbellek dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_documents(self, force_reload: bool = False) -> List:
        """Belgeleri yükle ve önbelleğe al"""
        cache_path = os.path.join(self.cache_dir, "documents_cache.json")
        
        # Önbellekten yükle (eğer varsa ve force_reload False ise)
        if os.path.exists(cache_path) and not force_reload:
            logger.info("Belgeler önbellekten yükleniyor...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                self.documents = [dict_to_document(doc_dict) for doc_dict in cached_data['documents']]
                self.metadata = cached_data['metadata']
            logger.info(f"{len(self.documents)} belge önbellekten yüklendi.")
            return self.documents
            
        logger.info("PDF belgeleri dizinden yükleniyor...")
        loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        self.documents = loader.load()
        logger.info(f"{len(self.documents)} belge yüklendi.")
        
        # Belge metadatasını çıkar
        self._extract_metadata()
        
        # Önbelleğe kaydet - Document nesnelerini sözlüklere dönüştür
        document_dicts = [document_to_dict(doc) for doc in self.documents]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': document_dicts,
                'metadata': self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        return self.documents
    
    def _extract_metadata(self):
        """PDF belgelerinden metadata çıkar"""
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            
            if source not in self.metadata:
                # PDF'den başlık, yazar vb. bilgileri çıkarmaya çalış
                try:
                    with open(source, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        info = reader.metadata
                        
                        self.metadata[source] = {
                            'title': info.title if info.title else os.path.basename(source),
                            'author': info.author,
                            'date': info.creation_date.strftime('%Y-%m-%d') if info.creation_date else None,
                            'document_type': self._determine_document_type(source, reader)
                        }
                except Exception as e:
                    logger.warning(f"Metadata çıkarılamadı: {source}, hata: {str(e)}")
                    self.metadata[source] = {
                        'title': os.path.basename(source),
                        'author': None,
                        'date': None,
                        'document_type': None
                    }
    
    def _determine_document_type(self, source: str, pdf_reader: PyPDF2.PdfReader) -> str:
        """Belge türünü belirlemeye çalışır (karar, yönetmelik, kanun vb.)"""
        # İlk sayfadan belge türünü çıkarmaya çalış
        first_page_text = pdf_reader.pages[0].extract_text().lower()
        
        document_types = {
            'karar': ['karar', 'kararı', 'mahkeme kararı'],
            'kanun': ['kanun', 'yasa', 'kanunu'],
            'yönetmelik': ['yönetmelik', 'yönetmeliği'],
            'tebliğ': ['tebliğ', 'genelge'],
            'sözleşme': ['sözleşme', 'akit', 'kontrat'],
            'dilekçe': ['dilekçe', 'başvuru'],
            'rapor': ['rapor', 'bilirkişi raporu']
        }
        
        for doc_type, keywords in document_types.items():
            if any(keyword in first_page_text for keyword in keywords):
                return doc_type
                
        # Dosya adından tahmin et
        filename = os.path.basename(source).lower()
        for doc_type, keywords in document_types.items():
            if any(keyword in filename for keyword in keywords):
                return doc_type
                
        return "bilinmeyen"
    
    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """Belgeleri parçalara böler"""
        logger.info("Belgeler parçalara bölünüyor...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        
        # Metadatayı parçalara ekle
        for chunk in self.chunks:
            source = chunk.metadata.get('source', '')
            if source in self.metadata:
                for key, value in self.metadata[source].items():
                    chunk.metadata[key] = value
        
        logger.info(f"{len(self.chunks)} parça oluşturuldu.")
        return self.chunks

class VectorStoreManager:
    """Vektör veritabanı yöneticisi"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.embedding_model = None
        self.vector_store = None
        
        # Önbellek dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "faiss_index"), exist_ok=True)
    
    def initialize_embeddings(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Embedding modelini başlatır"""
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            cache_folder=os.path.join(self.cache_dir, "embedding_model")
        )
        
        return self.embedding_model
    
    def create_vector_store(self, documents: List, force_recreate: bool = False) -> Any:
        """Vektör veritabanını oluşturur veya yükler"""
        index_path = os.path.join(self.cache_dir, "faiss_index")
        
        # Eğer embedding modeli yoksa, yükle
        if self.embedding_model is None:
            self.initialize_embeddings()
        
        # Eğer vektör store zaten varsa ve yeniden oluşturma istenmemişse, yükle
        if os.path.exists(index_path) and os.listdir(index_path) and not force_recreate:
            logger.info("Vektör veritabanı önbellekten yükleniyor...")
            self.vector_store = FAISS.load_local(
                index_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Vektör veritabanı yüklendi.")
            
        else:
            logger.info("Vektör veritabanı oluşturuluyor...")
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embedding_model
            )
            
            # Veritabanını kaydet
            self.vector_store.save_local(index_path)
            logger.info("Vektör veritabanı oluşturuldu ve kaydedildi.")
        
        return self.vector_store
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """Vektör veritabanından bir retriever döndürür"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        if self.vector_store is None:
            raise ValueError("Vektör veritabanı henüz oluşturulmadı.")
            
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

class LLMManager:
    """LLM modellerini yöneten sınıf"""
    
    def __init__(self):
        self.small_model = None  # Sorgu analizi için küçük model
        self.main_model = None   # Yanıt oluşturma için ana model
        self.query_analyzer_chain = None
        self.qa_chain = None     # Yeni zincir yapısı için
    
    def initialize_small_model(self, model_path: str) -> Any:
        """Sorgu analizi için küçük modeli yükler"""
        logger.info(f"Küçük model yükleniyor: {model_path}")
        
        self.small_model = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=256,
            top_p=0.95,
            callbacks=[StreamingStdOutCallbackHandler()],
            n_ctx=2048,
            verbose=False
        )
        
        return self.small_model
    
    def initialize_main_model(self, model_path: str) -> Any:
        """Ana LLM modelini yükler"""
        logger.info(f"Ana model yükleniyor: {model_path}")
        
        self.main_model = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=2000,
            top_p=0.95,
            callbacks=[StreamingStdOutCallbackHandler()],
            n_ctx=4096,
            verbose=False
        )
        
        return self.main_model
    
    def create_query_analyzer(self) -> Any:
        """Sorgu analizi için zincir oluşturur"""
        if self.small_model is None:
            raise ValueError("Önce küçük modeli başlatın.")
            
        # Çıktı ayrıştırıcı
        parser = PydanticOutputParser(pydantic_object=QueryAnalysisResult)
        
        # Sorgu analizi şablonu
        template = """
        Kullanıcının aşağıdaki hukuki sorusunu analiz et. Sorgunun ana konusunu, spesifik soruyu ve anahtar kelimeleri belirle.
        Ayrıca, bu sorguyla ilgili olabilecek belge türlerini de tahmin et (örn: kanun, yönetmelik, mahkeme kararı, sözleşme vb.).
        
        Kullanıcı sorgusu: {input}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Yeni zincir yapısı - LLMChain yerine RunnablePassthrough kullanarak
        self.query_analyzer_chain = (
            {"input": RunnablePassthrough()} 
            | prompt 
            | self.small_model 
            | parser
        )
        
        return self.query_analyzer_chain
    
    def create_qa_chain(self, retriever: Any) -> Any:
        """Yeni API kullanarak soru-cevap zinciri oluşturur"""
        if self.main_model is None:
            raise ValueError("Önce ana modeli başlatın.")
            
        # Prompt şablonu - yeni API ile tutarlı parametre adları
        template = """
        Sen bir hukuk asistanısın. Aşağıdaki bilgilere dayanarak kullanıcının sorusuna cevap ver.
        
        SADECE verilen belgelerden bilgi kullan. Uydurma bilgi verme veya hallüsinasyon yapma.
        Eğer cevabı bilmiyorsan veya belgelerde yeterli bilgi yoksa, bilmediğini açıkça belirt.
        
        Cevabında, hangi belgeden bilgi aldığını referans olarak göster. Örneğin: [Belge: İş Kanunu, Sayfa: 5]
        
        BELGELER:
        {context}
        
        SORU:
        {input}
        
        CEVAP:
        """
        
        # Şablonu PromptTemplate'e dönüştür
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )
        
        # Belgeleri birleştiren zincir
        document_chain = create_stuff_documents_chain(
            llm=self.main_model,
            prompt=prompt
        )
        
        # Getirme zinciri
        self.qa_chain = create_retrieval_chain(
            retriever=retriever, 
            combine_docs_chain=document_chain
        )
        
        return self.qa_chain

class SearchEngine:
    """Belge arama motoru"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.vector_store
    
    def keyword_search(self, query: str, documents: List, top_k: int = 10) -> List:
        """Anahtar kelime tabanlı arama yapar"""
        logger.info(f"Anahtar kelime araması: {query}")
        
        results = []
        # Sorguyu temizle ve anahtar kelimeleri çıkar
        keywords = re.sub(r'[^\w\s]', ' ', query.lower()).split()
        
        # Belgeleri anahtar kelimelere göre puanla
        for doc in documents:
            content = doc.page_content.lower()
            score = sum(content.count(keyword) for keyword in keywords)
            if score > 0:
                results.append((doc, score))
        
        # Sonuçları puana göre sırala ve en iyi top_k'yi döndür
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in results[:top_k]]
    
    def semantic_search(self, query: str, top_k: int = 5) -> List:
        """Semantik arama yapar"""
        logger.info(f"Semantik arama: {query}")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        # Yeni API kullanımı - get_relevant_documents yerine invoke
        return retriever.invoke(query)
    
    def hybrid_search(self, query: str, documents: List, top_k: int = 7) -> List:
        """Karma arama yapar (anahtar kelime + semantik)"""
        logger.info(f"Karma arama: {query}")
        
        # Anahtar kelime araması (daha fazla sonuç)
        keyword_results = self.keyword_search(query, documents, top_k * 2)
        # Semantik arama
        semantic_results = self.semantic_search(query, top_k)
        
        # Sonuçları birleştir (öncelik semantik sonuçlarda)
        combined_results = []
        doc_sources = set()
        
        # Önce semantik sonuçları ekle
        for doc in semantic_results:
            source = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}"
            if source not in doc_sources:
                combined_results.append(doc)
                doc_sources.add(source)
        
        # Sonra anahtar kelime sonuçlarını ekle (tekrar etmeyenleri)
        for doc in keyword_results:
            source = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}"
            if source not in doc_sources:
                combined_results.append(doc)
                doc_sources.add(source)
                
                # Top_k sınırına ulaşıldıysa dur
                if len(combined_results) >= top_k:
                    break
        
        return combined_results
    
    def filtered_search(self, query: str, filters: Dict, top_k: int = 7) -> List:
        """Filtrelere göre arama yapar"""
        logger.info(f"Filtreli arama: {query}, filtreler: {filters}")
        
        # Metadataya göre filtreleme parametreleri oluştur
        filter_criteria = {
            "$and": []
        }
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Birden fazla değer varsa $or operatörü kullan
                or_conditions = []
                for v in value:
                    or_conditions.append({key: v})
                if or_conditions:
                    filter_criteria["$and"].append({"$or": or_conditions})
            else:
                # Tek değer varsa doğrudan ekle
                filter_criteria["$and"].append({key: value})
        
        # Eğer filtre kriterleri boşsa, filtresiz arama yap
        if not filter_criteria["$and"]:
            return self.semantic_search(query, top_k)
        
        # Filtreleri uygula
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": filter_criteria
            }
        )
        
        # Yeni API kullanımı - get_relevant_documents yerine invoke
        return retriever.invoke(query)

class LegalAssistant:
    """Ana hukuki asistan sınıfı"""
    
    def __init__(self, pdf_directory: str, models_directory: str, cache_dir: str = "./cache"):
        self.pdf_directory = pdf_directory
        self.models_directory = models_directory
        self.cache_dir = cache_dir
        
        # Alt bileşenleri başlat
        self.document_processor = DocumentProcessor(pdf_directory, cache_dir)
        self.vector_store_manager = VectorStoreManager(cache_dir)
        self.llm_manager = LLMManager()
        
        # Başlangıç durumları
        self.documents = []
        self.chunks = []
        self.search_engine = None
        self.query_analyzer = None
        
        # İlk yükleme gecikmesini önlemek için önbelleği denetle
        self._check_and_warm_cache()
    
    def _check_and_warm_cache(self):
        """Önbelleği kontrol eder ve gerekirse hazırlar"""
        cache_path = os.path.join(self.cache_dir, "documents_cache.json")
        index_path = os.path.join(self.cache_dir, "faiss_index")
        
        # Belge önbelleği varsa ve vektör veritabanı da varsa, hazır demektir
        if os.path.exists(cache_path) and os.path.exists(index_path) and os.listdir(index_path):
            logger.info("Sistem önbelleği hazır.")
            return True
        
        # Önbellek yoksa, başlangıç ayarlarını yap
        logger.info("Önbellek bulunamadı. İlk çalıştırma sırasında sistem hazırlanacak.")
        return False
    
    def initialize(self, force_reload: bool = False):
        """Sistemi başlatır - belgeleri yükler, vektör veritabanını oluşturur, modelleri yükler"""
        logger.info("Sistem başlatılıyor...")
        
        # 1. Belgeleri yükle
        self.documents = self.document_processor.load_documents(force_reload)
        
        # 2. Belgeleri parçalara böl
        self.chunks = self.document_processor.split_documents()
        
        # 3. Embedding modelini başlat
        self.vector_store_manager.initialize_embeddings()
        
        # 4. Vektör veritabanını oluştur
        self.vector_store_manager.create_vector_store(self.chunks, force_reload)
        
        # 5. Arama motorunu başlat
        self.search_engine = SearchEngine(self.vector_store_manager)
        
        # 6. LLM modellerini yükle
      #  small_model_path = os.path.join(self.models_directory, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        main_model_path = os.path.join(self.models_directory, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        
        
        small_model_path = os.path.join(self.models_directory, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
       # small_model_path = os.path.join(self.models_directory, "phi-2.Q4_K_M.gguf")
        #main_model_path = os.path.join(self.models_directory, "DeepSeek-R1-Distill-Qwen-7B-IQ4_XS.gguf")
        
        self.llm_manager.initialize_small_model(small_model_path)
        self.llm_manager.initialize_main_model(main_model_path)
        
        # 7. Sorgu analiz zincirini oluştur
        self.query_analyzer = self.llm_manager.create_query_analyzer()
        
        # 8. QA zincirini oluştur
        retriever = self.vector_store_manager.get_retriever()
        self.llm_manager.create_qa_chain(retriever)
        
        logger.info("Sistem başlatıldı ve kullanıma hazır.")
    
    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """Kullanıcı sorgusunu analiz eder"""
        if not self.query_analyzer:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        logger.info(f"Sorgu analiz ediliyor: {query}")
        
        try:
            # Yeni API - artık query parametresi kullanmıyoruz
            return self.query_analyzer.invoke(query)
        except Exception as e:
            logger.warning(f"Sorgu analizi hatalı: {str(e)}")
            
            # Hata durumunda basit bir analiz yap
            is_hukuku_keywords = ["iş", "işçi", "işveren", "kıdem", "tazminat", "ihbar", 
                                 "çalışan", "maaş", "ücret", "izin", "fesih", "sözleşme"]
            
            # Hangi anahtar kelimeler sorgu içinde var
            found_keywords = [kw for kw in is_hukuku_keywords if kw.lower() in query.lower()]
            
            # Belge türlerini tahmin et
            doc_types = ["kanun", "yönetmelik"]
            if "fesih" in query.lower() or "tazminat" in query.lower():
                doc_types.append("mahkeme kararı")
            
            # Manuel QueryAnalysisResult oluştur
            return QueryAnalysisResult(
                main_topic="İş Hukuku",
                specific_question=query,
                keywords=found_keywords if found_keywords else [query.split()[0]],
                document_types=doc_types
            )
    
    def retrieve_relevant_documents(self, query: str, query_analysis: QueryAnalysisResult) -> Tuple[List, str]:
        """Sorguyla ilgili belgeleri getirir"""
        if not self.search_engine:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        # Filtreler oluştur
        filters = {}
        if query_analysis.document_types:
            filters["document_type"] = query_analysis.document_types
            
        # Önce filtreli arama dene
        if filters:
            try:
                docs = self.search_engine.filtered_search(query, filters)
                if docs:
                    return docs, "filtered"
            except Exception as e:
                logger.warning(f"Filtreli arama başarısız: {str(e)}")
        
        # Karma arama yap
        try:
            docs = self.search_engine.hybrid_search(query, self.chunks)
            if docs:
                return docs, "hybrid"
        except Exception as e:
            logger.warning(f"Karma arama başarısız: {str(e)}")
            
        # Son çare olarak semantik arama
        docs = self.search_engine.semantic_search(query)
        return docs, "semantic"
    
    def _calculate_confidence(self, query: str, answer: str, relevant_docs: List) -> float:
        """Yanıt için güven skoru hesaplar"""
        # Basit bir yöntem: Yanıttaki bilgilerin belgelerde ne kadar var olduğunu kontrol et
        if not answer or not relevant_docs:
            return 0.0
            
        # Cevabın embedding'ini al
        answer_embedding = self.vector_store_manager.embedding_model.embed_query(answer)
        
        # İlgili belgelerin embedding'lerini al ve benzerlik ortalamasını hesapla
        similarities = []
        for doc in relevant_docs:
            doc_embedding = self.vector_store_manager.embedding_model.embed_query(doc.page_content)
            similarity = cos_sim([answer_embedding], [doc_embedding])[0][0]
            similarities.append(float(similarity))
        
        # Ortalama benzerlik
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Eğer cevap "bilmiyorum" içeriyorsa ve benzerlik düşükse, güven skoru yüksek olmalı
        if "bilmiyorum" in answer.lower() and avg_similarity < 0.3:
            return 0.95
        
        # Benzerlik skoru 0-1 arasında, ortalama benzerliğe göre güven skoru döndür
        return min(1.0, max(0.0, avg_similarity))
    
    def generate_answer(self, query: str, relevant_docs: List) -> Dict:
        """Cevap oluşturur - yeni API ile"""
        if not self.llm_manager.qa_chain:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        logger.info("Cevap oluşturuluyor...")
        
        try:
            # Yeni API ile invoke kullanarak
            result = self.llm_manager.qa_chain.invoke({
                "input": query,
                "context": relevant_docs  # Direkt belgeleri geçirebiliriz
            })
            
            # Zincirin çıktısı farklı yapıda olabilir
            answer = result.get("answer", "")
            if not answer and "result" in result:
                answer = result["result"]
            
            # Belge kaynaklarını çıkar (eğer yoksa, ilgili belgeleri kullan)
            sources = []
            if "source_documents" in result:
                source_docs = result["source_documents"]
            else:
                source_docs = relevant_docs
                
            for doc in source_docs:
                sources.append(
                    f"{doc.metadata.get('title', 'Bilinmeyen Belge')} (Sayfa {doc.metadata.get('page', 0)})"
                )
            
            # Güven skorunu hesapla
            confidence_score = self._calculate_confidence(query, answer, relevant_docs)
            
            return {
                "answer": answer,
                "metadata": {
                    "confidence_score": confidence_score,
                    "source_documents": sources
                }
            }
        except Exception as e:
            logger.error(f"Cevap oluşturma hatası: {str(e)}")
            # Hata durumunda basit bir yanıt döndür
            return {
                "answer": "Üzgünüm, cevap oluştururken bir hata meydana geldi. Sorgunuzla ilgili belgeler bulundu ancak işlenemedi.",
                "metadata": {
                    "confidence_score": 0.0,
                    "source_documents": [
                        f"{doc.metadata.get('title', 'Bilinmeyen Belge')} (Sayfa {doc.metadata.get('page', 0)})"
                        for doc in relevant_docs[:3]
                    ],
                    "error": str(e)
                }
            }
    
    def process_query(self, query: str) -> Dict:
        """Kullanıcı sorgusunu işler - ana işlev akışı"""
        try:
            logger.info(f"Sorgu işleniyor: {query}")
            
            # 1. Sorguyu analiz et
            query_analysis = self.analyze_query(query)
            logger.info(f"Sorgu analizi: {query_analysis}")
            
            # 2. İlgili belgeleri getir
            relevant_docs, retrieval_method = self.retrieve_relevant_documents(query, query_analysis)
            
            if not relevant_docs:
                return {
                    "answer": "Üzgünüm, sorgunuzla ilgili belge bulunamadı.",
                    "metadata": {
                        "confidence_score": 0.0,
                        "source_documents": [],
                        "retrieval_method": "none"
                    }
                }
                
            logger.info(f"{len(relevant_docs)} ilgili belge bulundu, getirme yöntemi: {retrieval_method}")
            
            # 3. Cevap oluştur
            result = self.generate_answer(query, relevant_docs)
            
            # 4. Metadata ekle
            result["metadata"]["retrieval_method"] = retrieval_method
            
            return result
            
        except Exception as e:
            logger.error(f"Sorgu işleme hatası: {str(e)}", exc_info=True)
            return {
                "answer": f"Üzgünüm, sorgunuzu işlerken bir hata oluştu: {str(e)}",
                "metadata": {
                    "confidence_score": 0.0,
                    "source_documents": [],
                    "error": str(e)
                }
            }

# Demo ve test için komut satırı arayüzü
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hukuki Asistan')
    parser.add_argument('--pdf_dir', type=str, default='./hukuk_belgeleri', help='PDF belgelerinin bulunduğu dizin')
    parser.add_argument('--models_dir', type=str, default='./models', help='LLM modellerinin bulunduğu dizin')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Önbellek dizini')
    parser.add_argument('--force_reload', action='store_true', help='Önbelleği yeniden oluşturma')
    
    args = parser.parse_args()
    
    # Asistanı başlat
    assistant = LegalAssistant(
        pdf_directory=args.pdf_dir,
        models_directory=args.models_dir,
        cache_dir=args.cache_dir
    )
    
    # Sistemi başlat
    assistant.initialize(force_reload=args.force_reload)
    
    print("\n" + "="*50)
    print("Hukuki Asistan hazır! Sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'q' veya 'exit' yazın.")
    print("="*50 + "\n")
    
    while True:
        user_query = input("\nSoru: ")
        if user_query.lower() in ['q', 'exit', 'quit']:
            break
            
        # Sorguyu işle
        result = assistant.process_query(user_query)
        
        # Sonuçları görüntüle
        print("\n" + "="*50)
        print("CEVAP:\n")
        print(result["answer"])
        print("\n" + "-"*50)
        print(f"Güven skoru: {result['metadata']['confidence_score']:.2f}")
        print(f"Getirme yöntemi: {result['metadata'].get('retrieval_method', 'bilinmeyen')}")
        
        if result["metadata"]["source_documents"]:
            print("\nKaynaklar:")
            for i, source in enumerate(result["metadata"]["source_documents"], 1):
                print(f"{i}. {source}")
        print("="*50 + "\n")
        
if __name__ == "__main__":
    main()