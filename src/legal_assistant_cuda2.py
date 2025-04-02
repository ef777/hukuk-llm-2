import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch  # CUDA kontrolü için

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
    
    def split_documents(self, chunk_size: int = 600, chunk_overlap: int = 100) -> List:
        """Belgeleri parçalara böler - VRAM optimizasyonu için daha küçük parçalar"""
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
    
    def __init__(self, cache_dir: str = "./cache", use_gpu: bool = True):
        self.cache_dir = cache_dir
        self.embedding_model = None
        self.vector_store = None
        self.use_gpu = use_gpu
        
        # Önbellek dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "faiss_index"), exist_ok=True)
    
    def initialize_embeddings(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Embedding modelini başlatır - GPU kullanımını etkinleştir"""
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        
        # CUDA ile embedding modelini yükle
        device = 'cuda' if self.use_gpu else 'cpu'
        logger.info(f"Embedding modeli {device} üzerinde çalışacak")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
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
            # Belgeleri işlemek için batch_size kullanarak VRAM kullanımını sınırla
            batch_size = 64  # GTX 1050Ti için optimize edilmiş değer
            
            # Belgeleri daha küçük gruplara böl
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
            logger.info(f"Toplam {len(batches)} batch işlenecek")
            
            # İlk batch ile vektör database oluştur
            self.vector_store = FAISS.from_documents(
                batches[0], 
                self.embedding_model
            )
            
            # Kalan batchleri ekle
            for i, batch in enumerate(batches[1:], 1):
                logger.info(f"Batch {i}/{len(batches)-1} işleniyor...")
                self.vector_store.add_documents(batch)
            
            # Veritabanını kaydet
            self.vector_store.save_local(index_path)
            logger.info("Vektör veritabanı oluşturuldu ve kaydedildi.")
        
        return self.vector_store
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """Vektör veritabanından bir retriever döndürür"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # Bellek kullanımını azaltmak için daha az belge getir
            
        if self.vector_store is None:
            raise ValueError("Vektör veritabanı henüz oluşturulmadı.")
            
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

class LLMManager:
    """LLM modellerini yöneten sınıf"""
    
    def __init__(self, use_gpu: bool = True):
        self.main_model = None   # Tek model kullanacağız
        self.qa_chain = None     # Soru-cevap zinciri
        self.use_gpu = use_gpu
    
    def initialize_main_model(self, model_path: str) -> Any:
        """Ana LLM modelini yükler - CUDA desteği ile"""
        logger.info(f"Model yükleniyor: {model_path}")
        
        n_gpu_layers = -1 if self.use_gpu else 0  # -1: mümkün olan tüm katmanları GPU'ya yükle
        n_batch = 2048  # 1050Ti için optimize edilmiş değer
        
        self.main_model = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            repeat_penalty=1.2, 
            max_tokens=500,  # Daha kısa cevaplar için
            top_p=0.6,
            n_ctx=4072,  # Daha az bellek kullanımı için
            n_gpu_layers=n_gpu_layers,  # CUDA desteğini etkinleştir
            n_batch=n_batch,  # Batch büyüklüğünü belirle
           # f16_kv=True,  # Yarı kesinlikli KV önbelleğini etkinleştirerek VRAM kullanımını azalt
            callbacks=[StreamingStdOutCallbackHandler()],
            verbose=False
        )
        
        return self.main_model
    
    def create_query_analyzer(self):
        """Sorgu analizi için LLM'yi doğrudan kullan (ayrı bir model yok)"""
        if self.main_model is None:
            raise ValueError("Önce modeli başlatın.")
            
        # Model çıktıları Pydantic ile düzgün parse edilemiyor, manuel analiz kullan
        template = """
        Kullanıcının hukuki sorusunu analiz et:
        
        Kullanıcı sorgusu: {input}
        
        Lütfen aşağıdaki formatta çıktı ver,herhangi ek bir şey ekleme bir şey söyleme sadece senden istediğim çıktı şeklidne ver ve tamamla:
        
        Ana Konu: [Ana konu]
        Spesifik Soru: [Sorgu detayı]
        Anahtar Kelimeler: [kelime1, kelime2, kelime3]
        Belge Türleri: [tür1, tür2]
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"]
        )
        
        # Basit zincir
        analyzer_chain = (
            {"input": RunnablePassthrough()} 
            | prompt 
            | self.main_model 
            | StrOutputParser()
        )
        
        return analyzer_chain
    
    def create_qa_chain(self, retriever: Any) -> Any:
        """Soru-cevap zinciri oluşturur"""
        if self.main_model is None:
            raise ValueError("Önce modeli başlatın.")
            
        # Prompt şablonu - kısa ve öz
        template = """
        Sen bir hukuk asistanısın. Aşağıdaki bilgilere dayanarak kullanıcının sorusuna kısa ve öz bir cevap ver.
        SADECE verilen belgelerden bilgi kullan. Eğer cevabı bilmiyorsan, bilmediğini belirt. hukuk dışıdna bir konu varsa bilmiyorum de.
        ÖNEMLİ: Tekrar yapma. Aynı bilgiyi veya cümleyi tekrarlama. Her bir fikri sadece bir kere ifade et.
        Cevabın kısa, net ve tekrarsız olmalı.
        BELGELER:
        {context}
        
        SORU:
        {input}
        
        CEVAP:
        """
        
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
    """Belge arama motoru - Optimize edilmiş"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.vector_store
    
    def keyword_search(self, query: str, documents: List, top_k: int = 5) -> List:
        """Anahtar kelime tabanlı arama - Daha az belge döndür"""
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
    
    def semantic_search(self, query: str, top_k: int = 4) -> List:
        """Semantik arama - Daha az belge döndür"""
        logger.info(f"Semantik arama: {query}")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)
    
    def hybrid_search(self, query: str, documents: List, top_k: int = 5) -> List:
        """Karma arama - Optimize edilmiş"""
        logger.info(f"Karma arama: {query}")
        
        # Anahtar kelime araması
        keyword_results = self.keyword_search(query, documents, top_k)
        # Semantik arama
        semantic_results = self.semantic_search(query, top_k)
        
        # Sonuçları birleştir
        combined_results = []
        doc_sources = set()
        
        # Önce semantik sonuçları ekle
        for doc in semantic_results:
            source = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}"
            if source not in doc_sources:
                combined_results.append(doc)
                doc_sources.add(source)
        
        # Sonra anahtar kelime sonuçlarını ekle
        for doc in keyword_results:
            source = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}"
            if source not in doc_sources:
                combined_results.append(doc)
                doc_sources.add(source)
                
                # Top_k sınırına ulaşıldıysa dur
                if len(combined_results) >= top_k:
                    break
        
        return combined_results
    
    def filtered_search(self, query: str, filters: Dict, top_k: int = 5) -> List:
        """Filtrelere göre arama yapar"""
        logger.info(f"Filtreli arama: {query}, filtreler: {filters}")
        
        # Filtreleme parametreleri
        filter_criteria = {
            "$and": []
        }
        
        for key, value in filters.items():
            if isinstance(value, list):
                or_conditions = []
                for v in value:
                    or_conditions.append({key: v})
                if or_conditions:
                    filter_criteria["$and"].append({"$or": or_conditions})
            else:
                filter_criteria["$and"].append({key: value})
        
        if not filter_criteria["$and"]:
            return self.semantic_search(query, top_k)
        
        # Filtreleri uygula
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": filter_criteria
            }
        )
        
        return retriever.invoke(query)

class LegalAssistant:
    """Ana hukuki asistan sınıfı - CUDA Optimizasyonlu"""
    
    def __init__(self, pdf_directory: str, models_directory: str, cache_dir: str = "./cache", use_gpu: bool = True):
        self.pdf_directory = pdf_directory
        self.models_directory = models_directory
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        
        # CUDA kullanım bilgisini göster
        if self.use_gpu:
            import torch
            logger.info(f"CUDA kullanılabilir: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Alt bileşenleri başlat
        self.document_processor = DocumentProcessor(pdf_directory, cache_dir)
        self.vector_store_manager = VectorStoreManager(cache_dir, use_gpu=use_gpu)
        self.llm_manager = LLMManager(use_gpu=use_gpu)
        
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
        """Sistemi başlatır - VRAM kullanımı optimize edilmiş"""
        logger.info("Sistem başlatılıyor...")
        
        # 1. Belgeleri yükle
        self.documents = self.document_processor.load_documents(force_reload)
        
        # 2. Belgeleri parçalara böl - daha küçük parçalar
        self.chunks = self.document_processor.split_documents()
        
        # 3. Embedding modelini başlat - CUDA ile
        self.vector_store_manager.initialize_embeddings()
        
        # 4. Vektör veritabanını oluştur - batched işleme ile
        self.vector_store_manager.create_vector_store(self.chunks, force_reload)
        
        # 5. Arama motorunu başlat
        self.search_engine = SearchEngine(self.vector_store_manager)
        
        # 6. LLM modelini yükle - Tek model kullanacağız
        model_path = os.path.join(self.models_directory, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
       
       # model_path = os.path.join(self.models_directory, " DeepSeek-R1-Distill-Qwen-7B-IQ4_XS.gguf")

        self.llm_manager.initialize_main_model(model_path)
        
        # 7. Sorgu analiz zincirini oluştur - ana model üzerinden
        self.query_analyzer = self.llm_manager.create_query_analyzer()
        
        # 8. QA zincirini oluştur - daha az belge getiren retriever ile
        retriever = self.vector_store_manager.get_retriever()
        self.llm_manager.create_qa_chain(retriever)
        
        logger.info("Sistem başlatıldı ve kullanıma hazır.")
    
    

    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """Kullanıcı sorgusunu analiz eder - Basitleştirilmiş"""
        if not self.query_analyzer:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        logger.info(f"Sorgu analiz ediliyor: {query}")
        
        try:
            # Ana model ile analiz
            result_text = self.query_analyzer.invoke(query)
            #logger.debug(f"Ham analiz sonucu: {result_text}")

            # Manuel olarak sonucu analiz et
            main_topic = "İş Hukuku"  # Varsayılan değer
            specific_question = query
            keywords = []
            document_types = ["kanun", "yönetmelik"]  # Varsayılan değer
            
            # Metin analizi
            for line in result_text.strip().split("\n"):
                if "Ana Konu:" in line:
                    main_topic = line.split("Ana Konu:")[1].strip()
                elif "Spesifik Soru:" in line:
                    specific_question = line.split("Spesifik Soru:")[1].strip()
                elif "Anahtar Kelimeler:" in line:
                    keywords_text = line.split("Anahtar Kelimeler:")[1].strip()
                    keywords = [k.strip() for k in keywords_text.replace("[", "").replace("]", "").split(",")]
                elif "Belge Türleri:" in line:
                    types_text = line.split("Belge Türleri:")[1].strip()
                    document_types = [t.strip() for t in types_text.replace("[", "").replace("]", "").split(",")]
            
            # Eğer anahtar kelime bulunamadıysa, sorgudan çıkar
            if not keywords:
                keywords = [w for w in query.split() if len(w) > 2]
            
            return QueryAnalysisResult(
                main_topic=main_topic,
                specific_question=specific_question,
                keywords=keywords,
                document_types=document_types
            )
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
        """Sorguyla ilgili belgeleri getirir - Daha az belge getiren optimizasyon"""
        if not self.search_engine:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        # Filtreler oluştur
        filters = {}
        if query_analysis.document_types:
            filters["document_type"] = query_analysis.document_types
            
        # Önce filtreli arama dene
        if filters:
            try:
                docs = self.search_engine.filtered_search(query, filters, top_k=3)
                if docs:
                    return docs, "filtered"
            except Exception as e:
                logger.warning(f"Filtreli arama başarısız: {str(e)}")
        
        # Karma arama yap
        try:
            docs = self.search_engine.hybrid_search(query, self.chunks , top_k=3)
            if docs:
                return docs, "hybrid"
        except Exception as e:
            logger.warning(f"Karma arama başarısız: {str(e)}")
            
        # Son çare olarak semantik arama
        docs = self.search_engine.semantic_search(query , top_k=2)
        return docs, "semantic"
    
    def _calculate_confidence(self, query: str, answer: str, relevant_docs: List) -> float:
        """Yanıt için güven skoru hesaplar - Basitleştirilmiş"""
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
        """Cevap oluşturur - Daha az bellek kullanımı için optimizasyon"""
        if not self.llm_manager.qa_chain:
            raise ValueError("Sistem henüz başlatılmadı.")
            
        logger.info("Cevap oluşturuluyor...")
        
        try:
            # QA zincirine sorguyu ve belgeleri gönder
            result = self.llm_manager.qa_chain.invoke({
                "input": query,
                "context": relevant_docs  # Direkt belgeleri geçirebiliriz
            })
            
            # Zincirin çıktısını işle
            answer = result.get("answer", "")
            if not answer and "result" in result:
                answer = result["result"]
            
            # Belge kaynaklarını çıkar
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
    
    parser = argparse.ArgumentParser(description='CUDA Optimizasyonlu Hukuki Asistan')
    parser.add_argument('--pdf_dir', type=str, default='./hukuk_belgeleri', help='PDF belgelerinin bulunduğu dizin')
    parser.add_argument('--models_dir', type=str, default='./models', help='LLM modellerinin bulunduğu dizin')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Önbellek dizini')
    parser.add_argument('--force_reload', action='store_true', help='Önbelleği yeniden oluşturma')
    parser.add_argument('--no_gpu', action='store_true', help='GPU kullanımını devre dışı bırak')
    
    args = parser.parse_args()
    
    # GPU kullanılabilirliğini kontrol et
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    if use_gpu:
        print(f"\nGPU Bilgileri:")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Sürümü: {torch.version.cuda}")
    else:
        print("\nGPU kullanılamıyor veya devre dışı bırakıldı. CPU kullanılacak.")
    
    # Asistanı başlat
    assistant = LegalAssistant(
        pdf_directory=args.pdf_dir,
        models_directory=args.models_dir,
        cache_dir=args.cache_dir,
        use_gpu=use_gpu
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