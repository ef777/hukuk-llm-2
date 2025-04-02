import torch

def check_cuda_availability():
    """CUDA ve GPU kullanılabilirliğini kontrol eder, ayrıntılı bilgi verir"""
    print("\n==== GPU/CUDA Durum Kontrolü ====")
    
    # CUDA kullanılabilirliğini kontrol et
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Kullanılabilir: {cuda_available}")
    
    if cuda_available:
        # GPU sayısı
        device_count = torch.cuda.device_count()
        print(f"Kullanılabilir GPU Sayısı: {device_count}")
        
        # GPU bilgileri
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"CUDA Sürümü: {torch.version.cuda}")
            # CUDA Mimarisi
            props = torch.cuda.get_device_properties(i)
            print(f"CUDA Mimarisi: {props.major}.{props.minor}")
            print(f"Toplam Bellek: {props.total_memory / 1024**3:.2f} GB")
            
        # PyTorch'un hangi cihazı kullandığı
        device = torch.device("cuda" if cuda_available else "cpu")
        print(f"\nPyTorch'un kullanacağı cihaz: {device}")
    else:
        print("\nCUDA kullanılamıyor. Olası nedenler:")
        print("1. NVIDIA GPU sürücüleri düzgün yüklenmemiş olabilir")
        print("2. PyTorch CUDA desteği ile yüklenmemiş olabilir")
        print("3. CUDA Toolkit yüklü değil veya PATH'e eklenmemiş olabilir")
        print("4. GPU'nuz PyTorch'un desteklediği CUDA sürümüyle uyumlu olmayabilir")
        
        print("\nÇözüm Önerileri:")
        print("1. PyTorch'u CUDA desteğiyle yeniden yükleyin:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("2. NVIDIA sürücülerini güncelleyin")
        print("3. CUDA Toolkit'i yükleyin ve PATH'e ekleyin")
        
    print("\n===================================")
    
    return cuda_available

if __name__ == "__main__":
    check_cuda_availability()