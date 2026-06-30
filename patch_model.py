import zipfile
import json
import os
import shutil
import tempfile

def patch_keras_model(input_path, output_path):
    print(f"Membaca {input_path}...")
    
    # Buat temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Ekstrak semua file
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        config_path = os.path.join(temp_dir, 'config.json')
        if not os.path.exists(config_path):
            print("❌ config.json tidak ditemukan di dalam model.")
            return False
            
        # Baca config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            
        config_dict = json.loads(config_str)
        
        # Fungsi rekursif untuk menghapus quantization_config
        def remove_quant_config(d):
            if isinstance(d, dict):
                if 'quantization_config' in d:
                    del d['quantization_config']
                for k, v in list(d.items()):
                    remove_quant_config(v)
            elif isinstance(d, list):
                for item in d:
                    remove_quant_config(item)
                    
        remove_quant_config(config_dict)
        
        # Simpan kembali config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f)
            
        # Zip kembali
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_ref.write(file_path, arcname)
                    
        print(f"✅ Berhasil! Model baru disimpan sebagai: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat patching: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "lstm_model_last_fold.keras")
    output_file = os.path.join(BASE_DIR, "lstm_model_last_fold_patched.keras")
    
    if os.path.exists(input_file):
        patch_keras_model(input_file, output_file)
    else:
        print(f"❌ File tidak ditemukan: {input_file}")
