"""Install the experiment dependencies before importing Transformers in a new Colab VM."""
import subprocess,sys,os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
subprocess.run([sys.executable,'-m','pip','install','transformers==4.57.1','numpy==2.4.6','Pillow==11.3.0','matplotlib==3.10.0','requests==2.32.4'],check=True)
print('SETUP_DONE: if Transformers was already imported, restart the kernel before inference.')
