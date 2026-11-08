from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path('/content/visionai_week10')
with ZipFile('/content/week10-results.zip','w',ZIP_DEFLATED) as z:
    for folder in ['results','figures']:
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.name!='image_features.npy':z.write(p,p.relative_to(root))
    z.write(root/'data/manifest.json','data/manifest.json')
print('ARCHIVED',Path('/content/week10-results.zip').stat().st_size)
