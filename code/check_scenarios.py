import sys
sys.path.insert(0, r'D:\fly\doomfly')
sys.path.insert(0, r'D:\fly\experiment\code')
from pathlib import Path
import vizdoom as vzd

scenarios = ['combat_survival', 'defend_the_center', 'health_gathering', 'deadly_corridor', 'basic']
base = Path(vzd.scenarios_path)
for s in scenarios:
    cfg = base / f'{s}.cfg'
    wad = base / f'{s}.wad'
    print(f'{s}: cfg={cfg.exists()} wad={wad.exists()}')
