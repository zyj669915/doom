# doomfly-training

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。

鎶婄湡瀹炴灉铦囧叏鑴戣繛鎺ョ粍锛?*MaleCNS v1.0锛?66,700 绁炵粡鍏?/ 25.58M 绐佽Е**锛夋帴鍒?ViZDoom 涓婄帺 Doom 鐨勫疄楠屼粨搴撱€?
鍩轰簬涓婃父 [nftechie/doomfly](https://github.com/nftechie/doomfly)锛圓lex Wormuth, 2026-09锛夛紝鏈粨搴撳湪鍏跺熀纭€涓婂畬鎴愪簡锛?

1. **杩炴帴瀹¤**锛氬師鐗?澶ц剳鈫掓父鎴忓姩浣?灞備粠鏈帴閫氾紙10 涓湡瀹炶繍鍔ㄨ鍑哄叏 0 鏀剧數锛屾父鎴忛潬宸ョ▼ BCI 鏃佽矾椹卞姩鈥斺€斾綔鑰?`validation.json` 鑷 `biologically_validated: false`锛夈€?
2. **鐗╃悊杩炴帴淇**锛氳皟浠跨湡鍙傛暟锛堟潈閲?scale=1.5銆佽啘鐢典綅鍋忕疆 bias=6銆佹斁鐢甸槇鍊?th=鈭?8锛夛紝璁╃湡瀹炶繍鍔ㄧ缁忓厓 MN9 浠?0 鏀剧數鍒扮ǔ瀹氭斁鐢靛苟椹卞姩寮€鐏?杩愬姩銆?
3. **doomfly-rl 寮忚捀棣?*锛氳涓€涓?PPO 鑰佸笀鐜?Doom 鈫?姣忓抚鎶?娓告垙鐢婚潰 鈫?鏋滆潎鑴?14 涓鍑虹缁忓厓鏀剧數"浣滅壒寰?鈫?璁粌涓€涓鍑哄眰妯′豢鑰佸笀鍔ㄤ綔銆?

## 鐩綍缁撴瀯

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。

```
code/                 鎵€鏈夊疄楠岃剼鏈紙鑰佸笀/钂搁/鏍″噯/璇婃柇/褰曞埗锛?
docs/                 PROGRESS.md 鏉冨▉杩涘害妗ｆ銆丼NAPSHOT.md銆佸師鐗堝弬鑰?
data-reference/        灏忔暟鎹紙manifest/validation/璇勪及 json/璇婃柇缁撴灉锛?
models/               teacher.pt锛圥PO 鑰佸笀锛夈€乻tudent.pt锛堣捀棣忚鍑哄眰锛?
TRAINING_PLAN.md      涓嬩竴姝ヨ缁冪殑瀹屾暣鎻愮ず璇?绛栫暐锛堝彲鐩存帴鍠傜粰 agent锛?
```

## 澶ц剳鏁版嵁鎬庝箞澶嶇幇锛堥噸瑕侊級

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。

澶ц剳鏁版嵁鍒嗕袱閮ㄥ垎锛?

**宸插湪浠撳簱鍐咃紙灏忔枃浠讹紝<100MB锛?*锛歚data-reference/brain-snapshot-v0/`
- annotations.feather銆乶eurotransmitters.feather銆乻ource.lock.json銆乶ormalized/ 涓?catalog/neurons/npy/report

**澶ф枃浠讹紙瓒?GitHub 100MB 闄愬埗锛屼粠 Releases 涓嬭浇锛?*锛?

| 鏂囦欢 | 澶у皬 | 浣滅敤 |
|---|---|---|
| `graph.npz` | 239 MB | **浠跨湡鐩存帴鍔犺浇鐨勮剳鍥?*锛圔rain 绫荤敤瀹冿級 |
| `edges.feather` | 1.0 GB | MaleCNS 鍘熷杩炴帴杈癸紙涓婃父鏋勫缓锛?|
| `normalized/edges.arrow` | 293 MB | 褰掍竴鍖栧悗杈硅〃 |

**鎷夊彇鏂规硶锛堜竴閿級**锛?
```powershell
# 浠庝粨搴撴牴鐩綍璺戯細

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。
.\scripts\fetch_brain.ps1                    # 鎷夊師濮?v0 鑴?

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。
.\scripts\fetch_brain.ps1 -Tag brain-v1      # 鎷夎缁冨悗鐨勬柊鐗堟湰鑴?

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。
```
鑴氭湰浼氳嚜鍔ㄥ缓鐩綍銆佽烦杩囧凡瀛樺湪鐨勬枃浠躲€?

**鐗堟湰绠＄悊绾﹀畾**锛?
- `brain-v0-original`锛氳缁冨墠鍘熷蹇収锛堝凡鍥哄畾锛屼笉鍐嶅彉锛?
- `brain-v1`銆乣brain-v2`鈥︼細姣忔璁粌/澶ф敼鑴戠粨鏋勫悗鍙戞柊 tag锛屾棫鐗堟湰淇濈暀鍙洖婊?
- 鍙戞柊鐗?= 鍦?https://github.com/zyj669915/doom/releases/new 鎷栨枃浠朵笂浼狅紙鍗曟枃浠堕檺 2GB锛?

## 鐜

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。

- Python venv锛歚.venv-neural`锛?*numpy 蹇呴』閿?1.26.4**锛坰cipy 1.10.1 渚濊禆 <1.27锛涜鍖呭墠鍏堟煡渚濊禆锛屽嬁鍗?numpy锛?
- torch 2.x CPU 鐗堛€乶umba銆乵atplotlib 3.11銆乸illow
- 杩愯浠讳綍椤圭洰鑴氭湰鍓嶈 `$env:PYTHONPATH='D:\fly\doomfly'`
- ViZDoom 1.3.0 + freedoom2.wad锛涙灉铦囩帺鐨勬槸鑷缓灏忕珵鎶€鍦?`doom/scenarios/combat_survival.wad`

## 宸查獙璇佺殑鍏抽敭缁撹锛堣瑙?docs/PROGRESS.md锛?

> 新电脑从零训练？直接看 [SETUP.md](./SETUP.md)。

- 鍥哄畾鏉冮噸涓?浼氬姩涓嶄細鐜?鏄墿鐞嗘瀬闄愶細8 杞弬鏁版壂鎻?+ 3 杞棴鐜紝鏈€浣抽厤缃?kills鈮?.3 / 207 tics
- 钂搁 v1锛圥PO 鑰佸笀 30k 姝ワ級锛?0 绉掗暱鏃跺鎶椾笅 distilled kills=2.33 vs baseline=1.0锛?133%锛夛紝瀛樻椿 8-10 绉?vs 3 绉?
- 鑰佸笀閫€鍖栨槸鍑绘潃鐡堕锛歅PO 30k 姝ュ悗 argmax 閫€鍖栦负鍗曚竴鍔ㄤ綔锛岃捀棣忔棤绫充箣鐐?鈫?瑙?TRAINING_PLAN.md 涓嬩竴姝?
