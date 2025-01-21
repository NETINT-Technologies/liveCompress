# Instructions for downloading the Kodak Dataset

The kodak dataset is used in the `testing` folder for evaluation of previously trained models using their respective checkpoint file. To get this dataset, open up a new bash terminal, cd into the `kodak` folder, and run the following command, if using a linux or mac machine:

```
for i in $(seq -w 1 24); do wget -nc "http://r0k.us/graphics/kodak/kodak/kodim${i}.png" 2>/dev/null || echo "Failed to download kodim${i}.png"; done
```

If using a windows machine, run the following powershell command instead:

```powershell
1..24 | ForEach-Object { $num = $_.ToString('00'); Invoke-WebRequest -Uri "http://r0k.us/graphics/kodak/kodak/kodim$num.png" -OutFile "kodim$num.png" }
```