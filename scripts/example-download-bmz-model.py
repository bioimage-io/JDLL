from io.bioimage.modelrunner.bioimageio import BioimageioRepo


bmzModelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
print("start")
br = BioimageioRepo.connect()
print("connected")
modelDir = br.downloadByName(bmzModelName, r'C:\Users\angel\OneDrive\Documentos\deepimagej\fiji-win64-1\Fiji.app\models')

print(modelDir)