bbl = open("Bible_KJV.txt", "r", encoding="utf-8")
bbl_text = bbl.read()
bbl.close()

open("cleared.txt", "w", encoding="utf-8").write(bbl_text.replace("	"," "))
