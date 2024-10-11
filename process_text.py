def clean(input,output):
    bbl = open(input, "r", encoding="utf-8")
    bbl_text = bbl.read()
    bbl.close()

    open(output, "w", encoding="utf-8").write(bbl_text.split("	")[0]+" states that: "+bbl_text.split("	")[1].replace("[","").replace("]",""))


if __name__ == "__main__":
    clean("cleared.txt")
