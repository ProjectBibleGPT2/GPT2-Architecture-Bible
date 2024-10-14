def clean(input, output, version):
    bbl = open(input, "r", encoding="utf-8")
    bbl_text = bbl.read()
    bbl.close()
    bbl_list = bbl_text.split("\n")
    new_bbl = []
    for x in bbl_list:
        if x != "":
            new_bbl.append(f"In the Bible version {version}, {x.split('	')[0]} states that '"+x.split("	")[1].replace("[","").replace("]","")+"'\n")
    open(output, "w", encoding="utf-8").writelines(new_bbl)

