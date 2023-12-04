from pdf2image import convert_from_path

images = convert_from_path('/root/workspace/ex_data/Sharp_20231127_101629.pdf')
for i, image in enumerate(images):
    fname = "/root/workspace/ex_data/result/image" + str(i) + ".png"
    image.save(fname, "PNG")
