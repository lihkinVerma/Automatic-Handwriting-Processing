def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/emnist-byclass-train-images-idx3-ubyte", "C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/emnist-byclass-train-labels-idx1-ubyte",
        "C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/mnist_train.csv", 697932)
convert("C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/emnist-byclass-test-images-idx3-ubyte", "C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/emnist-byclass-test-labels-idx1-ubyte",
        "C:/Users/Nikhil/Desktop/finalized/3. ML-dataset-Conversion/mnist_test.csv", 116323)