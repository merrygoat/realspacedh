from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_images(image_directory, file_prefix, file_suffix, num_files, image_dimensions):
    data = np.zeros((num_files, image_dimensions[1], image_dimensions[0]))
    for file in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(file) + file_suffix)
        if tmp_image.mode == "RGB":
            tmp_image = tmp_image.convert(mode='L')
        tmp_array = np.array(tmp_image.copy())
        # Correct for bleaching by averaging the brightness across all images
        if file == 0:
            first_mean = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_mean / tmp_mean)
        data[file] = tmp_array
    return data


def setup_load_images(num_images, image_directory, file_prefix, file_suffix):
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    image_dimension = Image.open(image_directory + file_prefix + '0000' + file_suffix).size

    return num_files, image_dimension


def main(images_to_load, image_directory, file_prefix, file_suffix):

    num_files, image_dimension = setup_load_images(images_to_load, image_directory, file_prefix, file_suffix)
    data = load_images(image_directory, file_prefix, file_suffix, num_files, image_dimension)

    chunk_size = 20

    variance = np.zeros((num_files, 2))


    for first_time in range(num_files):
        for second_time in range(first_time + 1, num_files):
            diff = data[second_time, 0:100, 0:100] - data[first_time, 0:100, 0:100]
            variance[second_time-first_time, 0] += np.var(diff)
            variance[second_time - first_time, 1] += 1
    # normalise
    variance[:, 0] /= variance[:, 1]

    #plt.semilogx(np.arange(1, num_files), variance[1:, 0], marker='x')
    plt.semilogx(np.arange(1, num_files), variance[1:, 0], marker='x')
    plt.show()
    np.savetxt("var.txt", variance)

main(images_to_load=500, image_directory="D:/Confocal/STED/Hard spheres/17-02-02/RITC 23/ii/images/", file_prefix="ii_", file_suffix=".png")
