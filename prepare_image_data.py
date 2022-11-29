from PIL import Image
import glob

# def test():
#     img_path = "/home/timothy/Documents/Aicore/Airbnb-Property-Modelling/images/0a26e526-1adf-4a2a-888d-a05f7f0a2f33/0a26e526-1adf-4a2a-888d-a05f7f0a2f33-a.png"
#     img = Image.open(img_path)
#     print(img.format)
#     print(img.size)
#     print(img.mode)
#     img.show()

class PrepareImages:

    def __init__(self):
        self.image_dict = {}
        self.height_list = []
        self.minimum_height = 0
        self.resized_image_dict = []

    def __import_images(self):
        for filename in glob.glob("/home/timothy/Documents/Aicore/Airbnb-Property-Modelling/images/*/*.png"):
            print(filename)
            img_name = filename[-42:]
            print(img_name)
            img = Image.open(filename)
            if img.mode == "RGB":
                self.image_dict[img_name] = img
                self.height_list.append(img.height)
        print(f"There are {len(self.image_dict)} RGB images")

    def __find_minimum_image_height(self):
        for img in self.image_dict.values():
            self.height_list.append(img.height)
        self.minimum_height = min(self.height_list)
        print(f"Minimum height: {self.minimum_height}")

    def __resize_images(self):
        for img in self.image_dict.values():
            aspect_ratio = img.width / img.height
            new_width = int(self.minimum_height * aspect_ratio)
            new_size = (new_width, self.minimum_height)
            resized_image = img.resize(new_size)
            self.resized_image_dict.append(resized_image)

    def __install_resized_images(self):
        for img in self.resized_image_dict:
            pass
            
    def do_whole_resize(self):
        PrepareImages.__import_images(self)
        PrepareImages.__find_minimum_image_height(self)
        PrepareImages.__resize_images(self)
        # PrepareImages.__install_resized_images(self)

if __name__ == '__main__':
    resize = PrepareImages()
    resize.do_whole_resize()