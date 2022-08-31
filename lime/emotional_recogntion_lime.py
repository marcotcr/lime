from skimage.io import imread
from lime_text import LimeTextExplainer
from lime_image import LimeImageExplainer

class ITLIME:
    """
    Image Text Lime CLass
    """
    def __init__(self) -> None:
        self.text_explainer = LimeTextExplainer()
        self.image_explainer = LimeImageExplainer()
    
    def explain_instance(self,img,sentence):
        segments = self.image_explainer.get_segments(img)
        data,imgs = self.image_explainer.generate_imgs(img,segments,150)
        print(imgs.shape)
        print(data.shape)

    def predict(self,input):
        pass

if __name__=="__main__":
    image_path = "../../../../../../Stuff/man6.jpg"
    sentenct = "Happy White T-shirt Man"
    label = "positive"
    image_text_lime = ITLIME()
    img = imread(image_path,sentenct)
