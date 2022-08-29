import lime.lime_text as lime_text
import lime_image as lime_image

class ITLIME:
    """
    Image Text Lime CLass
    """
    def __init__(self) -> None:
        self.text_explainer = lime_text.LimeTextExplainer()
        self.image_explainer = lime_image.LimeImageExplainer()
    
    def explain_instance(self,img,sentence):
        pass

    def predict(self,input):
        pass
