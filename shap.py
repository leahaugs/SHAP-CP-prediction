import time
import numpy as np
from torch.autograd import Variable
import shap

def create_shap_explainer(model, background_data):
    """
    Create SHAP explainer object.
    
    Args:
        model: Model
            Model for SHAP explainer object
        background_data: Tensor
            Background data for SHAP explainer object
    Returns:
        SHAP explainer object
    """

    # create explainer object
    start_explainer = time.time()
    explainer = shap.GradientExplainer(model.cpu(), background_data)
    end_explainer = time.time()
    total_explainer = end_explainer - start_explainer
    print("Creating explainer: " + str(total_explainer) + " seconds in total")

    return explainer

def generate_shap_values(explainer, data):
    """
    Generate SHAP values.
    
    Args:
        explainer: Explainer
            SHAP explainer object
        data: Tensor
            data to generate SHAP values for"
    Returns:
        SHAP values
    """

    # CPU
    data = Variable(
        data.float(),
        requires_grad=True)

    start_time = time.time()
    shap_values = np.array(explainer.shap_values(data.cpu()))
    end_time = time.time()
    total_time = end_time - start_time

    print("Generating SHAP values " + str(total_time) + " seconds in total")

    xai_values = shap_values.tolist()

    return xai_values