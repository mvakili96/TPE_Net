import copy
import torchvision.models as models


from ptsemseg.models.hardnet import hardnet
from ptsemseg.models.rpnet_c import rpnet_c
from ptsemseg.models.comparison_models import DinkNet34
from ptsemseg.models.comparison_models import ERFNet




########################################################################################################################
### get_model
########################################################################################################################
def get_model(model_dict, n_classes_segmentation, n_channels_regression, version=None):
    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes_seg=n_classes_segmentation, n_channels_reg=n_channels_regression, **param_dict)

    return model
#end


########################################################################################################################
### _get_model_instance()
########################################################################################################################
def _get_model_instance(name):
    try:
        return {
            "hardnet": hardnet,
            "rpnet_c": rpnet_c,
            "dlinknet_34": DinkNet34,
            "erfnet": ERFNet
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################

