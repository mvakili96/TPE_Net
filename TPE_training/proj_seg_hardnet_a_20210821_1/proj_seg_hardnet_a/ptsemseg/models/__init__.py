import copy
import torchvision.models as models


from ptsemseg.models.hardnet import hardnet
from ptsemseg.models.rpnet_c import rpnet_c



########################################################################################################################
### get_model
########################################################################################################################
def get_model(model_dict, n_classes, version=None):
    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes=n_classes, **param_dict)

    return model
#end


########################################################################################################################
### _get_model_instance()
########################################################################################################################
def _get_model_instance(name):
    try:
        return {
            "hardnet": hardnet,
            "rpnet_c": rpnet_c
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################

