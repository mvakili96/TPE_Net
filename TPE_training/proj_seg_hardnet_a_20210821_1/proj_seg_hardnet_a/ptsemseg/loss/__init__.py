import logging
import functools

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    dice_ce_loss,
    CrossEntropyLoss2d_Weighted,
    PolyLaneNet_loss,
)


logger = logging.getLogger("ptsemseg")

key2loss_TPE = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "dice_ce_loss": dice_ce_loss,
    "ce_weighted": CrossEntropyLoss2d_Weighted,

}

key2loss_Poly = {
    "loss_poly": PolyLaneNet_loss,
}


def get_loss_function_TPE(cfg):
    if cfg["training"]["loss_TPE"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg["training"]["loss_TPE"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss_TPE:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss_TPE[loss_name], **loss_params)


def get_loss_function_Poly(cfg):
    if cfg["training"]["loss_Poly"] is None:
        logger.info("Using default cross entropy loss")
        return PolyLaneNet_loss

    else:
        loss_dict = cfg["training"]["loss_Poly"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss_Poly:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss_Poly[loss_name], **loss_params)
